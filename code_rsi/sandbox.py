"""Fail-closed WSL namespace runtime. The Windows evaluator never imports candidate code."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

MAX_BYTES = 2 * 1024 * 1024
PROJECT = Path(__file__).resolve().parent.parent
RUNTIME = Path(__file__).resolve().parent


class SandboxUnavailable(RuntimeError):
    pass


class SandboxExecutionError(RuntimeError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}
        if "kind" not in self.details:
            status = (self.details.get("finished") or {}).get("status")
            if "broker" in message or "handler" in message:
                kind = "broker_error"
            elif "timeout" in message or status == "timeout":
                kind = "timeout"
            elif "output_limit" in message or status == "output_limit":
                kind = "resource_limit"
            elif not self.details.get("isolation_checks"):
                kind = "isolation_error"
            else:
                kind = "candidate_error"
            self.details["kind"] = kind


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _no_links(path):
    path = Path(path).absolute()
    for part in [path, *path.parents]:
        info = part.lstat()
        if part.is_symlink() or (getattr(info, "st_file_attributes", 0) & 0x400):
            raise ValueError("links_or_reparse_points_forbidden:" + str(part))
    return path.resolve(strict=True)


def _wsl_path(path):
    value = str(Path(path).resolve(strict=True)).replace("\\", "/")
    if not re.fullmatch(r"[A-Za-z]:/[A-Za-z0-9_./-]+", value):
        raise ValueError("runtime_paths_must_be_absolute_ascii_without_shell_metacharacters")
    return "/mnt/" + value[0].lower() + value[2:]


def _json_line(obj):
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    if len(text) > MAX_BYTES:
        raise ValueError("input_frame_too_large")
    return text


class Sandbox:
    def __init__(self, runtime_root=None, distro="Ubuntu-22.04"):
        if runtime_root is not None and Path(runtime_root).resolve() != RUNTIME:
            raise ValueError("unverified_runtime_source_directory")
        if distro != "Ubuntu-22.04":
            raise ValueError("unverified_wsl_distribution")
        self.distro = distro

    def _command(self, *args):
        wsl = shutil.which("wsl.exe") or shutil.which("wsl")
        if os.name != "nt" or not wsl:
            raise SandboxUnavailable("verified_Windows_WSL_runtime_required")
        return [wsl, "--distribution", self.distro, "--exec", "/bin/bash", _wsl_path(RUNTIME / "linux_launcher.sh"), *args]

    def run(self, program_dir, corpus_file, question, handler, seconds=180, *, budget=None, mode="solve"):
        if mode not in ("solve", "auxiliary_tests"):
            raise ValueError("invalid_runtime_mode")
        if type(seconds) is not int or not 1 <= seconds <= 180:
            raise ValueError("seconds_must_be_1_to_180")
        if not isinstance(question, str) or not question or len(question.encode("utf-8")) > 256 * 1024:
            raise ValueError("invalid_question")
        if not callable(handler):
            raise ValueError("trusted_service_handler_required")
        program = _no_links(program_dir)
        corpus = _no_links(corpus_file)
        if not program.is_dir() or not corpus.is_file() or corpus.suffix != ".json":
            raise ValueError("invalid_program_or_corpus_path")
        files = []
        for path in sorted(program.rglob("*")):
            _no_links(path)
            if path.is_file():
                files.append(path)
        if not (program / "rag.py").is_file() or len(files) > 256 or sum(p.stat().st_size for p in files) > 8 * 1024 * 1024:
            raise ValueError("candidate_source_manifest_limit")
        if any(p.name in {".env", "metadata.json"} or ".git" in p.parts for p in files):
            raise ValueError("candidate_export_contains_forbidden_artifact")
        manifest = {str(p.relative_to(program)).replace("\\", "/"): _hash(p) for p in files}
        runtime_hashes = {name: _hash(RUNTIME / name) for name in ("linux_launcher.sh", "candidate_runner.py", "sdk.py")}
        resource_budget = {"seconds": seconds, "max_rpc_calls": 64, "max_output_bytes": MAX_BYTES}
        if budget is not None:
            if not isinstance(budget, dict):
                raise ValueError("budget_object_required")
            resource_budget.update(budget)
            if resource_budget.get("seconds") != seconds or type(resource_budget.get("max_rpc_calls")) is not int or not 0 <= resource_budget["max_rpc_calls"] <= 64:
                raise ValueError("budget_exceeds_runtime_envelope")
        initial = _json_line({"question": question, "budget": resource_budget, "mode": mode})
        # This directory only contains empty per-run mount points. Candidate files are never copied from arbitrary host trees.
        scratch_base = PROJECT / "runs" / ".code_rsi_sandbox"
        scratch_base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="run_", dir=scratch_base) as temporary:
            scratch = Path(temporary)
            nonce = uuid.uuid4().hex
            launcher = _wsl_path(RUNTIME / "linux_launcher.sh")
            command = self._command(launcher, _wsl_path(program), _wsl_path(corpus), _wsl_path(RUNTIME), _wsl_path(scratch), str(seconds), nonce)
            started = time.monotonic()
            events = queue.Queue()
            stderr = bytearray()
            counters = {"stdout": 0, "stderr": 0}
            trace = []
            supervisor = None
            finished = None
            preflight = None
            final = None
            candidate_error = None
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            def read_stdout():
                buffer = bytearray()
                try:
                    while True:
                        chunk = process.stdout.read(4096)
                        if not chunk:
                            if buffer:
                                events.put(("broken_frame", None))
                            break
                        counters["stdout"] += len(chunk)
                        if counters["stdout"] > MAX_BYTES + 4096:
                            events.put(("output_limit", None))
                            break
                        buffer.extend(chunk)
                        while b"\n" in buffer:
                            line, _, rest = buffer.partition(b"\n")
                            buffer = bytearray(rest)
                            try:
                                obj = json.loads(line.decode("utf-8"))
                            except (UnicodeDecodeError, ValueError):
                                events.put(("invalid_json", None))
                                return
                            if not isinstance(obj, dict):
                                events.put(("invalid_json", None))
                                return
                            events.put(("frame", obj))
                finally:
                    events.put(("eof", None))
            def read_stderr():
                while True:
                    chunk = process.stderr.read(4096)
                    if not chunk:
                        break
                    counters["stderr"] += len(chunk)
                    if len(stderr) < 16384:
                        stderr.extend(chunk[:16384 - len(stderr)])
                    if counters["stderr"] > MAX_BYTES + 4096:
                        events.put(("output_limit", None))
                        break
            readers = [threading.Thread(target=read_stdout, daemon=True), threading.Thread(target=read_stderr, daemon=True)]
            for reader in readers:
                reader.start()
            def details():
                return {"runtime": "wsl_namespaces_tmpfs_chroot_v1", "elapsed_seconds": time.monotonic() - started,
                        "supervisor": supervisor, "finished": finished, "isolation_checks": preflight,
                        "stderr": stderr.decode("utf-8", errors="replace")[-6000:], "trace": trace,
                        "source_hashes": manifest, "runtime_hashes": runtime_hashes, "output_bytes": dict(counters)}
            def cancel():
                if supervisor:
                    try:
                        subprocess.run(self._command("--cancel", str(supervisor["pid"]), str(supervisor["start_ticks"]), nonce), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8, check=False)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                try:
                    process.stdin.close()
                except OSError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            try:
                process.stdin.write(initial)
                process.stdin.flush()
                expected_id = 1
                while True:
                    remaining = seconds + 10 - (time.monotonic() - started)
                    if remaining <= 0:
                        raise SandboxExecutionError("host_watchdog_timeout", details())
                    try:
                        event, obj = events.get(timeout=min(remaining, 1.0))
                    except queue.Empty:
                        continue
                    if event == "eof":
                        break
                    if event != "frame":
                        raise SandboxExecutionError(event, details())
                    kind = obj.get("kind")
                    if supervisor is None:
                        if kind != "supervisor_started" or obj.get("nonce") != nonce or type(obj.get("pid")) is not int or not str(obj.get("start_ticks", "")).isdigit():
                            raise SandboxExecutionError("supervisor_handshake_failed", details())
                        supervisor = obj
                        continue
                    if kind == "isolation_preflight" and preflight is None:
                        checks = obj.get("checks")
                        if not isinstance(checks, dict) or not checks or not all(v is True for v in checks.values()):
                            raise SandboxExecutionError("isolation_preflight_rejected", details())
                        preflight = checks
                    elif kind == "call":
                        if not preflight or final is not None or obj.get("id") != expected_id or expected_id > resource_budget["max_rpc_calls"]:
                            raise SandboxExecutionError("rpc_sequence_or_budget_rejected", details())
                        name, payload = obj.get("name"), obj.get("payload")
                        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) or not isinstance(payload, dict):
                            raise SandboxExecutionError("rpc_schema_rejected", details())
                        call_start = time.monotonic()
                        call_remaining = max(0.0, seconds - (call_start - started))
                        if call_remaining <= 0:
                            raise SandboxExecutionError("service_time_budget_exhausted", details())
                        result_queue = queue.Queue(maxsize=1)
                        def invoke():
                            try:
                                result_queue.put((True, handler(name, payload, call_remaining)))
                            except Exception as error:
                                result_queue.put((False, type(error).__name__ + ":" + str(error)[:2000]))
                        threading.Thread(target=invoke, daemon=True).start()
                        try:
                            ok, result = result_queue.get(timeout=call_remaining)
                        except queue.Empty:
                            raise SandboxExecutionError("trusted_handler_timeout", details())
                        record = {"id": expected_id, "name": name, "payload": payload, "ok": ok,
                                  "elapsed_seconds": time.monotonic() - call_start}
                        if ok:
                            response = {"id": expected_id, "ok": True, "result": result}
                        else:
                            record["error"] = result
                            response = {"id": expected_id, "ok": False, "error": result}
                        trace.append(record)
                        if not ok:
                            info = details()
                            info["kind"] = "broker_error"
                            info["broker_error"] = result
                            info["broker_error_type"] = result.split(":", 1)[0]
                            raise SandboxExecutionError("trusted_broker_rejected", info)
                        process.stdin.write(_json_line(response))
                        process.stdin.flush()
                        expected_id += 1
                    elif kind == "final":
                        if not preflight or final is not None or not isinstance(obj.get("result"), dict):
                            raise SandboxExecutionError("unexpected_final", details())
                        final = obj["result"]
                    elif kind == "error":
                        candidate_error = obj.get("error")
                    elif kind == "supervisor_finished":
                        finished = obj
                    else:
                        raise SandboxExecutionError("unexpected_protocol_message", details())
                code = process.wait(timeout=5)
                if code != 0 or candidate_error or not final or not preflight or not finished or finished.get("status") != "complete" or finished.get("returncode") != 0:
                    info = details()
                    info["candidate_error"] = candidate_error
                    raise SandboxExecutionError("isolated_candidate_failed", info)
                expected_fields = {"answer", "citations", "abstention_reason"}
                if mode == "auxiliary_tests":
                    expected_fields.add("auxiliary_tests")
                if set(final) != expected_fields or not isinstance(final["answer"], str) or not isinstance(final["citations"], list):
                    raise SandboxExecutionError("host_output_schema_rejected", details())
                if manifest != {str(p.relative_to(program)).replace("\\", "/"): _hash(p) for p in files}:
                    raise SandboxExecutionError("candidate_sources_changed_during_run", details())
                return {"ok": True, "error": None, "result": final, "trace": trace, "runtime_seconds": time.monotonic() - started, "runtime": details()}
            except (BrokenPipeError, OSError) as error:
                raise SandboxExecutionError("runtime_transport_failed:" + type(error).__name__, details()) from error
            finally:
                cancel()
                for reader in readers:
                    reader.join(timeout=2)
                process.stdout.close()
                process.stderr.close()

    def preflight(self):
        """Actual namespace execution plus process-tree timeout; no candidate or API provider."""
        directory = PROJECT / "runs" / ".code_rsi_sandbox_tests"
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="preflight_", dir=directory) as temporary:
            folder = Path(temporary)
            program = folder / "source"
            program.mkdir()
            corpus = folder / "corpus.json"
            corpus.write_text("[]", encoding="utf-8")
            source = program / "rag.py"
            source.write_text("def solve(question, services):\n    return {\"answer\":\"isolation_ok\",\"citations\":[],\"abstention_reason\":None}\n", encoding="utf-8")
            def deny(name, payload, remaining):
                raise RuntimeError("preflight_services_disabled")
            try:
                good = self.run(program, corpus, "Isolation capability fixture", deny, seconds=12)
                source.write_text("import os\ndef solve(question, services):\n    try:\n        child=os.fork()\n        if child == 0:\n            os.setsid()\n    except OSError:\n        pass\n    while True:\n        pass\n", encoding="utf-8")
                try:
                    self.run(program, corpus, "Timeout fixture", deny, seconds=2)
                except SandboxExecutionError as error:
                    timeout = error.details
                    if not timeout.get("finished") or not timeout["finished"].get("descendants_terminated"):
                        raise SandboxUnavailable("process_tree_timeout_unverified") from error
                else:
                    raise SandboxUnavailable("process_tree_timeout_not_enforced")
            except SandboxExecutionError as error:
                raise SandboxUnavailable("isolation_preflight_failed:" + json.dumps(error.details, ensure_ascii=True)[:6000]) from error
            return {"ok": True, "runtime": good["runtime"], "process_tree_timeout": timeout, "new_api_calls": 0}
