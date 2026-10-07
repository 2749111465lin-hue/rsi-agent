"""Frozen runner inside the mount/PID/network namespace. Never imported by scorer."""
import contextlib
import ctypes
import importlib.util
import json
import os
import resource
import socket
import sys
import traceback

sys.path.insert(0, "/runtime")
from sdk import Services, emit, read_frame


def self_check():
    checks = {}
    for path in ("/mnt/c", "/mnt/d", "/home", "/root", "/run", "/sys"):
        checks["absent:" + path] = not os.path.exists(path)
    for path in ("/usr/__isolation_write_probe__", "/bin/__isolation_write_probe__", "/lib/__isolation_write_probe__", "/candidate/__isolation_write_probe__", "/runtime/__isolation_write_probe__", "/__isolation_write_probe__", "/data/corpus.json"):
        try:
            # O_WRONLY without O_TRUNC preserves corpus even if isolation failed.
            fd = os.open(path, os.O_WRONLY | (os.O_CREAT if path != "/data/corpus.json" else 0), 0o600)
        except OSError:
            checks["read_only:" + path] = True
        else:
            os.close(fd)
            checks["read_only:" + path] = False
    for path in ("/work/.write_probe", "/tmp/.write_probe"):
        with open(path, "w", encoding="utf-8") as stream:
            stream.write("ok")
        os.unlink(path)
        checks["writable:" + path] = True
    libc = ctypes.CDLL(None, use_errno=True)
    with open("/proc/self/uid_map", encoding="ascii") as stream:
        uid_map = [int(x) for x in stream.read().split()]
    checks["mapped_to_unprivileged_host_user"] = len(uid_map) == 3 and uid_map[0] == 0 and uid_map[1] != 0 and uid_map[2] == 1
    checks["private_pid_namespace"] = os.getpid() == 1
    checks["no_new_privileges"] = libc.prctl(39, 0, 0, 0, 0) == 1
    checks["empty_capability_bounding_set"] = all(libc.prctl(23, cap, 0, 0, 0) == 0 for cap in range(41))
    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]
    class Data(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]
    header, data = Header(0x20080522, 0), (Data * 2)()
    checks["empty_capabilities"] = libc.capget(ctypes.byref(header), data) == 0 and all(not (x.effective or x.permitted or x.inheritable) for x in data)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.2)
        try:
            connection.connect(("192.0.2.1", 9))
        except OSError as error:
            checks["network_unreachable"] = error.errno in (101, 100)
        else:
            checks["network_unreachable"] = False
    checks["no_sensitive_environment"] = not any(any(token in name.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")) for name in os.environ)
    checks["address_space_limit"] = resource.getrlimit(resource.RLIMIT_AS)[0] == 384 * 1024 * 1024
    checks["process_limit"] = resource.getrlimit(resource.RLIMIT_NPROC)[0] == 8
    checks["file_size_limit"] = resource.getrlimit(resource.RLIMIT_FSIZE)[0] == 64 * 1024 * 1024
    if not all(checks.values()):
        raise RuntimeError("isolation_preflight_failed:" + json.dumps(checks, sort_keys=True))
    return checks


def main():
    output = sys.stdout
    checks = self_check()
    emit(output, {"kind": "isolation_preflight", "checks": checks})
    init = read_frame(sys.stdin)
    if set(init) != {"question", "budget", "mode"} or not isinstance(init["question"], str) or not isinstance(init["budget"], dict):
        raise ValueError("invalid_candidate_input")
    mode = init["mode"]
    if mode not in ("solve", "auxiliary_tests"):
        raise ValueError("invalid_runner_mode")
    services = Services(init["budget"], reader=sys.stdin, writer=output)
    sys.path.insert(0, "/candidate")
    with contextlib.redirect_stdout(sys.stderr):
        if mode == "auxiliary_tests":
            import unittest
            suite = unittest.defaultTestLoader.discover("/candidate", pattern="test*.py")
            tested = unittest.TextTestRunner(stream=sys.stderr, verbosity=1).run(suite)
            result = {"answer": "", "citations": [], "abstention_reason": "", "auxiliary_tests": {
                "run": tested.testsRun, "failures": len(tested.failures), "errors": len(tested.errors),
                "skipped": len(tested.skipped), "passed": tested.wasSuccessful()}}
        else:
            spec = importlib.util.spec_from_file_location("candidate_program", "/candidate/rag.py")
            program = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(program)
            if not callable(getattr(program, "solve", None)):
                raise ValueError("solve_function_missing")
            result = program.solve(init["question"], services)
    expected_fields = {"answer", "citations", "abstention_reason"}
    if mode == "auxiliary_tests":
        expected_fields.add("auxiliary_tests")
    if not isinstance(result, dict) or set(result) != expected_fields:
        raise ValueError("invalid_candidate_output_fields")
    if not isinstance(result["answer"], str) or len(result["answer"]) > 16384:
        raise ValueError("invalid_answer")
    if not isinstance(result["citations"], list) or len(result["citations"]) > 256:
        raise ValueError("invalid_citations")
    if result["abstention_reason"] is not None and (not isinstance(result["abstention_reason"], str) or len(result["abstention_reason"]) > 16384):
        raise ValueError("invalid_abstention_reason")
    emit(output, {"kind": "final", "result": result})


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        emit(sys.__stdout__, {"kind": "error", "error": {"type": type(error).__name__, "message": str(error)[:2000], "traceback": traceback.format_exc()[-6000:]}})
        raise SystemExit(1)
