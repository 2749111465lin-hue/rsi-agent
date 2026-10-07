#!/bin/bash
# Immutable namespace launcher. Outer branch is trusted supervision only.
set -eu
if [ "${1:-}" = "--inner" ]; then
    shift
    candidate=$1
    corpus=$2
    runtime=$3
    sandbox_root=$4
    seconds=$5
    nonce=$6
    umask 077
    mount --make-rprivate /
    mount -t tmpfs -o size=16m,nosuid,nodev,mode=755 tmpfs "$sandbox_root"
    mkdir -p "$sandbox_root/usr" "$sandbox_root/bin" "$sandbox_root/lib" "$sandbox_root/candidate" "$sandbox_root/runtime" "$sandbox_root/data" "$sandbox_root/tmp" "$sandbox_root/work" "$sandbox_root/dev" "$sandbox_root/proc"
    for part in usr bin lib; do
        mount --rbind "/$part" "$sandbox_root/$part"
        while IFS= read -r mountpoint; do
            mount -o remount,bind,ro,nosuid,nodev "$mountpoint"
        done < <(findmnt -rn -o TARGET -R "$sandbox_root/$part" | tac)
    done
    if [ -d /lib64 ]; then
        mkdir "$sandbox_root/lib64"
        mount --rbind /lib64 "$sandbox_root/lib64"
        mount -o remount,bind,ro,nosuid,nodev "$sandbox_root/lib64"
    fi
    mount --bind "$candidate" "$sandbox_root/candidate"
    mount -o remount,bind,ro,nosuid,nodev,noexec "$sandbox_root/candidate"
    touch "$sandbox_root/data/corpus.json" "$sandbox_root/runtime/candidate_runner.py" "$sandbox_root/runtime/sdk.py" "$sandbox_root/dev/null"
    mount --bind "$corpus" "$sandbox_root/data/corpus.json"
    mount -o remount,bind,ro,nosuid,nodev,noexec "$sandbox_root/data/corpus.json"
    for part in candidate_runner.py sdk.py; do
        mount --bind "$runtime/$part" "$sandbox_root/runtime/$part"
        mount -o remount,bind,ro,nosuid,nodev,noexec "$sandbox_root/runtime/$part"
    done
    mount --bind /dev/null "$sandbox_root/dev/null"
    mount -t proc -o ro,nosuid,nodev,noexec proc "$sandbox_root/proc"
    mount -t tmpfs -o size=128m,nosuid,nodev,noexec,mode=1777 tmpfs "$sandbox_root/tmp"
    mount -t tmpfs -o size=128m,nosuid,nodev,noexec,mode=1777 tmpfs "$sandbox_root/work"
    mount -t tmpfs -o remount,ro,nosuid,nodev tmpfs "$sandbox_root"
    cd "$sandbox_root"
    exec chroot "$sandbox_root" /usr/bin/setpriv --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 HOME=/work TMPDIR=/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/prlimit --as=402653184 --nproc=8 --fsize=67108864 --nofile=64 --core=0 --cpu="$seconds" /usr/bin/python3 -I -B /runtime/candidate_runner.py
fi
exec /usr/bin/python3 -c '
import hashlib,json,os,pathlib,signal,subprocess,sys,threading,time

def start_ticks(pid):
    return pathlib.Path("/proc/%s/stat" % pid).read_text().rsplit(")",1)[1].split()[19]

if sys.argv[1] == "--cancel":
    pid, ticks, nonce = int(sys.argv[2]), sys.argv[3], sys.argv[4]
    try:
        command = pathlib.Path("/proc/%s/cmdline" % pid).read_bytes().split(b"\0")
        if start_ticks(pid) == ticks and nonce.encode() in command and os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGKILL)
    except (FileNotFoundError,ProcessLookupError):
        pass
    sys.exit(0)

launcher,candidate,corpus,runtime,root,seconds,nonce = sys.argv[1:]
seconds = int(seconds)
limit = 2 * 1024 * 1024
command = ["/usr/bin/unshare","--user","--map-root-user","--mount","--pid","--fork","--net","--ipc","--uts","--kill-child","/bin/bash",launcher,"--inner",candidate,corpus,runtime,root,str(seconds),nonce]
allowed_cpus=sorted(os.sched_getaffinity(0))[:2]
def setup():
    os.sched_setaffinity(0,allowed_cpus)
process=subprocess.Popen(command,stdin=sys.stdin.buffer,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,close_fds=True,preexec_fn=setup)
pid=process.pid
ticks=start_ticks(pid)
namespace_id=None
for attempt in range(20):
    try:
        children=pathlib.Path("/proc/%s/task/%s/children" % (pid,pid)).read_text().split()
        if children:
            namespace_id=os.readlink("/proc/%s/ns/pid" % children[0])
            if namespace_id==os.readlink("/proc/self/ns/pid"):
                namespace_id=None
            else:
                break
    except (FileNotFoundError,ProcessLookupError):
        break
    if process.poll() is not None:break
    time.sleep(0.005)
print(json.dumps({"kind":"supervisor_started","pid":pid,"start_ticks":ticks,"nonce":nonce,"cpu_affinity":allowed_cpus}),flush=True)
state={"bytes":0,"limit":False}
lock=threading.Lock()
def kill():
    try:os.killpg(pid,signal.SIGKILL)
    except ProcessLookupError:pass

def forward(source,destination):
    while True:
        chunk=source.read1(4096)
        if not chunk:break
        with lock:
            state["bytes"]+=len(chunk)
            if state["bytes"]>limit:
                state["limit"]=True
                kill()
                return
        destination.write(chunk)
        destination.flush()
threads=[threading.Thread(target=forward,args=(process.stdout,sys.stdout.buffer),daemon=True),threading.Thread(target=forward,args=(process.stderr,sys.stderr.buffer),daemon=True)]
for thread in threads:thread.start()
status="complete"
try:
    code=process.wait(timeout=seconds)
except subprocess.TimeoutExpired:
    status="timeout"
    kill()
    code=process.wait(timeout=5)
finally:
    kill()
for thread in threads:thread.join(timeout=2)
if state["limit"]:status="output_limit"
def namespace_survivors():
    if namespace_id is None:return []
    survivors=[]
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():continue
        try:
            if os.readlink(str(entry / "ns/pid"))==namespace_id:
                survivors.append(int(entry.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return survivors
survivors=namespace_survivors()
for survivor in survivors:
    try:os.kill(survivor,signal.SIGKILL)
    except ProcessLookupError:pass
for attempt in range(20):
    survivors=namespace_survivors()
    if not survivors:break
    time.sleep(0.01)
if survivors:status="namespace_cleanup_failed"
print(json.dumps({"kind":"supervisor_finished","status":status,"returncode":code,"output_bytes":state["bytes"],"process_group":pid,"namespace_id":namespace_id,"surviving_namespace_pids":survivors,"descendants_terminated":not survivors}),flush=True)
sys.exit(0 if code==0 and status=="complete" else 1)
' "$@"
