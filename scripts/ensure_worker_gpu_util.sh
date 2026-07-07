#!/usr/bin/env bash
# Keep worker GPU utilization above a threshold by running adaptive resident matmul load.
#
# Usage examples:
#   bash scripts/ensure_worker_gpu_util.sh start --host fdbd:dccd:cdc2:12c8:0:14:: --port <ssh_port>
#   bash scripts/ensure_worker_gpu_util.sh status --host fdbd:dccd:cdc2:12c8:0:14:: --port <ssh_port>
#   bash scripts/ensure_worker_gpu_util.sh stop --host fdbd:dccd:cdc2:12c8:0:14:: --port <ssh_port>
#   bash scripts/ensure_worker_gpu_util.sh local --target 45 --min-util 40 --gpus 0,1,2,3
#
# If --port is omitted, the script tries to infer it from ~/.ssh/known_hosts first,
# then from /root/worker_ssh_notes.md.

set -euo pipefail

CMD="${1:-start}"
if [[ $# -gt 0 ]]; then shift; fi

SSH_HOST="${WORKER_HOST:-fdbd:dccd:cdc2:12c8:0:14::}"
SSH_PORT="${WORKER_PORT:-}"
SSH_USER="${WORKER_USER:-tiger}"
if [[ -n "${WORKER_KEY:-}" ]]; then
  SSH_KEY="$WORKER_KEY"
elif [[ -r /home/tiger/.ssh/id_rsa ]]; then
  SSH_KEY="/home/tiger/.ssh/id_rsa"
else
  SSH_KEY="$HOME/.ssh/id_rsa"
fi
NOTES_PATH="${WORKER_SSH_NOTES:-/root/worker_ssh_notes.md}"
REMOTE_DIR="${REMOTE_GPU_KEEPALIVE_DIR:-/tmp/gpu_keepalive_${USER:-codex}}"
TARGET_UTIL="45"
MIN_UTIL="40"
HIGH_UTIL="70"
MAX_SLEEP="10"
GPUS="0,1,2,3"
MATRIX_SIZE="8192"
INTERVAL="3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host|--ip) SSH_HOST="$2"; shift 2 ;;
    --port) SSH_PORT="$2"; shift 2 ;;
    --user) SSH_USER="$2"; shift 2 ;;
    --key) SSH_KEY="$2"; shift 2 ;;
    --notes) NOTES_PATH="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --target) TARGET_UTIL="$2"; shift 2 ;;
    --min-util) MIN_UTIL="$2"; shift 2 ;;
    --high-util) HIGH_UTIL="$2"; shift 2 ;;
    --max-sleep) MAX_SLEEP="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --matrix-size) MATRIX_SIZE="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    -h|--help) sed -n '1,35p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

infer_port() {
  local host="$1"
  local port=""
  local kh="$HOME/.ssh/known_hosts"
  if [[ ! -r "$kh" && -r /home/tiger/.ssh/known_hosts ]]; then
    kh="/home/tiger/.ssh/known_hosts"
  fi
  if [[ -r "$kh" ]]; then
    port=$(python3 - "$host" "$kh" <<'PY' || true
import re, sys
host, path = sys.argv[1], sys.argv[2]
pat = re.compile(r'^\[' + re.escape(host) + r'\]:(\d+)\s')
for line in open(path, errors='ignore'):
    m = pat.search(line)
    if m:
        print(m.group(1)); break
PY
)
  fi
  if [[ -z "$port" && -f "$NOTES_PATH" ]]; then
    port=$(python3 - "$NOTES_PATH" <<'PY' || true
import re, sys
text = open(sys.argv[1], errors='ignore').read()
m = re.search(r'SSH\s*端口[:：]\s*(\d+)', text)
if m:
    print(m.group(1))
PY
)
  fi
  printf '%s' "$port"
}

write_payload() {
  local out="$1"
  cat > "$out" <<'PYTHON'
#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import time
from multiprocessing import Event, Process, Value


def run(cmd):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()


def gpu_count():
    out = run(["nvidia-smi", "-L"])
    return sum(1 for line in out.splitlines() if line.strip().startswith("GPU "))


def parse_gpus(s):
    if s == "all":
        return list(range(gpu_count()))
    return [int(x) for x in s.split(",") if x.strip() != ""]


def query_utils(gpus):
    idx = ",".join(map(str, gpus))
    out = run([
        "nvidia-smi",
        f"--id={idx}",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ])
    vals = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            vals[int(parts[0])] = {
                "util": int(parts[1]),
                "mem_used": int(parts[2]),
                "mem_total": int(parts[3]),
            }
    return vals


def _worker_ctypes(gpu, sleep_value, burst_value, stop_event, matrix_size):
    import ctypes
    import ctypes.util
    rt = None
    for libname in ["libcudart.so", "libcudart.so.12", "libcudart.so.12.9", "libcudart.so.11.0"]:
        try:
            rt = ctypes.CDLL(libname)
            break
        except OSError:
            pass
    if rt is None:
        raise RuntimeError("cannot load libcudart.so")
    rt.cudaSetDevice(gpu)
    # two large buffers for back-and-forth memcpy to generate bandwidth/utilization
    n_bytes = matrix_size * matrix_size * 4  # fp32 equivalent size
    src = ctypes.c_void_p()
    dst = ctypes.c_void_p()
    rt.cudaMalloc(ctypes.byref(src), n_bytes)
    rt.cudaMalloc(ctypes.byref(dst), n_bytes)
    rt.cudaMemset(src, 0x5A, n_bytes)
    while not stop_event.is_set():
        burst = max(1, int(burst_value.value))
        for _ in range(burst):
            rt.cudaMemcpy(dst, src, n_bytes, ctypes.c_int(3))  # cudaMemcpyDeviceToDevice
        rt.cudaDeviceSynchronize()
        s = float(sleep_value.value)
        if s > 0:
            time.sleep(s)
    rt.cudaFree(src)
    rt.cudaFree(dst)


def worker(gpu, sleep_value, burst_value, stop_event, matrix_size, dtype_name):
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    import torch

    use_torch = False
    try:
        torch.cuda.set_device(gpu)
        # smoke test: allocate a tiny tensor to confirm runtime works
        _ = torch.zeros(1, device=f"cuda:{gpu}")
        torch.cuda.synchronize(gpu)
        use_torch = True
    except Exception as e:
        print(f"[GPU {gpu}] torch CUDA unavailable ({e}), using ctypes cudart fallback", flush=True)

    if not use_torch:
        _worker_ctypes(gpu, sleep_value, burst_value, stop_event, matrix_size)
        return

    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    dev = torch.device(f"cuda:{gpu}")
    n = matrix_size
    # Allocate once and keep resident. Three bf16 8192x8192 tensors use ~384 MiB/GPU.
    a = torch.randn((n, n), device=dev, dtype=dtype)
    b = torch.randn((n, n), device=dev, dtype=dtype)
    c = torch.empty((n, n), device=dev, dtype=dtype)
    for _ in range(3):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize(dev)

    while not stop_event.is_set():
        burst = max(1, int(burst_value.value))
        for _ in range(burst):
            torch.matmul(a, b, out=c)
        torch.cuda.synchronize(dev)
        s = float(sleep_value.value)
        if s > 0:
            time.sleep(s)


def main():
    p = argparse.ArgumentParser(description="Adaptive resident GPU matmul load, targeting utilization > threshold.")
    p.add_argument("--target", type=int, default=45, help="controller target utilization; use > min-util as buffer")
    p.add_argument("--min-util", type=int, default=40, help="minimum desired nvidia-smi GPU utilization")
    p.add_argument("--high-util", type=int, default=70, help="if total utilization is at or above this, back off hard to avoid competing with real work")
    p.add_argument("--max-sleep", type=float, default=10.0, help="maximum sleep between keepalive matmuls when backing off")
    p.add_argument("--gpus", default="0,1,2,3", help="comma-separated GPU ids, or all")
    p.add_argument("--matrix-size", type=int, default=8192)
    p.add_argument("--interval", type=float, default=3.0)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = p.parse_args()

    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs selected")

    stop_event = Event()
    sleeps = {g: Value("d", 0.08) for g in gpus}
    bursts = {g: Value("i", 1) for g in gpus}
    procs = []

    def stop(*_):
        stop_event.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    print(f"[{time.strftime('%F %T')}] starting adaptive matmul keepalive", flush=True)
    print(f"gpus={gpus} min_util={args.min_util} target={args.target} high_util={args.high_util} max_sleep={args.max_sleep} matrix_size={args.matrix_size} dtype={args.dtype}", flush=True)

    for g in gpus:
        proc = Process(target=worker, args=(g, sleeps[g], bursts[g], stop_event, args.matrix_size, args.dtype), daemon=False)
        proc.start()
        procs.append(proc)

    time.sleep(max(5.0, args.interval))

    try:
        while not stop_event.is_set():
            try:
                stats = query_utils(gpus)
            except Exception as e:
                print(f"[{time.strftime('%F %T')}] failed to query nvidia-smi: {e}", flush=True)
                time.sleep(args.interval)
                continue

            fields = []
            for g in gpus:
                util = stats.get(g, {}).get("util", -1)
                with sleeps[g].get_lock(), bursts[g].get_lock():
                    s = float(sleeps[g].value)
                    b = int(bursts[g].value)
                    if util >= args.high_util:
                        # Real training/eval is probably active, or total load is already high.
                        # Back off aggressively: keep only a very sparse heartbeat matmul.
                        b = 1
                        s = min(args.max_sleep, max(2.0, s * 1.8 + 0.5))
                    elif util >= 0 and util < args.min_util:
                        if s > 0.002:
                            s = max(0.0, s * 0.55 - 0.001)
                        else:
                            b = min(16, b + 1)
                    elif util > args.target + 25:
                        if b > 1:
                            b = max(1, b - 1)
                        else:
                            s = min(args.max_sleep, s * 1.35 + 0.01)
                    elif util > args.target + 10:
                        s = min(args.max_sleep, s * 1.15 + 0.003)
                    elif args.min_util <= util < args.target:
                        s = max(0.0, s * 0.85 - 0.001)
                    sleeps[g].value = s
                    bursts[g].value = b
                mem = stats.get(g, {})
                fields.append(f"gpu{g}: util={util}% mem={mem.get('mem_used','?')}/{mem.get('mem_total','?')}MiB sleep={s:.4f}s burst={b}")

            print(f"[{time.strftime('%F %T')}] " + " | ".join(fields), flush=True)
            time.sleep(args.interval)
    finally:
        stop_event.set()
        for proc in procs:
            proc.join(timeout=8)
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        print(f"[{time.strftime('%F %T')}] stopped", flush=True)


if __name__ == "__main__":
    main()
PYTHON
}

if [[ "$CMD" != "local" && -z "$SSH_PORT" ]]; then
  SSH_PORT="$(infer_port "$SSH_HOST")"
fi

ssh_base() {
  if [[ -z "$SSH_PORT" ]]; then
    echo "Cannot infer SSH port for $SSH_HOST. Pass --port <port>, or update $NOTES_PATH / ~/.ssh/known_hosts." >&2
    return 2
  fi
  ssh -i "$SSH_KEY" -p "$SSH_PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 "$SSH_USER@$SSH_HOST" "$@"
}

case "$CMD" in
  local)
    tmp_py="/tmp/gpu_util_keepalive_$$.py"
    write_payload "$tmp_py"
    exec python3 "$tmp_py" --target "$TARGET_UTIL" --min-util "$MIN_UTIL" --high-util "$HIGH_UTIL" --max-sleep "$MAX_SLEEP" --gpus "$GPUS" --matrix-size "$MATRIX_SIZE" --interval "$INTERVAL"
    ;;
  start)
    if [[ -z "$SSH_PORT" ]]; then
      echo "Cannot infer SSH port for $SSH_HOST. Pass --port <port>." >&2
      exit 2
    fi
    tmp_py="/tmp/gpu_util_keepalive_$$.py"
    write_payload "$tmp_py"
    ssh_base "mkdir -p '$REMOTE_DIR'"
    scp -i "$SSH_KEY" -P "$SSH_PORT" -o StrictHostKeyChecking=no "$tmp_py" "$SSH_USER@[$SSH_HOST]:$REMOTE_DIR/gpu_util_keepalive.py"
    rm -f "$tmp_py"
    ssh_base "if [[ -f '$REMOTE_DIR/pid' ]] && kill -0 \$(cat '$REMOTE_DIR/pid') 2>/dev/null; then echo 'Already running: PID='\$(cat '$REMOTE_DIR/pid'); exit 0; fi; nohup env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH} python3 '$REMOTE_DIR/gpu_util_keepalive.py' --target '$TARGET_UTIL' --min-util '$MIN_UTIL' --high-util '$HIGH_UTIL' --max-sleep '$MAX_SLEEP' --gpus '$GPUS' --matrix-size '$MATRIX_SIZE' --interval '$INTERVAL' > '$REMOTE_DIR/keepalive.log' 2>&1 < /dev/null & echo \$! > '$REMOTE_DIR/pid'; echo 'Started GPU keepalive PID='\$(cat '$REMOTE_DIR/pid')' log=$REMOTE_DIR/keepalive.log'; sleep 2; tail -n 20 '$REMOTE_DIR/keepalive.log'"
    ;;
  stop)
    ssh_base "if [[ -f '$REMOTE_DIR/pid' ]]; then pid=\$(cat '$REMOTE_DIR/pid'); kill \$pid 2>/dev/null || true; sleep 2; kill -9 \$pid 2>/dev/null || true; rm -f '$REMOTE_DIR/pid'; echo stopped PID=\$pid; else echo 'No pid file found'; fi"
    ;;
  status)
    ssh_base "echo '=== nvidia-smi utilization ==='; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits; echo; if [[ -f '$REMOTE_DIR/pid' ]]; then pid=\$(cat '$REMOTE_DIR/pid'); if kill -0 \$pid 2>/dev/null; then echo 'keepalive running PID='\$pid; else echo 'pid file exists but process is not running PID='\$pid; fi; else echo 'keepalive not started by this script'; fi; echo; echo '=== log tail ==='; tail -n 40 '$REMOTE_DIR/keepalive.log' 2>/dev/null || true"
    ;;
  restart)
    "$0" stop --host "$SSH_HOST" --port "$SSH_PORT" --user "$SSH_USER" --key "$SSH_KEY" --remote-dir "$REMOTE_DIR" || true
    "$0" start --host "$SSH_HOST" --port "$SSH_PORT" --user "$SSH_USER" --key "$SSH_KEY" --remote-dir "$REMOTE_DIR" --target "$TARGET_UTIL" --min-util "$MIN_UTIL" --gpus "$GPUS" --matrix-size "$MATRIX_SIZE" --interval "$INTERVAL"
    ;;
  *)
    echo "Unknown command: $CMD. Use start|stop|status|restart|local" >&2
    exit 2
    ;;
esac
