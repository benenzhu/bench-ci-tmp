#!/usr/bin/env bash
# GLM-5-FP8 on ATOM built for ROCm 10, TP8 — diagnose the torch.compile failure.
#
# The first ROCm 10 attempt died in all 8 ranks during warmup with
#   AssertionError: VllmBackend can only be called once   (atom/utils/backends.py:649)
# meaning Dynamo produced a SECOND graph for the model forward. Defaults here
# reproduce that path (--level 3 --cudagraph-mode FULL, ATOM's own defaults)
# with TORCH_LOGS=graph_breaks,recompiles so the log shows WHERE the second
# graph comes from.
#
# Two notes on things that look like the cause but are not:
#  * VllmBackend is ATOM's OWN class (atom/utils/backends.py:478), forked from
#    the vLLM project — hence the name. ATOM server mode does not import real
#    vllm (those imports live under atom/plugin/vllm/, plugin mode only) and the
#    crash traceback has zero site-packages/vllm frames. IMG=atom-rocm10:novllm
#    tests this directly: same image with vllm pip-uninstalled.
#  * --enforce-eager would NOT help: it only gates CUDA graph capture and is
#    never read near compilation_config. The compile switch is --level
#    (arg_utils.py:195, default 3=PIECEWISE). atom/plugin/config.py:560 claims
#    the decorator "checks enforce_eager to skip" — that comment is wrong for
#    server mode.
#
# Prime suspect is the torch version: official ATOM images ship torch 2.13
# (rocm/atom-dev:nightly_*) or 2.10 (rocm/atom:*), ours has 2.12 because of the
# vLLM ROCm 10 base.
#
# Usage:
#   ./remote_glm5_atom.sh                              # reproduce w/ diagnostics
#   IMG=atom-rocm10:novllm ./remote_glm5_atom.sh       # is vllm's presence to blame?
#   LEVEL=0 CGMODE=NONE ./remote_glm5_atom.sh          # bypass compile entirely
# At LEVEL=0 any throughput is a FLOOR (no compile, no cuda graphs) and is NOT
# comparable to the 344.61 baseline.
set -uo pipefail

MODEL=/dev/shm/model/GLM-5-FP8
IMG=${IMG:-atom-rocm10:local}   # :novllm variant has vllm pip-uninstalled
TP=${TP:-8}
ISL=8192 OSL=1024 CONC=8 PROMPT_MULT=3 MAX_MODEL_LEN=10240 PORT=19000
LEVEL=${LEVEL:-3}          # 3=PIECEWISE (ATOM default, the path that asserts); 0=NO_COMPILATION
CGMODE=${CGMODE:-FULL}     # ATOM default; only meaningful at level 3
OUT=${OUT:-/dev/shm/glm5_atom_run}
LOG=$OUT/run.log
CTR=${CTR:-glm5-atom-rocm10}

mkdir -p "$OUT"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== GLM-5 on ATOM ROCm10, TP$TP, --level $LEVEL --cudagraph-mode $CGMODE ==="
docker rm -f "$CTR" >/dev/null 2>&1

# ATOM_DISABLE_MMAP is deliberately NOT set: the bundle's bench script hardcodes
# it to true, but with weights on tmpfs mmap-enabled loading is far faster
# (seconds vs minutes).
docker run -d --name "$CTR" \
  --network host --ipc host --privileged \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODEL:$MODEL:ro" -v "$OUT:/workspace" \
  -e HF_HUB_OFFLINE=1 \
  -e TORCH_LOGS="${TORCH_LOGS:-graph_breaks,recompiles}" \
  -e TORCHDYNAMO_VERBOSE="${TORCHDYNAMO_VERBOSE:-1}" \
  --entrypoint /bin/bash "$IMG" -lc "
    python3 -m atom.entrypoints.openai_server \
      --model '$MODEL' --server-port $PORT -tp $TP \
      --kv_cache_dtype fp8 \
      --max-model-len $MAX_MODEL_LEN \
      --gpu-memory-utilization 0.9 \
      --level $LEVEL --cudagraph-mode $CGMODE \
      --default-chat-template-kwargs '{\"enable_thinking\": false}' \
      --trust-remote-code > /workspace/server.log 2>&1
  " >/dev/null || { say "docker run failed"; exit 1; }

say "waiting for server (up to 40 min)"
ready=0
for i in $(seq 1 480); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; say "server READY after $((i*5))s"; break; fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$CTR"; then say "container exited early"; break; fi
  sleep 5
done

if [[ "$ready" != 1 ]]; then
  say "=== SERVER DID NOT COME UP ==="
  # Report whether the original compile assert is still the blocker, or something new.
  if grep -q "VllmBackend can only be called once" "$OUT/server.log" 2>/dev/null; then
    say "  STILL the torch.compile assert -- --level $LEVEL did not take effect"
  else
    say "  compile assert GONE; new failure below"
  fi
  grep -hoE "(OSError|TypeError|RuntimeError|AssertionError|ValueError|ImportError): .*" \
    "$OUT/server.log" 2>/dev/null | sort -u | head -6 | cut -c1-200 | tee -a "$LOG"
  say "--- graph breaks / recompiles (why a 2nd graph?) ---"
  grep -hiE "graph break|Recompiling|torch._dynamo.*cache_size|reason:" "$OUT/server.log" 2>/dev/null \
    | sed "s/^.*\] //" | sort -u | head -20 | cut -c1-220 | tee -a "$LOG"
  say "--- did real vllm participate at all? ---"
  n=$(grep -c "site-packages/vllm/" "$OUT/server.log" 2>/dev/null)
  say "  site-packages/vllm frames in log: ${n:-0}"
  docker rm -f "$CTR" >/dev/null 2>&1; exit 1
fi

say "=== benchmark: ${ISL}x${OSL} conc=$CONC prompts=$((CONC*PROMPT_MULT)) ==="
docker exec "$CTR" bash -lc "
  cd /app/ATOM && python3 -m atom.benchmarks.benchmark_serving \
    --backend vllm --model '$MODEL' --port $PORT \
    --dataset-name random --random-input-len $ISL --random-output-len $OSL \
    --random-range-ratio 0 \
    --num-prompts $((CONC*PROMPT_MULT)) --max-concurrency $CONC \
    --save-result --result-dir /workspace --result-filename bench.json
" 2>&1 | tail -30 | tee -a "$LOG"

say "=== result ==="
python3 - "$OUT" "$TP" "$LEVEL" <<'PY' 2>&1 | tee -a "$LOG"
import json,glob,sys,os
out,tp,level=sys.argv[1],int(sys.argv[2]),sys.argv[3]
f=glob.glob(f"{out}/bench.json")
if not f: print("  no bench.json -- benchmark did not complete"); raise SystemExit
raw=open(f[0]).read().strip()
j=json.loads(raw.splitlines()[-1]) if raw.startswith("{") else json.loads(raw)
tot=j.get("total_token_throughput") or j.get("total_throughput") or 0.0
print(f"  completed        = {j.get('completed')}")
print(f"  total throughput = {tot:.2f} tok/s over TP{tp}")
print(f"  tok/s/GPU        = {tot/tp:.2f}")
if level=="0":
    print("  NOTE: --level 0 disables torch.compile AND cuda graphs -- this is a")
    print("        FLOOR, not comparable to the 344.61 baseline.")
PY

docker rm -f "$CTR" >/dev/null 2>&1
say "=== DONE ==="
