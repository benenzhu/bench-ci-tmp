#!/usr/bin/env bash
# Kimi-K2.5-MXFP4 on ROCm 7 (vLLM v0.22.0), TP4 — control for the ROCm 10 TP4 run.
#
# Self-contained: no InferenceX checkout on this host, so serve/bench args are
# lifted verbatim from the bundle's patched kimik2.5_fp4_mi355x.sh.
#
# Why TP4: rocm_aiter_mla.py:333 gates the ASM prefill on `num_heads % 16 == 0`
# and num_heads is PER RANK. Kimi has 64 heads, so TP8 -> 8 (fails the gate,
# falls back to flash_attn_varlen_func and dies on an aiter_tensor_t pybind
# type clash) while TP4 -> 16 (passes, stays on the aiter ASM path).
#
# --logprobs-mode processed_logits is NOT needed on ROCm 7 (the hipcub::Traits
# sampler breakage is ROCm-10-only), but it is kept here deliberately: this run
# exists to be compared against the ROCm 10 TP4 run, so every serve flag must
# match. Dropping it here would make the two numbers incomparable.
set -uo pipefail

MODEL=/dev/shm/model/Kimi-K2.5-MXFP4
IMG=vllm/vllm-openai-rocm:v0.22.0
TP=${TP:-4}
ISL=8192 OSL=1024 CONC=128 PROMPT_MULT=3 MAX_MODEL_LEN=9416 PORT=19000
# vllm bench serve samples input_len from [len*(1-r), len*(1+r)], so the
# bundle's r=0.8 would draw up to 14746 input tokens against max_model_len
# 9416 and the server 400s those requests. InferenceX's run_benchmark_serving
# wrapper evidently clamps differently; here we pin r=0 (fixed 8192x1024) so
# every request is in-range and the run measures the kernel path, not rejects.
RRR=${RRR:-0}
OUT=/dev/shm/kimi_tp${TP}_rocm7_run
LOG=$OUT/run.log
CTR=kimi-tp${TP}-rocm7

mkdir -p "$OUT"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== Kimi TP$TP on ROCm7 (v0.22.0) -- control for the ROCm10 TP4 run ==="
docker rm -f "$CTR" >/dev/null 2>&1

docker run -d --name "$CTR" \
  --network host --ipc host --privileged \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODEL:$MODEL:ro" -v "$OUT:/workspace" \
  -e HF_HUB_OFFLINE=1 \
  --entrypoint /bin/bash "$IMG" -lc "
    # Env block lifted from the bundle's kimik2.5_fp4_mi355x.sh (lines 44-55).
    # VLLM_ROCM_USE_AITER=1 is REQUIRED: without it _get_backend_priorities()
    # never even offers ROCM_AITER_MLA, so --block-size=1 leaves zero valid
    # backends (TRITON_MLA rejects block_size=1) and the engine aborts.
    version=\$(rocm-smi --showfw | grep MEC | head -n 1 | awk '{print \$NF}')
    if [[ \"\$version\" == \"\" || \$version -lt 177 ]]; then
      export HSA_NO_SCRATCH_RECLAIM=1
    fi
    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
    # Upstream disables aiter RMSNorm below TP8 for accuracy; TP4 hits this.
    if [ '$TP' -lt 8 ]; then export VLLM_ROCM_USE_AITER_RMSNORM=0; fi

    vllm serve '$MODEL' --port $PORT \
      --tensor-parallel-size=$TP \
      --logprobs-mode processed_logits \
      --gpu-memory-utilization 0.80 \
      --max-model-len $MAX_MODEL_LEN \
      --block-size=1 \
      --no-enable-prefix-caching \
      --trust-remote-code \
      --mm-encoder-tp-mode data > /workspace/server.log 2>&1
  " >/dev/null || { say "docker run failed"; exit 1; }

say "waiting for server (up to 40 min: weight load + aiter JIT)"
ready=0
for i in $(seq 1 480); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; say "server READY after $((i*5))s"; break; fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$CTR"; then say "container exited early"; break; fi
  sleep 5
done

if [[ "$ready" != 1 ]]; then
  say "=== SERVER DID NOT COME UP ==="
  grep -iE "error|assert|Traceback" "$OUT/server.log" 2>/dev/null | grep -v "Internal Server Error" | tail -15 | tee -a "$LOG"
  docker rm -f "$CTR" >/dev/null 2>&1; exit 1
fi

# Which prefill path did we actually land on? This is the whole point of the run.
say "=== prefill path check ==="
if grep -q "fmha_fwd_bf16_opus_fwd" "$OUT/server.log" 2>/dev/null; then
  say "  WARNING: fmha_fwd_bf16_opus_fwd present -> still on flash_attn_varlen fallback"
else
  say "  no flash_attn_varlen fallback error so far"
fi

say "=== benchmark: ${ISL}x${OSL} conc=$CONC prompts=$((CONC*PROMPT_MULT)) ==="
docker exec "$CTR" bash -lc "
  vllm bench serve --model '$MODEL' --port $PORT --backend vllm \
    --dataset-name random \
    --random-input-len $ISL --random-output-len $OSL \
    --random-range-ratio $RRR \
    --num-prompts $((CONC*PROMPT_MULT)) --max-concurrency $CONC \
    --ignore-eos --trust-remote-code \
    --save-result --result-dir /workspace --result-filename bench.json
" 2>&1 | tail -40 | tee -a "$LOG"

say "=== result ==="
python3 - "$OUT" "$TP" <<'PY' 2>&1 | tee -a "$LOG"
import json,glob,sys
out,tp=sys.argv[1],int(sys.argv[2])
f=glob.glob(f"{out}/bench.json")
if not f: print("  no bench.json -- benchmark did not complete"); raise SystemExit
j=json.load(open(f[0]))
comp,fail=j.get("completed",0),j.get("failed",0)
if fail:
    print(f"  *** {fail} FAILED / {comp} completed -- throughput below is NOT valid ***")
tot=j.get("total_token_throughput",0.0)
print(f"  total_token_throughput = {tot:.2f} tok/s over TP{tp}")
print(f"  tok/s/GPU              = {tot/tp:.2f}")
print(f"  output_throughput      = {j.get('output_throughput',0):.2f}")
print(f"  completed              = {j.get('completed')}")
print(f"  mean TTFT / TPOT       = {j.get('mean_ttft_ms',0):.1f} ms / {j.get('mean_tpot_ms',0):.2f} ms")
print(f"  (compare against the ROCm 10 TP4 run: 3617.35 tok/s/GPU, same flags/host)")
PY

say "=== final prefill path verdict ==="
grep -c "fmha_fwd_bf16_opus_fwd" "$OUT/server.log" 2>/dev/null | xargs -I{} say "  fmha_fwd_bf16_opus_fwd occurrences: {}"

docker rm -f "$CTR" >/dev/null 2>&1
say "=== DONE ==="
