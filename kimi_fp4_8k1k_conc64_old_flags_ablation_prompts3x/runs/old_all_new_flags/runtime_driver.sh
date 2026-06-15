#!/usr/bin/env bash
set -euo pipefail

source /workspace/benchmarks/benchmark_lib.sh

check_env_vars \
  MODEL \
  TP \
  CONC \
  ISL \
  OSL \
  MAX_MODEL_LEN \
  RANDOM_RANGE_RATIO \
  RESULT_FILENAME \
  PROMPT_MULTIPLIER \
  ENABLE_HSA_NO_SCRATCH_RECLAIM \
  ENABLE_AITER \
  QUICK_REDUCE_QUANTIZATION \
  GPU_MEMORY_UTILIZATION \
  BLOCK_SIZE \
  DISABLE_PREFIX_CACHING \
  DISABLE_LOG_REQUESTS

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

hf download "$MODEL"

# Install amd-quark for MXFP4 quantization support, matching the original script.
pip install amd-quark

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

if [[ "$ENABLE_HSA_NO_SCRATCH_RECLAIM" == "1" ]]; then
  version=$(rocm-smi --showfw | grep MEC | head -n 1 | awk '{print $NF}' || true)
  if [[ "$version" == "" || "$version" -lt 177 ]]; then
    export HSA_NO_SCRATCH_RECLAIM=1
  fi
fi

if [[ "$ENABLE_AITER" == "1" ]]; then
  export VLLM_ROCM_USE_AITER=1
fi

if [[ -n "$QUICK_REDUCE_QUANTIZATION" ]]; then
  export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="$QUICK_REDUCE_QUANTIZATION"
fi

if [[ "${TP}" -lt 8 ]]; then
  export VLLM_ROCM_USE_AITER_RMSNORM=0
fi

EP_FLAG=()
if [[ "${EP_SIZE:-1}" -gt 1 ]]; then
  EP_FLAG=(--enable-expert-parallel)
fi

PREFIX_FLAG=()
if [[ "$DISABLE_PREFIX_CACHING" == "1" ]]; then
  PREFIX_FLAG=(--no-enable-prefix-caching)
fi

LOG_FLAG=()
if [[ "$DISABLE_LOG_REQUESTS" == "1" ]]; then
  LOG_FLAG=(--disable-log-requests)
fi

set -x
vllm serve "$MODEL" --port "$PORT" \
  --tensor-parallel-size="$TP" \
  "${EP_FLAG[@]}" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --block-size="$BLOCK_SIZE" \
  "${PREFIX_FLAG[@]}" \
  "${LOG_FLAG[@]}" \
  --trust-remote-code \
  --mm-encoder-tp-mode data > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
  --model "$MODEL" \
  --port "$PORT" \
  --backend vllm \
  --input-len "$ISL" \
  --output-len "$OSL" \
  --random-range-ratio "$RANDOM_RANGE_RATIO" \
  --num-prompts "$((CONC * PROMPT_MULTIPLIER))" \
  --max-concurrency "$CONC" \
  --result-filename "$RESULT_FILENAME" \
  --result-dir /workspace/ \
  --trust-remote-code

set +x
