#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/InferenceX}
WT_ROOT=${WT_ROOT:-/scratch/inferencex_worktrees/kimi_fp4_old_flags_ablation}
OUT_DIR=${OUT_DIR:-/scratch/inferencex_runs/kimi_fp4_8k1k_conc64_old_flags_ablation_prompts3x}
MODEL_DIR=${MODEL_DIR:-/scratch/hf_hub_cache}
HF_HOME_DIR=${HF_HOME_DIR:-/scratch/hf_home}
PORT=${PORT:-18888}
PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER:-3}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.8}
SKIP_EXISTING=${SKIP_EXISTING:-1}
TARGETS=${TARGETS:-old_all_new_flags}
HF_TOKEN_FILE=${HF_TOKEN_FILE:-/root/.cache/huggingface/token}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-0}
DRY_RUN=${DRY_RUN:-0}

REPO_SLUG=SemiAnalysisAI/InferenceX
COMMIT=bc818474fdba28c4802e823fdc474be56fb79452
IMAGE=vllm/vllm-openai-rocm:v0.16.0
MODEL=amd/Kimi-K2.5-MXFP4
MODEL_PREFIX=kimik2.5
PRECISION=fp4
FRAMEWORK=vllm
ISL=8192
OSL=1024
CONC=64
TP=8
EP_SIZE=1
DP_ATTENTION=false
RUN_ID=22555187591
ATTEMPT=2
JOB_ID=65962398629
REFERENCE_NEW_RUN_ID=23614589869
REFERENCE_NEW_ATTEMPT=1
REFERENCE_NEW_JOB_ID=68778764973
REFERENCE_OLD_TPUT_PER_GPU=348.5322178041199
REFERENCE_NEW_TPUT_PER_GPU=1647.2603593393117

LABEL_ORDER=(
  old_baseline
  old_all_new_flags
  old_server_flags_only
  old_server_flags_no_block1
  old_env_flags_only
  old_aiter_quick_only
  old_aiter_only
  old_block1_only
  old_no_prefix_only
  old_mem090_only
)

declare -A VARIANT_NOTE=(
  [old_baseline]="old image with original 03-01 env/server flags"
  [old_all_new_flags]="old image plus 03-26 env/server flags: HSA conditional, AITER, INT4 quick reduce, gpu_memory=0.90, block_size=1, no prefix cache, no disable-log-requests"
  [old_server_flags_only]="old image plus server flags only: gpu_memory=0.90, block_size=1, no prefix cache, no disable-log-requests"
  [old_server_flags_no_block1]="old image plus compatible server flags only: gpu_memory=0.90, no prefix cache, no disable-log-requests; block_size remains 64"
  [old_env_flags_only]="old image plus env flags only: HSA conditional, AITER, INT4 quick reduce; old server flags retained"
  [old_aiter_quick_only]="old image plus VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 only"
  [old_aiter_only]="old image plus VLLM_ROCM_USE_AITER=1 only"
  [old_block1_only]="old image plus --block-size=1 only"
  [old_no_prefix_only]="old image plus --no-enable-prefix-caching only"
  [old_mem090_only]="old image plus --gpu-memory-utilization 0.90 only"
)

declare -A ENABLE_HSA=(
  [old_baseline]=0
  [old_all_new_flags]=1
  [old_server_flags_only]=0
  [old_server_flags_no_block1]=0
  [old_env_flags_only]=1
  [old_aiter_quick_only]=0
  [old_aiter_only]=0
  [old_block1_only]=0
  [old_no_prefix_only]=0
  [old_mem090_only]=0
)

declare -A ENABLE_AITER=(
  [old_baseline]=0
  [old_all_new_flags]=1
  [old_server_flags_only]=0
  [old_server_flags_no_block1]=0
  [old_env_flags_only]=1
  [old_aiter_quick_only]=1
  [old_aiter_only]=1
  [old_block1_only]=0
  [old_no_prefix_only]=0
  [old_mem090_only]=0
)

declare -A QUICK_REDUCE=(
  [old_baseline]=
  [old_all_new_flags]=INT4
  [old_server_flags_only]=
  [old_server_flags_no_block1]=
  [old_env_flags_only]=INT4
  [old_aiter_quick_only]=INT4
  [old_aiter_only]=
  [old_block1_only]=
  [old_no_prefix_only]=
  [old_mem090_only]=
)

declare -A GPU_MEMORY_UTILIZATION=(
  [old_baseline]=0.95
  [old_all_new_flags]=0.90
  [old_server_flags_only]=0.90
  [old_server_flags_no_block1]=0.90
  [old_env_flags_only]=0.95
  [old_aiter_quick_only]=0.95
  [old_aiter_only]=0.95
  [old_block1_only]=0.95
  [old_no_prefix_only]=0.95
  [old_mem090_only]=0.90
)

declare -A BLOCK_SIZE=(
  [old_baseline]=64
  [old_all_new_flags]=1
  [old_server_flags_only]=1
  [old_server_flags_no_block1]=64
  [old_env_flags_only]=64
  [old_aiter_quick_only]=64
  [old_aiter_only]=64
  [old_block1_only]=1
  [old_no_prefix_only]=64
  [old_mem090_only]=64
)

declare -A DISABLE_PREFIX_CACHING=(
  [old_baseline]=0
  [old_all_new_flags]=1
  [old_server_flags_only]=1
  [old_server_flags_no_block1]=1
  [old_env_flags_only]=0
  [old_aiter_quick_only]=0
  [old_aiter_only]=0
  [old_block1_only]=0
  [old_no_prefix_only]=1
  [old_mem090_only]=0
)

declare -A DISABLE_LOG_REQUESTS=(
  [old_baseline]=1
  [old_all_new_flags]=0
  [old_server_flags_only]=0
  [old_server_flags_no_block1]=0
  [old_env_flags_only]=1
  [old_aiter_quick_only]=1
  [old_aiter_only]=1
  [old_block1_only]=1
  [old_no_prefix_only]=1
  [old_mem090_only]=1
)

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

ensure_hf_token() {
  if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
    HF_TOKEN=$(tr -d '\r\n' < "$HF_TOKEN_FILE")
    export HF_TOKEN
  fi
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is empty and $HF_TOKEN_FILE was not readable" >&2
    exit 1
  fi
}

ensure_commit() {
  if ! git -C "$REPO" cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
    git -C "$REPO" fetch origin "$COMMIT"
  fi
}

max_model_len() {
  echo $((ISL + OSL + 200))
}

worktree_for() {
  local label=$1
  echo "$WT_ROOT/$label-${COMMIT:0:7}"
}

run_dir_for() {
  local label=$1
  echo "$OUT_DIR/runs/$label"
}

result_filename_for() {
  local label=$1
  echo "${MODEL_PREFIX}_${ISL}x${OSL}_${PRECISION}_${FRAMEWORK}_mi355x_tp${TP}-ep${EP_SIZE}-dpa${DP_ATTENTION}_disagg-false_spec-none_conc${CONC}_prompts${PROMPT_MULTIPLIER}x_${label}"
}

selected_labels() {
  local normalized=${TARGETS//,/ }
  if [[ "$normalized" == "all" ]]; then
    printf '%s\n' "${LABEL_ORDER[@]}"
    return 0
  fi
  local token
  for token in $normalized; do
    case "$token" in
      old_all_new_flags|all_new_flags) printf '%s\n' old_all_new_flags ;;
      old_server_flags_only|server_flags_only) printf '%s\n' old_server_flags_only ;;
      old_server_flags_no_block1|server_flags_no_block1|compatible_server_flags) printf '%s\n' old_server_flags_no_block1 ;;
      old_env_flags_only|env_flags_only) printf '%s\n' old_env_flags_only ;;
      old_aiter_quick_only|aiter_quick_only) printf '%s\n' old_aiter_quick_only ;;
      old_aiter_only|aiter_only) printf '%s\n' old_aiter_only ;;
      old_block1_only|block1_only) printf '%s\n' old_block1_only ;;
      old_no_prefix_only|no_prefix_only) printf '%s\n' old_no_prefix_only ;;
      old_mem090_only|mem090_only) printf '%s\n' old_mem090_only ;;
      old_baseline|baseline) printf '%s\n' old_baseline ;;
      bisect) printf '%s\n' old_server_flags_only old_server_flags_no_block1 old_env_flags_only old_aiter_quick_only old_aiter_only old_block1_only old_no_prefix_only old_mem090_only ;;
      *)
        echo "Unknown TARGETS token: $token" >&2
        exit 1
        ;;
    esac
  done | awk '!seen[$0]++'
}

prepare_worktree() {
  local label=$1
  local dest
  dest=$(worktree_for "$label")
  mkdir -p "$WT_ROOT" "$(run_dir_for "$label")"
  ensure_commit

  if [[ -e "$dest" && ! -f "$dest/.kimi_fp4_flags_ablation_owner" ]]; then
    echo "Refusing to modify existing unowned worktree path: $dest" >&2
    exit 1
  fi

  if [[ ! -e "$dest/.git" ]]; then
    git -C "$REPO" worktree add --force --detach "$dest" "$COMMIT"
    touch "$dest/.kimi_fp4_flags_ablation_owner"
  else
    git -C "$dest" reset --hard "$COMMIT" >/dev/null
    git -C "$dest" clean -fdx >/dev/null
    touch "$dest/.kimi_fp4_flags_ablation_owner"
  fi
}

write_runtime_driver() {
  local label=$1
  local wt
  wt=$(worktree_for "$label")
  local result_filename
  result_filename=$(result_filename_for "$label")

  cat > "$wt/.kimi_fp4_flags_driver.sh" <<'EOF'
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
EOF
  chmod +x "$wt/.kimi_fp4_flags_driver.sh"
  cp -f "$wt/.kimi_fp4_flags_driver.sh" "$(run_dir_for "$label")/runtime_driver.sh"
  git -C "$wt" diff -- benchmarks/single_node/kimik2.5_fp4_mi355x.sh > "$(run_dir_for "$label")/local_patch.diff" 2>/dev/null || true
  printf '%s\n' "$result_filename" > "$(run_dir_for "$label")/result_filename.txt"
}

write_run_env_file() {
  local label=$1
  local run_dir
  run_dir=$(run_dir_for "$label")
  mkdir -p "$run_dir"
  cat > "$run_dir/env.repro" <<EOF
LABEL=$label
VARIANT_NOTE=${VARIANT_NOTE[$label]}
COMMIT=$COMMIT
IMAGE=$IMAGE
MODEL=$MODEL
MODEL_PREFIX=$MODEL_PREFIX
PRECISION=$PRECISION
FRAMEWORK=$FRAMEWORK
RUNNER_TYPE=mi355x
TP=$TP
EP_SIZE=$EP_SIZE
DP_ATTENTION=$DP_ATTENTION
CONC=$CONC
ISL=$ISL
OSL=$OSL
MAX_MODEL_LEN=$(max_model_len)
RANDOM_RANGE_RATIO=$RANDOM_RANGE_RATIO
PROMPT_MULTIPLIER=$PROMPT_MULTIPLIER
NUM_PROMPTS=$((CONC * PROMPT_MULTIPLIER))
SPEC_DECODING=none
DISAGG=false
RUN_EVAL=false
EVAL_ONLY=false
PORT=$PORT
HF_HUB_CACHE=$MODEL_DIR
HF_HOME=$HF_HOME_DIR
HF_XET_CACHE=$HF_HOME_DIR/xet
WORKFLOW_RUN_ID=$RUN_ID
WORKFLOW_ATTEMPT=$ATTEMPT
WORKFLOW_JOB_ID=$JOB_ID
REFERENCE_NEW_RUN_ID=$REFERENCE_NEW_RUN_ID
REFERENCE_NEW_ATTEMPT=$REFERENCE_NEW_ATTEMPT
REFERENCE_NEW_JOB_ID=$REFERENCE_NEW_JOB_ID
REFERENCE_OLD_TPUT_PER_GPU=$REFERENCE_OLD_TPUT_PER_GPU
REFERENCE_NEW_TPUT_PER_GPU=$REFERENCE_NEW_TPUT_PER_GPU
ENABLE_HSA_NO_SCRATCH_RECLAIM=${ENABLE_HSA[$label]}
ENABLE_AITER=${ENABLE_AITER[$label]}
QUICK_REDUCE_QUANTIZATION=${QUICK_REDUCE[$label]}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION[$label]}
BLOCK_SIZE=${BLOCK_SIZE[$label]}
DISABLE_PREFIX_CACHING=${DISABLE_PREFIX_CACHING[$label]}
DISABLE_LOG_REQUESTS=${DISABLE_LOG_REQUESTS[$label]}
EOF
}

download_workflow_metadata() {
  local dest="$OUT_DIR/workflow_metadata"
  mkdir -p "$dest/old" "$dest/new_reference"
  curl --compressed -fsSL "https://api.github.com/repos/$REPO_SLUG/actions/runs/$RUN_ID/attempts/$ATTEMPT" -o "$dest/old/run.json" || true
  curl --compressed -fsSL "https://api.github.com/repos/$REPO_SLUG/actions/runs/$RUN_ID/attempts/$ATTEMPT/jobs?per_page=100" -o "$dest/old/jobs.json" || true
  curl --compressed -fsSL "https://api.github.com/repos/$REPO_SLUG/actions/runs/$REFERENCE_NEW_RUN_ID/attempts/$REFERENCE_NEW_ATTEMPT" -o "$dest/new_reference/run.json" || true
  curl --compressed -fsSL "https://api.github.com/repos/$REPO_SLUG/actions/runs/$REFERENCE_NEW_RUN_ID/attempts/$REFERENCE_NEW_ATTEMPT/jobs?per_page=100" -o "$dest/new_reference/jobs.json" || true
}

cleanup_named_container() {
  local label=$1
  docker rm -f "mi355-kimi-fp4-flags-$label" >/dev/null 2>&1 || true
}

process_result() {
  local label=$1
  local wt
  wt=$(worktree_for "$label")
  local result_filename
  result_filename=$(result_filename_for "$label")
  (
    cd "$wt"
    RUNNER_TYPE=mi355x \
    FRAMEWORK="$FRAMEWORK" \
    PRECISION="$PRECISION" \
    SPEC_DECODING=none \
    RESULT_FILENAME="$result_filename" \
    ISL="$ISL" \
    OSL="$OSL" \
    DISAGG=false \
    MODEL_PREFIX="$MODEL_PREFIX" \
    IMAGE="$IMAGE" \
    TP="$TP" \
    EP_SIZE="$EP_SIZE" \
    DP_ATTENTION="$DP_ATTENTION" \
    python3 utils/process_result.py
  )
}

copy_run_artifacts() {
  local label=$1
  local status=$2
  local wt
  wt=$(worktree_for "$label")
  local run_dir
  run_dir=$(run_dir_for "$label")
  local result_filename
  result_filename=$(result_filename_for "$label")
  mkdir -p "$run_dir"
  printf '%s\n' "$status" > "$run_dir/status.txt"
  [[ -f "$wt/${result_filename}.json" ]] && cp -f "$wt/${result_filename}.json" "$run_dir/"
  [[ -f "$wt/agg_${result_filename}.json" ]] && cp -f "$wt/agg_${result_filename}.json" "$run_dir/"
  [[ -f "$wt/server.log" ]] && cp -f "$wt/server.log" "$run_dir/server.log"
  [[ -f "$wt/gpu_metrics.csv" ]] && cp -f "$wt/gpu_metrics.csv" "$run_dir/gpu_metrics.csv"
  [[ -f "$wt/.kimi_fp4_flags_driver.sh" ]] && cp -f "$wt/.kimi_fp4_flags_driver.sh" "$run_dir/runtime_driver.sh"
}

run_one() {
  local label=$1
  local wt
  wt=$(worktree_for "$label")
  local run_dir
  run_dir=$(run_dir_for "$label")
  local result_filename
  result_filename=$(result_filename_for "$label")
  local result_path="$run_dir/${result_filename}.json"
  local docker_log="$run_dir/docker.log"

  mkdir -p "$run_dir"
  write_run_env_file "$label"
  prepare_worktree "$label"
  write_runtime_driver "$label"

  if [[ "$SKIP_EXISTING" == "1" && -s "$result_path" && -s "$run_dir/agg_${result_filename}.json" ]]; then
    echo "[skip] $label already has raw and aggregate results"
    return 0
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    copy_run_artifacts "$label" "dry_run"
    echo "[dry-run] $label prepared"
    return 0
  fi

  cleanup_named_container "$label"
  rm -f "$wt/${result_filename}.json" "$wt/agg_${result_filename}.json" "$wt/server.log" "$wt/gpu_metrics.csv"
  if [[ -s "$docker_log" ]]; then
    mv "$docker_log" "$run_dir/docker.$(date -u +%Y%m%dT%H%M%SZ).log"
  fi

  echo "[run] $label image=$IMAGE seq=$ISL/$OSL conc=$CONC tp=$TP prompts=$((CONC * PROMPT_MULTIPLIER))"

  set +e
  docker run --rm \
    --name "mi355-kimi-fp4-flags-$label" \
    --network host \
    --ipc host \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --entrypoint /bin/bash \
    -w /workspace \
    -v "$wt:/workspace" \
    -v "$MODEL_DIR:$MODEL_DIR" \
    -v "$HF_HOME_DIR:$HF_HOME_DIR" \
    -e HF_TOKEN \
    -e HF_HUB_CACHE="$MODEL_DIR" \
    -e HF_HOME="$HF_HOME_DIR" \
    -e HF_XET_CACHE="$HF_HOME_DIR/xet" \
    -e MODEL="$MODEL" \
    -e MODEL_PREFIX="$MODEL_PREFIX" \
    -e IMAGE="$IMAGE" \
    -e FRAMEWORK="$FRAMEWORK" \
    -e PRECISION="$PRECISION" \
    -e RUNNER_TYPE=mi355x \
    -e TP="$TP" \
    -e EP_SIZE="$EP_SIZE" \
    -e DP_ATTENTION="$DP_ATTENTION" \
    -e CONC="$CONC" \
    -e ISL="$ISL" \
    -e OSL="$OSL" \
    -e MAX_MODEL_LEN="$(max_model_len)" \
    -e RANDOM_RANGE_RATIO="$RANDOM_RANGE_RATIO" \
    -e RESULT_FILENAME="$result_filename" \
    -e SPEC_DECODING=none \
    -e DISAGG=false \
    -e RUN_EVAL=false \
    -e EVAL_ONLY=false \
    -e PORT="$PORT" \
    -e PROMPT_MULTIPLIER="$PROMPT_MULTIPLIER" \
    -e ENABLE_HSA_NO_SCRATCH_RECLAIM="${ENABLE_HSA[$label]}" \
    -e ENABLE_AITER="${ENABLE_AITER[$label]}" \
    -e QUICK_REDUCE_QUANTIZATION="${QUICK_REDUCE[$label]}" \
    -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION[$label]}" \
    -e BLOCK_SIZE="${BLOCK_SIZE[$label]}" \
    -e DISABLE_PREFIX_CACHING="${DISABLE_PREFIX_CACHING[$label]}" \
    -e DISABLE_LOG_REQUESTS="${DISABLE_LOG_REQUESTS[$label]}" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPYCACHEPREFIX=/tmp/inferencex-pycache \
    "$IMAGE" \
    -lc "bash /workspace/.kimi_fp4_flags_driver.sh" > "$docker_log" 2>&1
  local docker_status=$?
  set -e

  if (( docker_status != 0 )); then
    copy_run_artifacts "$label" "docker_failed:$docker_status"
    echo "[error] docker failed for $label status=$docker_status, log=$docker_log" >&2
    return "$docker_status"
  fi

  if [[ ! -s "$wt/${result_filename}.json" ]]; then
    copy_run_artifacts "$label" "missing_result"
    echo "[error] missing benchmark JSON for $label, log=$docker_log" >&2
    return 1
  fi

  if ! process_result "$label" > "$run_dir/process_result.stdout" 2> "$run_dir/process_result.stderr"; then
    copy_run_artifacts "$label" "process_failed"
    echo "[error] process_result failed for $label" >&2
    return 1
  fi

  copy_run_artifacts "$label" "success"
}

write_readme() {
  mkdir -p "$OUT_DIR"
  cat > "$OUT_DIR/README.md" <<EOF
# Kimi-K2.5 MXFP4 Old-Image Flag Ablation

Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)

Scope:

- Model: \`$MODEL\`
- Old workflow: \`$RUN_ID/attempts/$ATTEMPT\`, job \`$JOB_ID\`, commit \`${COMMIT:0:12}\`
- New reference workflow: \`$REFERENCE_NEW_RUN_ID/attempts/$REFERENCE_NEW_ATTEMPT\`, job \`$REFERENCE_NEW_JOB_ID\`
- Image under test: \`$IMAGE\`
- Hardware/framework/precision: MI355X / vLLM / FP4
- Shape: ISL/OSL \`$ISL/$OSL\`, CONC \`$CONC\`, TP \`$TP\`, EP \`$EP_SIZE\`
- Prompt count patch: \`num_prompts = CONC * $PROMPT_MULTIPLIER = $((CONC * PROMPT_MULTIPLIER))\`
- Dashboard reference for this same row: \`$REFERENCE_OLD_TPUT_PER_GPU -> $REFERENCE_NEW_TPUT_PER_GPU tok/s/GPU\`, gain \`$(python3 - <<PY
old=$REFERENCE_OLD_TPUT_PER_GPU
new=$REFERENCE_NEW_TPUT_PER_GPU
print(f"{new/old:.4f}x")
PY
)\`

Variants:

| label | note |
|---|---|
EOF
  local label
  for label in "${LABEL_ORDER[@]}"; do
    printf '| `%s` | %s |\n' "$label" "${VARIANT_NOTE[$label]}" >> "$OUT_DIR/README.md"
  done
}

write_combined_csv() {
  python3 - "$OUT_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
labels = [
    "old_baseline",
    "old_all_new_flags",
    "old_server_flags_only",
    "old_server_flags_no_block1",
    "old_env_flags_only",
    "old_aiter_quick_only",
    "old_aiter_only",
    "old_block1_only",
    "old_no_prefix_only",
    "old_mem090_only",
]

rows = []
for label in labels:
    run_dir = out / "runs" / label
    status_path = run_dir / "status.txt"
    status = status_path.read_text().strip() if status_path.exists() else "not_run"
    env = {}
    env_path = run_dir / "env.repro"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                env[k] = v
    agg_paths = sorted(run_dir.glob("agg_*.json"))
    raw_paths = [p for p in sorted(run_dir.glob("*.json")) if not p.name.startswith("agg_")]
    agg = json.loads(agg_paths[0].read_text()) if agg_paths else {}
    raw = json.loads(raw_paths[0].read_text()) if raw_paths else {}
    rows.append({
        "label": label,
        "status": status,
        "note": env.get("VARIANT_NOTE", ""),
        "image": env.get("IMAGE", ""),
        "commit": env.get("COMMIT", ""),
        "isl": env.get("ISL", ""),
        "osl": env.get("OSL", ""),
        "conc": env.get("CONC", ""),
        "tp": env.get("TP", ""),
        "ep": env.get("EP_SIZE", ""),
        "num_prompts": env.get("NUM_PROMPTS", raw.get("num_prompts", "")),
        "enable_hsa_no_scratch_reclaim": env.get("ENABLE_HSA_NO_SCRATCH_RECLAIM", ""),
        "enable_aiter": env.get("ENABLE_AITER", ""),
        "quick_reduce_quantization": env.get("QUICK_REDUCE_QUANTIZATION", ""),
        "gpu_memory_utilization": env.get("GPU_MEMORY_UTILIZATION", ""),
        "block_size": env.get("BLOCK_SIZE", ""),
        "disable_prefix_caching": env.get("DISABLE_PREFIX_CACHING", ""),
        "disable_log_requests": env.get("DISABLE_LOG_REQUESTS", ""),
        "completed": raw.get("completed", ""),
        "duration": raw.get("duration", ""),
        "total_input_tokens": raw.get("total_input_tokens", ""),
        "total_output_tokens": raw.get("total_output_tokens", ""),
        "total_token_throughput": raw.get("total_token_throughput", ""),
        "output_throughput": raw.get("output_throughput", ""),
        "tput_per_gpu": agg.get("tput_per_gpu", ""),
        "output_tput_per_gpu": agg.get("output_tput_per_gpu", ""),
        "input_tput_per_gpu": agg.get("input_tput_per_gpu", ""),
        "mean_ttft_ms": raw.get("mean_ttft_ms", ""),
        "mean_tpot_ms": raw.get("mean_tpot_ms", ""),
        "mean_itl_ms": raw.get("mean_itl_ms", ""),
        "reference_old_tput_per_gpu": env.get("REFERENCE_OLD_TPUT_PER_GPU", ""),
        "reference_new_tput_per_gpu": env.get("REFERENCE_NEW_TPUT_PER_GPU", ""),
    })

fields = list(rows[0]) if rows else []
with (out / "combined_results.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
PY
}

append_results_to_readme() {
  python3 - "$OUT_DIR" <<'PY'
import csv
from pathlib import Path
import sys

out = Path(sys.argv[1])
csv_path = out / "combined_results.csv"
if not csv_path.exists():
    raise SystemExit(0)
rows = list(csv.DictReader(csv_path.open()))
readme = out / "README.md"
text = readme.read_text()
marker = "\n## Results\n"
if marker in text:
    text = text.split(marker, 1)[0].rstrip() + "\n"
text += "\n## Results\n\n"
text += "| label | status | flags | tok/s/GPU | output tok/s/GPU | duration s | completed |\n"
text += "|---|---|---|---:|---:|---:|---:|\n"
for r in rows:
    flags = (
        f"hsa={r['enable_hsa_no_scratch_reclaim']}, "
        f"aiter={r['enable_aiter']}, "
        f"quick={r['quick_reduce_quantization'] or 'none'}, "
        f"mem={r['gpu_memory_utilization']}, "
        f"block={r['block_size']}, "
        f"no_prefix={r['disable_prefix_caching']}, "
        f"disable_log={r['disable_log_requests']}"
    )
    def fmt(x):
        return f"{float(x):.6f}" if x not in ("", None) else ""
    text += (
        f"| `{r['label']}` | {r['status']} | {flags} | "
        f"{fmt(r['tput_per_gpu'])} | {fmt(r['output_tput_per_gpu'])} | "
        f"{fmt(r['duration'])} | {r['completed']} |\n"
    )
readme.write_text(text)
PY
}

main() {
  need_cmd git
  need_cmd docker
  need_cmd curl
  need_cmd python3
  ensure_hf_token

  mkdir -p "$OUT_DIR/runs" "$OUT_DIR/workflow_metadata" "$MODEL_DIR" "$HF_HOME_DIR"
  write_readme
  download_workflow_metadata

  local -a labels=()
  mapfile -t labels < <(selected_labels)

  local label
  for label in "${labels[@]}"; do
    if run_one "$label"; then
      write_combined_csv
      append_results_to_readme
    else
      local status=$?
      write_combined_csv
      append_results_to_readme
      echo "[error] $label failed with status=$status" >&2
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
        exit "$status"
      fi
    fi
  done

  write_combined_csv
  append_results_to_readme
  echo "[done] results: $OUT_DIR"
  echo "[done] combined CSV: $OUT_DIR/combined_results.csv"
}

main "$@"
