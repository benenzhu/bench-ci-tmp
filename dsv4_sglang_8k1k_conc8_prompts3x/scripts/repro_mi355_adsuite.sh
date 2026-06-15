#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/InferenceX}
WT_ROOT=${WT_ROOT:-/scratch/inferencex_worktrees/mi355_adsuite_repro}
OUT_DIR=${OUT_DIR:-/scratch/inferencex_runs/mi355_adsuite_prompts3x}
MODEL_DIR=${MODEL_DIR:-/scratch/hf_hub_cache}
HF_HOME_DIR=${HF_HOME_DIR:-/scratch/hf_home}
BACKUP_CACHE_DIR=${BACKUP_CACHE_DIR:-/scratch/hf_hub_cache}
PORT=${PORT:-18888}
PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER:-3}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.8}
SKIP_EXISTING=${SKIP_EXISTING:-1}
TARGETS=${TARGETS:-all}
HF_TOKEN_FILE=${HF_TOKEN_FILE:-/root/.cache/huggingface/token}
DRY_RUN=${DRY_RUN:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-0}

REPO_SLUG=SemiAnalysisAI/InferenceX

LABEL_ORDER=(
  kimi_old
  kimi_new
  dsv4_old
  dsv4_new
  glm_old
  glm_new
)

declare -A MODEL_GROUP=(
  [kimi_old]=kimi
  [kimi_new]=kimi
  [dsv4_old]=dsv4
  [dsv4_new]=dsv4
  [glm_old]=glm
  [glm_new]=glm
)

declare -A VARIANT=(
  [kimi_old]=old
  [kimi_new]=new
  [dsv4_old]=old
  [dsv4_new]=new
  [glm_old]=old
  [glm_new]=new
)

declare -A COMMIT=(
  [kimi_old]=bc818474fdba28c4802e823fdc474be56fb79452
  [kimi_new]=a187fa20e7785c88e136c834c411fc39022c75ba
  [dsv4_old]=ac04da00630f0fc1cc127ac62eae9783c89ed6ac
  [dsv4_new]=4e52736885b61250765966a2b10d6cfa9d82d16f
  [glm_old]=f3d0fc5bd7b148daad4e16a7ae79cbf762e9e278
  [glm_new]=6a3ca2d293795be4b567fb61589fb80ca010d4c7
)

declare -A IMAGE=(
  [kimi_old]=vllm/vllm-openai-rocm:v0.16.0
  [kimi_new]=vllm/vllm-openai-rocm:v0.22.0
  [dsv4_old]=vllm/vllm-openai-rocm:v0.21.0
  [dsv4_new]=vllm/vllm-openai-rocm:v0.22.0
  [glm_old]=rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260219
  [glm_new]=lmsysorg/sglang-rocm:v0.5.10rc0-rocm720-mi35x-20260413
)

declare -A ORIGINAL_IMAGE=(
  [kimi_old]=vllm/vllm-openai-rocm:v0.16.0
  [kimi_new]=vllm/vllm-openai-rocm:v0.22.0
  [dsv4_old]=vllm/vllm-openai-rocm:nightly-b50646e5effd7cb5884cd96fdff4c53c18521198
  [dsv4_new]=vllm/vllm-openai-rocm:v0.22.0
  [glm_old]=rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260219
  [glm_new]=lmsysorg/sglang-rocm:v0.5.10rc0-rocm720-mi35x-20260413
)

declare -A IMAGE_NOTE=(
  [kimi_old]=
  [kimi_new]=
  [dsv4_old]="original workflow image tag is no longer available from Docker Hub; using v0.21.0 released 2026-05-15 as the closest public release before the 2026-05-19 workflow"
  [dsv4_new]=
  [glm_old]=
  [glm_new]=
)

declare -A RUNTIME_PATCH_NOTE=(
  [kimi_old]=
  [kimi_new]=
  [dsv4_old]="patch vLLM 0.21.0 MHC TileLang kernels inside the ephemeral container to disable PDL calls on ROCm; without this, server startup fails with 'PDL is not supported'"
  [dsv4_new]=
  [glm_old]=
  [glm_new]=
)

declare -A MODEL=(
  [kimi_old]=amd/Kimi-K2.5-MXFP4
  [kimi_new]=amd/Kimi-K2.5-MXFP4
  [dsv4_old]=deepseek-ai/DeepSeek-V4-Pro
  [dsv4_new]=deepseek-ai/DeepSeek-V4-Pro
  [glm_old]=zai-org/GLM-5-FP8
  [glm_new]=zai-org/GLM-5-FP8
)

declare -A MODEL_PREFIX=(
  [kimi_old]=kimik2.5
  [kimi_new]=kimik2.5
  [dsv4_old]=dsv4
  [dsv4_new]=dsv4
  [glm_old]=glm5
  [glm_new]=glm5
)

declare -A PRECISION=(
  [kimi_old]=fp4
  [kimi_new]=fp4
  [dsv4_old]=fp4
  [dsv4_new]=fp4
  [glm_old]=fp8
  [glm_new]=fp8
)

declare -A FRAMEWORK=(
  [kimi_old]=vllm
  [kimi_new]=vllm
  [dsv4_old]=vllm
  [dsv4_new]=vllm
  [glm_old]=sglang
  [glm_new]=sglang
)

declare -A SCRIPT=(
  [kimi_old]=benchmarks/single_node/kimik2.5_fp4_mi355x.sh
  [kimi_new]=benchmarks/single_node/kimik2.5_fp4_mi355x.sh
  [dsv4_old]=benchmarks/single_node/dsv4_fp4_mi355x_vllm.sh
  [dsv4_new]=benchmarks/single_node/dsv4_fp4_mi355x_vllm.sh
  [glm_old]=benchmarks/single_node/glm5_fp8_mi355x.sh
  [glm_new]=benchmarks/single_node/glm5_fp8_mi355x.sh
)

declare -A ISL=(
  [kimi_old]=8192
  [kimi_new]=8192
  [dsv4_old]=1024
  [dsv4_new]=1024
  [glm_old]=1024
  [glm_new]=1024
)

declare -A OSL=(
  [kimi_old]=1024
  [kimi_new]=1024
  [dsv4_old]=1024
  [dsv4_new]=1024
  [glm_old]=1024
  [glm_new]=1024
)

declare -A CONC=(
  [kimi_old]=4
  [kimi_new]=4
  [dsv4_old]=4
  [dsv4_new]=4
  [glm_old]=4
  [glm_new]=4
)

declare -A TP=(
  [kimi_old]=8
  [kimi_new]=8
  [dsv4_old]=8
  [dsv4_new]=8
  [glm_old]=8
  [glm_new]=8
)

declare -A EP_SIZE=(
  [kimi_old]=1
  [kimi_new]=1
  [dsv4_old]=1
  [dsv4_new]=1
  [glm_old]=1
  [glm_new]=1
)

declare -A DP_ATTENTION=(
  [kimi_old]=false
  [kimi_new]=false
  [dsv4_old]=false
  [dsv4_new]=false
  [glm_old]=false
  [glm_new]=false
)

declare -A RUN_ID=(
  [kimi_old]=22555187591
  [kimi_new]=26696253037
  [dsv4_old]=26110663181
  [dsv4_new]=26696268345
  [glm_old]=22792161490
  [glm_new]=24574269364
)

declare -A ATTEMPT=(
  [kimi_old]=2
  [kimi_new]=1
  [dsv4_old]=1
  [dsv4_new]=1
  [glm_old]=5
  [glm_new]=1
)

declare -A JOB_ID=(
  [kimi_old]=65962398410
  [kimi_new]=78682149908
  [dsv4_old]=76787086168
  [dsv4_new]=78681278252
  [glm_old]=66170603755
  [glm_new]=71855382514
)

declare -A EXPECTED_OLD_TPUT_PER_GPU=(
  [kimi]=28.7
  [dsv4]=3.3
  [glm]=17.5
)

declare -A EXPECTED_NEW_TPUT_PER_GPU=(
  [kimi]=374.2
  [dsv4]=23.3
  [glm]=44.2
)

declare -A CACHE_DIR=(
  [kimi]=models--amd--Kimi-K2.5-MXFP4
  [dsv4]=models--deepseek-ai--DeepSeek-V4-Pro
  [glm]=models--zai-org--GLM-5-FP8
)

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

max_model_len_for() {
  local label=$1
  echo $(( ISL[$label] + OSL[$label] + 200 ))
}

worktree_for() {
  local label=$1
  local short=${COMMIT[$label]:0:7}
  echo "$WT_ROOT/$label-$short"
}

run_dir_for() {
  local label=$1
  echo "$OUT_DIR/runs/$label"
}

result_filename_for() {
  local label=$1
  local model=${MODEL_PREFIX[$label]}
  local seq="${ISL[$label]}x${OSL[$label]}"
  local precision=${PRECISION[$label]}
  local framework=${FRAMEWORK[$label]}
  local tp=${TP[$label]}
  local ep=${EP_SIZE[$label]}
  local dpa=${DP_ATTENTION[$label]}
  local conc=${CONC[$label]}
  local variant=${VARIANT[$label]}
  echo "${model}_${seq}_${precision}_${framework}_mi355x_tp${tp}-ep${ep}-dpa${dpa}_disagg-false_spec-none_conc${conc}_prompts${PROMPT_MULTIPLIER}x_${variant}"
}

selected_labels() {
  local normalized
  normalized=${TARGETS//,/ }
  if [[ "$normalized" == "all" ]]; then
    printf '%s\n' "${LABEL_ORDER[@]}"
    return 0
  fi

  local token
  for token in $normalized; do
    case "$token" in
      kimi) printf '%s\n' kimi_old kimi_new ;;
      dsv4|deepseek|deepseek-v4-pro) printf '%s\n' dsv4_old dsv4_new ;;
      glm|glm5|glm-5) printf '%s\n' glm_old glm_new ;;
      kimi_old|kimi_new|dsv4_old|dsv4_new|glm_old|glm_new) printf '%s\n' "$token" ;;
      *)
        echo "Unknown TARGETS token: $token" >&2
        exit 1
        ;;
    esac
  done | awk '!seen[$0]++'
}

ensure_hf_token() {
  if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
    HF_TOKEN=$(<"$HF_TOKEN_FILE")
    export HF_TOKEN
  fi
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is empty and $HF_TOKEN_FILE was not readable" >&2
    exit 1
  fi
}

ensure_commit() {
  local commit=$1
  if ! git -C "$REPO" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    git -C "$REPO" fetch origin "$commit"
  fi
}

prepare_worktree() {
  local label=$1
  local commit=${COMMIT[$label]}
  local dest
  dest=$(worktree_for "$label")
  local script=${SCRIPT[$label]}

  ensure_commit "$commit"
  mkdir -p "$WT_ROOT"

  if [[ -e "$dest" && ! -f "$dest/.mi355_adsuite_owner" ]]; then
    echo "Refusing to modify existing unowned worktree path: $dest" >&2
    exit 1
  fi

  if [[ ! -e "$dest/.git" ]]; then
    git -C "$REPO" worktree add --force --detach "$dest" "$commit"
    touch "$dest/.mi355_adsuite_owner"
  else
    git -C "$dest" reset --hard "$commit" >/dev/null
    git -C "$dest" clean -fdx >/dev/null
    touch "$dest/.mi355_adsuite_owner"
  fi

  if [[ ! -f "$dest/$script" ]]; then
    echo "Benchmark script not found: $dest/$script" >&2
    exit 1
  fi

  perl -0pi -e 's/--num-prompts "\$\(\(CONC \* 10\)\)"/--num-prompts "\$\(\(CONC * PROMPT_MULTIPLIER\)\)"/g' "$dest/$script"
  if ! rg -F -q -- '--num-prompts "$((CONC * PROMPT_MULTIPLIER))"' "$dest/$script"; then
    echo "Failed to patch num-prompts in $dest/$script" >&2
    exit 1
  fi

  git -C "$dest" diff -- "$script" > "$(run_dir_for "$label")/local_patch.diff" 2>/dev/null || true
}

write_suite_readme() {
  mkdir -p "$OUT_DIR"
  cat > "$OUT_DIR/README.md" <<EOF
# MI355X Ad Benchmark Repro

Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)

Scope:

- Hardware: MI355X
- Mode: single-node non-disaggregated
- Spec decoding: none
- Prompt count patch: \`num_prompts = CONC * $PROMPT_MULTIPLIER\`
- HF cache: \`$MODEL_DIR\`
- HF home/xet cache: \`$HF_HOME_DIR\`
- Output root: \`$OUT_DIR\`

Targets:

| label | model | precision | framework | image | commit | seq | conc | tp | workflow run | job |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
EOF
  local label
  for label in "${LABEL_ORDER[@]}"; do
    printf '| %s | `%s` | %s | %s | `%s` | `%s` | %s/%s | %s | %s | %s attempt %s | %s |\n' \
      "$label" "${MODEL[$label]}" "${PRECISION[$label]}" "${FRAMEWORK[$label]}" \
      "${IMAGE[$label]}" "${COMMIT[$label]:0:12}" "${ISL[$label]}" "${OSL[$label]}" \
      "${CONC[$label]}" "${TP[$label]}" "${RUN_ID[$label]}" "${ATTEMPT[$label]}" "${JOB_ID[$label]}" \
      >> "$OUT_DIR/README.md"
  done

  cat >> "$OUT_DIR/README.md" <<'EOF'

Image notes:

EOF
  for label in "${LABEL_ORDER[@]}"; do
    if [[ -n "${IMAGE_NOTE[$label]}" ]]; then
      printf -- '- `%s`: original workflow image `%s`; running image `%s`. %s.\n' \
        "$label" "${ORIGINAL_IMAGE[$label]}" "${IMAGE[$label]}" "${IMAGE_NOTE[$label]}" \
        >> "$OUT_DIR/README.md"
    fi
  done

  cat >> "$OUT_DIR/README.md" <<'EOF'

Runtime patch notes:

EOF
  for label in "${LABEL_ORDER[@]}"; do
    if [[ -n "${RUNTIME_PATCH_NOTE[$label]}" ]]; then
      printf -- '- `%s`: %s.\n' "$label" "${RUNTIME_PATCH_NOTE[$label]}" >> "$OUT_DIR/README.md"
    fi
  done
}

write_repro_commands_header() {
  mkdir -p "$OUT_DIR"
  cat > "$OUT_DIR/repro_commands.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

# Token is intentionally not written here. Load it at runtime:
# export HF_TOKEN="\$(cat $HF_TOKEN_FILE)"

export REPO=$REPO
export WT_ROOT=$WT_ROOT
export OUT_DIR=$OUT_DIR
export MODEL_DIR=$MODEL_DIR
export HF_HOME_DIR=$HF_HOME_DIR
export BACKUP_CACHE_DIR=$BACKUP_CACHE_DIR
export PORT=$PORT
export PROMPT_MULTIPLIER=$PROMPT_MULTIPLIER
export RANDOM_RANGE_RATIO=$RANDOM_RANGE_RATIO
export SKIP_EXISTING=1
export DRY_RUN=0
export CONTINUE_ON_ERROR=0

# Full suite:
TARGETS=all /root/repro_mi355_adsuite.sh

# Individual model pairs:
TARGETS=kimi /root/repro_mi355_adsuite.sh
TARGETS=dsv4 /root/repro_mi355_adsuite.sh
TARGETS=glm /root/repro_mi355_adsuite.sh

# Individual labels:
EOF
  local label
  for label in "${LABEL_ORDER[@]}"; do
    printf 'TARGETS=%s /root/repro_mi355_adsuite.sh\n' "$label" >> "$OUT_DIR/repro_commands.sh"
  done
  chmod +x "$OUT_DIR/repro_commands.sh"
}

write_run_env_file() {
  local label=$1
  local run_dir
  run_dir=$(run_dir_for "$label")
  mkdir -p "$run_dir"
  cat > "$run_dir/env.repro" <<EOF
LABEL=$label
MODEL_GROUP=${MODEL_GROUP[$label]}
VARIANT=${VARIANT[$label]}
COMMIT=${COMMIT[$label]}
IMAGE=${IMAGE[$label]}
ORIGINAL_IMAGE=${ORIGINAL_IMAGE[$label]}
IMAGE_NOTE=${IMAGE_NOTE[$label]}
RUNTIME_PATCH_NOTE=${RUNTIME_PATCH_NOTE[$label]}
MODEL=${MODEL[$label]}
MODEL_PREFIX=${MODEL_PREFIX[$label]}
PRECISION=${PRECISION[$label]}
FRAMEWORK=${FRAMEWORK[$label]}
SCRIPT=${SCRIPT[$label]}
RUNNER_TYPE=mi355x
TP=${TP[$label]}
EP_SIZE=${EP_SIZE[$label]}
DP_ATTENTION=${DP_ATTENTION[$label]}
CONC=${CONC[$label]}
ISL=${ISL[$label]}
OSL=${OSL[$label]}
MAX_MODEL_LEN=$(max_model_len_for "$label")
RANDOM_RANGE_RATIO=$RANDOM_RANGE_RATIO
PROMPT_MULTIPLIER=$PROMPT_MULTIPLIER
NUM_PROMPTS=$(( CONC[$label] * PROMPT_MULTIPLIER ))
SPEC_DECODING=none
DISAGG=false
RUN_EVAL=false
EVAL_ONLY=false
PORT=$PORT
HF_HUB_CACHE=$MODEL_DIR
HF_HOME=$HF_HOME_DIR
HF_XET_CACHE=$HF_HOME_DIR/xet
WORKFLOW_RUN_ID=${RUN_ID[$label]}
WORKFLOW_ATTEMPT=${ATTEMPT[$label]}
WORKFLOW_JOB_ID=${JOB_ID[$label]}
EOF
}

write_runtime_patch_script() {
  local label=$1
  local wt
  wt=$(worktree_for "$label")
  local patch_file="$wt/.mi355_adsuite_runtime_patch.sh"

  if [[ "$label" == "dsv4_old" ]]; then
    cat > "$patch_file" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mhc.py")
text = path.read_text()
patched = text.replace(
    "T.pdl_sync()",
    "pass  # mi355_adsuite: disable TileLang PDL on ROCm",
).replace(
    "T.pdl_trigger()",
    "pass  # mi355_adsuite: disable TileLang PDL on ROCm",
)
if patched == text:
    print(f"[runtime-patch] no PDL calls found in {path}")
else:
    path.write_text(patched)
    print(f"[runtime-patch] disabled TileLang PDL calls in {path}")
PY
EOF
  else
    cat > "$patch_file" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
EOF
  fi
  chmod +x "$patch_file"
}

download_workflow_logs() {
  local label=$1
  local dest="$OUT_DIR/workflow_logs/$label"
  mkdir -p "$dest"

  local run_api="https://api.github.com/repos/$REPO_SLUG/actions/runs/${RUN_ID[$label]}"
  local jobs_api="https://api.github.com/repos/$REPO_SLUG/actions/runs/${RUN_ID[$label]}/attempts/${ATTEMPT[$label]}/jobs?per_page=100"
  local job_log_api="https://api.github.com/repos/$REPO_SLUG/actions/jobs/${JOB_ID[$label]}/logs"

  curl -fsSL "$run_api" -o "$dest/run.json" || true
  curl -fsSL "$jobs_api" -o "$dest/jobs.json" || true
  curl -fsSL -L "$job_log_api" -o "$dest/job_${JOB_ID[$label]}.log" || true
}

cleanup_named_container() {
  local label=$1
  docker rm -f "mi355-adsuite-$label" >/dev/null 2>&1 || true
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
  local script=${SCRIPT[$label]}

  mkdir -p "$run_dir"
  printf '%s\n' "$status" > "$run_dir/status.txt"
  [[ -f "$wt/${result_filename}.json" ]] && cp -f "$wt/${result_filename}.json" "$run_dir/"
  [[ -f "$wt/agg_${result_filename}.json" ]] && cp -f "$wt/agg_${result_filename}.json" "$run_dir/"
  [[ -f "$wt/server.log" ]] && cp -f "$wt/server.log" "$run_dir/server.log"
  [[ -f "$wt/gpu_metrics.csv" ]] && cp -f "$wt/gpu_metrics.csv" "$run_dir/gpu_metrics.csv"
  [[ -f "$wt/$script" ]] && cp -f "$wt/$script" "$run_dir/$(basename "$script").patched"
  [[ -f "$wt/.mi355_adsuite_runtime_patch.sh" ]] && cp -f "$wt/.mi355_adsuite_runtime_patch.sh" "$run_dir/runtime_patch.sh"
  git -C "$wt" diff -- "$script" > "$run_dir/local_patch.diff" 2>/dev/null || true
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
    FRAMEWORK="${FRAMEWORK[$label]}" \
    PRECISION="${PRECISION[$label]}" \
    SPEC_DECODING=none \
    RESULT_FILENAME="$result_filename" \
    ISL="${ISL[$label]}" \
    OSL="${OSL[$label]}" \
    DISAGG=false \
    MODEL_PREFIX="${MODEL_PREFIX[$label]}" \
    IMAGE="${IMAGE[$label]}" \
    TP="${TP[$label]}" \
    EP_SIZE="${EP_SIZE[$label]}" \
    DP_ATTENTION="${DP_ATTENTION[$label]}" \
    python3 utils/process_result.py
  )
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
  local script=${SCRIPT[$label]}

  mkdir -p "$run_dir"
  write_run_env_file "$label"
  prepare_worktree "$label"
  write_runtime_patch_script "$label"

  if [[ "$SKIP_EXISTING" == "1" && -s "$result_path" && -s "$run_dir/agg_${result_filename}.json" ]]; then
    echo "[skip] $label already has raw and aggregate results"
    return 0
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    copy_run_artifacts "$label" "dry_run"
    echo "[dry-run] $label prepared worktree and patched ${SCRIPT[$label]}"
    return 0
  fi

  download_workflow_logs "$label"
  cleanup_named_container "$label"
  rm -f "$wt/${result_filename}.json" "$wt/agg_${result_filename}.json" "$wt/server.log" "$wt/gpu_metrics.csv"
  if [[ -s "$docker_log" ]]; then
    mv "$docker_log" "$run_dir/docker.$(date -u +%Y%m%dT%H%M%SZ).log"
  fi

  echo "[run] $label model=${MODEL[$label]} image=${IMAGE[$label]} seq=${ISL[$label]}/${OSL[$label]} conc=${CONC[$label]} tp=${TP[$label]} prompts=$(( CONC[$label] * PROMPT_MULTIPLIER ))"

  set +e
  docker run --rm \
    --name "mi355-adsuite-$label" \
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
    -e MODEL="${MODEL[$label]}" \
    -e MODEL_PREFIX="${MODEL_PREFIX[$label]}" \
    -e IMAGE="${IMAGE[$label]}" \
    -e FRAMEWORK="${FRAMEWORK[$label]}" \
    -e PRECISION="${PRECISION[$label]}" \
    -e RUNNER_TYPE=mi355x \
    -e TP="${TP[$label]}" \
    -e EP_SIZE="${EP_SIZE[$label]}" \
    -e DP_ATTENTION="${DP_ATTENTION[$label]}" \
    -e CONC="${CONC[$label]}" \
    -e ISL="${ISL[$label]}" \
    -e OSL="${OSL[$label]}" \
    -e MAX_MODEL_LEN="$(max_model_len_for "$label")" \
    -e RANDOM_RANGE_RATIO="$RANDOM_RANGE_RATIO" \
    -e RESULT_FILENAME="$result_filename" \
    -e SPEC_DECODING=none \
    -e DISAGG=false \
    -e RUN_EVAL=false \
    -e EVAL_ONLY=false \
    -e PORT="$PORT" \
    -e PROMPT_MULTIPLIER="$PROMPT_MULTIPLIER" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPYCACHEPREFIX=/tmp/inferencex-pycache \
    "${IMAGE[$label]}" \
    -lc "bash /workspace/.mi355_adsuite_runtime_patch.sh && bash /workspace/$script" > "$docker_log" 2>&1
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

backup_model_cache_for_group() {
  local group=$1
  local cache_name=${CACHE_DIR[$group]}
  local src="$MODEL_DIR/$cache_name"
  local dst="$BACKUP_CACHE_DIR/$cache_name"
  mkdir -p "$BACKUP_CACHE_DIR" "$OUT_DIR/cache_status"

  {
    echo "group=$group"
    echo "cache_name=$cache_name"
    echo "source=$src"
    echo "destination=$dst"
    date -u +timestamp=%Y-%m-%dT%H:%M:%SZ
  } > "$OUT_DIR/cache_status/$group.txt"

  if [[ ! -e "$src" ]]; then
    echo "status=missing_source" >> "$OUT_DIR/cache_status/$group.txt"
    return 0
  fi

  du -sh "$src" >> "$OUT_DIR/cache_status/$group.txt" 2>/dev/null || true

  if [[ "$(readlink -f "$MODEL_DIR")" == "$(readlink -f "$BACKUP_CACHE_DIR")" ]]; then
    echo "status=already_on_persistent_cache" >> "$OUT_DIR/cache_status/$group.txt"
    return 0
  fi

  rsync -aH --whole-file --info=progress2 "$src" "$BACKUP_CACHE_DIR/" \
    > "$OUT_DIR/cache_status/${group}_rsync.log" 2>&1
  echo "status=rsynced" >> "$OUT_DIR/cache_status/$group.txt"
}

backup_results_root() {
  mkdir -p "$OUT_DIR/cache_status"
  if [[ "$OUT_DIR" == /scratch/* ]]; then
    echo "results already under /scratch: $OUT_DIR" > "$OUT_DIR/cache_status/results_backup.txt"
    return 0
  fi
  local dst="/scratch/inferencex_runs/$(basename "$OUT_DIR")"
  rsync -aH --whole-file --info=progress2 "$OUT_DIR/" "$dst/" \
    > "$OUT_DIR/cache_status/results_rsync.log" 2>&1
  echo "results copied to $dst" > "$OUT_DIR/cache_status/results_backup.txt"
}

write_combined_csv() {
  python3 - "$OUT_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
labels = ["kimi_old", "kimi_new", "dsv4_old", "dsv4_new", "glm_old", "glm_new"]
groups = {"kimi_old": "kimi", "kimi_new": "kimi", "dsv4_old": "dsv4", "dsv4_new": "dsv4", "glm_old": "glm", "glm_new": "glm"}
variants = {label: label.rsplit("_", 1)[1] for label in labels}
expected = {
    "kimi": {"old": 28.7, "new": 374.2},
    "dsv4": {"old": 3.3, "new": 23.3},
    "glm": {"old": 17.5, "new": 44.2},
}

rows = []
by_group = {}
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
    group = groups[label]
    variant = variants[label]
    row = {
        "model_group": group,
        "variant": variant,
        "label": label,
        "status": status,
        "model": env.get("MODEL", agg.get("model", raw.get("model_id", ""))),
        "precision": env.get("PRECISION", agg.get("precision", "")),
        "framework": env.get("FRAMEWORK", agg.get("framework", "")),
        "image": env.get("IMAGE", agg.get("image", "")),
        "original_image": env.get("ORIGINAL_IMAGE", env.get("IMAGE", agg.get("image", ""))),
        "image_note": env.get("IMAGE_NOTE", ""),
        "runtime_patch_note": env.get("RUNTIME_PATCH_NOTE", ""),
        "commit": env.get("COMMIT", ""),
        "workflow_run_id": env.get("WORKFLOW_RUN_ID", ""),
        "workflow_attempt": env.get("WORKFLOW_ATTEMPT", ""),
        "workflow_job_id": env.get("WORKFLOW_JOB_ID", ""),
        "isl": env.get("ISL", agg.get("isl", "")),
        "osl": env.get("OSL", agg.get("osl", "")),
        "conc": env.get("CONC", agg.get("conc", raw.get("max_concurrency", ""))),
        "tp": env.get("TP", agg.get("tp", "")),
        "ep": env.get("EP_SIZE", agg.get("ep", "")),
        "dp_attention": env.get("DP_ATTENTION", agg.get("dp_attention", "")),
        "num_prompts": env.get("NUM_PROMPTS", raw.get("num_prompts", "")),
        "total_token_throughput": raw.get("total_token_throughput", ""),
        "output_throughput": raw.get("output_throughput", ""),
        "tput_per_gpu": agg.get("tput_per_gpu", ""),
        "output_tput_per_gpu": agg.get("output_tput_per_gpu", ""),
        "input_tput_per_gpu": agg.get("input_tput_per_gpu", ""),
        "mean_ttft_ms": raw.get("mean_ttft_ms", ""),
        "mean_tpot_ms": raw.get("mean_tpot_ms", ""),
        "mean_itl_ms": raw.get("mean_itl_ms", ""),
        "expected_dashboard_tput_per_gpu": expected[group][variant],
    }
    rows.append(row)
    by_group.setdefault(group, {})[variant] = row

combined_path = out / "combined_results.csv"
fields = list(rows[0].keys()) if rows else []
with combined_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

comp_fields = [
    "model_group",
    "old_label",
    "new_label",
    "old_status",
    "new_status",
    "old_tput_per_gpu",
    "new_tput_per_gpu",
    "measured_gain_x",
    "measured_improvement_pct",
    "expected_old_tput_per_gpu",
    "expected_new_tput_per_gpu",
    "expected_gain_x",
]
comp_rows = []
for group in ["dsv4", "kimi", "glm"]:
    old = by_group.get(group, {}).get("old", {})
    new = by_group.get(group, {}).get("new", {})
    try:
        old_t = float(old.get("tput_per_gpu", ""))
        new_t = float(new.get("tput_per_gpu", ""))
        gain = new_t / old_t if old_t else ""
        pct = (gain - 1.0) * 100 if gain != "" else ""
    except (TypeError, ValueError):
        old_t = old.get("tput_per_gpu", "")
        new_t = new.get("tput_per_gpu", "")
        gain = ""
        pct = ""
    exp_old = expected[group]["old"]
    exp_new = expected[group]["new"]
    comp_rows.append({
        "model_group": group,
        "old_label": old.get("label", f"{group}_old"),
        "new_label": new.get("label", f"{group}_new"),
        "old_status": old.get("status", "not_run"),
        "new_status": new.get("status", "not_run"),
        "old_tput_per_gpu": old_t,
        "new_tput_per_gpu": new_t,
        "measured_gain_x": gain,
        "measured_improvement_pct": pct,
        "expected_old_tput_per_gpu": exp_old,
        "expected_new_tput_per_gpu": exp_new,
        "expected_gain_x": exp_new / exp_old,
    })

with (out / "comparison.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=comp_fields)
    writer.writeheader()
    writer.writerows(comp_rows)
PY
}

main() {
  need_cmd git
  need_cmd docker
  need_cmd curl
  need_cmd python3
  need_cmd rg
  ensure_hf_token

  mkdir -p "$OUT_DIR/runs" "$OUT_DIR/workflow_logs" "$MODEL_DIR" "$HF_HOME_DIR" "$BACKUP_CACHE_DIR"
  write_suite_readme
  write_repro_commands_header

  local -a labels=()
  mapfile -t labels < <(selected_labels)

  local label
  for label in "${labels[@]}"; do
    if run_one "$label"; then
      if [[ "$DRY_RUN" != "1" ]]; then
        backup_model_cache_for_group "${MODEL_GROUP[$label]}"
      fi
      write_combined_csv
    else
      local status=$?
      write_combined_csv
      echo "[error] stopping after $label failure; set CONTINUE_ON_ERROR=1 to continue remaining labels" >&2
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
        exit "$status"
      fi
    fi
  done

  if [[ "$DRY_RUN" != "1" ]]; then
    backup_results_root
  fi
  write_combined_csv
  echo "[done] results: $OUT_DIR"
  echo "[done] combined CSV: $OUT_DIR/combined_results.csv"
  echo "[done] comparison CSV: $OUT_DIR/comparison.csv"
}

main "$@"
