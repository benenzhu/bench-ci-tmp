#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/InferenceX}
WT_ROOT=${WT_ROOT:-/scratch/inferencex_worktrees/dsv4_sglang_geomean}
OUT_DIR=${OUT_DIR:-/scratch/inferencex_runs/dsv4_sglang_geomean}
MODEL_DIR=${MODEL_DIR:-/scratch/hf_hub_cache}
HF_HOME_DIR=${HF_HOME_DIR:-/scratch/hf_home}
PORT=${PORT:-18888}
PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER:-3}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.8}
SKIP_EXISTING=${SKIP_EXISTING:-1}
TARGETS=${TARGETS:-all}
HF_TOKEN_FILE=${HF_TOKEN_FILE:-/root/.cache/huggingface/token}
DRY_RUN=${DRY_RUN:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-0}

REPO_SLUG=SemiAnalysisAI/InferenceX
MODEL_CACHE_NAME=models--deepseek-ai--DeepSeek-V4-Pro
MODEL_CACHE_ROOT="$MODEL_DIR/$MODEL_CACHE_NAME"
CONFIG_BACKUP_ROOT="$OUT_DIR/cache_status/hf_config_backups"

LABEL_ORDER=(
  dsv4_8k1k_c8_old
  dsv4_8k1k_c8_new
  dsv4_1k1k_c1_old
  dsv4_1k1k_c1_new
  dsv4_1k1k_c2_old
  dsv4_1k1k_c2_new
  dsv4_1k1k_c3_old
  dsv4_1k1k_c3_new
  dsv4_1k1k_c4_old
  dsv4_1k1k_c4_new
  dsv4_1k1k_c8_old
  dsv4_1k1k_c8_new
  dsv4_1k1k_c16_old
  dsv4_1k1k_c16_new
  dsv4_1k512_c1_old
  dsv4_1k512_c1_new
  dsv4_1k8k_c8_old
  dsv4_1k8k_c8_new
)

declare -A VARIANT=(
  [dsv4_8k1k_c8_old]=old
  [dsv4_8k1k_c8_new]=new
  [dsv4_1k1k_c8_old]=old
  [dsv4_1k1k_c8_new]=new
  [dsv4_1k8k_c8_old]=old
  [dsv4_1k8k_c8_new]=new
)

declare -A CONFIG_ID=(
  [dsv4_8k1k_c8_old]=8k1k_c8
  [dsv4_8k1k_c8_new]=8k1k_c8
  [dsv4_1k1k_c8_old]=1k1k_c8
  [dsv4_1k1k_c8_new]=1k1k_c8
  [dsv4_1k8k_c8_old]=1k8k_c8
  [dsv4_1k8k_c8_new]=1k8k_c8
)

declare -A COMMIT=(
  [dsv4_8k1k_c8_old]=f93f91dfb0efbc46e363dbcdf96e983ea96d214d
  [dsv4_8k1k_c8_new]=8d206016537a519824e527f7524ca52d17303512
  [dsv4_1k1k_c8_old]=f93f91dfb0efbc46e363dbcdf96e983ea96d214d
  [dsv4_1k1k_c8_new]=8d206016537a519824e527f7524ca52d17303512
  [dsv4_1k8k_c8_old]=f93f91dfb0efbc46e363dbcdf96e983ea96d214d
  [dsv4_1k8k_c8_new]=8d206016537a519824e527f7524ca52d17303512
)

declare -A IMAGE=(
  [dsv4_8k1k_c8_old]=rocm/sgl-dev:rocm720-mi35x-583b1b6-20260501-DSv4
  [dsv4_8k1k_c8_new]=lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612
  [dsv4_1k1k_c8_old]=rocm/sgl-dev:rocm720-mi35x-583b1b6-20260501-DSv4
  [dsv4_1k1k_c8_new]=lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612
  [dsv4_1k8k_c8_old]=rocm/sgl-dev:rocm720-mi35x-583b1b6-20260501-DSv4
  [dsv4_1k8k_c8_new]=lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612
)

declare -A MODEL=(
  [dsv4_8k1k_c8_old]=deepseek-ai/DeepSeek-V4-Pro
  [dsv4_8k1k_c8_new]=deepseek-ai/DeepSeek-V4-Pro
  [dsv4_1k1k_c8_old]=deepseek-ai/DeepSeek-V4-Pro
  [dsv4_1k1k_c8_new]=deepseek-ai/DeepSeek-V4-Pro
  [dsv4_1k8k_c8_old]=deepseek-ai/DeepSeek-V4-Pro
  [dsv4_1k8k_c8_new]=deepseek-ai/DeepSeek-V4-Pro
)

declare -A MODEL_PREFIX=(
  [dsv4_8k1k_c8_old]=dsv4
  [dsv4_8k1k_c8_new]=dsv4
  [dsv4_1k1k_c8_old]=dsv4
  [dsv4_1k1k_c8_new]=dsv4
  [dsv4_1k8k_c8_old]=dsv4
  [dsv4_1k8k_c8_new]=dsv4
)

declare -A PRECISION=(
  [dsv4_8k1k_c8_old]=fp4
  [dsv4_8k1k_c8_new]=fp4
  [dsv4_1k1k_c8_old]=fp4
  [dsv4_1k1k_c8_new]=fp4
  [dsv4_1k8k_c8_old]=fp4
  [dsv4_1k8k_c8_new]=fp4
)

declare -A FRAMEWORK=(
  [dsv4_8k1k_c8_old]=sglang
  [dsv4_8k1k_c8_new]=sglang
  [dsv4_1k1k_c8_old]=sglang
  [dsv4_1k1k_c8_new]=sglang
  [dsv4_1k8k_c8_old]=sglang
  [dsv4_1k8k_c8_new]=sglang
)

declare -A SCRIPT=(
  [dsv4_8k1k_c8_old]=benchmarks/single_node/dsv4_fp4_mi355x_sglang.sh
  [dsv4_8k1k_c8_new]=benchmarks/single_node/fixed_seq_len/dsv4_fp4_mi355x_sglang.sh
  [dsv4_1k1k_c8_old]=benchmarks/single_node/dsv4_fp4_mi355x_sglang.sh
  [dsv4_1k1k_c8_new]=benchmarks/single_node/fixed_seq_len/dsv4_fp4_mi355x_sglang.sh
  [dsv4_1k8k_c8_old]=benchmarks/single_node/dsv4_fp4_mi355x_sglang.sh
  [dsv4_1k8k_c8_new]=benchmarks/single_node/fixed_seq_len/dsv4_fp4_mi355x_sglang.sh
)

declare -A ISL=(
  [dsv4_8k1k_c8_old]=8192
  [dsv4_8k1k_c8_new]=8192
  [dsv4_1k1k_c8_old]=1024
  [dsv4_1k1k_c8_new]=1024
  [dsv4_1k8k_c8_old]=1024
  [dsv4_1k8k_c8_new]=1024
)

declare -A OSL=(
  [dsv4_8k1k_c8_old]=1024
  [dsv4_8k1k_c8_new]=1024
  [dsv4_1k1k_c8_old]=1024
  [dsv4_1k1k_c8_new]=1024
  [dsv4_1k8k_c8_old]=8192
  [dsv4_1k8k_c8_new]=8192
)

declare -A CONC=(
  [dsv4_8k1k_c8_old]=8
  [dsv4_8k1k_c8_new]=8
  [dsv4_1k1k_c8_old]=8
  [dsv4_1k1k_c8_new]=8
  [dsv4_1k8k_c8_old]=8
  [dsv4_1k8k_c8_new]=8
)

declare -A TP=(
  [dsv4_8k1k_c8_old]=8
  [dsv4_8k1k_c8_new]=8
  [dsv4_1k1k_c8_old]=8
  [dsv4_1k1k_c8_new]=8
  [dsv4_1k8k_c8_old]=8
  [dsv4_1k8k_c8_new]=8
)

declare -A EP_SIZE=(
  [dsv4_8k1k_c8_old]=1
  [dsv4_8k1k_c8_new]=1
  [dsv4_1k1k_c8_old]=1
  [dsv4_1k1k_c8_new]=1
  [dsv4_1k8k_c8_old]=1
  [dsv4_1k8k_c8_new]=1
)

declare -A DP_ATTENTION=(
  [dsv4_8k1k_c8_old]=false
  [dsv4_8k1k_c8_new]=false
  [dsv4_1k1k_c8_old]=false
  [dsv4_1k1k_c8_new]=false
  [dsv4_1k8k_c8_old]=false
  [dsv4_1k8k_c8_new]=false
)

declare -A RUN_ID=(
  [dsv4_8k1k_c8_old]=25262278289
  [dsv4_8k1k_c8_new]=27475314602
  [dsv4_1k1k_c8_old]=25262278289
  [dsv4_1k1k_c8_new]=27475314602
  [dsv4_1k8k_c8_old]=25262278289
  [dsv4_1k8k_c8_new]=27475314602
)

declare -A ATTEMPT=(
  [dsv4_8k1k_c8_old]=1
  [dsv4_8k1k_c8_new]=3
  [dsv4_1k1k_c8_old]=1
  [dsv4_1k1k_c8_new]=3
  [dsv4_1k8k_c8_old]=1
  [dsv4_1k8k_c8_new]=3
)

declare -A EXPECTED_TPUT_PER_GPU=(
  [dsv4_8k1k_c8_old]=115.801807
  [dsv4_8k1k_c8_new]=483.774241
  [dsv4_1k1k_c8_old]=
  [dsv4_1k1k_c8_new]=
  [dsv4_1k8k_c8_old]=
  [dsv4_1k8k_c8_new]=
)

declare -A RUNTIME_PATCH_NOTE=(
  [dsv4_8k1k_c8_old]="reused from prior strict 8k1k bundle unless rerun; host restores DeepSeek-V4-Pro HF config.json after old SGLang patches model_type deepseek_v4 -> deepseek_v3"
  [dsv4_8k1k_c8_new]="reused from prior strict 8k1k bundle unless rerun"
  [dsv4_1k1k_c8_old]="host restores DeepSeek-V4-Pro HF config.json after old SGLang patches model_type deepseek_v4 -> deepseek_v3"
  [dsv4_1k1k_c8_new]="none"
  [dsv4_1k8k_c8_old]="host restores DeepSeek-V4-Pro HF config.json after old SGLang patches model_type deepseek_v4 -> deepseek_v3"
  [dsv4_1k8k_c8_new]="none"
)

for spec in 1k1k_c1:1024:1024:1 1k1k_c2:1024:1024:2 1k1k_c3:1024:1024:3 1k1k_c4:1024:1024:4 1k1k_c16:1024:1024:16 1k512_c1:1024:512:1; do
  IFS=: read -r config_id isl osl conc <<< "$spec"
  for variant in old new; do
    label="dsv4_${config_id}_${variant}"
    VARIANT[$label]="$variant"
    CONFIG_ID[$label]="$config_id"
    MODEL[$label]=deepseek-ai/DeepSeek-V4-Pro
    MODEL_PREFIX[$label]=dsv4
    PRECISION[$label]=fp4
    FRAMEWORK[$label]=sglang
    ISL[$label]="$isl"
    OSL[$label]="$osl"
    CONC[$label]="$conc"
    TP[$label]=8
    EP_SIZE[$label]=1
    DP_ATTENTION[$label]=false
    RUN_ID[$label]=${RUN_ID[dsv4_1k1k_c8_${variant}]}
    ATTEMPT[$label]=${ATTEMPT[dsv4_1k1k_c8_${variant}]}
    EXPECTED_TPUT_PER_GPU[$label]=
    if [[ "$variant" == "old" ]]; then
      COMMIT[$label]=${COMMIT[dsv4_1k1k_c8_old]}
      IMAGE[$label]=${IMAGE[dsv4_1k1k_c8_old]}
      SCRIPT[$label]=${SCRIPT[dsv4_1k1k_c8_old]}
      RUNTIME_PATCH_NOTE[$label]="host restores DeepSeek-V4-Pro HF config.json after old SGLang patches model_type deepseek_v4 -> deepseek_v3"
    else
      COMMIT[$label]=${COMMIT[dsv4_1k1k_c8_new]}
      IMAGE[$label]=${IMAGE[dsv4_1k1k_c8_new]}
      SCRIPT[$label]=${SCRIPT[dsv4_1k1k_c8_new]}
      RUNTIME_PATCH_NOTE[$label]=none
    fi
  done
done

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
      old) printf '%s\n' dsv4_8k1k_c8_old dsv4_1k1k_c1_old dsv4_1k1k_c2_old dsv4_1k1k_c3_old dsv4_1k1k_c4_old dsv4_1k1k_c8_old dsv4_1k1k_c16_old dsv4_1k512_c1_old dsv4_1k8k_c8_old ;;
      new) printf '%s\n' dsv4_8k1k_c8_new dsv4_1k1k_c1_new dsv4_1k1k_c2_new dsv4_1k1k_c3_new dsv4_1k1k_c4_new dsv4_1k1k_c8_new dsv4_1k1k_c16_new dsv4_1k512_c1_new dsv4_1k8k_c8_new ;;
      8k1k|8k1k_c8) printf '%s\n' dsv4_8k1k_c8_old dsv4_8k1k_c8_new ;;
      1k1k) printf '%s\n' dsv4_1k1k_c1_old dsv4_1k1k_c1_new dsv4_1k1k_c2_old dsv4_1k1k_c2_new dsv4_1k1k_c3_old dsv4_1k1k_c3_new dsv4_1k1k_c4_old dsv4_1k1k_c4_new dsv4_1k1k_c8_old dsv4_1k1k_c8_new dsv4_1k1k_c16_old dsv4_1k1k_c16_new ;;
      1k1k_c1) printf '%s\n' dsv4_1k1k_c1_old dsv4_1k1k_c1_new ;;
      1k1k_c2) printf '%s\n' dsv4_1k1k_c2_old dsv4_1k1k_c2_new ;;
      1k1k_c3) printf '%s\n' dsv4_1k1k_c3_old dsv4_1k1k_c3_new ;;
      1k1k_c4) printf '%s\n' dsv4_1k1k_c4_old dsv4_1k1k_c4_new ;;
      1k1k_c8) printf '%s\n' dsv4_1k1k_c8_old dsv4_1k1k_c8_new ;;
      1k1k_c16) printf '%s\n' dsv4_1k1k_c16_old dsv4_1k1k_c16_new ;;
      1k512|1k512_c1) printf '%s\n' dsv4_1k512_c1_old dsv4_1k512_c1_new ;;
      1k8k|1k8k_c8) printf '%s\n' dsv4_1k8k_c8_old dsv4_1k8k_c8_new ;;
      dsv4|deepseek|deepseek-v4-pro) printf '%s\n' "${LABEL_ORDER[@]}" ;;
      dsv4_*)
        if [[ -n "${VARIANT[$token]+x}" ]]; then
          printf '%s\n' "$token"
        else
          echo "Unknown TARGETS token: $token" >&2
          exit 1
        fi
        ;;
      *)
        echo "Unknown TARGETS token: $token" >&2
        exit 1
        ;;
    esac
  done | awk '!seen[$0]++'
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

  if [[ -e "$dest" && ! -f "$dest/.dsv4_sglang_repro_owner" ]]; then
    echo "Refusing to modify existing unowned worktree path: $dest" >&2
    exit 1
  fi

  if [[ ! -e "$dest/.git" ]]; then
    git -C "$REPO" worktree add --force --detach "$dest" "$commit"
    touch "$dest/.dsv4_sglang_repro_owner"
  else
    git -C "$dest" reset --hard "$commit" >/dev/null
    git -C "$dest" clean -fdx >/dev/null
    touch "$dest/.dsv4_sglang_repro_owner"
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
# DeepSeek-V4-Pro FP4 SGLang MI355X Geomean Repro

Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)

Scope:

- Hardware: MI355X
- Model: \`deepseek-ai/DeepSeek-V4-Pro\`
- Precision/framework: FP4 / SGLang
- Mode: single-node non-disaggregated
- Parallelism: TP \`8\`, EP \`1\`, DP attention \`false\`
- Spec decoding: none
- Prompt count patch: \`num_prompts = CONC * $PROMPT_MULTIPLIER\`
- HF cache: \`$MODEL_DIR\`
- HF home/xet cache: \`$HF_HOME_DIR\`
- Output root: \`$OUT_DIR\`

Targets:

| label | config | run | attempt | commit | image | script | dashboard tok/s/GPU |
|---|---|---:|---:|---|---|---|---:|
EOF
  local label
  for label in "${LABEL_ORDER[@]}"; do
    printf '| %s | %s/%s conc=%s | %s | %s | `%s` | `%s` | `%s` | %s |\n' \
      "$label" "${ISL[$label]}" "${OSL[$label]}" "${CONC[$label]}" \
      "${RUN_ID[$label]}" "${ATTEMPT[$label]}" "${COMMIT[$label]:0:12}" \
      "${IMAGE[$label]}" "${SCRIPT[$label]}" "${EXPECTED_TPUT_PER_GPU[$label]:-}" \
      >> "$OUT_DIR/README.md"
  done

  cat >> "$OUT_DIR/README.md" <<EOF

Cache note:

- The old SGLang script mutates the cached HF \`config.json\` by changing \`model_type\` from \`deepseek_v4\` to \`deepseek_v3\`.
- This runner backs up the real blob target before the old run and restores it afterward, then records status in \`cache_status/\`.
- The model cache already lives on \`/scratch\`, which is the persistent large disk on this host.

Run:

\`\`\`bash
TARGETS=all /root/repro_dsv4_sglang_geomean.sh
\`\`\`
EOF
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
export PORT=$PORT
export PROMPT_MULTIPLIER=$PROMPT_MULTIPLIER
export RANDOM_RANGE_RATIO=$RANDOM_RANGE_RATIO
export SKIP_EXISTING=1
export DRY_RUN=0
export CONTINUE_ON_ERROR=0

TARGETS=all /root/repro_dsv4_sglang_geomean.sh
TARGETS=1k1k /root/repro_dsv4_sglang_geomean.sh
TARGETS=1k1k_c1 /root/repro_dsv4_sglang_geomean.sh
TARGETS=1k1k_c2 /root/repro_dsv4_sglang_geomean.sh
TARGETS=1k1k_c3 /root/repro_dsv4_sglang_geomean.sh
TARGETS=1k1k_c4 /root/repro_dsv4_sglang_geomean.sh
TARGETS=1k1k_c16 /root/repro_dsv4_sglang_geomean.sh
TARGETS=1k512_c1 /root/repro_dsv4_sglang_geomean.sh
TARGETS=1k8k /root/repro_dsv4_sglang_geomean.sh
TARGETS=old /root/repro_dsv4_sglang_geomean.sh
TARGETS=new /root/repro_dsv4_sglang_geomean.sh
EOF
  chmod +x "$OUT_DIR/repro_commands.sh"
}

write_run_env_file() {
  local label=$1
  local run_dir
  run_dir=$(run_dir_for "$label")
  mkdir -p "$run_dir"
  cat > "$run_dir/env.repro" <<EOF
LABEL=$label
CONFIG_ID=${CONFIG_ID[$label]}
VARIANT=${VARIANT[$label]}
COMMIT=${COMMIT[$label]}
IMAGE=${IMAGE[$label]}
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
EXPECTED_DASHBOARD_TPUT_PER_GPU=${EXPECTED_TPUT_PER_GPU[$label]}
RUNTIME_PATCH_NOTE=${RUNTIME_PATCH_NOTE[$label]}
EOF
}

write_runtime_patch_script() {
  local label=$1
  local wt
  wt=$(worktree_for "$label")
  local patch_file="$wt/.dsv4_sglang_runtime_patch.sh"
  cat > "$patch_file" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
EOF
  chmod +x "$patch_file"
}

snapshot_config_paths() {
  if [[ ! -d "$MODEL_CACHE_ROOT/snapshots" ]]; then
    return 0
  fi
  find "$MODEL_CACHE_ROOT/snapshots" -mindepth 2 -maxdepth 2 -name config.json -print 2>/dev/null | sort
}

config_model_type() {
  python3 - "$1" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(json.load(f).get("model_type", ""))
PY
}

record_hf_config_status() {
  local stage=$1
  local dest="$OUT_DIR/cache_status/${stage}.txt"
  mkdir -p "$OUT_DIR/cache_status"
  {
    echo "stage=$stage"
    date -u +timestamp=%Y-%m-%dT%H:%M:%SZ
    echo "model_cache_root=$MODEL_CACHE_ROOT"
  } > "$dest"

  local -a paths=()
  mapfile -t paths < <(snapshot_config_paths)
  if ((${#paths[@]} == 0)); then
    echo "status=missing_config" >> "$dest"
    return 1
  fi

  local path real mt sha
  for path in "${paths[@]}"; do
    real=$(readlink -f "$path")
    mt=$(config_model_type "$path" 2>/dev/null || printf 'json_error')
    sha=$(sha256sum "$real" | awk '{print $1}')
    {
      echo "config=$path"
      echo "realpath=$real"
      echo "model_type=$mt"
      echo "sha256=$sha"
    } >> "$dest"
  done
}

verify_hf_config_model_type() {
  local expected=$1
  local -a paths=()
  mapfile -t paths < <(snapshot_config_paths)
  if ((${#paths[@]} == 0)); then
    echo "No DeepSeek-V4-Pro config.json found under $MODEL_CACHE_ROOT" >&2
    return 1
  fi

  local path mt bad=0
  for path in "${paths[@]}"; do
    mt=$(config_model_type "$path" 2>/dev/null || printf 'json_error')
    if [[ "$mt" != "$expected" ]]; then
      echo "Unexpected model_type in $path: $mt (expected $expected)" >&2
      bad=1
    fi
  done
  return "$bad"
}

backup_hf_config() {
  local label=$1
  local backup_dir="$CONFIG_BACKUP_ROOT/$label"
  local manifest="$backup_dir/manifest.tsv"
  mkdir -p "$backup_dir"
  : > "$manifest"

  local -a paths=()
  mapfile -t paths < <(snapshot_config_paths)
  if ((${#paths[@]} == 0)); then
    echo "No config paths to back up under $MODEL_CACHE_ROOT" >&2
    return 1
  fi

  local path real rel backup_file mt sha
  for path in "${paths[@]}"; do
    real=$(readlink -f "$path")
    rel=${path#"$MODEL_DIR"/}
    backup_file="$backup_dir/${rel//\//__}"
    mt=$(config_model_type "$path")
    sha=$(sha256sum "$real" | awk '{print $1}')
    cp -f "$real" "$backup_file"
    printf '%s\t%s\t%s\t%s\t%s\n' "$path" "$real" "$backup_file" "$sha" "$mt" >> "$manifest"
  done
}

restore_hf_config() {
  local label=$1
  local backup_dir="$CONFIG_BACKUP_ROOT/$label"
  local manifest="$backup_dir/manifest.tsv"
  if [[ ! -s "$manifest" ]]; then
    echo "Missing HF config backup manifest: $manifest" >&2
    return 1
  fi

  local path real backup_file sha mt current_real
  while IFS=$'\t' read -r path real backup_file sha mt; do
    if [[ ! -s "$backup_file" ]]; then
      echo "Missing backup file: $backup_file" >&2
      return 1
    fi
    if [[ -e "$real" ]]; then
      cp -f "$backup_file" "$real"
    fi
    if [[ -e "$path" ]]; then
      current_real=$(readlink -f "$path")
      if [[ "$current_real" != "$real" ]]; then
        cp -f "$backup_file" "$current_real"
      fi
    fi
  done < "$manifest"

  record_hf_config_status "after_restore_$label" || return 1
  verify_hf_config_model_type deepseek_v4
}

write_model_cache_status() {
  mkdir -p "$OUT_DIR/cache_status"
  local dest="$OUT_DIR/cache_status/model_cache.txt"
  {
    date -u +timestamp=%Y-%m-%dT%H:%M:%SZ
    echo "MODEL_DIR=$MODEL_DIR"
    echo "MODEL_CACHE_ROOT=$MODEL_CACHE_ROOT"
    if [[ -e "$MODEL_CACHE_ROOT" ]]; then
      du -sh "$MODEL_CACHE_ROOT" 2>/dev/null || true
      echo "status=already_on_persistent_scratch"
    else
      echo "status=missing_model_cache"
    fi
    df -h "$MODEL_DIR" 2>/dev/null || true
  } > "$dest"
}

check_image_manifest() {
  local label=$1
  local dest="$OUT_DIR/cache_status/${label}_image_manifest"
  mkdir -p "$OUT_DIR/cache_status"
  if docker manifest inspect "${IMAGE[$label]}" > "${dest}.json" 2> "${dest}.stderr"; then
    echo "status=ok" > "${dest}.txt"
  else
    echo "status=manifest_inspect_failed" > "${dest}.txt"
  fi
}

download_workflow_metadata() {
  local label=$1
  local dest="$OUT_DIR/workflow_metadata/$label"
  mkdir -p "$dest"

  local run_api="https://api.github.com/repos/$REPO_SLUG/actions/runs/${RUN_ID[$label]}"
  local jobs_api="https://api.github.com/repos/$REPO_SLUG/actions/runs/${RUN_ID[$label]}/attempts/${ATTEMPT[$label]}/jobs?per_page=100"

  curl -fsSL "$run_api" -o "$dest/run.json" || true
  curl -fsSL "$jobs_api" -o "$dest/jobs.json" || true
}

cleanup_named_container() {
  local label=$1
  docker rm -f "mi355-dsv4-sglang-$label" >/dev/null 2>&1 || true
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
  [[ -f "$wt/.dsv4_sglang_runtime_patch.sh" ]] && cp -f "$wt/.dsv4_sglang_runtime_patch.sh" "$run_dir/runtime_patch.sh"
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

  record_hf_config_status "before_$label" || true
  verify_hf_config_model_type deepseek_v4
  if [[ "${VARIANT[$label]}" == "old" ]]; then
    backup_hf_config "$label"
  fi

  download_workflow_metadata "$label"
  cleanup_named_container "$label"
  rm -f "$wt/${result_filename}.json" "$wt/agg_${result_filename}.json" "$wt/server.log" "$wt/gpu_metrics.csv"
  if [[ -s "$docker_log" ]]; then
    mv "$docker_log" "$run_dir/docker.$(date -u +%Y%m%dT%H%M%SZ).log"
  fi

  echo "[run] $label model=${MODEL[$label]} image=${IMAGE[$label]} seq=${ISL[$label]}/${OSL[$label]} conc=${CONC[$label]} tp=${TP[$label]} prompts=$(( CONC[$label] * PROMPT_MULTIPLIER ))"

  set +e
  docker run --rm \
    --name "mi355-dsv4-sglang-$label" \
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
    -lc "bash /workspace/.dsv4_sglang_runtime_patch.sh && bash /workspace/$script" > "$docker_log" 2>&1
  local docker_status=$?
  set -e

  if [[ "${VARIANT[$label]}" == "old" ]]; then
    if ! restore_hf_config "$label"; then
      copy_run_artifacts "$label" "cache_restore_failed"
      echo "[error] failed to restore HF config after $label" >&2
      return 1
    fi
  else
    record_hf_config_status "after_$label" || true
    verify_hf_config_model_type deepseek_v4
  fi

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

write_combined_csv() {
  python3 - "$OUT_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
label_order = [
    "dsv4_8k1k_c8_old",
    "dsv4_8k1k_c8_new",
    "dsv4_1k1k_c1_old",
    "dsv4_1k1k_c1_new",
    "dsv4_1k1k_c2_old",
    "dsv4_1k1k_c2_new",
    "dsv4_1k1k_c3_old",
    "dsv4_1k1k_c3_new",
    "dsv4_1k1k_c4_old",
    "dsv4_1k1k_c4_new",
    "dsv4_1k1k_c8_old",
    "dsv4_1k1k_c8_new",
    "dsv4_1k1k_c16_old",
    "dsv4_1k1k_c16_new",
    "dsv4_1k512_c1_old",
    "dsv4_1k512_c1_new",
    "dsv4_1k8k_c8_old",
    "dsv4_1k8k_c8_new",
]
rank = {label: idx for idx, label in enumerate(label_order)}
labels = [
    p.name
    for p in sorted((out / "runs").iterdir(), key=lambda p: (rank.get(p.name, 10_000), p.name))
    if p.is_dir() and (p / "env.repro").exists()
]

rows = []
by_config = {}
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
    variant = env.get("VARIANT", "old" if label.endswith("_old") else "new")
    config_id = env.get("CONFIG_ID", label.rsplit("_", 1)[0])
    row = {
        "model_group": "dsv4",
        "config_id": config_id,
        "variant": variant,
        "label": label,
        "status": status,
        "model": env.get("MODEL", agg.get("model", raw.get("model_id", ""))),
        "precision": env.get("PRECISION", agg.get("precision", "")),
        "framework": env.get("FRAMEWORK", agg.get("framework", "")),
        "image": env.get("IMAGE", agg.get("image", "")),
        "commit": env.get("COMMIT", ""),
        "workflow_run_id": env.get("WORKFLOW_RUN_ID", ""),
        "workflow_attempt": env.get("WORKFLOW_ATTEMPT", ""),
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
        "expected_dashboard_tput_per_gpu": env.get("EXPECTED_DASHBOARD_TPUT_PER_GPU", ""),
    }
    rows.append(row)
    by_config.setdefault(config_id, {})[variant] = row

fields = list(rows[0].keys()) if rows else []
with (out / "combined_results.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

comp_fields = [
    "model_group",
    "config_id",
    "isl",
    "osl",
    "conc",
    "num_prompts",
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
gains = []
config_order = []
seen_configs = set()
for label in labels:
    run_dir = out / "runs" / label
    env_path = run_dir / "env.repro"
    config_id = None
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("CONFIG_ID="):
                config_id = line.split("=", 1)[1]
                break
    if config_id and config_id not in seen_configs:
        seen_configs.add(config_id)
        config_order.append(config_id)
for config_id in config_order:
    pair = by_config.get(config_id, {})
    old = pair.get("old", {})
    new = pair.get("new", {})
    try:
        old_t = float(old.get("tput_per_gpu", ""))
        new_t = float(new.get("tput_per_gpu", ""))
        gain = new_t / old_t if old_t else ""
        pct = (gain - 1.0) * 100 if gain != "" else ""
        if gain:
            gains.append(gain)
    except (TypeError, ValueError):
        old_t = old.get("tput_per_gpu", "")
        new_t = new.get("tput_per_gpu", "")
        gain = ""
        pct = ""
    try:
        exp_old = float(old.get("expected_dashboard_tput_per_gpu", ""))
        exp_new = float(new.get("expected_dashboard_tput_per_gpu", ""))
        exp_gain = exp_new / exp_old if exp_old else ""
    except (TypeError, ValueError):
        exp_old = old.get("expected_dashboard_tput_per_gpu", "")
        exp_new = new.get("expected_dashboard_tput_per_gpu", "")
        exp_gain = ""
    comp_rows.append({
        "model_group": "dsv4",
        "config_id": config_id,
        "isl": old.get("isl") or new.get("isl", ""),
        "osl": old.get("osl") or new.get("osl", ""),
        "conc": old.get("conc") or new.get("conc", ""),
        "num_prompts": old.get("num_prompts") or new.get("num_prompts", ""),
        "old_label": old.get("label", f"dsv4_{config_id}_old"),
        "new_label": new.get("label", f"dsv4_{config_id}_new"),
        "old_status": old.get("status", "not_run"),
        "new_status": new.get("status", "not_run"),
        "old_tput_per_gpu": old_t,
        "new_tput_per_gpu": new_t,
        "measured_gain_x": gain,
        "measured_improvement_pct": pct,
        "expected_old_tput_per_gpu": exp_old,
        "expected_new_tput_per_gpu": exp_new,
        "expected_gain_x": exp_gain,
    })

if gains:
    import math
    geomean = math.prod(gains) ** (1.0 / len(gains))
    comp_rows.append({
        "model_group": "dsv4",
        "config_id": "GEOMEAN",
        "isl": "",
        "osl": "",
        "conc": "",
        "num_prompts": "",
        "old_label": "",
        "new_label": "",
        "old_status": f"{len(gains)} configs",
        "new_status": f"{len(gains)} configs",
        "old_tput_per_gpu": "",
        "new_tput_per_gpu": "",
        "measured_gain_x": geomean,
        "measured_improvement_pct": (geomean - 1.0) * 100,
        "expected_old_tput_per_gpu": "",
        "expected_new_tput_per_gpu": "",
        "expected_gain_x": "",
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
  need_cmd find
  need_cmd sha256sum
  ensure_hf_token

  mkdir -p "$OUT_DIR/runs" "$OUT_DIR/workflow_metadata" "$OUT_DIR/cache_status" "$MODEL_DIR" "$HF_HOME_DIR"
  write_suite_readme
  write_repro_commands_header
  write_model_cache_status
  record_hf_config_status initial || true
  verify_hf_config_model_type deepseek_v4

  local -a labels=()
  mapfile -t labels < <(selected_labels)

  local label
  for label in "${labels[@]}"; do
    check_image_manifest "$label"
  done

  for label in "${labels[@]}"; do
    if run_one "$label"; then
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

  record_hf_config_status final || true
  verify_hf_config_model_type deepseek_v4
  write_model_cache_status
  write_combined_csv
  echo "[done] results: $OUT_DIR"
  echo "[done] combined CSV: $OUT_DIR/combined_results.csv"
  echo "[done] comparison CSV: $OUT_DIR/comparison.csv"
}

main "$@"
