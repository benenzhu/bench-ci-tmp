#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/InferenceX}
WT_ROOT=${WT_ROOT:-/scratch/inferencex_worktrees/kimi_int4_8k1k_conc16}
OUT_DIR=${OUT_DIR:-/scratch/inferencex_runs/kimi_int4_8k1k_conc16_prompts3x}
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

LABEL_ORDER=(
  kimi_int4_old
  kimi_int4_new
)

declare -A VARIANT=(
  [kimi_int4_old]=old
  [kimi_int4_new]=new
)

declare -A COMMIT=(
  [kimi_int4_old]=d4f6998d1e5a4b44953d4618810143c599ce505b
  [kimi_int4_new]=79a42e20a8559784641ca2279994c75c0608dc39
)

declare -A IMAGE=(
  [kimi_int4_old]=vllm/vllm-openai-rocm:v0.15.1
  [kimi_int4_new]=vllm/vllm-openai-rocm:v0.18.0
)

declare -A MODEL=(
  [kimi_int4_old]=moonshotai/Kimi-K2.5
  [kimi_int4_new]=moonshotai/Kimi-K2.5
)

declare -A MODEL_PREFIX=(
  [kimi_int4_old]=kimik2.5
  [kimi_int4_new]=kimik2.5
)

declare -A PRECISION=(
  [kimi_int4_old]=int4
  [kimi_int4_new]=int4
)

declare -A FRAMEWORK=(
  [kimi_int4_old]=vllm
  [kimi_int4_new]=vllm
)

declare -A SCRIPT=(
  [kimi_int4_old]=benchmarks/kimik2.5_int4_mi355x.sh
  [kimi_int4_new]=benchmarks/single_node/kimik2.5_int4_mi355x.sh
)

declare -A ISL=(
  [kimi_int4_old]=8192
  [kimi_int4_new]=8192
)

declare -A OSL=(
  [kimi_int4_old]=1024
  [kimi_int4_new]=1024
)

declare -A CONC=(
  [kimi_int4_old]=16
  [kimi_int4_new]=16
)

declare -A TP=(
  [kimi_int4_old]=8
  [kimi_int4_new]=8
)

declare -A EP_SIZE=(
  [kimi_int4_old]=1
  [kimi_int4_new]=1
)

declare -A DP_ATTENTION=(
  [kimi_int4_old]=false
  [kimi_int4_new]=false
)

declare -A RUN_ID=(
  [kimi_int4_old]=22129277189
  [kimi_int4_new]=23669977901
)

declare -A ATTEMPT=(
  [kimi_int4_old]=3
  [kimi_int4_new]=6
)

declare -A JOB_ID=(
  [kimi_int4_old]=64160961756
  [kimi_int4_new]=70014502875
)

declare -A EXPECTED_TPUT_PER_GPU=(
  [kimi_int4_old]=157.19541980732237
  [kimi_int4_new]=631.0259966432371
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
      old|kimi_old|kimi_int4_old) printf '%s\n' kimi_int4_old ;;
      new|kimi_new|kimi_int4_new) printf '%s\n' kimi_int4_new ;;
      kimi|kimi_int4|kimik2.5) printf '%s\n' kimi_int4_old kimi_int4_new ;;
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
  mkdir -p "$WT_ROOT" "$(run_dir_for "$label")"

  if [[ -e "$dest" && ! -f "$dest/.kimi_int4_repro_owner" ]]; then
    echo "Refusing to modify existing unowned worktree path: $dest" >&2
    exit 1
  fi

  if [[ ! -e "$dest/.git" ]]; then
    git -C "$REPO" worktree add --force --detach "$dest" "$commit"
    touch "$dest/.kimi_int4_repro_owner"
  else
    git -C "$dest" reset --hard "$commit" >/dev/null
    git -C "$dest" clean -fdx >/dev/null
    touch "$dest/.kimi_int4_repro_owner"
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
# Kimi-K2.5 INT4 vLLM MI355X Repro

Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)

Scope:

- Hardware: MI355X
- Model: \`moonshotai/Kimi-K2.5\`
- Precision/framework: INT4 / vLLM
- Mode: single-node non-disaggregated
- Sequence/concurrency: ISL/OSL \`8192/1024\`, CONC \`16\`
- Parallelism: TP \`8\`, EP \`1\`, DP attention \`false\`
- Spec decoding: none
- Prompt count patch: \`num_prompts = CONC * $PROMPT_MULTIPLIER\`
- Local num prompts: \`$(( 16 * PROMPT_MULTIPLIER ))\`
- HF cache: \`$MODEL_DIR\`
- HF home/xet cache: \`$HF_HOME_DIR\`
- Output root: \`$OUT_DIR\`

Targets:

| label | run | attempt | job | commit | image | script | dashboard tok/s/GPU |
|---|---:|---:|---:|---|---|---|---:|
EOF
  local label
  for label in "${LABEL_ORDER[@]}"; do
    printf '| %s | %s | %s | %s | `%s` | `%s` | `%s` | %.6f |\n' \
      "$label" "${RUN_ID[$label]}" "${ATTEMPT[$label]}" "${JOB_ID[$label]}" "${COMMIT[$label]:0:12}" \
      "${IMAGE[$label]}" "${SCRIPT[$label]}" "${EXPECTED_TPUT_PER_GPU[$label]}" \
      >> "$OUT_DIR/README.md"
  done

  cat >> "$OUT_DIR/README.md" <<EOF

Run:

\`\`\`bash
cd $OUT_DIR
TARGETS=all scripts/repro_kimi_int4_8k1k_conc16_tp8.sh
\`\`\`
EOF
}

write_repro_commands_header() {
  mkdir -p "$OUT_DIR"
  cat > "$OUT_DIR/repro_commands.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR=\$(cd -- "\$(dirname -- "\${BASH_SOURCE[0]}")" && pwd)

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

TARGETS=all "\$BUNDLE_DIR/scripts/repro_kimi_int4_8k1k_conc16_tp8.sh"
TARGETS=old "\$BUNDLE_DIR/scripts/repro_kimi_int4_8k1k_conc16_tp8.sh"
TARGETS=new "\$BUNDLE_DIR/scripts/repro_kimi_int4_8k1k_conc16_tp8.sh"
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
MODEL_GROUP=kimi_int4
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
WORKFLOW_JOB_ID=${JOB_ID[$label]}
EXPECTED_DASHBOARD_TPUT_PER_GPU=${EXPECTED_TPUT_PER_GPU[$label]}
EOF
}

write_runtime_patch_script() {
  local label=$1
  local wt
  wt=$(worktree_for "$label")
  local patch_file="$wt/.kimi_int4_runtime_patch.sh"
  cat > "$patch_file" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
EOF
  chmod +x "$patch_file"
}

download_workflow_metadata() {
  local label=$1
  local dest="$OUT_DIR/workflow_metadata/$label"
  mkdir -p "$dest"
  curl --compressed -fsSL "https://api.github.com/repos/$REPO_SLUG/actions/runs/${RUN_ID[$label]}/attempts/${ATTEMPT[$label]}" -o "$dest/run.json" || true
  curl --compressed -fsSL "https://api.github.com/repos/$REPO_SLUG/actions/runs/${RUN_ID[$label]}/attempts/${ATTEMPT[$label]}/jobs?per_page=100" -o "$dest/jobs.json" || true
  curl --compressed -fsSL -L "https://api.github.com/repos/$REPO_SLUG/actions/jobs/${JOB_ID[$label]}/logs" -o "$dest/job_${JOB_ID[$label]}.log" || true
}

record_model_cache_status() {
  local stage=$1
  mkdir -p "$OUT_DIR/cache_status"
  local root="$MODEL_DIR/models--moonshotai--Kimi-K2.5"
  {
    echo "stage=$stage"
    date -u +timestamp=%Y-%m-%dT%H:%M:%SZ
    echo "MODEL_DIR=$MODEL_DIR"
    echo "MODEL_CACHE_ROOT=$root"
    if [[ -e "$root" ]]; then
      du -sh "$root" 2>/dev/null || true
      echo "status=present"
    else
      echo "status=missing"
    fi
    df -h "$MODEL_DIR" 2>/dev/null || true
  } > "$OUT_DIR/cache_status/${stage}.txt"
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

cleanup_named_container() {
  local label=$1
  docker rm -f "mi355-kimi-int4-$label" >/dev/null 2>&1 || true
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
  [[ -f "$wt/.kimi_int4_runtime_patch.sh" ]] && cp -f "$wt/.kimi_int4_runtime_patch.sh" "$run_dir/runtime_patch.sh"
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
  check_image_manifest "$label"

  if [[ "$SKIP_EXISTING" == "1" && -s "$result_path" && -s "$run_dir/agg_${result_filename}.json" ]]; then
    echo "[skip] $label already has raw and aggregate results"
    return 0
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    copy_run_artifacts "$label" "dry_run"
    echo "[dry-run] $label prepared worktree and patched ${SCRIPT[$label]}"
    return 0
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
    --name "mi355-kimi-int4-$label" \
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
    -lc "bash /workspace/.kimi_int4_runtime_patch.sh && bash /workspace/$script" > "$docker_log" 2>&1
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

write_combined_csv() {
  python3 - "$OUT_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
labels = ["kimi_int4_old", "kimi_int4_new"]
expected = {"old": 157.19541980732237, "new": 631.0259966432371}

rows = []
by_variant = {}
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
    variant = env.get("VARIANT", label.rsplit("_", 1)[1])
    row = {
        "model_group": "kimi_int4",
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
        "expected_dashboard_tput_per_gpu": expected.get(variant, ""),
    }
    rows.append(row)
    by_variant[variant] = row

fields = list(rows[0].keys()) if rows else []
with (out / "combined_results.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

old = by_variant.get("old", {})
new = by_variant.get("new", {})
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
with (out / "comparison.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=comp_fields)
    writer.writeheader()
    writer.writerow({
        "model_group": "kimi_int4",
        "old_label": old.get("label", "kimi_int4_old"),
        "new_label": new.get("label", "kimi_int4_new"),
        "old_status": old.get("status", "not_run"),
        "new_status": new.get("status", "not_run"),
        "old_tput_per_gpu": old_t,
        "new_tput_per_gpu": new_t,
        "measured_gain_x": gain,
        "measured_improvement_pct": pct,
        "expected_old_tput_per_gpu": expected["old"],
        "expected_new_tput_per_gpu": expected["new"],
        "expected_gain_x": expected["new"] / expected["old"],
    })
PY
}

append_results_to_readme() {
  python3 - "$OUT_DIR" <<'PY'
import csv
from pathlib import Path
import sys

out = Path(sys.argv[1])
readme = out / "README.md"
if not (out / "comparison.csv").exists():
    raise SystemExit(0)
with (out / "comparison.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit(0)
row = rows[0]
marker = "## Local Repro Result"
text = readme.read_text()
if marker in text:
    text = text.split(marker, 1)[0].rstrip() + "\n\n"
old = row["old_tput_per_gpu"]
new = row["new_tput_per_gpu"]
gain = row["measured_gain_x"]
pct = row["measured_improvement_pct"]
if not old or not new or not gain or not pct:
    raise SystemExit(0)
text += f"""{marker}

| Old status | New status | Old tok/s/GPU | New tok/s/GPU | Gain | Improvement |
|---|---|---:|---:|---:|---:|
| {row['old_status']} | {row['new_status']} | {float(old):.6f} | {float(new):.6f} | {float(gain):.4f}x | +{float(pct):.2f}% |

Dashboard reference: `{float(row['expected_old_tput_per_gpu']):.6f} -> {float(row['expected_new_tput_per_gpu']):.6f} tok/s/GPU`, `{float(row['expected_gain_x']):.4f}x`.

## Outcome

This is a completed local reproduction attempt, but it did **not** reproduce the
`4.0143x` dashboard claim for the requested `8192/1024`, `CONC=16`, `TP=8`
Kimi INT4 row. Treat this bundle as evidence that the previously shortlisted
8k/1k Kimi INT4 middle-state candidate is not usable without more investigation.

## Follow-Up Candidate

The same workflow pair still has a dashboard candidate in the desired
3.5x-4.5x band, but it is `1024/8192`, not this `8192/1024` local run:

| Precision | Seq | CONC | TP | Dashboard tok/s/GPU | Gain | Runs |
|---|---|---:|---:|---:|---:|---|
| int4 | 1024/8192 | 16 | 8 | `22.890 -> 95.645` | `4.178x` | `22129277189/attempts/3` -> `23669977901/attempts/6` |

That `1024/8192` candidate has not been run locally in this bundle.

## Acceptance

Validate the saved bundle:

```bash
cd /scratch/inferencex_runs/kimi_int4_8k1k_conc16_prompts3x
scripts/verify_kimi_int4_8k1k_conc16.sh
```
"""
readme.write_text(text)
PY
}

main() {
  need_cmd git
  need_cmd docker
  need_cmd curl
  need_cmd python3
  need_cmd rg
  ensure_hf_token

  mkdir -p "$OUT_DIR/runs" "$OUT_DIR/workflow_metadata" "$OUT_DIR/cache_status" "$MODEL_DIR" "$HF_HOME_DIR"
  write_suite_readme
  write_repro_commands_header
  record_model_cache_status initial

  local -a labels=()
  mapfile -t labels < <(selected_labels)

  local label
  for label in "${labels[@]}"; do
    if run_one "$label"; then
      write_combined_csv
      append_results_to_readme || true
      record_model_cache_status "after_$label"
    else
      local status=$?
      write_combined_csv
      record_model_cache_status "after_${label}_failure"
      echo "[error] stopping after $label failure; set CONTINUE_ON_ERROR=1 to continue remaining labels" >&2
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
        exit "$status"
      fi
    fi
  done

  record_model_cache_status final
  write_combined_csv
  append_results_to_readme || true
  echo "[done] results: $OUT_DIR"
  echo "[done] combined CSV: $OUT_DIR/combined_results.csv"
  echo "[done] comparison CSV: $OUT_DIR/comparison.csv"
}

main "$@"
