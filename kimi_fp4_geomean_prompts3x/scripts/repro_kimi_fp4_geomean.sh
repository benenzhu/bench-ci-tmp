#!/usr/bin/env bash
set -euo pipefail

# Kimi-K2.5 MXFP4 vLLM MI355X multi-config geomean repro.
# Modeled on repro_dsv4_sglang_geomean.sh. Key differences:
# - Model weights staged on /dev/shm only (machine constraint).
# - vLLM framework, no HF config.json mutation (DSV4-specific) needed.
# - Sweeps multiple CONC values to build a geomean across configs.

REPO=${REPO:-/root/InferenceX}
WT_ROOT=${WT_ROOT:-/dev/shm/inferencex_worktrees/kimi_fp4_geomean}
OUT_DIR=${OUT_DIR:-/home/tazhu/bench-ci-tmp/kimi_fp4_geomean_prompts3x}
MODEL_DIR=${MODEL_DIR:-/dev/shm/hf_hub_cache}
HF_HOME_DIR=${HF_HOME_DIR:-/dev/shm/hf_home}
PORT=${PORT:-19000}  # 18888 is held by a shared tinyproxy on this machine; use a free port
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.80}  # lowered from upstream 0.95: leaves headroom for other users on GPU3 AND for CUDA-graph capture (0.85 OOM'd during graph capture at CONC=8)
PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER:-3}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.8}
SKIP_EXISTING=${SKIP_EXISTING:-1}
TARGETS=${TARGETS:-all}
HF_TOKEN_FILE=${HF_TOKEN_FILE:-/root/.cache/huggingface/token}
DRY_RUN=${DRY_RUN:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}

REPO_SLUG=SemiAnalysisAI/InferenceX

# Base (anchor) config carries commit/image/script/run metadata shared by all CONC variants.
OLD_COMMIT=bc818474fdba28c4802e823fdc474be56fb79452
NEW_COMMIT=a187fa20e7785c88e136c834c411fc39022c75ba
OLD_IMAGE=vllm/vllm-openai-rocm:v0.16.0
NEW_IMAGE=vllm/vllm-openai-rocm:v0.22.0
BENCH_SCRIPT=benchmarks/single_node/kimik2.5_fp4_mi355x.sh
MODEL_ID=amd/Kimi-K2.5-MXFP4
MODEL_PREFIX_VAL=kimik2.5
MODEL_CACHE_NAME=models--amd--Kimi-K2.5-MXFP4

declare -A OLD_RUN=([run]=22555187591 [attempt]=2 [job]=65962398410)
declare -A NEW_RUN=([run]=26696253037 [attempt]=1 [job]=78682149908)

# Config sweep: id:isl:osl:conc
CONFIG_SPECS=(
  8k1k_c4:8192:1024:4
  8k1k_c8:8192:1024:8
  8k1k_c16:8192:1024:16
  8k1k_c32:8192:1024:32
  8k1k_c64:8192:1024:64
  1k1k_c4:1024:1024:4
  1k1k_c8:1024:1024:8
)

declare -A VARIANT CONFIG_ID COMMIT IMAGE MODEL MODEL_PREFIX PRECISION FRAMEWORK
declare -A SCRIPT ISL OSL CONC TP EP_SIZE DP_ATTENTION RUN_ID ATTEMPT JOB_ID
LABEL_ORDER=()

for spec in "${CONFIG_SPECS[@]}"; do
  IFS=: read -r cfg isl osl conc <<< "$spec"
  for variant in old new; do
    label="kimi_fp4_${cfg}_${variant}"
    LABEL_ORDER+=("$label")
    VARIANT[$label]="$variant"
    CONFIG_ID[$label]="$cfg"
    MODEL[$label]="$MODEL_ID"
    MODEL_PREFIX[$label]="$MODEL_PREFIX_VAL"
    PRECISION[$label]=fp4
    FRAMEWORK[$label]=vllm
    SCRIPT[$label]="$BENCH_SCRIPT"
    ISL[$label]="$isl"
    OSL[$label]="$osl"
    CONC[$label]="$conc"
    TP[$label]=8
    EP_SIZE[$label]=1
    DP_ATTENTION[$label]=false
    if [[ "$variant" == "old" ]]; then
      COMMIT[$label]="$OLD_COMMIT"; IMAGE[$label]="$OLD_IMAGE"
      RUN_ID[$label]="${OLD_RUN[run]}"; ATTEMPT[$label]="${OLD_RUN[attempt]}"; JOB_ID[$label]="${OLD_RUN[job]}"
    else
      COMMIT[$label]="$NEW_COMMIT"; IMAGE[$label]="$NEW_IMAGE"
      RUN_ID[$label]="${NEW_RUN[run]}"; ATTEMPT[$label]="${NEW_RUN[attempt]}"; JOB_ID[$label]="${NEW_RUN[job]}"
    fi
  done
done

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }
}

max_model_len_for() { local l=$1; echo $(( ISL[$l] + OSL[$l] + 200 )); }
worktree_for() { local l=$1; echo "$WT_ROOT/$l-${COMMIT[$l]:0:7}"; }
run_dir_for() { local l=$1; echo "$OUT_DIR/runs/$l"; }

result_filename_for() {
  local l=$1
  echo "${MODEL_PREFIX[$l]}_${ISL[$l]}x${OSL[$l]}_${PRECISION[$l]}_${FRAMEWORK[$l]}_mi355x_tp${TP[$l]}-ep${EP_SIZE[$l]}-dpa${DP_ATTENTION[$l]}_disagg-false_spec-none_conc${CONC[$l]}_prompts${PROMPT_MULTIPLIER}x_${VARIANT[$l]}"
}

selected_labels() {
  local normalized=${TARGETS//,/ }
  if [[ "$normalized" == "all" ]]; then printf '%s\n' "${LABEL_ORDER[@]}"; return 0; fi
  local token
  for token in $normalized; do
    case "$token" in
      old) for l in "${LABEL_ORDER[@]}"; do [[ "${VARIANT[$l]}" == "old" ]] && printf '%s\n' "$l"; done ;;
      new) for l in "${LABEL_ORDER[@]}"; do [[ "${VARIANT[$l]}" == "new" ]] && printf '%s\n' "$l"; done ;;
      kimi_fp4_*)
        if [[ -n "${VARIANT[$token]+x}" ]]; then printf '%s\n' "$token"
        else echo "Unknown TARGETS token: $token" >&2; exit 1; fi ;;
      *)
        # treat as a config id (e.g. 8k1k_c8)
        local matched=0
        for l in "${LABEL_ORDER[@]}"; do
          if [[ "${CONFIG_ID[$l]}" == "$token" ]]; then printf '%s\n' "$l"; matched=1; fi
        done
        if (( ! matched )); then echo "Unknown TARGETS token: $token" >&2; exit 1; fi ;;
    esac
  done | awk '!seen[$0]++'
}

ensure_hf_token() {
  if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
    HF_TOKEN=$(tr -d '\r\n' < "$HF_TOKEN_FILE"); export HF_TOKEN
  fi
  [[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN empty and $HF_TOKEN_FILE unreadable" >&2; exit 1; }
}

ensure_commit() {
  local c=$1
  git -C "$REPO" cat-file -e "${c}^{commit}" 2>/dev/null || git -C "$REPO" fetch origin "$c"
}

prepare_worktree() {
  local label=$1 commit=${COMMIT[$1]} script=${SCRIPT[$1]}
  local dest; dest=$(worktree_for "$label")
  ensure_commit "$commit"
  mkdir -p "$WT_ROOT" "$(run_dir_for "$label")"
  if [[ -e "$dest" && ! -f "$dest/.kimi_fp4_repro_owner" ]]; then
    echo "Refusing to modify unowned worktree path: $dest" >&2; exit 1
  fi
  if [[ ! -e "$dest/.git" ]]; then
    git -C "$REPO" worktree add --force --detach "$dest" "$commit"
    touch "$dest/.kimi_fp4_repro_owner"
  else
    git -C "$dest" reset --hard "$commit" >/dev/null
    git -C "$dest" clean -fdx >/dev/null
    touch "$dest/.kimi_fp4_repro_owner"
  fi
  [[ -f "$dest/$script" ]] || { echo "Benchmark script not found: $dest/$script" >&2; exit 1; }
  perl -0pi -e 's/--num-prompts "\$\(\(CONC \* 10\)\)"/--num-prompts "\$\(\(CONC * PROMPT_MULTIPLIER\)\)"/g' "$dest/$script"
  rg -F -q -- '--num-prompts "$((CONC * PROMPT_MULTIPLIER))"' "$dest/$script" || {
    echo "Failed to patch num-prompts in $dest/$script" >&2; exit 1; }
  # Lower GPU memory utilization to leave headroom for other users on this shared machine.
  # old commit ships 0.95, new commit ships 0.90; clamp both down to $GPU_MEM_UTIL.
  perl -0pi -e "s/--gpu-memory-utilization 0\.9[05]/--gpu-memory-utilization $GPU_MEM_UTIL/g" "$dest/$script"
  rg -F -q -- "--gpu-memory-utilization $GPU_MEM_UTIL" "$dest/$script" || {
    echo "Failed to patch gpu-memory-utilization in $dest/$script" >&2; exit 1; }
  git -C "$dest" diff -- "$script" > "$(run_dir_for "$label")/local_patch.diff" 2>/dev/null || true
}

write_run_env_file() {
  local label=$1 run_dir; run_dir=$(run_dir_for "$label"); mkdir -p "$run_dir"
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
WORKFLOW_JOB_ID=${JOB_ID[$label]}
EOF
}

write_runtime_patch_script() {
  local wt; wt=$(worktree_for "$1")
  printf '#!/usr/bin/env bash\nset -euo pipefail\n' > "$wt/.kimi_fp4_runtime_patch.sh"
  chmod +x "$wt/.kimi_fp4_runtime_patch.sh"
}

record_model_cache_status() {
  local stage=$1; mkdir -p "$OUT_DIR/cache_status"
  local root="$MODEL_DIR/$MODEL_CACHE_NAME"
  {
    echo "stage=$stage"
    date -u +timestamp=%Y-%m-%dT%H:%M:%SZ
    echo "MODEL_DIR=$MODEL_DIR"
    echo "MODEL_CACHE_ROOT=$root"
    if [[ -e "$root" ]]; then du -sh "$root" 2>/dev/null || true; echo "status=present"; else echo "status=missing"; fi
    df -h "$MODEL_DIR" 2>/dev/null || true
  } > "$OUT_DIR/cache_status/${stage}.txt"
}

check_image_manifest() {
  local label=$1 dest="$OUT_DIR/cache_status/${1}_image_manifest"; mkdir -p "$OUT_DIR/cache_status"
  if docker manifest inspect "${IMAGE[$label]}" > "${dest}.json" 2> "${dest}.stderr"; then
    echo "status=ok" > "${dest}.txt"; else echo "status=manifest_inspect_failed" > "${dest}.txt"; fi
}

cleanup_named_container() { docker rm -f "mi355-kimi-fp4-$1" >/dev/null 2>&1 || true; }

copy_run_artifacts() {
  local label=$1 status=$2 wt run_dir result_filename script
  wt=$(worktree_for "$label"); run_dir=$(run_dir_for "$label")
  result_filename=$(result_filename_for "$label"); script=${SCRIPT[$label]}
  mkdir -p "$run_dir"; printf '%s\n' "$status" > "$run_dir/status.txt"
  [[ -f "$wt/${result_filename}.json" ]] && cp -f "$wt/${result_filename}.json" "$run_dir/"
  [[ -f "$wt/agg_${result_filename}.json" ]] && cp -f "$wt/agg_${result_filename}.json" "$run_dir/"
  [[ -f "$wt/server.log" ]] && cp -f "$wt/server.log" "$run_dir/server.log"
  [[ -f "$wt/gpu_metrics.csv" ]] && cp -f "$wt/gpu_metrics.csv" "$run_dir/gpu_metrics.csv"
  [[ -f "$wt/$script" ]] && cp -f "$wt/$script" "$run_dir/$(basename "$script").patched"
  git -C "$wt" diff -- "$script" > "$run_dir/local_patch.diff" 2>/dev/null || true
}

process_result() {
  local label=$1 wt result_filename
  wt=$(worktree_for "$label"); result_filename=$(result_filename_for "$label")
  ( cd "$wt"
    RUNNER_TYPE=mi355x FRAMEWORK="${FRAMEWORK[$label]}" PRECISION="${PRECISION[$label]}" \
    SPEC_DECODING=none RESULT_FILENAME="$result_filename" ISL="${ISL[$label]}" OSL="${OSL[$label]}" \
    DISAGG=false MODEL_PREFIX="${MODEL_PREFIX[$label]}" IMAGE="${IMAGE[$label]}" TP="${TP[$label]}" \
    EP_SIZE="${EP_SIZE[$label]}" DP_ATTENTION="${DP_ATTENTION[$label]}" python3 utils/process_result.py )
}

run_one() {
  local label=$1 wt run_dir result_filename result_path docker_log script
  wt=$(worktree_for "$label"); run_dir=$(run_dir_for "$label")
  result_filename=$(result_filename_for "$label")
  result_path="$run_dir/${result_filename}.json"; docker_log="$run_dir/docker.log"; script=${SCRIPT[$label]}

  mkdir -p "$run_dir"
  write_run_env_file "$label"
  prepare_worktree "$label"
  write_runtime_patch_script "$label"
  check_image_manifest "$label"

  if [[ "$SKIP_EXISTING" == "1" && -s "$result_path" && -s "$run_dir/agg_${result_filename}.json" ]]; then
    echo "[skip] $label already has results"; return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    copy_run_artifacts "$label" "dry_run"; echo "[dry-run] $label prepared"; return 0
  fi

  cleanup_named_container "$label"
  rm -f "$wt/${result_filename}.json" "$wt/agg_${result_filename}.json" "$wt/server.log" "$wt/gpu_metrics.csv"
  [[ -s "$docker_log" ]] && mv "$docker_log" "$run_dir/docker.$(date -u +%Y%m%dT%H%M%SZ).log"

  echo "[run] $label image=${IMAGE[$label]} seq=${ISL[$label]}/${OSL[$label]} conc=${CONC[$label]} tp=${TP[$label]} prompts=$(( CONC[$label] * PROMPT_MULTIPLIER ))"

  set +e
  docker run --rm \
    --name "mi355-kimi-fp4-$label" \
    --network host --ipc host --privileged \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --entrypoint /bin/bash -w /workspace \
    -v "$wt:/workspace" \
    -v "$MODEL_DIR:$MODEL_DIR" \
    -v "$HF_HOME_DIR:$HF_HOME_DIR" \
    -v /dev/shm/pip_cache:/root/.cache/pip \
    -e HF_TOKEN \
    -e HF_HUB_CACHE="$MODEL_DIR" -e HF_HOME="$HF_HOME_DIR" -e HF_XET_CACHE="$HF_HOME_DIR/xet" \
    -e MODEL="${MODEL[$label]}" -e MODEL_PREFIX="${MODEL_PREFIX[$label]}" -e IMAGE="${IMAGE[$label]}" \
    -e FRAMEWORK="${FRAMEWORK[$label]}" -e PRECISION="${PRECISION[$label]}" -e RUNNER_TYPE=mi355x \
    -e TP="${TP[$label]}" -e EP_SIZE="${EP_SIZE[$label]}" -e DP_ATTENTION="${DP_ATTENTION[$label]}" \
    -e CONC="${CONC[$label]}" -e ISL="${ISL[$label]}" -e OSL="${OSL[$label]}" \
    -e MAX_MODEL_LEN="$(max_model_len_for "$label")" -e RANDOM_RANGE_RATIO="$RANDOM_RANGE_RATIO" \
    -e RESULT_FILENAME="$result_filename" -e SPEC_DECODING=none -e DISAGG=false \
    -e RUN_EVAL=false -e EVAL_ONLY=false -e PORT="$PORT" -e PROMPT_MULTIPLIER="$PROMPT_MULTIPLIER" \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPYCACHEPREFIX=/tmp/inferencex-pycache \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${IMAGE[$label]}" \
    -lc "bash /workspace/.kimi_fp4_runtime_patch.sh && bash /workspace/$script" > "$docker_log" 2>&1
  local docker_status=$?
  set -e

  if (( docker_status != 0 )); then
    copy_run_artifacts "$label" "docker_failed:$docker_status"
    echo "[error] docker failed for $label status=$docker_status, log=$docker_log" >&2; return 1
  fi
  if [[ ! -s "$wt/${result_filename}.json" ]]; then
    copy_run_artifacts "$label" "missing_result"
    echo "[error] missing benchmark JSON for $label, log=$docker_log" >&2; return 1
  fi
  if ! process_result "$label" > "$run_dir/process_result.stdout" 2> "$run_dir/process_result.stderr"; then
    copy_run_artifacts "$label" "process_failed"
    echo "[error] process_result failed for $label" >&2; return 1
  fi
  copy_run_artifacts "$label" "success"
}

write_combined_csv() {
  python3 - "$OUT_DIR" <<'PY'
import csv, json, math, sys
from pathlib import Path

out = Path(sys.argv[1])
runs = out / "runs"
labels = [p.name for p in sorted(runs.iterdir()) if p.is_dir() and (p / "env.repro").exists()] if runs.exists() else []

rows = []
by_config = {}
config_order = []
for label in labels:
    rd = runs / label
    status = (rd / "status.txt").read_text().strip() if (rd / "status.txt").exists() else "not_run"
    env = {}
    for line in (rd / "env.repro").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1); env[k] = v
    agg_paths = sorted(rd.glob("agg_*.json"))
    raw_paths = [p for p in sorted(rd.glob("*.json")) if not p.name.startswith("agg_")]
    agg = json.loads(agg_paths[0].read_text()) if agg_paths else {}
    raw = json.loads(raw_paths[0].read_text()) if raw_paths else {}
    variant = env.get("VARIANT", "old" if label.endswith("_old") else "new")
    cfg = env.get("CONFIG_ID", "")
    row = {
        "model_group": "kimi_fp4", "config_id": cfg, "variant": variant, "label": label,
        "status": status, "model": env.get("MODEL", ""), "precision": env.get("PRECISION", ""),
        "framework": env.get("FRAMEWORK", ""), "image": env.get("IMAGE", ""), "commit": env.get("COMMIT", ""),
        "isl": env.get("ISL", ""), "osl": env.get("OSL", ""), "conc": env.get("CONC", ""),
        "tp": env.get("TP", ""), "ep": env.get("EP_SIZE", ""), "dp_attention": env.get("DP_ATTENTION", ""),
        "num_prompts": env.get("NUM_PROMPTS", ""),
        "total_token_throughput": raw.get("total_token_throughput", ""),
        "output_throughput": raw.get("output_throughput", ""),
        "tput_per_gpu": agg.get("tput_per_gpu", ""),
        "output_tput_per_gpu": agg.get("output_tput_per_gpu", ""),
        "input_tput_per_gpu": agg.get("input_tput_per_gpu", ""),
        "mean_ttft_ms": raw.get("mean_ttft_ms", ""), "mean_tpot_ms": raw.get("mean_tpot_ms", ""),
    }
    rows.append(row)
    by_config.setdefault(cfg, {})[variant] = row
    if cfg not in config_order:
        config_order.append(cfg)

if rows:
    with (out / "combined_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

comp_fields = ["model_group","config_id","isl","osl","conc","num_prompts","old_label","new_label",
               "old_status","new_status","old_tput_per_gpu","new_tput_per_gpu","measured_gain_x","measured_improvement_pct"]
comp_rows = []
gains = []
for cfg in config_order:
    pair = by_config.get(cfg, {})
    old = pair.get("old", {}); new = pair.get("new", {})
    try:
        ot = float(old.get("tput_per_gpu", "")); nt = float(new.get("tput_per_gpu", ""))
        gain = nt / ot if ot else ""; pct = (gain - 1.0) * 100 if gain != "" else ""
        if gain: gains.append(gain)
    except (TypeError, ValueError):
        ot = old.get("tput_per_gpu", ""); nt = new.get("tput_per_gpu", ""); gain = ""; pct = ""
    comp_rows.append({
        "model_group": "kimi_fp4", "config_id": cfg,
        "isl": old.get("isl") or new.get("isl", ""), "osl": old.get("osl") or new.get("osl", ""),
        "conc": old.get("conc") or new.get("conc", ""), "num_prompts": old.get("num_prompts") or new.get("num_prompts", ""),
        "old_label": old.get("label", f"kimi_fp4_{cfg}_old"), "new_label": new.get("label", f"kimi_fp4_{cfg}_new"),
        "old_status": old.get("status", "not_run"), "new_status": new.get("status", "not_run"),
        "old_tput_per_gpu": ot, "new_tput_per_gpu": nt, "measured_gain_x": gain, "measured_improvement_pct": pct,
    })
if gains:
    geomean = math.prod(gains) ** (1.0 / len(gains))
    comp_rows.append({
        "model_group": "kimi_fp4", "config_id": "GEOMEAN", "isl": "", "osl": "", "conc": "", "num_prompts": "",
        "old_label": "", "new_label": "", "old_status": f"{len(gains)} configs", "new_status": f"{len(gains)} configs",
        "old_tput_per_gpu": "", "new_tput_per_gpu": "", "measured_gain_x": geomean, "measured_improvement_pct": (geomean - 1.0) * 100,
    })
with (out / "comparison.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=comp_fields); w.writeheader(); w.writerows(comp_rows)
PY
}

main() {
  need_cmd git; need_cmd docker; need_cmd curl; need_cmd python3; need_cmd rg
  ensure_hf_token
  mkdir -p "$OUT_DIR/runs" "$OUT_DIR/cache_status" "$MODEL_DIR" "$HF_HOME_DIR"
  record_model_cache_status initial

  local -a labels=(); mapfile -t labels < <(selected_labels)
  local label
  for label in "${labels[@]}"; do
    if run_one "$label"; then
      write_combined_csv
      record_model_cache_status "after_$label"
    else
      local status=$?
      write_combined_csv
      record_model_cache_status "after_${label}_failure"
      echo "[error] $label failed" >&2
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then exit "$status"; fi
    fi
  done

  record_model_cache_status final
  write_combined_csv
  echo "[done] results: $OUT_DIR"
  echo "[done] comparison CSV: $OUT_DIR/comparison.csv"
}

main "$@"
