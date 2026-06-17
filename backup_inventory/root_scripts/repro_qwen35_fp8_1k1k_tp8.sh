#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/InferenceX}
OUT_DIR=${OUT_DIR:-/root/qwen35_fp8_1k1k_tp8_results_prompts3x}
MODEL_DIR=${MODEL_DIR:-/dev/shm/hf_hub_cache}
MODEL=${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}

OLD_COMMIT=${OLD_COMMIT:-bc3e73fd0ceb0088ef1ca8556ef1d43a212a49e1}
NEW_COMMIT=${NEW_COMMIT:-fe3633af915d99c82a7c753e2e0f8bc4f8b86253}

OLD_IMAGE=${OLD_IMAGE:-rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260218}
NEW_IMAGE=${NEW_IMAGE:-lmsysorg/sglang-rocm:v0.5.12.post1-rocm720-mi35x-20260528}

TP=${TP:-8}
EP_SIZE=${EP_SIZE:-1}
ISL=${ISL:-1024}
OSL=${OSL:-1024}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.8}
CONC_LIST=${CONC_LIST:-"4 8 16 32 64"}
PORT=${PORT:-18888}
PROMPT_MULTIPLIER=${PROMPT_MULTIPLIER:-3}

# Set CLEAN_DOCKER=1 to mimic the GitHub Actions cleanup:
# docker ps -aq | xargs -r docker rm -f; docker network prune -f
CLEAN_DOCKER=${CLEAN_DOCKER:-0}
SKIP_EXISTING=${SKIP_EXISTING:-0}

WT_ROOT=${WT_ROOT:-/root/inferencex-qwen35-fp8-repro-worktrees}
OLD_WT="$WT_ROOT/old-${OLD_COMMIT:0:7}"
NEW_WT="$WT_ROOT/new-${NEW_COMMIT:0:7}"

OLD_SCRIPT=benchmarks/single_node/qwen3.5_fp8_mi355x.sh
NEW_SCRIPT=benchmarks/single_node/fixed_seq_len/qwen3.5_fp8_mi355x.sh

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

ensure_commit() {
  local commit=$1
  if ! git -C "$REPO" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    git -C "$REPO" fetch origin "$commit"
  fi
}

prepare_worktree() {
  local commit=$1
  local dest=$2
  local script=$3

  ensure_commit "$commit"
  mkdir -p "$WT_ROOT"
  if [[ ! -e "$dest/.git" ]]; then
    git -C "$REPO" worktree add --force --detach "$dest" "$commit"
  else
    git -C "$dest" fetch origin "$commit" || true
    git -C "$dest" checkout --detach "$commit"
    git -C "$dest" reset --hard "$commit"
  fi

  if [[ ! -f "$dest/$script" ]]; then
    echo "Benchmark script not found: $dest/$script" >&2
    exit 1
  fi

  # Local-only patch requested for quick reproduction: 10 * CONC -> PROMPT_MULTIPLIER * CONC.
  perl -0pi -e 's/--num-prompts "\$\(\(CONC \* 10\)\)"/--num-prompts "\$\(\(CONC * '"$PROMPT_MULTIPLIER"'\)\)"/g' "$dest/$script"
  if ! rg -F -q -- "--num-prompts \"\$((CONC * ${PROMPT_MULTIPLIER}))\"" "$dest/$script"; then
    echo "Failed to patch num-prompts in $dest/$script" >&2
    exit 1
  fi

  # The old qwen3.5 script predates explicit cuda graph sizing and lets SGLang
  # capture up to its default bs=512. On the current gfx950 stack that old
  # Triton path asserts during capture. Capping to the tested concurrency keeps
  # the serving shape aligned with this sweep and matches the newer recipe style.
  if ! rg -q -- '--cuda-graph-max-bs' "$dest/$script"; then
    perl -0pi -e 's/(--tensor-parallel-size \$TP \\\n)/$1    --max-running-requests \$CONC \\\n    --cuda-graph-max-bs \$CONC \\\n/g' "$dest/$script"
  fi

  # The old ROCm image has AITER installed but needs PYTHONPATH set correctly.
  # Force block FP8 GEMM away from the Triton path that fails to compile on this host.
  if [[ "$script" == "$OLD_SCRIPT" ]] && ! rg -q -- '--fp8-gemm-backend' "$dest/$script"; then
    perl -0pi -e 's/(--attention-backend triton \\\n)/$1    --fp8-gemm-backend aiter \\\n/g' "$dest/$script"
  fi
}

cleanup_docker() {
  if [[ "$CLEAN_DOCKER" == "1" ]]; then
    docker ps -aq | xargs -r docker rm -f
    docker network prune -f
  else
    docker rm -f qwen35-fp8-old qwen35-fp8-new >/dev/null 2>&1 || true
  fi
}

copy_run_artifacts() {
  local label=$1
  local worktree=$2
  local result_filename=$3
  local conc=$4
  local suffix=${5:-}

  mkdir -p "$OUT_DIR/$label"
  [[ -f "$worktree/${result_filename}.json" ]] && cp -f "$worktree/${result_filename}.json" "$OUT_DIR/$label/"
  [[ -f "$worktree/agg_${result_filename}.json" ]] && cp -f "$worktree/agg_${result_filename}.json" "$OUT_DIR/$label/"
  [[ -f "$worktree/server.log" ]] && cp -f "$worktree/server.log" "$OUT_DIR/$label/server_conc${conc}${suffix}.log"
  [[ -f "$worktree/gpu_metrics.csv" ]] && cp -f "$worktree/gpu_metrics.csv" "$OUT_DIR/$label/gpu_metrics_conc${conc}${suffix}.csv"
  return 0
}

run_one() {
  local label=$1
  local worktree=$2
  local image=$3
  local script=$4
  local conc=$5

  local result_filename="qwen3.5_1k1k_fp8_sglang_tp${TP}-ep${EP_SIZE}-dpafalse_disagg-false_spec-none_conc${conc}_prompts${PROMPT_MULTIPLIER}x_${label}"
  local result_path="$OUT_DIR/$label/${result_filename}.json"

  mkdir -p "$OUT_DIR/$label"
  if [[ "$SKIP_EXISTING" == "1" && -s "$result_path" ]]; then
    echo "[skip] $result_path already exists"
    return 0
  fi

  cleanup_docker
  rm -f "$worktree/${result_filename}.json" "$worktree/agg_${result_filename}.json"
  rm -f "$worktree/server.log" "$worktree/gpu_metrics.csv"

  echo "[run] label=$label image=$image tp=$TP ep=$EP_SIZE isl=$ISL osl=$OSL conc=$conc prompts=$((conc * PROMPT_MULTIPLIER))"

  set +e
  docker run --rm \
    --name "qwen35-fp8-${label}" \
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
    -w /workspace \
    -v "$worktree:/workspace" \
    -v "$MODEL_DIR:$MODEL_DIR" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HUB_CACHE="$MODEL_DIR" \
    -e MODEL="$MODEL" \
    -e MODEL_PREFIX="qwen3.5" \
    -e IMAGE="$image" \
    -e FRAMEWORK="sglang" \
    -e PRECISION="fp8" \
    -e RUNNER_TYPE="mi355x" \
    -e TP="$TP" \
    -e EP_SIZE="$EP_SIZE" \
    -e DP_ATTENTION="false" \
    -e CONC="$conc" \
    -e ISL="$ISL" \
    -e OSL="$OSL" \
    -e RANDOM_RANGE_RATIO="$RANDOM_RANGE_RATIO" \
    -e RESULT_FILENAME="$result_filename" \
    -e SPEC_DECODING="none" \
    -e DISAGG="false" \
    -e RUN_EVAL="false" \
    -e EVAL_ONLY="false" \
    -e PORT="$PORT" \
    -e SGLANG_USE_AITER=1 \
    -e SGLANG_USE_AITER_UNIFIED_ATTN=0 \
    -e PYTHONPATH=/sgl-workspace/aiter \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPYCACHEPREFIX=/tmp/inferencex-pycache \
    "$image" \
    bash "/workspace/$script"
  local docker_status=$?
  set -e
  if (( docker_status != 0 )); then
    copy_run_artifacts "$label" "$worktree" "$result_filename" "$conc" "_failed"
    echo "[error] docker run failed for label=$label conc=$conc status=$docker_status" >&2
    return "$docker_status"
  fi
  if [[ ! -s "$worktree/${result_filename}.json" ]]; then
    copy_run_artifacts "$label" "$worktree" "$result_filename" "$conc" "_failed"
    echo "[error] benchmark result JSON was not created for label=$label conc=$conc" >&2
    return 1
  fi

  if ! (
    cd "$worktree"
    RUNNER_TYPE=mi355x \
    FRAMEWORK=sglang \
    PRECISION=fp8 \
    SPEC_DECODING=none \
    RESULT_FILENAME="$result_filename" \
    ISL="$ISL" \
    OSL="$OSL" \
    DISAGG=false \
    MODEL_PREFIX="qwen3.5" \
    IMAGE="$image" \
    TP="$TP" \
    EP_SIZE="$EP_SIZE" \
    DP_ATTENTION=false \
    python3 utils/process_result.py
  ); then
    copy_run_artifacts "$label" "$worktree" "$result_filename" "$conc" "_failed"
    echo "[error] result processing failed for label=$label conc=$conc" >&2
    return 1
  fi
  copy_run_artifacts "$label" "$worktree" "$result_filename" "$conc"
}

write_compare_csv() {
  python3 - "$OUT_DIR" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])

def load(label):
    rows = {}
    for path in (out / label).glob("qwen3.5_1k1k_fp8_sglang_*.json"):
        if path.name.startswith("agg_"):
            continue
        data = json.loads(path.read_text())
        conc = int(data["max_concurrency"])
        rows[conc] = {
            "total_token_throughput": float(data.get("total_token_throughput", 0.0)),
            "output_throughput": float(data.get("output_throughput", 0.0)),
            "mean_ttft_ms": float(data.get("mean_ttft_ms", 0.0)),
            "mean_tpot_ms": float(data.get("mean_tpot_ms", 0.0)),
            "mean_itl_ms": float(data.get("mean_itl_ms", 0.0)),
        }
    return rows

old = load("old")
new = load("new")
fields = [
    "conc",
    "old_total_tps",
    "new_total_tps",
    "total_tps_delta_pct",
    "old_output_tps",
    "new_output_tps",
    "output_tps_delta_pct",
    "old_mean_ttft_ms",
    "new_mean_ttft_ms",
    "old_mean_tpot_ms",
    "new_mean_tpot_ms",
    "old_mean_itl_ms",
    "new_mean_itl_ms",
]
rows = []
for conc in sorted(set(old) & set(new)):
    o = old[conc]
    n = new[conc]
    def pct(key):
        return "" if o[key] == 0 else (n[key] / o[key] - 1.0) * 100.0
    rows.append({
        "conc": conc,
        "old_total_tps": o["total_token_throughput"],
        "new_total_tps": n["total_token_throughput"],
        "total_tps_delta_pct": pct("total_token_throughput"),
        "old_output_tps": o["output_throughput"],
        "new_output_tps": n["output_throughput"],
        "output_tps_delta_pct": pct("output_throughput"),
        "old_mean_ttft_ms": o["mean_ttft_ms"],
        "new_mean_ttft_ms": n["mean_ttft_ms"],
        "old_mean_tpot_ms": o["mean_tpot_ms"],
        "new_mean_tpot_ms": n["mean_tpot_ms"],
        "old_mean_itl_ms": o["mean_itl_ms"],
        "new_mean_itl_ms": n["mean_itl_ms"],
    })

prompt_multiplier = next(
    (part.removeprefix("prompts").removesuffix("x")
     for label in ("old", "new")
     for path in (out / label).glob("qwen3.5_1k1k_fp8_sglang_*_prompts*x_*.json")
     for part in path.stem.split("_")
     if part.startswith("prompts") and part.endswith("x")),
    "3",
)
tp = next(
    (m.group(1)
     for label in ("old", "new")
     for path in (out / label).glob("qwen3.5_1k1k_fp8_sglang_tp*_prompts*x_*.json")
     for m in [re.search(r"_tp(\d+)-", path.stem)]
     if m),
    "8",
)
csv_path = out / f"compare_1k1k_tp{tp}_prompts{prompt_multiplier}x.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {csv_path}")
PY
}

main() {
  need_cmd git
  need_cmd docker
  need_cmd perl
  need_cmd rg
  need_cmd python3

  if [[ ! -d "$REPO/.git" ]]; then
    echo "Repo not found at $REPO" >&2
    exit 1
  fi

  mkdir -p "$OUT_DIR" "$MODEL_DIR"
  echo "[info] model cache: $MODEL_DIR"
  df -h "$MODEL_DIR" || true

  prepare_worktree "$OLD_COMMIT" "$OLD_WT" "$OLD_SCRIPT"
  prepare_worktree "$NEW_COMMIT" "$NEW_WT" "$NEW_SCRIPT"

  for conc in $CONC_LIST; do
    run_one old "$OLD_WT" "$OLD_IMAGE" "$OLD_SCRIPT" "$conc"
    run_one new "$NEW_WT" "$NEW_IMAGE" "$NEW_SCRIPT" "$conc"
    write_compare_csv
  done

  echo "[done] results: $OUT_DIR"
}

main "$@"
