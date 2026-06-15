#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$ROOT"

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

ok() {
  echo "[OK] $*"
}

require_file() {
  [[ -f "$1" ]] || fail "missing file: $1"
}

require_success() {
  local path=$1
  require_file "$path"
  [[ "$(tr -d '\r\n' < "$path")" == "success" ]] || fail "$path is not success"
}

require_status() {
  local path=$1
  local expected=$2
  require_file "$path"
  [[ "$(tr -d '\r\n' < "$path")" == "$expected" ]] || fail "$path is not $expected"
}

required_files=(
  README.md
  comparison.csv
  combined_results.csv
  repro_commands.sh
  scripts/repro_dsv4_sglang_8k1k_conc8_tp8.sh
  scripts/repro_mi355_adsuite.sh
  docs/dsv4_sglang_8k1k_conc8_repro_plan.md
  docs/candidates/mi355_nondisagg_no_atom_best_all_models.md
  docs/candidates/mi355_nondisagg_no_atom_best_all_models.csv
  docs/candidates/mi355_nondisagg_gt200pct_best_per_model_no_atom.md
  docs/candidates/mi355_nondisagg_gt200pct_best_per_model_no_atom.csv
  cache_status/initial.txt
  cache_status/final.txt
  cache_status/after_restore_dsv4_sglang_old.txt
  runs/dsv4_sglang_old/status.txt
  runs/dsv4_sglang_new/status.txt
  runs/dsv4_sglang_old/docker.log
  runs/dsv4_sglang_new/docker.log
  runs/dsv4_sglang_old/server.log
  runs/dsv4_sglang_new/server.log
  runs/dsv4_sglang_old/gpu_metrics.csv
  runs/dsv4_sglang_new/gpu_metrics.csv
  runs/dsv4_sglang_old/dsv4_8192x1024_fp4_sglang_mi355x_tp8-ep1-dpafalse_disagg-false_spec-none_conc8_prompts3x_old.json
  runs/dsv4_sglang_new/dsv4_8192x1024_fp4_sglang_mi355x_tp8-ep1-dpafalse_disagg-false_spec-none_conc8_prompts3x_new.json
  runs/dsv4_sglang_old/agg_dsv4_8192x1024_fp4_sglang_mi355x_tp8-ep1-dpafalse_disagg-false_spec-none_conc8_prompts3x_old.json
  runs/dsv4_sglang_new/agg_dsv4_8192x1024_fp4_sglang_mi355x_tp8-ep1-dpafalse_disagg-false_spec-none_conc8_prompts3x_new.json
  workflow_metadata/dsv4_sglang_old/run.json
  workflow_metadata/dsv4_sglang_old/jobs.json
  workflow_metadata/dsv4_sglang_new/run.json
  workflow_metadata/dsv4_sglang_new/jobs.json
  suites/mi355_adsuite_prompts3x/README.md
  suites/mi355_adsuite_prompts3x/comparison.csv
  suites/mi355_adsuite_prompts3x/combined_results.csv
  suites/mi355_adsuite_prompts3x/repro_commands.sh
  suites/mi355_adsuite_prompts3x/runs/kimi_old/status.txt
  suites/mi355_adsuite_prompts3x/runs/kimi_new/status.txt
  suites/mi355_adsuite_prompts3x/runs/glm_old/status.txt
  suites/mi355_adsuite_prompts3x/runs/glm_new/status.txt
  suites/mi355_adsuite_prompts3x/runs/dsv4_old/status.txt
  suites/mi355_adsuite_prompts3x/runs/dsv4_new/status.txt
  suites/mi355_adsuite_prompts3x/runs/kimi_old/agg_kimik2.5_8192x1024_fp4_vllm_mi355x_tp8-ep1-dpafalse_disagg-false_spec-none_conc4_prompts3x_old.json
  suites/mi355_adsuite_prompts3x/runs/kimi_new/agg_kimik2.5_8192x1024_fp4_vllm_mi355x_tp8-ep1-dpafalse_disagg-false_spec-none_conc4_prompts3x_new.json
  suites/mi355_adsuite_prompts3x/runs/glm_old/agg_glm5_1024x1024_fp8_sglang_mi355x_tp8-ep1-dpafalse_disagg-false_spec-none_conc4_prompts3x_old.json
  suites/mi355_adsuite_prompts3x/runs/glm_new/agg_glm5_1024x1024_fp8_sglang_mi355x_tp8-ep1-dpafalse_disagg-false_spec-none_conc4_prompts3x_new.json
)

for path in "${required_files[@]}"; do
  require_file "$path"
done
ok "required files are present"

require_success runs/dsv4_sglang_old/status.txt
require_success runs/dsv4_sglang_new/status.txt
ok "old/new status files are success"

require_success suites/mi355_adsuite_prompts3x/runs/kimi_old/status.txt
require_success suites/mi355_adsuite_prompts3x/runs/kimi_new/status.txt
require_success suites/mi355_adsuite_prompts3x/runs/glm_old/status.txt
require_success suites/mi355_adsuite_prompts3x/runs/glm_new/status.txt
require_success suites/mi355_adsuite_prompts3x/runs/dsv4_new/status.txt
require_status suites/mi355_adsuite_prompts3x/runs/dsv4_old/status.txt docker_failed:1
ok "multi-model suite statuses match expected results"

if [[ -f MANIFEST.sha256 ]]; then
  sha256sum -c MANIFEST.sha256 >/dev/null
  ok "MANIFEST.sha256 matches current files"
else
  echo "[WARN] MANIFEST.sha256 is missing; skipping checksum validation"
fi

if command -v rg >/dev/null 2>&1; then
  if rg -n "hf_[A-Za-z0-9]{20,}" --glob '!.git/**' . >/dev/null; then
    fail "possible persisted Hugging Face token found"
  fi
else
  if grep -RIE "hf_[A-Za-z0-9]{20,}" --exclude-dir=.git . >/dev/null; then
    fail "possible persisted Hugging Face token found"
  fi
fi
ok "strict HF token scan has no matches"

grep -q '^model_type=deepseek_v4$' cache_status/initial.txt || fail "initial cache model_type is not deepseek_v4"
grep -q '^model_type=deepseek_v4$' cache_status/final.txt || fail "final cache model_type is not deepseek_v4"
initial_sha=$(awk -F= '/^sha256=/{print $2; exit}' cache_status/initial.txt)
final_sha=$(awk -F= '/^sha256=/{print $2; exit}' cache_status/final.txt)
[[ -n "$initial_sha" ]] || fail "initial cache sha256 missing"
[[ "$initial_sha" == "$final_sha" ]] || fail "cache sha256 changed: initial=$initial_sha final=$final_sha"
ok "HF cache config restored to deepseek_v4 with unchanged sha256"

python3 - <<'PY'
import csv
import json
import math
from pathlib import Path

root = Path(".")

with (root / "comparison.csv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if len(rows) != 1:
    raise SystemExit(f"comparison.csv should have exactly one row, got {len(rows)}")
row = rows[0]

expected = {
    "model_group": "dsv4",
    "old_label": "dsv4_sglang_old",
    "new_label": "dsv4_sglang_new",
    "old_status": "success",
    "new_status": "success",
}
for key, value in expected.items():
    if row.get(key) != value:
        raise SystemExit(f"comparison.csv {key}={row.get(key)!r}, expected {value!r}")

expected_numbers = {
    "old_tput_per_gpu": 104.04413451994634,
    "new_tput_per_gpu": 421.9827666910809,
    "measured_gain_x": 4.0558054390868366,
    "measured_improvement_pct": 305.58054390868364,
    "expected_old_tput_per_gpu": 115.801807,
    "expected_new_tput_per_gpu": 483.774241,
    "expected_gain_x": 4.177605285554828,
}
for key, value in expected_numbers.items():
    got = float(row[key])
    if not math.isclose(got, value, rel_tol=0, abs_tol=1e-9):
        raise SystemExit(f"comparison.csv {key}={got}, expected {value}")

old = float(row["old_tput_per_gpu"])
new = float(row["new_tput_per_gpu"])
gain = new / old
if gain <= 4.0:
    raise SystemExit(f"measured gain should be >4.0x, got {gain}")

agg_paths = {
    "old": root / "runs/dsv4_sglang_old/agg_dsv4_8192x1024_fp4_sglang_mi355x_tp8-ep1-dpafalse_disagg-false_spec-none_conc8_prompts3x_old.json",
    "new": root / "runs/dsv4_sglang_new/agg_dsv4_8192x1024_fp4_sglang_mi355x_tp8-ep1-dpafalse_disagg-false_spec-none_conc8_prompts3x_new.json",
}
expected_tput = {"old": old, "new": new}
for variant, path in agg_paths.items():
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    got = float(data["tput_per_gpu"])
    if not math.isclose(got, expected_tput[variant], rel_tol=0, abs_tol=1e-9):
        raise SystemExit(f"{path} tput_per_gpu={got}, expected {expected_tput[variant]}")
    if data["hw"] != "mi355x" or data["framework"] != "sglang" or data["precision"] != "fp4":
        raise SystemExit(f"{path} has unexpected hw/framework/precision metadata")
    if data["isl"] != 8192 or data["osl"] != 1024 or data["conc"] != 8 or data["tp"] != 8:
        raise SystemExit(f"{path} has unexpected benchmark shape")

print("summary:")
print(f"  old tok/s/GPU: {old:.6f}")
print(f"  new tok/s/GPU: {new:.6f}")
print(f"  measured gain: {gain:.4f}x")
print(f"  measured improvement: {(gain - 1.0) * 100:.2f}%")

suite_path = root / "suites/mi355_adsuite_prompts3x/comparison.csv"
with suite_path.open(newline="", encoding="utf-8") as f:
    suite_rows = {r["model_group"]: r for r in csv.DictReader(f)}

suite_expected = {
    "kimi": {
        "old_status": "success",
        "new_status": "success",
        "old_tput_per_gpu": 24.517506602596864,
        "new_tput_per_gpu": 359.12262861606865,
        "measured_gain_x": 14.64759995529198,
    },
    "glm": {
        "old_status": "success",
        "new_status": "success",
        "old_tput_per_gpu": 16.169319874259486,
        "new_tput_per_gpu": 41.51533759847504,
        "measured_gain_x": 2.567537652870903,
    },
    "dsv4": {
        "old_status": "docker_failed:1",
        "new_status": "success",
        "new_tput_per_gpu": 21.32887424788058,
    },
}

for group, checks in suite_expected.items():
    if group not in suite_rows:
        raise SystemExit(f"{suite_path} missing group {group}")
    suite_row = suite_rows[group]
    for key, value in checks.items():
        if key.endswith("status"):
            if suite_row[key] != value:
                raise SystemExit(f"{suite_path} {group} {key}={suite_row[key]!r}, expected {value!r}")
        else:
            got = float(suite_row[key])
            if not math.isclose(got, value, rel_tol=0, abs_tol=1e-9):
                raise SystemExit(f"{suite_path} {group} {key}={got}, expected {value}")

print("multi-model suite:")
print("  Kimi-K2.5: 24.517507 -> 359.122629 tok/s/GPU, 14.6476x")
print("  GLM-5: 16.169320 -> 41.515338 tok/s/GPU, 2.5675x")
print("  DSV4 vLLM old: expected docker_failed:1; DSV4 SGLang replacement is the primary success row")
PY
ok "numeric results match expected bundle values"

if [[ -d .git ]]; then
  git status --short
fi

echo "verification complete"
