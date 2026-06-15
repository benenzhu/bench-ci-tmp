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

required_files=(
  README.md
  comparison.csv
  combined_results.csv
  repro_commands.sh
  MANIFEST.sha256
  scripts/repro_kimi_int4_8k1k_conc16_tp8.sh
  scripts/verify_kimi_int4_8k1k_conc16.sh
  cache_status/initial.txt
  cache_status/final.txt
  cache_status/after_kimi_int4_old.txt
  cache_status/after_kimi_int4_new.txt
  cache_status/kimi_int4_old_image_manifest.txt
  cache_status/kimi_int4_new_image_manifest.txt
  runs/kimi_int4_old/status.txt
  runs/kimi_int4_new/status.txt
  runs/kimi_int4_old/docker.log
  runs/kimi_int4_new/docker.log
  runs/kimi_int4_old/server.log
  runs/kimi_int4_new/server.log
  runs/kimi_int4_old/env.repro
  runs/kimi_int4_new/env.repro
  runs/kimi_int4_old/kimik2.5_8192x1024_int4_vllm_mi355x_tp8-ep1-dpafalse_disagg-false_spec-none_conc16_prompts3x_old.json
  runs/kimi_int4_new/kimik2.5_8192x1024_int4_vllm_mi355x_tp8-ep1-dpafalse_disagg-false_spec-none_conc16_prompts3x_new.json
  runs/kimi_int4_old/agg_kimik2.5_8192x1024_int4_vllm_mi355x_tp8-ep1-dpafalse_disagg-false_spec-none_conc16_prompts3x_old.json
  runs/kimi_int4_new/agg_kimik2.5_8192x1024_int4_vllm_mi355x_tp8-ep1-dpafalse_disagg-false_spec-none_conc16_prompts3x_new.json
  workflow_metadata/kimi_int4_old/run.json
  workflow_metadata/kimi_int4_old/jobs.json
  workflow_metadata/kimi_int4_new/run.json
  workflow_metadata/kimi_int4_new/jobs.json
)

for path in "${required_files[@]}"; do
  require_file "$path"
done
ok "required files are present"

require_success runs/kimi_int4_old/status.txt
require_success runs/kimi_int4_new/status.txt
ok "old/new status files are success"

sha256sum -c MANIFEST.sha256 >/dev/null
ok "MANIFEST.sha256 matches current files"

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

python3 - <<'PY'
import csv
import json
import math
from pathlib import Path

root = Path(".")

def close(got, expected, key):
    if not math.isclose(float(got), float(expected), rel_tol=0, abs_tol=1e-9):
        raise SystemExit(f"{key}={got}, expected {expected}")

with (root / "comparison.csv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if len(rows) != 1:
    raise SystemExit(f"comparison.csv should have exactly one row, got {len(rows)}")
row = rows[0]

expected_strings = {
    "model_group": "kimi_int4",
    "old_label": "kimi_int4_old",
    "new_label": "kimi_int4_new",
    "old_status": "success",
    "new_status": "success",
}
for key, expected in expected_strings.items():
    if row.get(key) != expected:
        raise SystemExit(f"comparison.csv {key}={row.get(key)!r}, expected {expected!r}")

expected_numbers = {
    "old_tput_per_gpu": 142.1744402828832,
    "new_tput_per_gpu": 187.66949595644815,
    "measured_gain_x": 1.3199946177600126,
    "measured_improvement_pct": 31.999461776001258,
    "expected_old_tput_per_gpu": 157.19541980732237,
    "expected_new_tput_per_gpu": 631.0259966432371,
    "expected_gain_x": 4.014277244316015,
}
for key, expected in expected_numbers.items():
    close(row[key], expected, f"comparison.csv {key}")

with (root / "combined_results.csv").open(newline="", encoding="utf-8") as f:
    combined = {r["variant"]: r for r in csv.DictReader(f)}
if set(combined) != {"old", "new"}:
    raise SystemExit(f"combined_results.csv variants are {sorted(combined)}, expected old/new")

for variant, expected_tput in {"old": 142.1744402828832, "new": 187.66949595644815}.items():
    data = combined[variant]
    if data["status"] != "success":
        raise SystemExit(f"combined {variant} status={data['status']!r}")
    if data["model"] != "moonshotai/Kimi-K2.5" or data["precision"] != "int4" or data["framework"] != "vllm":
        raise SystemExit(f"combined {variant} has unexpected model/precision/framework")
    if data["isl"] != "8192" or data["osl"] != "1024" or data["conc"] != "16" or data["tp"] != "8":
        raise SystemExit(f"combined {variant} has unexpected benchmark shape")
    close(data["tput_per_gpu"], expected_tput, f"combined {variant} tput_per_gpu")

raw_paths = {
    "old": root / "runs/kimi_int4_old/kimik2.5_8192x1024_int4_vllm_mi355x_tp8-ep1-dpafalse_disagg-false_spec-none_conc16_prompts3x_old.json",
    "new": root / "runs/kimi_int4_new/kimik2.5_8192x1024_int4_vllm_mi355x_tp8-ep1-dpafalse_disagg-false_spec-none_conc16_prompts3x_new.json",
}
expected_raw = {
    "old": {
        "duration": 347.8235954480042,
        "total_token_throughput": 1137.3955222630657,
        "output_throughput": 127.4151626858939,
    },
    "new": {
        "duration": 263.503798249003,
        "total_token_throughput": 1501.3559676515852,
        "output_throughput": 168.18732896639634,
    },
}
for variant, path in raw_paths.items():
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if data["model_id"] != "moonshotai/Kimi-K2.5":
        raise SystemExit(f"{path} model_id={data['model_id']!r}")
    for key, expected in {
        "num_prompts": 48,
        "completed": 48,
        "max_concurrency": 16,
        "total_input_tokens": 351295,
        "total_output_tokens": 44318,
    }.items():
        if data[key] != expected:
            raise SystemExit(f"{path} {key}={data[key]}, expected {expected}")
    for key, expected in expected_raw[variant].items():
        close(data[key], expected, f"{path} {key}")

jobs = {
    "old": (root / "workflow_metadata/kimi_int4_old/jobs.json", 64160961756),
    "new": (root / "workflow_metadata/kimi_int4_new/jobs.json", 70014502875),
}
for variant, (path, job_id) in jobs.items():
    data = json.loads(path.read_text(encoding="utf-8"))
    matches = [job for job in data.get("jobs", []) if job.get("id") == job_id]
    if not matches:
        raise SystemExit(f"{path} missing job {job_id}")
    name = matches[0].get("name", "")
    if "8k1k" not in name or "int4" not in name or "conc-16" not in name:
        raise SystemExit(f"{path} job {job_id} has unexpected name: {name}")

print("summary:")
print("  local 8k/1k Kimi INT4 result: 142.174440 -> 187.669496 tok/s/GPU, 1.3200x")
print("  dashboard reference in candidate table: 157.195420 -> 631.025997 tok/s/GPU, 4.0143x")
print("  interpretation: completed local attempt, but 4.0143x was not reproduced")
PY
ok "numeric results and workflow metadata match expected bundle values"

echo "verification complete"
