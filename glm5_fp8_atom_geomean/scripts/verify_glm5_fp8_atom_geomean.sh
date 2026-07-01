#!/usr/bin/env bash
set -euo pipefail

# Verify the GLM-5 FP8 ATOM geomean bundle: NEW=ATOM vs OLD=SGLang v0.5.8 baseline.
# Run from anywhere; ROOT defaults to this script's parent bundle directory.
ROOT=${ROOT:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"}

python3 - "$ROOT" <<'PY'
import csv
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
comparison_path = root / "comparison.csv"
combined_path = root / "combined_results.csv"

expected_configs = {
    "1k1k_c1",
    "1k1k_c2",
    "1k1k_c4",
    "1k1k_c8",
    "1k1k_c16",
    "8k1k_c4",
    "8k1k_c8",
}
expected_geomean = 3.1414573381911475

if not comparison_path.exists():
    raise SystemExit(f"missing {comparison_path}")
if not combined_path.exists():
    raise SystemExit(f"missing {combined_path}")

with comparison_path.open(newline="") as f:
    rows = list(csv.DictReader(f))

data_rows = [row for row in rows if row["config_id"] != "GEOMEAN"]
seen_configs = {row["config_id"] for row in data_rows}
if seen_configs != expected_configs:
    raise SystemExit(f"config mismatch: {sorted(seen_configs)}")

for row in data_rows:
    if row["old_status"] != "success" or row["new_status"] != "success":
        raise SystemExit(f"non-success status in {row['config_id']}: {row['old_status']} / {row['new_status']}")
    gain = float(row["measured_gain_x"])
    if gain <= 0:
        raise SystemExit(f"bad gain for {row['config_id']}: {gain}")

geomean_rows = [row for row in rows if row["config_id"] == "GEOMEAN"]
if len(geomean_rows) != 1:
    raise SystemExit(f"expected one GEOMEAN row, got {len(geomean_rows)}")

geomean = float(geomean_rows[0]["measured_gain_x"])
if not (3.0 <= geomean <= 4.0):
    raise SystemExit(f"geomean outside target band [3.0, 4.0]: {geomean}")
if not math.isclose(geomean, expected_geomean, rel_tol=0, abs_tol=1e-9):
    raise SystemExit(f"unexpected geomean: {geomean}")

# NEW variant must be ATOM, OLD must be the reused SGLang baseline.
with combined_path.open(newline="") as f:
    combined_rows = list(csv.DictReader(f))
if len(combined_rows) != len(expected_configs) * 2:
    raise SystemExit(f"expected {len(expected_configs) * 2} combined rows, got {len(combined_rows)}")
for row in combined_rows:
    if row["status"] != "success":
        raise SystemExit(f"non-success combined row: {row['label']} {row['status']}")
    if row["variant"] == "new" and row.get("framework") != "atom":
        raise SystemExit(f"new variant not ATOM: {row['label']} framework={row.get('framework')}")
    if row["variant"] == "old" and row.get("framework") != "sglang":
        raise SystemExit(f"old variant not SGLang baseline: {row['label']} framework={row.get('framework')}")

print(f"ok: {len(expected_configs)} configs, ATOM-vs-SGLang geomean={geomean:.6f}x")
PY
