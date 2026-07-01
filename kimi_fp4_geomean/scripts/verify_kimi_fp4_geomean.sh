#!/usr/bin/env bash
set -euo pipefail

# Verify the Kimi-K2.5 FP4 vLLM geomean bundle.
# The bundle sweeps 9 configs; the reported headline is the credible high-concurrency
# geomean over 8k1k CONC 64/128/256 (both old/new near saturation).
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
    "1k1k_c4", "1k1k_c8",
    "8k1k_c4", "8k1k_c8", "8k1k_c16", "8k1k_c32", "8k1k_c64", "8k1k_c128", "8k1k_c256",
}
headline_configs = ["8k1k_c64", "8k1k_c128", "8k1k_c256"]
expected_headline_geomean = 3.709068175509587

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

gains = {}
for row in data_rows:
    if row["old_status"] != "success" or row["new_status"] != "success":
        raise SystemExit(f"non-success status in {row['config_id']}: {row['old_status']} / {row['new_status']}")
    g = float(row["measured_gain_x"])
    if g <= 0:
        raise SystemExit(f"bad gain for {row['config_id']}: {g}")
    gains[row["config_id"]] = g

# Headline = geomean over the high-concurrency subset.
hc = [gains[c] for c in headline_configs]
headline = math.prod(hc) ** (1.0 / len(hc))
if not (3.0 <= headline <= 4.0):
    raise SystemExit(f"high-conc geomean outside target band [3.0, 4.0]: {headline}")
if not math.isclose(headline, expected_headline_geomean, rel_tol=0, abs_tol=1e-9):
    raise SystemExit(f"unexpected high-conc geomean: {headline}")

with combined_path.open(newline="") as f:
    combined_rows = list(csv.DictReader(f))
if len(combined_rows) != len(expected_configs) * 2:
    raise SystemExit(f"expected {len(expected_configs) * 2} combined rows, got {len(combined_rows)}")
for row in combined_rows:
    if row["status"] != "success":
        raise SystemExit(f"non-success combined row: {row['label']} {row['status']}")

print(f"ok: {len(expected_configs)} configs; high-conc (c64/c128/c256) geomean={headline:.6f}x")
PY
