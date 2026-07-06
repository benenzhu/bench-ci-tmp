#!/usr/bin/env bash
set -euo pipefail

# Verify the DeepSeek-R1-0528 MXFP4 SGLang geomean bundle.
# Both OLD and NEW use the SAME model (amd/DeepSeek-R1-0528-MXFP4-Preview @ 6c94c74)
# and SAME benchmark logic; the only variable is the SGLang image
# (old v0.5.2-rocm7.0 + old server flags -> new v0.5.13 + current flags).
# 12 configs: 1k1k c4/8/16/32/64 + 8k1k c4/8/16/32/64/128/256.
# Run from anywhere; ROOT defaults to this script's parent bundle dir.
ROOT=${ROOT:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"}

python3 - "$ROOT" <<'PY'
import csv, math, sys
from pathlib import Path

root = Path(sys.argv[1])
comparison_path = root / "comparison.csv"
combined_path = root / "combined_results.csv"

expected_configs = {
    "1k1k_c4","1k1k_c8","1k1k_c16","1k1k_c32","1k1k_c64",
    "8k1k_c4","8k1k_c8","8k1k_c16","8k1k_c32","8k1k_c64","8k1k_c128","8k1k_c256",
}
expected_geomean = 1.7718158468204603          # all 12 configs
# high-concurrency window (8k1k c64+c128), the peak part of the curve
highconc_configs = ["8k1k_c64", "8k1k_c128"]
expected_highconc_geomean = 2.1110086395805694

if not comparison_path.exists(): raise SystemExit(f"missing {comparison_path}")
if not combined_path.exists(): raise SystemExit(f"missing {combined_path}")

with comparison_path.open(newline="") as f:
    rows = list(csv.DictReader(f))

data_rows = [r for r in rows if r["config_id"] != "GEOMEAN"]
seen = {r["config_id"] for r in data_rows}
if seen != expected_configs:
    raise SystemExit(f"config mismatch: {sorted(seen)}")

gains = {}
for r in data_rows:
    if r["old_status"] != "success" or r["new_status"] != "success":
        raise SystemExit(f"non-success {r['config_id']}: {r['old_status']}/{r['new_status']}")
    g = float(r["measured_gain_x"])
    if g <= 0: raise SystemExit(f"bad gain {r['config_id']}: {g}")
    gains[r["config_id"]] = g

# whole-sweep geomean from the GEOMEAN row
geo = [r for r in rows if r["config_id"] == "GEOMEAN"]
if len(geo) != 1: raise SystemExit(f"expected 1 GEOMEAN row, got {len(geo)}")
gm = float(geo[0]["measured_gain_x"])
if not math.isclose(gm, expected_geomean, rel_tol=0, abs_tol=1e-9):
    raise SystemExit(f"unexpected all-config geomean: {gm}")

# high-conc window
hc = [gains[c] for c in highconc_configs]
hc_gm = math.prod(hc) ** (1.0/len(hc))
if not math.isclose(hc_gm, expected_highconc_geomean, rel_tol=0, abs_tol=1e-9):
    raise SystemExit(f"unexpected high-conc geomean: {hc_gm}")

with combined_path.open(newline="") as f:
    crows = list(csv.DictReader(f))
if len(crows) != len(expected_configs) * 2:
    raise SystemExit(f"expected {len(expected_configs)*2} combined rows, got {len(crows)}")
for r in crows:
    if r["status"] != "success":
        raise SystemExit(f"non-success combined row: {r['label']} {r['status']}")

print(f"ok: {len(expected_configs)} configs, all-config geomean={gm:.6f}x; "
      f"high-conc (8k1k c64+c128) geomean={hc_gm:.6f}x; peak 8k1k_c128={gains['8k1k_c128']:.3f}x")
PY
