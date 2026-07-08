#!/usr/bin/env bash
set -euo pipefail

# Verify the GLM-5 FP8 rocm700->ATOM geomean bundle (TP8).
# OLD end: rocm/sgl-dev:v0.5.8.post1-rocm700-mi35x-20260219 (ROCm 7.0), SGLang,
#   run fresh here (needs transformers==5.3.0 in-container for the glm_moe_dsa arch,
#   and --kv-cache-dtype fp8_e4m3 dropped since tilelang NSA needs bf16 KV).
# NEW end: rocm/atom:...atom0.1.2.post (ATOM engine), reused from glm5_fp8_atom_geomean/.
# Model zai-org/GLM-5-FP8 staged at /dev/shm/model/GLM-5-FP8. 7 configs:
# 1k1k c1/2/4/8/16 + 8k1k c4/8. tput_per_gpu is per-GPU normalized.
# Run from anywhere; ROOT defaults to this script's parent bundle dir.
ROOT=${ROOT:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"}

python3 - "$ROOT" <<'PY'
import csv, math, sys
from pathlib import Path

root = Path(sys.argv[1])
comparison_path = root / "comparison.csv"
combined_path = root / "combined_results.csv"

expected_configs = {"1k1k_c1","1k1k_c2","1k1k_c4","1k1k_c8","1k1k_c16","8k1k_c4","8k1k_c8"}
expected_geomean = 3.516458195125949

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

geo = [r for r in rows if r["config_id"] == "GEOMEAN"]
if len(geo) != 1: raise SystemExit(f"expected 1 GEOMEAN row, got {len(geo)}")
gm = float(geo[0]["measured_gain_x"])
if not math.isclose(gm, expected_geomean, rel_tol=0, abs_tol=1e-9):
    raise SystemExit(f"unexpected geomean: {gm}")
if not (2.0 <= gm <= 4.0):
    raise SystemExit(f"geomean outside credible band [2.0,4.0]: {gm}")

with combined_path.open(newline="") as f:
    crows = list(csv.DictReader(f))
if len(crows) != len(expected_configs) * 2:
    raise SystemExit(f"expected {len(expected_configs)*2} combined rows, got {len(crows)}")
for r in crows:
    if r["status"] != "success":
        raise SystemExit(f"non-success combined row: {r['label']} {r['status']}")
    if r.get("tp", "") != "8":
        raise SystemExit(f"expected TP8, got tp={r.get('tp')} for {r['label']}")

print(f"ok: {len(expected_configs)} configs (1k1k c1-16, 8k1k c4/8, TP8, rocm700->ATOM) "
      f"geomean={gm:.6f}x; per-config " + " ".join(f"{c}={gains[c]:.2f}x" for c in
      ["1k1k_c1","1k1k_c2","1k1k_c4","1k1k_c8","1k1k_c16","8k1k_c4","8k1k_c8"]))
PY
