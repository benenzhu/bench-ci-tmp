#!/usr/bin/env bash
set -euo pipefail

# Verify the gpt-oss-120b FP4 vLLM geomean bundle.
# Both ends run the SAME model (openai/gpt-oss-120b, native MXFP4) and SAME
# benchmark logic at TP1; the only variables are the vLLM image + server flags:
#   OLD (commit 12de5c0f): rocm/7.0:...vllm_0.10.1_instinct_20250927_rc1 + old flags
#   NEW (HEAD):            vllm/vllm-openai-rocm:v0.22.0 + new flags
# so the gain reflects pure software optimization.
# 5 configs: 8k1k c16/c32/c64/c128/c256. tput_per_gpu is per-GPU normalized.
# This is a modest-gain model (geomean ~1.15x, peak 1.29x at c256): gpt-oss lacks
# the DeepSeek-specific MoE/MLA kernel optimizations that drive the 2-4x bundles.
# Run from anywhere; ROOT defaults to this script's parent bundle dir.
ROOT=${ROOT:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"}

python3 - "$ROOT" <<'PY'
import csv, math, sys
from pathlib import Path

root = Path(sys.argv[1])
comparison_path = root / "comparison.csv"
combined_path = root / "combined_results.csv"

expected_configs = {"8k1k_c16","8k1k_c32","8k1k_c64","8k1k_c128","8k1k_c256"}
expected_geomean = 1.1506636066991018

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

with combined_path.open(newline="") as f:
    crows = list(csv.DictReader(f))
if len(crows) != len(expected_configs) * 2:
    raise SystemExit(f"expected {len(expected_configs)*2} combined rows, got {len(crows)}")
for r in crows:
    if r["status"] != "success":
        raise SystemExit(f"non-success combined row: {r['label']} {r['status']}")
    if r.get("tp", "") != "1":
        raise SystemExit(f"expected TP1, got tp={r.get('tp')} for {r['label']}")

print(f"ok: {len(expected_configs)} configs (8k1k c16-c256, TP1) geomean={gm:.6f}x; "
      f"peak 8k1k_c256={gains['8k1k_c256']:.3f}x")
PY
