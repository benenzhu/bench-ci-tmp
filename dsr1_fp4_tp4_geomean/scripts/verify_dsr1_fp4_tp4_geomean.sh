#!/usr/bin/env bash
set -euo pipefail

# Verify the DeepSeek-R1-0528 MXFP4 SGLang geomean bundle — TP4 variant.
# Both ends run at TP4 (tput_per_gpu is per-GPU normalized). Unlike the sibling
# dsr1_fp4_geomean/ (TP8, same-checkpoint) bundle, here the OLD and NEW ends use
# DIFFERENT MXFP4 checkpoints:
#   - OLD: revision 6c94c74 (pre-2025-09-30) on the v0.5.2 image + old server flags.
#   - NEW: revision 913fc83b (HEAD, 2026-02-26) on the v0.5.13 image.
# So the gain reflects BOTH the software upgrade AND AMD's model-level weight
# optimization across releases (an end-to-end latest-vs-earliest comparison).
# Headline = geomean over the high-concurrency window 8k1k c32/c64/c128.
# Run from anywhere; ROOT defaults to this script's parent bundle dir.
ROOT=${ROOT:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"}

python3 - "$ROOT" <<'PY'
import csv, math, sys
from pathlib import Path

root = Path(sys.argv[1])
comparison_path = root / "comparison.csv"
combined_path = root / "combined_results.csv"

# headline = the three high-concurrency configs (8k1k c32/c64/c128)
headline_configs = ["8k1k_c32", "8k1k_c64", "8k1k_c128"]
expected_headline_geomean = 2.5546822539577927

if not comparison_path.exists(): raise SystemExit(f"missing {comparison_path}")
if not combined_path.exists(): raise SystemExit(f"missing {combined_path}")

with comparison_path.open(newline="") as f:
    rows = list(csv.DictReader(f))

data_rows = [r for r in rows if r["config_id"] != "GEOMEAN"]
seen = {r["config_id"] for r in data_rows}
for c in headline_configs:
    if c not in seen:
        raise SystemExit(f"missing headline config: {c}")

gains = {}
for r in data_rows:
    if r["config_id"] not in headline_configs:
        continue
    if r["old_status"] != "success" or r["new_status"] != "success":
        raise SystemExit(f"non-success {r['config_id']}: {r['old_status']}/{r['new_status']}")
    g = float(r["measured_gain_x"])
    if g <= 0: raise SystemExit(f"bad gain {r['config_id']}: {g}")
    # confirm TP4 on both ends
    if r.get("old_tput_per_gpu", "") == "" or r.get("new_tput_per_gpu", "") == "":
        raise SystemExit(f"missing tput for {r['config_id']}")
    gains[r["config_id"]] = g

hc = [gains[c] for c in headline_configs]
hc_gm = math.prod(hc) ** (1.0 / len(hc))
if not (2.0 <= hc_gm <= 4.0):
    raise SystemExit(f"headline geomean outside credible band [2.0, 4.0]: {hc_gm}")
if not math.isclose(hc_gm, expected_headline_geomean, rel_tol=0, abs_tol=1e-9):
    raise SystemExit(f"unexpected headline geomean: {hc_gm}")

# confirm both ends recorded TP=4 in the combined rows
with combined_path.open(newline="") as f:
    crows = list(csv.DictReader(f))
for r in crows:
    if r["config_id"] in headline_configs:
        if r["status"] != "success":
            raise SystemExit(f"non-success combined row: {r['label']} {r['status']}")
        if r.get("tp", "") != "4":
            raise SystemExit(f"expected TP4, got tp={r.get('tp')} for {r['label']}")

print(f"ok: headline (8k1k c32/c64/c128, TP4, old-ckpt->HEAD-ckpt) geomean={hc_gm:.6f}x; "
      f"c32={gains['8k1k_c32']:.3f}x c64={gains['8k1k_c64']:.3f}x c128={gains['8k1k_c128']:.3f}x")
PY
