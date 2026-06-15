#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

test -s combined_results.csv
test -s runs/old_server_flags_no_block1/status.txt
grep -qx 'success' runs/old_server_flags_no_block1/status.txt

python3 - <<'PY'
import csv
from pathlib import Path

rows = {row["label"]: row for row in csv.DictReader(Path("combined_results.csv").open())}
row = rows["old_server_flags_no_block1"]
assert row["status"] == "success", row
assert row["completed"] == "192", row
assert abs(float(row["tput_per_gpu"]) - 290.7628669348586) < 1e-6, row
assert abs(float(row["output_tput_per_gpu"]) - 32.307676296534865) < 1e-6, row

for label in ["old_all_new_flags", "old_aiter_only", "old_block1_only"]:
    assert rows[label]["status"] == "docker_failed:1", rows[label]
PY

rg -q "Aiter MLA only supports 16 or 128 number of heads" runs/old_all_new_flags/docker.log
rg -q "Provided 8 number of heads" runs/old_aiter_only/docker.log
rg -q "Provided 8 number of heads" runs/old_block1_only/docker.log
rg -q "Using Triton MLA backend" runs/old_server_flags_no_block1/server.log

if rg -n 'hf_[A-Za-z0-9]{20,}' . >/tmp/kimi_fp4_old_flags_token_hits.txt; then
  cat /tmp/kimi_fp4_old_flags_token_hits.txt >&2
  exit 1
fi

echo "Kimi FP4 old-image flag ablation verification passed."
