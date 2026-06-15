#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mhc.py")
text = path.read_text()
patched = text.replace(
    "T.pdl_sync()",
    "pass  # mi355_adsuite: disable TileLang PDL on ROCm",
).replace(
    "T.pdl_trigger()",
    "pass  # mi355_adsuite: disable TileLang PDL on ROCm",
)
if patched == text:
    print(f"[runtime-patch] no PDL calls found in {path}")
else:
    path.write_text(patched)
    print(f"[runtime-patch] disabled TileLang PDL calls in {path}")
PY
