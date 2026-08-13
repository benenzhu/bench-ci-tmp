#!/usr/bin/env bash
# Minimal standalone reproducer:
#   vllm/vllm-openai-rocm:nightly cannot serve Kimi-K2.5-MXFP4 on MI355X (gfx950).
#   The aiter Gluon MLA decode kernel fails to compile, every TP worker dies,
#   and the engine never initializes.
#
#   File "/aiter/ops/triton/gluon/mla_gluon.py", line 1029, in mla_gluon
#     gl.amd.cdna4.async_copy.buffer_load_to_shared(buf_q_pe, Q_pe, offs_q_pe, ...)
#   expected offsets type layout to be BlockedLayout or SliceLayout
#
# Works on: vllm/vllm-openai-rocm:v0.22.0 and rocm/atom-dev:vllm-v0.25.1-nightly_20260811
# Fails on: vllm/vllm-openai-rocm:nightly (2026-08-12)
#
# Usage:
#   ./repro_bug.sh                       # default: nightly (expect failure)
#   IMAGE=vllm/vllm-openai-rocm:v0.22.0 ./repro_bug.sh   # control: expect success
#   VLLM_ROCM_AITER_MLA_ASM_PADDING=asm ./repro_bug.sh    # workaround: force ASM decode
#
# Env knobs: IMAGE, MODEL_DIR, TP, PORT, ROCR_VISIBLE_DEVICES,
#            VLLM_ROCM_AITER_MLA_ASM_PADDING (auto|gluon|asm)
set -uo pipefail

IMAGE=${IMAGE:-vllm/vllm-openai-rocm:nightly}
MODEL_DIR=${MODEL_DIR:-/dev/shm/model/Kimi-K2.5-MXFP4}
MODEL_REPO=${MODEL_REPO:-amd/Kimi-K2.5-MXFP4}
TP=${TP:-8}
PORT=${PORT:-19000}
NAME=vllm-nightly-mla-repro
LOG=${LOG:-/tmp/${NAME}.log}

echo "image     : $IMAGE"
echo "model dir : $MODEL_DIR"
echo "TP        : $TP"
echo "log       : $LOG"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  cat <<MSG
ERROR: model not found at $MODEL_DIR

Download it first (~559 GB), e.g.:
  huggingface-cli download $MODEL_REPO --local-dir $MODEL_DIR
MSG
  exit 1
fi

docker rm -f "$NAME" >/dev/null 2>&1

# Note: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here --
# it independently breaks custom_all_reduce's hipIpcGetMemHandle (aiter #4174).
docker run --rm --name "$NAME" \
  --network host --ipc host --privileged \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODEL_DIR:$MODEL_DIR:ro" \
  ${ROCR_VISIBLE_DEVICES:+-e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"} \
  -e HF_HUB_OFFLINE=1 \
  ${VLLM_ROCM_AITER_MLA_ASM_PADDING:+-e VLLM_ROCM_AITER_MLA_ASM_PADDING="$VLLM_ROCM_AITER_MLA_ASM_PADDING"} \
  --entrypoint /bin/bash "$IMAGE" -lc "
    vllm serve '$MODEL_DIR' \
      --served-model-name kimi-k2.5-mxfp4 \
      --tensor-parallel-size $TP \
      --max-model-len 9416 \
      --port $PORT
  " 2>&1 | tee "$LOG"

echo
echo "================ verdict ================"
if grep -q "expected offsets type layout to be BlockedLayout or SliceLayout" "$LOG"; then
  echo "REPRODUCED: aiter Gluon MLA kernel failed to compile."
  grep -n "mla_gluon.py" "$LOG" | head -3
  exit 1
elif grep -qE "Application startup complete|Uvicorn running" "$LOG"; then
  echo "OK: server started on this image."
  exit 0
else
  echo "Server did not start, but not the known MLA-layout error. See $LOG"
  exit 2
fi
