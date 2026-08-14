# ROCm 10.1 + PyTorch base for building ATOM, on plain Ubuntu 24.04.
#
# Why this exists: no ROCm 10 rocm/pytorch image is published, and our previous
# workaround (layering ATOM onto a vLLM ROCm 10 image) produced two blockers,
# both from version skew rather than from ROCm 10 itself:
#   1. torch 2.12's Dynamo cannot trace aiter's pybind QuantType enum
#   2. aiter wheel 0.1.19's cache kernel aborts ("norm_weight dtype must match
#      q dtype") where ATOM's source-built aiter does not
# This base fixes both preconditions: torch 2.13 (what official ATOM images
# ship) and no prebuilt aiter at all, leaving atom_release.dockerfile to build
# aiter from source the way the official images do.
#
# The ROCm/torch installation is lifted from the Primus training Dockerfile,
# trimmed to just the ROCm SDK + torch (no TransformerEngine/Primus/FBGEMM/etc)
# and to gfx950 only. Layout matches official rocm/atom-dev: a /opt/venv
# virtualenv with ROCm delivered as _rocm_sdk_* wheels rather than /opt/rocm.
#
# Build:
#   DOCKER_BUILDKIT=1 docker build -f docker/rocm101_torch_base.dockerfile \
#     -t rocm101-torch:local .

ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE} AS rocm_torch_base

WORKDIR /workspace/
ENV MAX_JOBS=128
# gfx950 (MI355X) only — the benchmark boxes are all MI355X, and a single arch
# keeps the later aiter source build to a fraction of its multi-arch time.
ENV PYTORCH_ROCM_ARCH="gfx950"
ENV ROCM_AMDGPU_TARGETS="gfx950"
ENV GPU_ARCHS="${PYTORCH_ROCM_ARCH}"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update \
    && apt install --only-upgrade --no-install-recommends -y linux-libc-dev \
    && apt install --no-install-recommends -y \
    gfortran git git-lfs ninja-build g++ pkg-config xxd patchelf \
    automake libtool python3-venv python3-dev python3-pip python-is-python3 \
    libegl1-mesa-dev wget sudo flex liblzma-dev ccache libdw1 libdrm-dev \
    autoconf rdma-core ca-certificates \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip \
    && pip install \
        pybind11 typeguard wheel==0.46.2 cmake==3.31.6 ninja==1.11.1.3 \
        packaging==25.0 setuptools==80.10.2 \
    && rm -rf /root/.cache

# Workaround for the HSA_STATUS_ERROR_OUT_OF_RESOURCES issue
ENV HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0
ENV HSA_NO_SCRATCH_RECLAIM=1

# ROCm from TheRock nightly wheels.
# See: https://rocm.nightlies.amd.com/whl-multi-arch/rocm/
ARG THE_ROCK_VERSION=10.1.0a20260814
RUN python -m pip install \
        --index-url https://rocm.nightlies.amd.com/whl-multi-arch \
        --pre \
        rocm==${THE_ROCK_VERSION} \
        rocm-bootstrap \
        rocm-sdk-core==${THE_ROCK_VERSION} \
        rocm-sdk-devel==${THE_ROCK_VERSION} \
        rocm-sdk-device-gfx950==${THE_ROCK_VERSION} \
        rocm-sdk-libraries==${THE_ROCK_VERSION} \
    && rm -rf /root/.cache

# torch 2.13 — deliberately NOT the 2.12 the Primus file pins. Official ATOM
# images ship 2.13, and 2.12 is exactly what breaks Dynamo on aiter's pybind
# enums. The nightly index carries 2.13 for the same ROCm 10.1 date.
ARG PYTORCH_VERSION=2.13.0+rocm10.1.0a20260814
RUN pip install \
    cxxfilt==0.3.0 tqdm==4.67.3 pyyaml==6.0.3 py-cpuinfo==9.0.0 build==1.5.0 \
    && python -m pip uninstall -y torch \
    && python -m pip install \
        --index-url https://rocm.nightlies.amd.com/whl-multi-arch \
        --pre \
        torch==${PYTORCH_VERSION} \
        amd-torch-device-gfx950==${PYTORCH_VERSION} \
    && rm -rf /root/.cache

RUN rocm-sdk init
ENV ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
ENV ROCM_HOME=$ROCM_PATH
ENV HIP_PLATFORM=amd
ENV HIP_PATH=$ROCM_PATH
ENV HIP_CLANG_PATH=$ROCM_PATH/llvm/bin
ENV HIP_INCLUDE_PATH=$ROCM_PATH/include
ENV HIP_LIB_PATH=$ROCM_PATH/lib
ENV HIP_DEVICE_LIB_PATH=$ROCM_PATH/lib/llvm/amdgcn/bitcode
ENV PATH="$ROCM_PATH/bin:$HIP_CLANG_PATH:$PATH"
ENV LD_LIBRARY_PATH="$HIP_LIB_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:$ROCM_PATH/lib/host-math/lib:$ROCM_PATH/lib/rocm_sysdeps/lib"
ENV LIBRARY_PATH="$HIP_LIB_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64"
ENV CPATH=$HIP_INCLUDE_PATH
ENV PKG_CONFIG_PATH="$ROCM_PATH/lib/pkgconfig"

# transformers 5.x is required for GLM-5's glm_moe_dsa architecture.
RUN pip install transformers==5.5.0 einops sentencepiece \
    && rm -rf /root/.cache

# Report, don't assert: torch cannot see a GPU during docker build.
RUN python -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip)" && \
    python -c "import torch; assert torch.version.hip, 'not a ROCm torch build'" && \
    echo "ROCM_PATH=$ROCM_PATH" && ls "$ROCM_PATH/bin" | head -5

CMD ["/usr/bin/bash"]
