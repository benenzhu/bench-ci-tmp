# Build ATOM on top of the vLLM ROCm 10.0.0rc2 image.
#
# Unlike docker/atom_release.dockerfile (which builds aiter/RCCL from source on a
# rocm/pytorch base), this base already ships torch 2.12.0+rocm10.0.0rc2 and
# amd-aiter 0.1.19, so we only add ATOM itself. The base uses /opt/python, not
# the /opt/venv that the upstream ATOM dockerfiles assume.
#
# atomesh (the Rust mesh binary) is intentionally skipped: it is only needed for
# ATOM's mesh/router features, not for single-node serving benchmarks, and
# building it pulls the whole crates.io dependency tree.
#
# Build:
#   DOCKER_BUILDKIT=1 docker build -f docker/atom_on_vllm_rocm10.dockerfile \
#     -t atom-rocm10:local --build-arg CACHEBUST=$(date +%s) .

ARG BASE_IMAGE="rocm/ufb-private:vllm-0.27.0-ubuntu24.04-py3.14-prereleases-device-all-rocm10.0.0rc2-7d3ceb497f"
FROM ${BASE_IMAGE} AS atom_image

ARG ATOM_REPO="https://github.com/ROCm/ATOM.git"
ARG ATOM_COMMIT="HEAD"
ARG PY=python3

ENV PATH="/opt/python/bin:${PATH}"

RUN echo "========== [1/4] Base sanity ==========" && \
    ${PY} -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip)" && \
    ${PY} -m pip show amd-aiter | head -2 || true

RUN echo "========== [2/4] Build deps ==========" && \
    apt-get update && \
    apt-get install -y --no-install-recommends git curl build-essential pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*

ARG CACHEBUST=1
RUN echo "========== [3/4] Install ATOM (no mesh) ==========" && \
    git clone ${ATOM_REPO} /app/ATOM && \
    cd /app/ATOM && \
    git checkout ${ATOM_COMMIT} && \
    git rev-parse HEAD && \
    ${PY} -m pip install -e . && \
    ${PY} -m pip install --no-cache-dir msgpack msgspec quart && \
    ${PY} -m pip show atom | head -3

RUN echo "========== [4/4] Verify ATOM imports ==========" && \
    ${PY} -c "import atom; print('atom OK:', getattr(atom,'__version__','(no __version__)'))" && \
    ${PY} -m pip show atom amd-aiter torch | grep -E "^Name|^Version"

CMD ["/bin/bash"]
