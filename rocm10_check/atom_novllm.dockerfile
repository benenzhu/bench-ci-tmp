# atom-rocm10 with vLLM removed.
#
# Rationale: the official ATOM images (rocm/atom-dev:*, rocm/atom:*) ship NO
# vllm at all, while ours does because it is built on a vLLM base image. ATOM's
# server mode does not import vllm (its real-vllm imports live under
# atom/plugin/vllm/, plugin mode only) and the GLM-5 crash traceback contains no
# site-packages/vllm frame — so this is expected to change nothing. Building it
# to confirm that empirically rather than by argument.
FROM atom-rocm10:local
RUN python3 -m pip uninstall -y vllm && \
    python3 -c "import atom; print('atom still imports OK')" && \
    python3 -c "import atom.entrypoints.openai_server; print('server entrypoint OK')"
