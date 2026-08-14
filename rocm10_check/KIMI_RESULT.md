# Kimi-K2.5-MXFP4 on ROCm 10.0.0rc2 — 8k1k c128, TP8 (vLLM)

Image: `rocm/ufb-private:vllm-0.27.0-ubuntu24.04-py3.14-prereleases-device-all-rocm10.0.0rc2-7d3ceb497f`
Baseline: 2239.96 tok/s/GPU (`vllm/vllm-openai-rocm:v0.22.0`, ROCm 7.x)

**Result: does not run. No throughput number.** Two independent ROCm 10
incompatibilities in aiter, hit one after the other.

## Blocker 1 — aiter sampler fails to JIT-compile (fatal at startup)

All 8 workers die during engine init:

```
OSError: /root/.aiter/build/top_k_top_p_sampling_from_probs_<hash>/lib.so:
         cannot open shared object file: No such file or directory
```

The `.so` is missing because the JIT build failed. Root cause is in the
generated header, not the loader:

```
include/sampling.cuh:87:29: error: no type named 'Traits' in namespace 'hipcub'
   87 | __device__ typename hipcub::Traits<T>::UnsignedBits twiddle_in(T key, bool select_min)
```

ROCm 10's hipCUB no longer defines `hipcub::Traits` (absent from
`_rocm_sdk_devel/include/hipcub/util_type.hpp`); aiter's
`aiter_meta/csrc/cpp_itfs/sampling/sampling.cuh` still uses it. `make` exits 2,
but aiter logs `finish build ... cost 12.86s` anyway, so the failure only
surfaces later as the confusing "cannot open shared object file".

Reproduce standalone (no benchmark needed):

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --entrypoint bash \
  rocm/ufb-private:vllm-0.27.0-...rocm10.0.0rc2-7d3ceb497f -lc '
python3 -c "
import torch, aiter.ops.sampling
p   = torch.softmax(torch.randn(4, 1000, device=\"cuda\"), -1)
idx = torch.arange(4, device=\"cuda\", dtype=torch.int32)
k   = torch.full((4,), 50, device=\"cuda\", dtype=torch.int32)
tp  = torch.full((4,), 0.9, device=\"cuda\")
torch.ops.aiter.top_k_top_p_sampling_from_probs(p, idx, k, 0, tp, 0.0, False)
"'
```

Note the red herring in the log: `-mllvm -amdgpu-coerce-illegal-types=1 is not
supported by hipcc`. That flag is correctly detected and dropped by aiter's
`hip_flag_checker`; it is **not** the failure.

### Workaround

`--logprobs-mode processed_logits` routes sampling through `forward_native`
and skips the aiter sampler entirely (see
`vllm/v1/sample/ops/topk_topp_sampler.py:117`, gated on
`logprobs_mode not in PROCESSED_LOGPROBS_MODES`). Every other aiter kernel
stays enabled. Wired into `scripts/repro_kimi_nightly.sh` as `LOGPROBS_MODE=`.

With that, the server starts and serves — which exposes the next blocker.

## Blocker 2 — aiter `aiter_tensor_t` pybind type identity clash (fatal at request time)

Server reaches `Application startup complete`, then **every** request returns
500 and the run records 0 input / 0 output tokens:

```
TypeError: fmha_fwd_bf16_opus_fwd(): incompatible function arguments.
  The following argument types are supported:
    1. (q: aiter_tensor_t, k: aiter_tensor_t, v: aiter_tensor_t,
        out: aiter_tensor_t, causal: bool, softmax_scale: ..., seqstart_q: ...,
        seqstart_k: ..., seqstart_q_pad: ..., seqstart_k_pad: ...,
        max_seqlen_q: ..., max_seqlen_k: ...)

  Invoked with: <aiter.jit.module_aiter_core.aiter_tensor_t object>, ... x4,
                True, 0.14467962580268923, <...aiter_tensor_t> x4, 7237, 7237
```

This is **not** an arity or a vLLM/aiter version problem — expected and invoked
are both 12 arguments in the same order and the versions match (see below). The
mismatch is the *type identity* of `aiter_tensor_t`:

- `aiter_tensor_t` is registered by exactly one prebuilt module,
  `aiter/jit/module_aiter_core.so` (the only `.so` in `aiter/jit/` whose
  strings contain the class name), and `torch_to_aiter_pybind()`
  (`aiter/utility/dtypes.py:59`) constructs instances from *that* module.
- `module_fmha_fwd_bf16_opus` is **JIT-built at runtime** in this image
  (`finish build [module_fmha_fwd_bf16_opus], cost 17.8s`) and, via
  `FMHA_FWD_BF16_OPUS_PYBIND` in `aiter_meta/csrc/include/rocm_ops.hpp:1452`,
  registers its own `aiter_tensor_t`.

pybind11 keys type conversion on the registered C++ type in the local internals
namespace, so the freshly JIT-built module does not accept the
`module_aiter_core` instances the `develop=True` path feeds it
(`aiter/jit/core.py:1758`). Prebuilt-vs-JIT split of the same pybind class.

Call path: `rocm_aiter_mla.py:1023 forward_mha` → `run_prefill_new_tokens` →
`flash_attn_varlen_func` → `FlashAttnVarlenFunc.apply` →
`_fmha_fwd_bf16_opus_fwd(...)` (`aiter/ops/mha.py:344`, declared
`@compile_ops(..., develop=True)`).

Why it JIT-builds here at all is the thing to check upstream: on a working
image `module_fmha_fwd_bf16_opus` should be prebuilt into the wheel alongside
`module_aiter_core`, and then both share one type registry.

### Workaround: run TP4 — it stays on aiter

The gate that decides this is `rocm_aiter_mla.py:333`:

```python
# FP8 MLA prefill (kn_mla_reduce_v1) only supports 16-aligned heads.
self._fp8_prefill_enabled = (
    _fp8_mla_prefill_supported() and self.num_heads % 16 == 0
)
```

`num_heads` here is **per rank** (`ModelConfig.get_num_attention_heads()`
divides by TP). Kimi has 64 attention heads (`text_config`), so:

| TP | heads/rank | `% 16` | prefill path |
|---:|---:|---|---|
| 8 | 8 | ✗ | falls back to `flash_attn_varlen_func` → blocker 2 |
| 4 | **16** | ✓ | **stays on the aiter FP8 ASM prefill** |

The other two conditions in `_fp8_mla_prefill_supported()` already hold on this
image: `on_gfx950()` is True and aiter 0.1.19 exports both
`mla_prefill_ps_asm_fwd` and `mla_reduce_v1`.

Verified at TP4 — see the result section below. `fmha_fwd_bf16_opus_fwd`
appears **0 times** in the server log and the backend selected is
`ROCM_AITER_MLA`. So blocker 2 is not a dead end; TP8 simply excludes itself
from the ASM path.

## TP4 result — Kimi runs on ROCm 10

Run on `smci355-ccs-aus-n02-09` (8 idle GPUs), same image, 8192x1024 / CONC=128
/ 384 prompts:

| Metric | Value |
|---|---:|
| Successful / failed requests | **384 / 0** |
| Total token throughput | 14469.40 tok/s |
| **tok/s/GPU** | **3617.35** |
| Output throughput | 1607.71 tok/s |
| Mean TTFT / TPOT | 7517.6 ms / 71.74 ms |
| Attention backend | `ROCM_AITER_MLA` |
| `fmha_fwd_bf16_opus_fwd` errors | 0 |

**Do not compare 3617.35 against the 2239.96 TP8 baseline.** Three things
differ: TP4 vs TP8, the aiter sampler is disabled, and this is a different
host. `tok/s/GPU` normalises for GPU count but not for any of those. The
like-for-like comparison is the ROCm 7 TP4 control below.

## ROCm 7 vs ROCm 10 at TP4 — the like-for-like comparison

Control run on the **same host, same TP, same serve flags, same 8192x1024 /
CONC=128 / 384-prompt workload**, only the image differs. `--logprobs-mode
processed_logits` is kept on both ends even though ROCm 7 does not need it, so
the two configurations stay identical.

| | ROCm 7.2 (`vllm/vllm-openai-rocm:v0.22.0`) | ROCm 10.0.0rc2 | delta |
|---|---:|---:|---:|
| **tok/s/GPU** | **3270.65** | **3617.35** | **+10.6%** |
| Total token throughput | 13082.59 | 14469.40 | +10.6% |
| Output throughput | 1453.62 | 1607.71 | +10.6% |
| Mean TPOT | 78.51 ms | 71.74 ms | −8.6% |
| Mean TTFT | 9240.5 ms | 7517.6 ms | −18.6% |
| Benchmark duration | 270.51 s | 244.58 s | −9.6% |
| Successful / failed | 384 / 0 | 384 / 0 | — |

Baseline image versions: vLLM 0.22.0, torch 2.10.0+git8514f05, **hip 7.2.53211**,
amd-aiter 0.1.13.

**Verdict: no regression — ROCm 10 is ~10.6% faster at TP4**, with the gain
showing up consistently across throughput, TPOT and TTFT rather than in a single
metric. This mirrors DeepSeek-R1's +4.9% on the SGLang ROCm 10 image, so the
ROCm 10 kernels are healthy where they are reachable; Kimi's TP8 failures are
eligibility/toolchain gates, not performance problems.

Two config details this run pinned down, both of which are harness bugs rather
than ROCm 10 problems:

- `VLLM_ROCM_USE_AITER=1` is **required**. Without it `_get_backend_priorities()`
  never offers `ROCM_AITER_MLA` at all, and since `--block-size=1` is also in
  the recipe (and `TRITON_MLA` rejects `block_size=1`), the selector finds zero
  valid backends and the engine aborts with `No valid attention backend found`.
  The bundle's bench script exports it at line 49; a hand-rolled serve command
  must too.
- `--random-range-ratio 0.8` with `--max-model-len 9416` is invalid for
  `vllm bench serve`: it samples input length from
  `[len*(1-r), len*(1+r)]` = up to 14746 tokens, and the server 400s everything
  over 9416. A first attempt scored 181/384 successful for exactly this reason.
  Pinned to `0` (fixed 8192x1024) for the numbers above.

## Status

| Config | Outcome |
|---|---|
| TP8, default | dies at init (blocker 1) |
| TP8 `+ --logprobs-mode processed_logits` | serves, but all requests 500 (blocker 2) |
| **TP4 `+ --logprobs-mode processed_logits`** | **works — 384/384, on `ROCM_AITER_MLA`** |

Note on `VLLM_ROCM_AITER_MLA_ASM_PADDING`: that env var **does not exist in
this image**. It is present in the upstream nightly (vLLM 0.26.1, where it
works around the Gluon MLA decode compile bug — see `NIGHTLY_CHECK.md`) but
vLLM 0.27.1 rewrote that code path and dropped it, so setting it here is
silently ignored. An earlier version of this file listed it as a distinct
tested configuration; it was not.

## Versions in the image

From `pip show` and `/app/versions.txt` inside the container:

| Component | Version |
|---|---|
| vLLM | `0.27.1.dev1+g7d3ceb497.d20260812.rocm100` (commit `7d3ceb497`, 2026-08-12) |
| amd-aiter | `0.1.19` — `AITER_BRANCH: v0.1.19`, `AITER_REPO: https://github.com/ROCm/aiter.git` |
| torch | `2.12.0+rocm10.0.0rc2` (hip 7.15.26301) |
| flash-attention | `FA_BRANCH: 0e60e394` |
| base | `rocm/ufb-private:vllm-base-...rocm10.0.0rc2-7d3ceb497f` |

vLLM source is at `/app/vllm` but **with `.git` stripped**, so there is no repo
to diff against; the commit is recoverable only from the version string and the
image tag (both say `7d3ceb497`). The aiter pin is not guesswork — vLLM's own
`docker/Dockerfile.rocm_base:4` carries `ARG AITER_BRANCH="v0.1.19"`, and that
is what got built, so **vLLM and aiter are the pair vLLM intends**. That is why
blocker 2 is an aiter-internal problem, not version skew.

Artifacts: `logs/kimi/` here (`docker.log.gz`, `server.log.gz`,
`local_patch.diff`, patched bench script).
