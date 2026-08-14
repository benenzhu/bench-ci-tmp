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

## Blocker 2 — aiter prefill FMHA signature mismatch (fatal at request time)

Server reaches `Application startup complete`, then **every** request returns
500 and the run records 0 input / 0 output tokens:

```
TypeError: fmha_fwd_bf16_opus_fwd(): incompatible function arguments.
  The following argument types are supported:
    1. (q: aiter_tensor_t, k: aiter_tensor_t, v: aiter_tensor_t,
        out: aiter_tensor_t, causal: bool, softmax_scale: ...)
```

Call path: `rocm_aiter_mla.py:1023 forward_mha` →
`run_prefill_new_tokens` → `flash_attn_varlen_func` → `FlashAttnVarlenFunc.apply`
→ `_fmha_fwd_bf16_opus_fwd(...)`. vLLM 0.27.0 passes an argument list the
amd-aiter 0.1.19 binding in this image does not accept — a vLLM/aiter version
skew inside the image, independent of blocker 1.

No workaround found that keeps the MLA prefill path on aiter.

## Status

| Config | Outcome |
|---|---|
| default | dies at init (blocker 1) |
| `VLLM_ROCM_AITER_MLA_ASM_PADDING=asm` | dies at init (blocker 1) |
| `+ --logprobs-mode processed_logits` | serves, but all requests 500 (blocker 2) |

`VLLM_ROCM_AITER_MLA_ASM_PADDING=asm` does successfully avoid the *older* Gluon
MLA decode compile bug seen on the upstream nightly (see `NIGHTLY_CHECK.md`);
it is orthogonal to both blockers above.

Artifacts: `kimi_nightly_check/runs/kimi_fp4_8k1k_c128_new/`
(`docker.log`, `server.log`, `local_patch.diff`).
