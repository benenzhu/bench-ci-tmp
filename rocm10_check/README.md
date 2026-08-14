# ROCm 10.0.0rc2 check (Aug 2026)

Do our three headline models still run on ROCm 10? Same models and configs as
`../NIGHTLY_CHECK.md`, one high-concurrency point each.

Images (all prerelease):

- SGLang `rocm/ufb-private:sglang-v0.5.15.post1-ubuntu24.04-py3.14-prereleases-device-all-rocm10.0.0rc2-e35a33b34a`
- vLLM `rocm/ufb-private:vllm-0.27.0-ubuntu24.04-py3.14-prereleases-device-all-rocm10.0.0rc2-7d3ceb497f`
- ATOM `atom-rocm10:local` — built here on the vLLM image above
  (`atom_on_vllm_rocm10.dockerfile`), since no ROCm 10 ATOM image is published

| Model | Framework | Config | Baseline | ROCm 10 | Verdict |
|---|---|---|---:|---:|---|
| DeepSeek-R1-0528 | SGLang | TP4 8k1k c128 | 4519.73 | **4742.94** | +4.9%, no regression |
| Kimi-K2.5 | vLLM | TP8 8k1k c128 | 2239.96 | — | **broken** at TP8 |
| Kimi-K2.5 | vLLM | **TP4** 8k1k c128 | 3270.65 | **3617.35** | **+10.6%, no regression** |
| GLM-5 | ATOM | TP8 8k1k c8 | 344.61 | — | **broken** |

Units are `tok/s/GPU`. Detail per model in `DSR1_RESULT.md`,
`KIMI_RESULT.md`, `GLM5_ATOM_RESULT.md`.

The Kimi TP4 baseline (3270.65) is a control run measured for this comparison —
same host, same TP, same serve flags, only the image differs
(`vllm/vllm-openai-rocm:v0.22.0`, hip 7.2.53211). It is not the 2239.96 TP8
number, which is not comparable to a TP4 run.

## Conclusion

Where ROCm 10 kernels are reachable they are **faster**, not slower: DeepSeek-R1
+4.9% on SGLang and Kimi +10.6% at TP4 on vLLM. Nothing here is a performance
regression. What breaks are toolchain skew and kernel-eligibility gates:

- **Kimi / vLLM** — two separate aiter breakages. (1) ROCm 10's hipCUB dropped
  `hipcub::Traits`, so aiter's sampling kernel fails to JIT-compile and all
  workers die at init; `--logprobs-mode processed_logits` works around it.
  (2) Past that, `fmha_fwd_bf16_opus_fwd` rejects its arguments and every
  request 500s — not version skew (vLLM pins `AITER_BRANCH=v0.1.19` itself and
  that is what shipped), but two pybind registrations of `aiter_tensor_t`: one
  prebuilt in `module_aiter_core.so`, one in the runtime-JIT-built
  `module_fmha_fwd_bf16_opus`. **(2) only bites at TP8**: the aiter FP8 ASM MLA
  prefill requires `num_heads % 16 == 0` per rank, and Kimi's 64 heads give 8 at
  TP8 (falls back to the broken path) but 16 at TP4 (stays on aiter). At TP4 the
  model serves 384/384 requests on `ROCM_AITER_MLA`, and against a same-host
  ROCm 7 TP4 control it is **10.6% faster** (3270.65 -> 3617.35 tok/s/GPU).
- **GLM-5 / ATOM** — `AssertionError: VllmBackend can only be called once`
  during warmup. Graph-break logs pin it down: `LinearBase.forward`
  (`atom/model_ops/linear.py:866`) branches on `QuantType`, a **pybind11 enum**
  from `aiter.jit.module_aiter_core`, which torch 2.12's Dynamo cannot trace
  ("likely to be a Dynamo bug", per PyTorch's own hint). The break lands inside
  a loop, Dynamo falls back to eager, and the second entry into `VllmBackend`
  trips its single-shot assert. `atom_quanttype_dynamo.patch` here fixes that
  (mirror the enum to a plain int; bind-mounted and `git apply`d at container
  start, no rebuild) and the compile failure does go away — only to expose a
  **second, independent blocker**: an abort inside an aiter cache kernel
  (`norm_weight dtype must match q dtype`) during CUDA graph capture, which also
  happens with bf16 KV. Both are version skew from our improvised image: official
  ATOM ships torch **2.13**/**2.10** and builds aiter from source, while ours has
  torch 2.12 and aiter wheel 0.1.19 because the only ROCm 10 base is a vLLM image.
  Note `VllmBackend` is ATOM's own forked class and no real vLLM code runs here
  (0 `site-packages/vllm` frames) — uninstalling vLLM does not help, and
  `--enforce-eager` does not bypass it (`--level` is the compile switch).

## Building the ATOM ROCm 10 image

`atom_on_vllm_rocm10.dockerfile` layers ATOM onto the vLLM ROCm 10 image, which
already ships torch 2.12.0+rocm10.0.0rc2 and amd-aiter 0.1.19 — so unlike
ATOM's own `atom_release.dockerfile` it does not rebuild aiter/RCCL from source.
`atomesh` (Rust) is skipped; it is only needed for mesh/router features, not for
single-node serving benchmarks.

```bash
git clone https://github.com/ROCm/ATOM.git && cd ATOM
cp /path/to/atom_on_vllm_rocm10.dockerfile docker/
DOCKER_BUILDKIT=1 docker build -f docker/atom_on_vllm_rocm10.dockerfile \
  -t atom-rocm10:local --build-arg CACHEBUST=$(date +%s) .
```

Verified in the resulting image: `atom 0.1.6rc1.dev272+gc406d694a`,
`torch 2.12.0+rocm10.0.0rc2`, `hip 7.15.26301`, `amd-aiter 0.1.19`, and both
`import atom` and `import atom.entrypoints.openai_server` succeed. The image
builds and imports cleanly — the failure above is at runtime.
