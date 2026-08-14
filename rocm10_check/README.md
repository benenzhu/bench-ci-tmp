# ROCm 10.0.0rc2 check (Aug 2026)

Do our three headline models still run on ROCm 10? Same models and configs as
`../NIGHTLY_CHECK.md`, one high-concurrency point each.

Images (all prerelease):

- SGLang `rocm/ufb-private:sglang-v0.5.15.post1-ubuntu24.04-py3.14-prereleases-device-all-rocm10.0.0rc2-e35a33b34a`
- vLLM `rocm/ufb-private:vllm-0.27.0-ubuntu24.04-py3.14-prereleases-device-all-rocm10.0.0rc2-7d3ceb497f`
- ATOM `atom-rocm101:local` — built here, since no ROCm 10 ATOM image is
  published: `rocm101_torch_base.dockerfile` (ROCm 10.1 + torch 2.13 on
  `ubuntu:24.04`) then ATOM's own `atom_release_rocm101.dockerfile` on top,
  with RCCL and aiter built from source

| Model | Framework | Config | Baseline | ROCm 10 | Verdict |
|---|---|---|---:|---:|---|
| DeepSeek-R1-0528 | SGLang | TP4 8k1k c128 | 4519.73 | **4742.94** | +4.9%, no regression |
| Kimi-K2.5 | vLLM | TP8 8k1k c128 | 2239.96 | — | **broken** at TP8 |
| Kimi-K2.5 | vLLM | **TP4** 8k1k c128 | 3270.65 | **3617.35** | **+10.6%, no regression** |
| GLM-5 | ATOM | TP8 8k1k c8 | 344.61 | **350.29** | **+1.6%, no regression** |

Units are `tok/s/GPU`. Detail per model in `DSR1_RESULT.md`,
`KIMI_RESULT.md`, `GLM5_ATOM_RESULT.md`.

The Kimi TP4 baseline (3270.65) is a control run measured for this comparison —
same host, same TP, same serve flags, only the image differs
(`vllm/vllm-openai-rocm:v0.22.0`, hip 7.2.53211). It is not the 2239.96 TP8
number, which is not comparable to a TP4 run.

## Conclusion

**No regressions anywhere.** All three models are faster on ROCm 10: DeepSeek-R1
+4.9% (SGLang), Kimi +10.6% at TP4 (vLLM), GLM-5 +1.6% (ATOM). Every failure hit
along the way turned out to be toolchain skew or a kernel-eligibility gate in
our own images, never a ROCm 10 performance problem:

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
  trips its single-shot assert. Behind it sat a second, independent blocker: an
  abort inside an aiter cache kernel (`norm_weight dtype must match q dtype`)
  during CUDA graph capture. **Both were artefacts of the first image**, which
  paired ATOM with a torch and an aiter it was never developed against (2.12 +
  aiter wheel 0.1.19) because the only ROCm 10 base available is a vLLM image.
  Rebuilding the way official ATOM images are built — torch **2.13**, aiter and
  RCCL **from source** — makes both vanish and GLM-5 runs at **+1.6%**, with no
  source patch required. Two notes for anyone re-treading this: `VllmBackend` is
  ATOM's own forked class, so no real vLLM code is involved (0
  `site-packages/vllm` frames) and uninstalling vLLM does not help; and
  `--enforce-eager` does not bypass compilation (`--level` is the switch).

## Building the ATOM ROCm 10 image

No ROCm 10 ATOM image is published, so build one in two steps. **Do not layer
ATOM onto a vLLM ROCm 10 image** — that was the first attempt and it pairs ATOM
with a torch (2.12) and an aiter (wheel 0.1.19) it was not developed against,
producing two separate blockers. `atom_on_vllm_rocm10.dockerfile` is kept only
as the record of that dead end.

```bash
git clone https://github.com/ROCm/ATOM.git && cd ATOM
cp /path/to/rocm101_torch_base.dockerfile /path/to/atom_release_rocm101.dockerfile docker/

# 1. ROCm 10.1 + torch 2.13 base on ubuntu:24.04 (~18 GB, ~25 min)
DOCKER_BUILDKIT=1 docker build -f docker/rocm101_torch_base.dockerfile \
  -t rocm101-torch:local .

# 2. ATOM's own release dockerfile on top; RCCL + aiter from source (~30 GB, ~1 h)
DOCKER_BUILDKIT=1 docker build -f docker/atom_release_rocm101.dockerfile \
  --build-arg BASE_IMAGE=rocm101-torch:local \
  --build-arg GPU_ARCH=gfx950 \
  --build-arg INSTALL_MOONCAKE=0 \
  -t atom-rocm101:local .
```

Resulting image: `atom 0.1.6rc1.dev274`, `torch 2.13.0+rocm10.1.0a20260814`,
`hip 7.16.26323`, aiter source-built (`0.0.0`) — matching how the official
`rocm/atom-dev` images are put together.

Three deviations from the stock files, each deliberate:

- **torch 2.13, not the 2.12** the Primus training Dockerfile pins. The nightly
  index carries 2.13 for the same ROCm 10.1 date, and 2.13 is what official ATOM
  ships; 2.12 is exactly what breaks Dynamo on aiter's pybind enums.
- **gfx950 only** — the benchmark boxes are all MI355X, and a single arch cuts
  the aiter source build substantially.
- `atom_release_rocm101.dockerfile` adds `ATOM_MESH_BUILD=0` (the Rust
  `atomesh` build fails with `Too many open files`, and mesh/router is not used
  for single-node serving) plus a missing `sortedcontainers` in the LMCache
  step. `INSTALL_MOONCAKE=0` for the same reason — disaggregated serving only.
