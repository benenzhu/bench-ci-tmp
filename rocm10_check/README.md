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
| Kimi-K2.5 | vLLM | TP8 8k1k c128 | 2239.96 | — | **broken** |
| GLM-5 | ATOM | TP8 8k1k c8 | 344.61 | — | **broken** |

Units are `tok/s/GPU`. Detail per model in `DSR1_RESULT.md`,
`KIMI_RESULT.md`, `GLM5_ATOM_RESULT.md`.

## Conclusion

Only the SGLang path is healthy on ROCm 10. Both failures are toolchain/version
skew inside the images, not model or config problems:

- **Kimi / vLLM** — two separate aiter breakages. (1) ROCm 10's hipCUB dropped
  `hipcub::Traits`, so aiter's sampling kernel fails to JIT-compile and all
  workers die at init; `--logprobs-mode processed_logits` works around it.
  (2) Past that, `fmha_fwd_bf16_opus_fwd` rejects its arguments and every
  request 500s — not version skew (vLLM pins `AITER_BRANCH=v0.1.19` itself and
  that is what shipped), but two pybind registrations of `aiter_tensor_t`: one
  prebuilt in `module_aiter_core.so`, one in the runtime-JIT-built
  `module_fmha_fwd_bf16_opus`. No workaround for (2).
- **GLM-5 / ATOM** — `AssertionError: VllmBackend can only be called once`
  during warmup: Dynamo produces a second graph for the model forward under
  torch 2.12 / py3.14. Needs a fix in ATOM.

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
