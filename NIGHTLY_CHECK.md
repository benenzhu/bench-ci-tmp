# Nightly image regression check (Aug 2026)

Do the latest nightly images regress vs the images behind the committed
baselines? Same model, same config, one high-concurrency point per model;
baselines reused from the existing bundles.

| Model | Framework / config | Baseline image | Nightly image | Baseline | Nightly | Delta | Verdict |
|---|---|---|---|---:|---:|---:|---|
| DeepSeek-R1-0528 | SGLang, TP4, 8k1k c128 | `lmsysorg/sglang-rocm:v0.5.13-rocm720` | `lmsysorg/sglang-rocm:v0.5.17-rocm700-mi35x-20260811` | 1182.58 | **4591.87** | **+288.3%** | improved |
| GLM-5 | ATOM, TP8, 8k1k c8 | `rocm/atom:rocm7.2.2...atom0.1.2.post` | `rocm/atom-dev:nightly_202608111555` | 344.61 | 359.59 | **+4.3%** | neutral/better |
| Kimi-K2.5 | vLLM, TP8, 8k1k c128 | `vllm/vllm-openai-rocm:v0.22.0` | `vllm/vllm-openai-rocm:nightly` | 2239.96 | **n/a** | **server fails to start** | **REGRESSION** |

Units are `tok/s/GPU` (per-GPU normalized). Per-model detail in
`dsr1_nightly_check/RESULT.md`, `glm5_nightly_check/RESULT.md`,
`kimi_nightly_check/RESULT.md`.

## Conclusion

- **DeepSeek-R1 improves dramatically** on the SGLang nightly (TPOT 224.0 -> 56.0 ms).
- **GLM-5 is slightly better** on the ATOM nightly, with a much better TTFT (1304.9 -> 807.1 ms).
- **Kimi-K2.5 is broken on the upstream vLLM nightly**: the aiter Gluon MLA decode
  kernel fails to compile (`expected offsets type layout to be BlockedLayout or
  SliceLayout`, `mla_gluon.py:1029`), all 8 workers die, engine never initializes.
  The same config runs fine on the v0.22.0 baseline and on the `rocm/atom-dev`
  repackaged vLLM v0.25.1 (2236.69, -0.1%), so the breakage is specific to the
  newer upstream nightly.

## Notes for reruns

- Do **not** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — it breaks
  `custom_all_reduce`'s `hipIpcGetMemHandle` (aiter issue #4174). This, not GPU
  pinning, caused the IPC init crashes seen while setting these up.
- Pin to idle GPUs with `ROCR_VISIBLE_DEVICES` on this shared box.
- ATOM's bench script hardcodes `ATOM_DISABLE_MMAP=true`; with weights on
  `/dev/shm` (tmpfs), leaving mmap **enabled** loads far faster (64 shards in
  11 s vs several minutes).
- `rocm/atom-dev` also publishes repackaged `sglang-*` / `vllm-*` nightly tags;
  those are **not** the upstream images. Earlier runs against them are kept
  under each bundle's `_atomdev_rebuild_runs/`.
