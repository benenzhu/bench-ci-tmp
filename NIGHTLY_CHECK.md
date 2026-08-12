# Nightly image regression check (Aug 2026)

Do the latest `rocm/atom-dev` nightly images regress performance vs the images
behind the committed baselines? Same model, same config, one high-concurrency
point per model; baselines reused from the existing bundles.

| Model | Framework | Baseline image | Nightly image | Baseline | Nightly | Delta | Verdict |
|---|---|---|---|---:|---:|---:|---|
| DeepSeek-R1-0528 | SGLang, TP4, 8k1k c128 | `lmsysorg/sglang-rocm:v0.5.13-rocm720` | `rocm/atom-dev:sglang-v0.5.15.post1-nightly_20260811` | 1182.58 | **4052.97** | **+242.7%** | improved |
| Kimi-K2.5 | vLLM, TP8, 8k1k c128 | `vllm/vllm-openai-rocm:v0.22.0` | `rocm/atom-dev:vllm-v0.25.1-nightly_20260811` | 2239.96 | 2236.69 | **-0.1%** | neutral |
| GLM-5 | ATOM, TP8, 8k1k c8 | `rocm/atom:rocm7.2.2...atom0.1.2.post` | `rocm/atom-dev:nightly_202608111555` | 344.61 | 359.59 | **+4.3%** | neutral/better |

Units are `tok/s/GPU` (per-GPU normalized). Per-model detail in
`dsr1_nightly_check/RESULT.md`, `kimi_nightly_check/RESULT.md`,
`glm5_nightly_check/RESULT.md`.

## Conclusion

**No regressions on any of the three models.** DeepSeek-R1 on the SGLang
nightly is dramatically faster (TPOT 224.0 -> 62.3 ms); Kimi is unchanged
within noise; GLM-5 is slightly faster with a much better TTFT
(1304.9 -> 807.1 ms).

## Notes for reruns

- Do **not** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — it breaks
  `custom_all_reduce`'s `hipIpcGetMemHandle` (aiter issue #4174). This, not GPU
  pinning, was the cause of the IPC init crashes seen while setting these up.
- Pin to idle GPUs with `ROCR_VISIBLE_DEVICES` on this shared box.
- ATOM's bench script hardcodes `ATOM_DISABLE_MMAP=true`; with weights on
  `/dev/shm` (tmpfs) leaving mmap **enabled** loads far faster (64 shards in
  11 s vs several minutes).
