# Nightly image regression check (Aug 2026)

Do the latest `rocm/atom-dev` nightly images regress performance vs the images
used for the committed baselines? Same model, same config, single high-concurrency
point (`8k1k`, CONC=128); baselines reused from the existing bundles.

| Model | Framework | Baseline image | Nightly image | Baseline | Nightly | Delta | Verdict |
|---|---|---|---|---:|---:|---:|---|
| DeepSeek-R1-0528 | SGLang, TP4 | `lmsysorg/sglang-rocm:v0.5.13-rocm720` | `rocm/atom-dev:sglang-v0.5.15.post1-nightly_20260811` | 1182.58 | **4052.97** | **+242.7%** | improved |
| Kimi-K2.5 | vLLM, TP8 | `vllm/vllm-openai-rocm:v0.22.0` | `rocm/atom-dev:vllm-v0.25.1-nightly_20260811` | 2236.69 vs 2239.96 | 2236.69 | **-0.1%** | neutral |

Units are `tok/s/GPU` (per-GPU normalized). Details in
`dsr1_nightly_check/RESULT.md` and `kimi_nightly_check/RESULT.md`.

**No regressions found.** DSR1 on the SGLang nightly is dramatically faster
(TPOT 224.0 -> 62.3 ms); Kimi is unchanged within noise.

Notes for reruns:
- Do **not** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — it breaks
  `custom_all_reduce`'s `hipIpcGetMemHandle` (aiter issue #4174).
- Pin to idle GPUs with `ROCR_VISIBLE_DEVICES` on this shared box.
