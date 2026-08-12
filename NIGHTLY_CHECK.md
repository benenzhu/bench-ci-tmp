# Nightly image regression check (Aug 2026)

Do the latest nightly images regress vs the current baselines? Same model, same
config, one high-concurrency point per model.

| Model | Baseline image | Nightly image | Baseline | Nightly | Delta | Verdict |
|---|---|---|---:|---:|---:|---|
| DeepSeek-R1-0528 | `lmsysorg/sglang-rocm:v0.5.13-rocm720` | `lmsysorg/sglang-rocm:v0.5.17-rocm700-mi35x-20260811` | 4519.73 | 4591.87 | **+1.6%** | neutral |
| GLM-5 | `rocm/atom:rocm7.2.2...atom0.1.2.post` | `rocm/atom-dev:nightly_202608111555` | 344.61 | 359.59 | **+4.3%** | neutral/better |
| Kimi-K2.5 | `vllm/vllm-openai-rocm:v0.22.0` | `rocm/atom-dev:vllm-v0.25.1-nightly_20260811` | 2239.96 | 2236.69 | **-0.1%** | neutral |
| Kimi-K2.5 | `vllm/vllm-openai-rocm:v0.22.0` | upstream `vllm/vllm-openai-rocm:nightly` | 2239.96 | **n/a** | **fails to start** | **REGRESSION** |

Units are `tok/s/GPU` (per-GPU normalized). Per-model detail in
`dsr1_nightly_check/RESULT.md`, `glm5_nightly_check/RESULT.md`,
`kimi_nightly_check/RESULT.md`.

## Conclusion

- **DeepSeek-R1 and GLM-5: no regression** — both within noise / slightly better
  (GLM-5's TTFT improves 1304.9 -> 807.1 ms).
- **Kimi-K2.5 is broken on the upstream vLLM nightly**: the aiter Gluon MLA
  decode kernel fails to compile (`expected offsets type layout to be
  BlockedLayout or SliceLayout`, `mla_gluon.py:1029`), all 8 workers die, the
  engine never initializes. The **ATOM-packaged vLLM v0.25.1 works fine**
  (-0.1%), so only the upstream nightly build is affected.

## Notes

- DSR1 baseline is **4519.73**, from a re-run of the baseline image. The
  `dsr1_fp4_tp4_geomean` bundle records 1182.58 for the same config, but that
  run was affected by a machine issue.
- Do **not** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — it breaks
  `custom_all_reduce`'s `hipIpcGetMemHandle` (aiter issue #4174). This, not GPU
  pinning, caused the IPC init crashes seen while setting these up.
- Pin to idle GPUs with `ROCR_VISIBLE_DEVICES` on this shared box.
- ATOM's bench script hardcodes `ATOM_DISABLE_MMAP=true`; with weights on
  `/dev/shm` (tmpfs), leaving mmap **enabled** loads far faster (64 shards in
  11 s vs several minutes).
- `rocm/atom-dev` also publishes repackaged `sglang-*` / `vllm-*` nightly tags;
  those are **not** the upstream images. Runs against them are kept under each
  bundle's `_atomdev_rebuild_runs/`.
