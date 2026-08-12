# DSR1 nightly regression check — 8k1k c128, TP4 (SGLang)

| Image | tok/s/GPU | total tok/s | TTFT | TPOT |
|---|---:|---:|---:|---:|
| baseline `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` | 1182.58 | 4730.3 | 10491.7 ms | 223.98 ms |
| nightly `lmsysorg/sglang-rocm:v0.5.17-rocm700-mi35x-20260811` | **4591.87** | 18367.5 | 4650.7 ms | 56.00 ms |
| delta | **+288.3%** | +288% | -56% | -75% |

**No regression — large improvement.** Same model
(amd/DeepSeek-R1-0528-MXFP4-Preview), same TP4 / 8k1k / CONC=128 / 384 prompts,
384/384 completed on both runs.

`_atomdev_rebuild_runs/` holds an earlier run against the `rocm/atom-dev`
repackaged SGLang (+242.7%); the number above is the upstream `lmsysorg` image.
