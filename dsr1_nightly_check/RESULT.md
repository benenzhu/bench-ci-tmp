# DSR1 nightly regression check — 8k1k c128, TP4 (SGLang)

| Image | tok/s/GPU | total tok/s | TTFT | TPOT |
|---|---:|---:|---:|---:|
| baseline `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` | 4519.73 | — | — | — |
| nightly `lmsysorg/sglang-rocm:v0.5.17-rocm700-mi35x-20260811` | **4591.87** | 18367.5 | 4650.7 ms | 56.00 ms |
| delta | **+1.6%** | | | |

**No regression** (within noise). Same model
(amd/DeepSeek-R1-0528-MXFP4-Preview), same TP4 / 8k1k / CONC=128 / 384 prompts,
384/384 completed.

Baseline note: the `dsr1_fp4_tp4_geomean` bundle records 1182.58 for this
config, but that run hit a machine issue; the baseline image was re-run
afterwards and scored **4519.73**, which is the number used here.

`_atomdev_rebuild_runs/` holds an earlier run against the `rocm/atom-dev`
repackaged SGLang (4052.97); the number above is the upstream `lmsysorg` image.
