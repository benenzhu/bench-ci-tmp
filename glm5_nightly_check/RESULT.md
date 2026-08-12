# GLM-5 nightly regression check — 8k1k c8, TP8 (ATOM)

| Image | tok/s/GPU | total tok/s | TTFT | TPOT |
|---|---:|---:|---:|---:|
| baseline `rocm/atom:rocm7.2.2...atom0.1.2.post` | 344.61 | 2756.9 | 1304.9 ms | 23.15 ms |
| nightly `rocm/atom-dev:nightly_202608111555` | 359.59 | 2876.7 | 807.1 ms | 22.68 ms |
| delta | **+4.3%** | +4.3% | -38% | -2.0% |

**No regression** — slightly faster, with a notably better TTFT.
Same model (zai-org/GLM-5-FP8), same TP8 / 8k1k / CONC=8 / 24 prompts,
24/24 completed on both runs. CONC=8 is the highest concurrency present in
the committed GLM baseline bundle, so it is the comparable point.
