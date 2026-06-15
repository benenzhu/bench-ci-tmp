# Kimi-K2.5 Middle-State Candidates

Source: live InferenceX dashboard API `/api/v1/benchmarks/history`, fetched 2026-06-15 UTC. Filter: `hardware=mi355x`, `framework=vllm`, non-disaggregated, single-node, `spec_method=none`, equal prefill/decode TP/EP/GPU counts, no DP attention. Metric is `metrics.tput_per_gpu`.

## Conclusion

- Erratum after local reproduction on 2026-06-15: the previously recommended
  Kimi INT4 `8192/1024`, `CONC=16`, `TP=8` row did **not** reproduce locally.
  The saved local attempt is in `../../../kimi_int4_8k1k_conc16_prompts3x/`
  and measured `142.174 -> 187.669 tok/s/GPU`, `1.320x`, not the dashboard
  candidate-table value `157.195 -> 631.026 tok/s/GPU`, `4.014x`.
- The existing local Kimi FP4 result (`CONC=4`, `8192/1024`) should be preserved: it is a valid high-gain row, but it is very large at `14.65x` locally and `13.05x` on the dashboard.
- Strict Kimi `fp4` does **not** have a same-config 3.5x-4.5x pair in the available MI355X history. The closest FP4 middle-state candidate is `8192/1024`, `CONC=64`, `TP=8`, `8 GPUs`, `2026-03-01 -> 2026-03-26`, at `4.726x`, slightly above the requested band.
- If `int4` Kimi is acceptable, the cleaner same-workflow 3.5x-4.5x candidate is now `1024/8192`, `CONC=16`, `TP=8`, `8 GPUs`, `22.890 -> 95.645 tok/s/GPU`, `4.178x`. This row has not been run locally yet.

## Recommended 3.5x-4.5x Candidates

| Precision | Seq | CONC | TP | GPUs | Dates | tok/s/GPU | Gain | Runs |
|---|---|---:|---:|---:|---|---:|---:|---|
| int4 | 1024/8192 | 16 | 8 | 8 | 2026-02-18 -> 2026-03-27 | 22.890 -> 95.645 | 4.178x (+317.8%) | `22129277189/attempts/3` -> `23669977901/attempts/6` |
| int4 | 1024/8192 | 16 | 8 | 8 | 2026-02-18 -> 2026-03-27 | 22.890 -> 95.377 | 4.167x (+316.7%) | `22129277189/attempts/3` -> `23626527425/attempts/1` |
| int4 | 8192/1024 | 8 | 8 | 8 | 2026-02-18 -> 2026-03-27 | 104.084 -> 396.787 | 3.812x (+281.2%) | `22129277189/attempts/3` -> `23669977901/attempts/6` |
| int4 | 8192/1024 | 64 | 8 | 8 | 2026-02-18 -> 2026-03-27 | 328.504 -> 1228.645 | 3.740x (+274.0%) | `22129277189/attempts/3` -> `23669977901/attempts/6` |

## Rejected After Local Repro

| Precision | Seq | CONC | TP | GPUs | Dashboard tok/s/GPU | Dashboard Gain | Local Result |
|---|---|---:|---:|---:|---:|---:|---|
| int4 | 8192/1024 | 16 | 8 | 8 | 157.195 -> 631.026 | 4.014x | 142.174 -> 187.669 tok/s/GPU, 1.320x |

## FP4 Nearest Candidates

| Precision | Seq | CONC | TP | GPUs | Dates | tok/s/GPU | Gain | Runs |
|---|---|---:|---:|---:|---|---:|---:|---|
| fp4 | 8192/1024 | 64 | 8 | 8 | 2026-03-01 -> 2026-03-26 | 348.532 -> 1647.260 | 4.726x (+372.6%) | `22555187591/attempts/2` -> `23614589869/attempts/1` |
| fp4 | 8192/1024 | 64 | 8 | 8 | 2026-03-01 -> 2026-03-27 | 348.532 -> 1681.306 | 4.824x (+382.4%) | `22555187591/attempts/2` -> `23669977901/attempts/6` |
| fp4 | 8192/1024 | 64 | 8 | 8 | 2026-03-01 -> 2026-05-31 | 348.532 -> 1903.487 | 5.461x (+446.1%) | `22555187591/attempts/2` -> `26696253037/attempts/1` |
| fp4 | 8192/1024 | 64 | 8 | 8 | 2026-03-01 -> 2026-05-17 | 348.532 -> 1949.167 | 5.593x (+459.3%) | `22555187591/attempts/2` -> `25956503208/attempts/1` |

## Full Candidate CSV

See `kimi_middle_state_candidates.csv`. It includes all strict 3.5x-4.5x pairs plus the nearest FP4 out-of-band pairs from the original dashboard scan. The markdown conclusion above supersedes the CSV ranking for the rejected `8192/1024`, `CONC=16` Kimi INT4 row.
