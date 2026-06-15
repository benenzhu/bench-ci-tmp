# DeepSeek-V4-Pro SGLang MI355X Repro Plan

This is the handoff note for the proposed DeepSeek-V4-Pro SGLang replacement for the failed vLLM old-row repro. Do not start the benchmark until the user sets a goal or explicitly asks to run.

## Selected Claim

Use the strict same-config SGLang row with good gain and moderate runtime:

| Model | Precision | Framework | Hardware | Seq | Conc | TP | EP | DP attention | Disagg | Old tok/s/GPU | New tok/s/GPU | Gain |
|---|---|---|---|---|---:|---:|---:|---|---|---:|---:|---:|
| DeepSeek-V4-Pro | fp4 | SGLang | MI355X | 8192/1024 | 8 | 8 | 1 | false | false | 115.801807 | 483.774241 | 4.1776x |

Source: live InferenceX dashboard API `/api/v1/benchmarks/history`, fetched 2026-06-15 UTC.

This is a strict same-config comparison on model, hardware, framework, precision, spec method, TP/EP/DP-attention, GPU counts, ISL/OSL, and concurrency. Metric is `metrics.tput_per_gpu`.

## Rows To Reproduce

### Old Row

- Date in dashboard history: `2026-05-02`
- Run URL: `https://github.com/SemiAnalysisAI/InferenceX/actions/runs/25262278289/attempts/1`
- Head SHA: `f93f91dfb0efbc46e363dbcdf96e983ea96d214d`
- Branch at run time: `main`
- Workflow title: `Run Sweep - dsv4-mi355x-sgl (#1244)`
- Image: `rocm/sgl-dev:rocm720-mi35x-583b1b6-20260501-DSv4`
- Script path at that commit: `benchmarks/single_node/dsv4_fp4_mi355x_sglang.sh`
- Model: `deepseek-ai/DeepSeek-V4-Pro`
- Config: `ISL=8192`, `OSL=1024`, `CONC=8`, `TP=8`, `EP_SIZE=1`, `DP_ATTENTION=false`, `spec_method=none`, `disagg=false`
- Dashboard throughput: `115.801807 tok/s/GPU`

Important old-script behavior: this script patches the Hugging Face cached `config.json` in place from `model_type=deepseek_v4` to `model_type=deepseek_v3`. If using persistent `/scratch/hf_hub_cache`, back up and restore the config file/blob around the old run, or run with an isolated temporary HF cache. Do not let this silently mutate the shared model cache for later runs.

### New Row

- Date in dashboard history: `2026-06-15`
- Run URL: `https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27475314602/attempts/3`
- Head SHA: `8d206016537a519824e527f7524ca52d17303512`
- Branch at run time: `dsv4-mi355-sgl-0612`
- Workflow title: `Run Sweep - [AMD][MI35X] 0612 DSV4`
- Image: `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612`
- Script path at that commit: `benchmarks/single_node/fixed_seq_len/dsv4_fp4_mi355x_sglang.sh`
- Model: `deepseek-ai/DeepSeek-V4-Pro`
- Config: `ISL=8192`, `OSL=1024`, `CONC=8`, `TP=8`, `EP_SIZE=1`, `DP_ATTENTION=false`, `spec_method=none`, `disagg=false`
- Dashboard throughput: `483.774241 tok/s/GPU`

## Image Availability

Checked with `docker manifest inspect` on 2026-06-15:

| Image | Availability |
|---|---|
| `rocm/sgl-dev:rocm720-mi35x-583b1b6-20260501-DSv4` | OK |
| `rocm/sgl-dev:rocm720-mi35x-a8410de-20260502-DSv4` | OK |
| `rocm/sgl-dev:rocm720-mi35x-b19052c-20260518-DSv4` | OK |
| `rocm/sgl-dev:rocm720-mi35x-8c3b5aa-20260521-DSv4` | OK |
| `rocm/sgl-dev:rocm720-mi35x-f96ac98-20260526-DSv4` | OK |
| `lmsysorg/sglang-rocm:v0.5.12.post1-rocm720-mi35x-20260610` | OK |
| `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` | OK |

This is materially better than the failed vLLM old-row repro, where the original old workflow image was unavailable.

## Recommended Local Repro Setup

Recommended script path:

```text
/root/repro_dsv4_sglang_8k1k_conc8_tp8.sh
```

Recommended output root:

```text
/scratch/inferencex_runs/dsv4_sglang_8k1k_conc8_prompts3x
```

Recommended worktree root:

```text
/scratch/inferencex_worktrees/dsv4_sglang_8k1k_conc8
```

Use persistent cache:

```text
HF_HUB_CACHE=/scratch/hf_hub_cache
HF_HOME=/scratch/hf_home
```

DeepSeek-V4-Pro cache is already present from the previous run and is about `806G`, so do not try to copy the full model to `/dev/shm` unless there is clearly enough free tmpfs space. Prefer running from `/scratch/hf_hub_cache`.

Use the prior local policy unless the user changes it:

- Patch benchmark prompt count from `CONC * 10` to `CONC * 3`.
- For this selected row, local `num_prompts = 8 * 3 = 24`.
- Keep `RANDOM_RANGE_RATIO=0.8`.
- Use a non-conflicting port such as `18888`.
- Preserve `docker.log`, `server.log`, `gpu_metrics.csv`, raw JSON, aggregate JSON, `env.repro`, `local_patch.diff`, exact command line, and status files.
- Do not run `docker purge`; remove only named benchmark containers created by the script.
- Do not print or persist the HF token in logs, docs, scripts, or command history.

## Repro Risks / Notes

- Old script mutates HF `config.json`; handle this explicitly.
- Old and new scripts live at different paths because benchmark scripts were reorganized into `fixed_seq_len/` later.
- Old row uses `rocm/sgl-dev` DSv4 side-branch image. New row uses `lmsysorg/sglang-rocm` image.
- Local result may differ from dashboard because we plan to use `CONC * 3` prompts for speed instead of the workflow default `CONC * 10`.
- The broader blog claim of `110.5x` is not strict same-config: it compares 2026-04-25 FP8 first-light to a later FP4 high-concurrency/DP-attention point. Do not mix it with this strict 4.18x claim.

## Suggested Goal

```text
Run local MI355X reproduction for DeepSeek-V4-Pro FP4 SGLang non-disaggregated strict same-config row: old run 25262278289 attempt 1 vs new run 27475314602 attempt 3, ISL/OSL 8192/1024, CONC=8, TP=8, EP=1, DP_ATTENTION=false, num_prompts=CONC*3; preserve logs, raw/aggregate JSON, exact commands, local patch notes, and cache/backup status under /scratch; produce old/new tok/s/GPU table and document any cache-mutation or runtime patch needed.
```

## Local Repro Result

Run completed on 2026-06-15 UTC with:

```bash
TARGETS=all /root/repro_dsv4_sglang_8k1k_conc8_tp8.sh
```

Output root:

```text
/scratch/inferencex_runs/dsv4_sglang_8k1k_conc8_prompts3x
```

Both old and new runs completed with `success`.

| Variant | Run attempt | Image | Commit | ISL/OSL | CONC | TP | Prompts | Local tok/s/GPU | Local total tok/s | Local output tok/s/GPU | Mean TTFT ms | Mean TPOT ms | Dashboard tok/s/GPU |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old | 25262278289/1 | `rocm/sgl-dev:rocm720-mi35x-583b1b6-20260501-DSv4` | `f93f91dfb0ef` | 8192/1024 | 8 | 8 | 24 | 104.044135 | 832.353076 | 11.774508 | 6143.49 | 75.4615 | 115.801807 |
| new | 27475314602/3 | `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` | `8d206016537a` | 8192/1024 | 8 | 8 | 24 | 421.982767 | 3375.862134 | 47.755112 | 1310.04 | 18.6352 | 483.774241 |

Measured strict same-config gain:

| Old tok/s/GPU | New tok/s/GPU | Gain | Improvement |
|---:|---:|---:|---:|
| 104.044135 | 421.982767 | 4.0558x | +305.58% |

For reference, the dashboard row gain is `483.774241 / 115.801807 = 4.1776x`.
The local run used `num_prompts = CONC * 3 = 24`, so absolute throughput should not be treated as a byte-for-byte reproduction of the workflow default prompt count.

Artifact highlights:

- Combined CSV: `/scratch/inferencex_runs/dsv4_sglang_8k1k_conc8_prompts3x/combined_results.csv`
- Comparison CSV: `/scratch/inferencex_runs/dsv4_sglang_8k1k_conc8_prompts3x/comparison.csv`
- Old raw and aggregate JSON: `/scratch/inferencex_runs/dsv4_sglang_8k1k_conc8_prompts3x/runs/dsv4_sglang_old/`
- New raw and aggregate JSON: `/scratch/inferencex_runs/dsv4_sglang_8k1k_conc8_prompts3x/runs/dsv4_sglang_new/`
- Logs and GPU metrics: each run directory contains `docker.log`, `server.log`, and `gpu_metrics.csv`.

Cache mutation verification:

- Initial cache state: `model_type=deepseek_v4`, sha256 `5fe4568daee51c208cb8a79538eaeda090ae011ade1dee2c386aa95f569c810e`
- Final cache state: `model_type=deepseek_v4`, sha256 `5fe4568daee51c208cb8a79538eaeda090ae011ade1dee2c386aa95f569c810e`
- The old run's temporary `deepseek_v3` config mutation was restored successfully.
- Strict HF token scan found no persisted token in the runner or output artifacts.
