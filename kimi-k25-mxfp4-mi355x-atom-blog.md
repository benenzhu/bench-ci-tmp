# Serving Kimi K2.5 in MXFP4 on AMD Instinct™ MI355X GPUs with ATOM

**Authors:** [FILL: author list — e.g. benenzhu, RadeonFlow Team, AMD contributors]
**Date:** [FILL: publication date]
**Tags:** AI/ML, LLM, Inference, Optimization, Performance

---

## Introduction

Trillion-parameter Mixture-of-Experts (MoE) models are no longer research curiosities — they are production workloads. Moonshot AI's Kimi K2.5 packs 1T total parameters with roughly 32B activated per token, routed across 384 experts (8 routed plus 1 shared per token) on top of Multi-head Latent Attention (MLA). Serving a model of this shape efficiently stresses every layer of the inference stack at once: low-precision GEMM throughput in the MoE experts, expert-routing overhead, inter-GPU communication, KV-cache capacity, and the scheduler that decides when prefill is allowed to interrupt decode.

In a [previous post](https://rocm.blogs.amd.com/software-tools-optimization/atom-optimiztion/README.html) we showed how ATOM, AMD's open inference engine, uses DP Attention scheduling and Two-Batch Overlap to deliver strong DeepSeek-V4 performance on AMD Instinct™ MI355X GPUs. In this post we turn to Kimi K2.5 — one of the most widely deployed open-weight models today, and the same backbone that powers Cursor's Composer 2 coding model [FILL: link/citation for the Composer 2 claim]. That popularity makes its serving efficiency a question with immediate practical stakes for anyone running it at scale. As with DeepSeek-V4, we serve it end-to-end in **MXFP4**: the MI355X's CDNA™ 4 architecture natively supports the OCP microscaling MXFP4 data type, and with the latest ATOM release and AITER (AI Tensor Engine for ROCm) kernels, the entire MoE path — activations and weights — runs in 4-bit.

All results below are produced with [InferenceX](https://github.com/SemiAnalysisAI/InferenceX), SemiAnalysis's open, continuously-run inference benchmark, and are reproducible from the public configuration merged in [InferenceX PR #2132](https://github.com/SemiAnalysisAI/InferenceX/pull/2132). Every submission must pass the InferenceX compliance checklist — no model-architecture changes, no engine patching, no evaluation shortcuts — so what is benchmarked is what you can `docker pull` today.

## At a Glance

- Kimi K2.5 (1T-parameter MoE) served in MXFP4 on a **single MI355X node at TP=4** — the 4-bit weights (~[FILL: ~5xx] GB) fit comfortably in 4 × 288 GB of HBM3E, leaving ample room for KV cache; one 8-GPU node hosts two independent TP4 replicas.
- Peak throughput of **[FILL: X,XXX] tok/s/GPU** on the 8k/1k workload and **[FILL: X,XXX] tok/s/GPU** on 1k/1k, versus the published single-node NVIDIA B200 (vLLM, NVFP4) figure of ~4,021 tok/s/GPU on 8k/1k — [FILL: ratio / positioning statement].
- A fully 4-bit MoE path: AITER MXFP4 (A4W4) fused MoE kernels for gfx950 with MXFP4 intermediate activations.
- Everything ships in the public `rocm/atom` Docker image — no private branches, no patches.

## Background: Kimi K2.5 Meets MI355X

Kimi K2.5 inherits the DeepSeek-style recipe — MLA attention plus a wide, sparsely-activated FFN — but at 1T scale with 384 experts. Two properties dominate its serving profile:

1. **It is memory-bound at the weights.** Even though only ~32B parameters activate per token, the full 1T parameters must be resident. At 4 bits per weight this is roughly [FILL: ~5xx] GB plus scales and metadata. On MI355X (288 GB HBM3E, 8 TB/s per GPU), TP=4 holds the model with headroom for large KV caches; competing 192 GB-class GPUs need TP=8 for the same model, spending twice the GPUs per replica.
2. **It is communication-sensitive.** With tensor-parallel MoE, every layer ends in an all-reduce. At interactive batch sizes these collectives sit on the critical path of each decode step, so shrinking or hiding them translates directly into tok/s/user.

ATOM ([ROCm/atom](https://github.com/ROCm/atom)) is AMD's lightweight, PyTorch-native LLM inference engine, and AITER ([ROCm/aiter](https://github.com/ROCm/aiter)) supplies its hand-tuned kernels.

## Key Optimizations

The gains come from four threads of work, each visible as an upstream PR. We describe them in the order a token experiences them.

### 1. MXFP4 (A4W4) fused MoE kernels for gfx950

The heart of Kimi K2.5 inference is the grouped GEMM inside the MoE block. Earlier MXFP4 support quantized weights only (W4A8/W4A16): activations were upcast, halving the arithmetic benefit of the format. The new AITER backends ([aiter #3470](https://github.com/ROCm/aiter/pull/3470), with a RadeonFlow-contributed variant in [aiter #3832](https://github.com/ROCm/aiter/pull/3832)) execute the full **A4W4** path on gfx950: both activations and weights enter the matrix units as MXFP4, using CDNA 4's native microscaling instructions with per-32-element E8M0 scales.

Two backends are provided because MoE GEMMs live in two regimes. At decode-time batch sizes the kernel is launched over many small expert problems and is bound by scheduling and memory; at prefill it approaches dense GEMM behavior. The dispatcher picks per-shape, so both ends of the Pareto curve benefit.

Setting `AITER_MXFP4_INTERMEDIATE=1` extends 4-bit to the **intermediate activations between the up- and down-projection** of each expert. The gate/up output is quantized to MXFP4 on the fly (fused into the activation kernel, so no extra pass over memory) before the down-projection consumes it. This cuts the intermediate tensor traffic — the largest activation in the model, `topk × tokens × 2 × moe_intermediate_dim` — by [FILL: 2×/4× vs previous format], which matters most at high concurrency where prefill and decode share bandwidth.

### 2. Single-stage fused all-reduce at production batch sizes

With TP=4, every transformer layer ends in an all-reduce over the residual-stream tensor, and at interactive batch sizes these collectives sit on the critical path of every decode step. AITER's **Quick Reduce**, the fused all-reduce ATOM uses over XGMI links, has two internal strategies: a **one-stage** algorithm (each rank pulls and reduces peers' buffers directly — lowest latency, more redundant traffic) and a **two-stage** reduce-scatter + all-gather (less traffic, more synchronization). The crossover had been tuned for small messages, so the batch sizes that matter for serving — concurrency 32–64 — silently fell onto the slower two-stage path. [aiter #3458](https://github.com/ROCm/aiter/pull/3458) raises the one-stage limit to cover concurrency 64, and [aiter #3880](https://github.com/ROCm/aiter/pull/3880) fixes the gating condition so the decision is made on actual message bytes ([FILL: exact condition]). This is a two-line class of fix that is invisible in kernel microbenchmarks and worth [FILL: X%] in end-to-end throughput — precisely the kind of regression a continuous benchmark like InferenceX exists to catch.

### 3. Faster expert routing: the TopK kernel

Routing every token to 8 of 384 experts sounds negligible next to the GEMMs, but at decode batch sizes the router runs once per layer per step, and Kimi K2.5 has [FILL: N] MoE layers. The original top-k selection launched [FILL: multiple kernels / used a generic sort]; [aiter #3466](https://github.com/ROCm/aiter/pull/3466) replaces it with a fused single-kernel reduction specialized for K2.5's expert count and group-routing scheme, cutting router time by [FILL: X×]. Together with the MoE-metadata parallelization work for MLA models that landed alongside it, the non-GEMM overhead per decode step dropped from [FILL: X µs to Y µs].

### 4. Scheduling: `--scheduler-delay-factor 1`

ATOM, like vLLM, must decide when to admit new prefills into a busy decode loop. Admitting them greedily maximizes time-to-first-token at low load but causes **decode stalls** at high concurrency: a long prefill lands in the middle of a decode stream and every in-flight user sees a latency spike; worse, decode batches fragment and per-step efficiency drops.

`--scheduler-delay-factor 1` delays prefill admission by one previous-prompt-latency unit, letting the scheduler accumulate prefills and co-schedule them instead of dribbling them into the decode stream. On the fixed-sequence-length InferenceX workloads this smooths the concurrency-64/128 operating points — [FILL: X% throughput at conc 128, X% p99 TPOT improvement] — at negligible TTFT cost.

## Performance Results

We benchmark with InferenceX's single-node fixed-sequence-length harness on two standard workloads — 1k/1k (ISL 1024 / OSL 1024) and 8k/1k (ISL 8192 / OSL 1024) — sweeping concurrency from 4 to 128 and plotting the resulting Pareto frontier of throughput (tok/s/GPU) versus interactivity (tok/s/user).

**Figure 1: [FILL: Pareto frontier, 8k/1k — MI355X ATOM MXFP4 (TP4) vs. NVIDIA B200 vLLM NVFP4 (single node)]**

**Figure 2: [FILL: Pareto frontier, 1k/1k — same systems]**

On the 8k/1k workload, MI355X with ATOM reaches **[FILL: X,XXX] tok/s/GPU** at concurrency 128 while sustaining **[FILL: XX] tok/s/user**, and **[FILL: XX] tok/s/user** at concurrency 4 on the interactive end. For reference, the published single-node B200 result on this workload (vLLM, NVFP4, TP[FILL]) is **~4,021 tok/s/GPU** ([InferenceX](https://inferencex.semianalysis.com/)); the best previously published MI355X number on the vLLM path was 2,687 tok/s/GPU. [FILL: one-sentence positioning — e.g. "The ATOM MXFP4 stack closes/reverses this gap, reaching XX% of / exceeding B200 single-node throughput while offering YY% more HBM per GPU."]

**Figure 3: [FILL: optimization waterfall — baseline ATOM 0.1.3 → +A4W4 MoE → +MXFP4 intermediate → +AR 1-stage fix → +TopK → +scheduler, at conc 64, 8k/1k]**

Beyond the peak number, the shape of the frontier is the story: the collective-strategy work (optimization #2) lifts the low-concurrency, interactivity-critical end, while the A4W4 MoE kernels and scheduler change lift the high-concurrency end. [FILL: 1–2 sentences on where the biggest deltas landed vs. the previous ATOM image, e.g. "Relative to the previous atom0.1.3 image, the new stack is X% faster at conc 4 and Y% faster at conc 128."]

Note also what the comparison does *not* include: rack-scale disaggregated serving (e.g. wide-EP on NVL72-class systems) is a different deployment class with its own InferenceX category; here we compare single-node, buy-it-today configurations.

## How to Reproduce

Everything runs from the public image and the open InferenceX harness.

```bash
# 1. Pull the ATOM release image
docker pull rocm/atom:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom0.1.4_202607091539

# 2. Get the benchmark harness
git clone https://github.com/SemiAnalysisAI/InferenceX.git
cd InferenceX

# 3. Run the Kimi K2.5 MXFP4 benchmark (TP4, concurrency sweep 4–128)
bash benchmarks/single_node/fixed_seq_len/kimik2.5_fp4_mi355x_atom.sh
```

The key server-side settings, as merged in [PR #2132](https://github.com/SemiAnalysisAI/InferenceX/pull/2132):

```bash
export ATOM_DISABLE_MMAP=true              # faster model load
export AITER_QUICK_REDUCE_QUANTIZATION=INT4  # INT4-compressed TP all-reduce
export AITER_MXFP4_INTERMEDIATE=1          # MXFP4 intermediate MoE activations

atom serve amd/Kimi-K2.5-MXFP4 \
  --tensor-parallel-size 4 \
  --scheduler-delay-factor 1 \
  [FILL: remaining serve flags from the script]
```

Both the 1k/1k and 8k/1k workloads run at concurrencies {4, …, 128}; results land in the standard InferenceX output format and are directly comparable with the public dashboard at [inferencex.semianalysis.com](https://inferencex.semianalysis.com/).

## From a Speedrun to Upstream: Community-Built Performance

Part of this work has an unusual origin story: a public competition. Earlier this year, AMD and GPU MODE ran the [E2E Model Speedrun](https://luma.com/cqq4mojz) — a $1.1M challenge on MI355X GPUs whose qualifier round targeted exactly the kernels that dominate this workload (MXFP4 MoE, MLA decode, MXFP4 GEMM), and whose finals asked for end-to-end inference optimization, with Kimi K2.5 1T FP4 as one of the two tracks.

The RadeonFlow team competed in the Speedrun, and after the competition contributed their work upstream: their MXFP4 (A4W4) MoE backend ([aiter #3832](https://github.com/ROCm/aiter/pull/3832)) now ships in the same public `rocm/atom` image benchmarked in this post, side by side with AMD's own kernels, picked by the dispatcher whenever it wins on a given shape. Competition code becoming production code within weeks is exactly the loop this stack is designed for: the kernels live in open repositories, the serving engine is open, and an open, continuously-run benchmark decides publicly what is actually faster.

That loop is open to everyone. If you have a faster kernel, a better scheduling heuristic, or a sharper configuration, the path the RadeonFlow team took is available to you too: send a PR to [AITER](https://github.com/ROCm/aiter) or [ATOM](https://github.com/ROCm/atom), and let [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) measure it in the open. We want the ROCm software stack — from kernels to serving engine to benchmark harness, fully open source — to be iterated by the community that runs on it, and results like the ones in this post are what that iteration looks like.

## Summary

Kimi K2.5 on MI355X is now a fully 4-bit serving stack: MXFP4 weights *and* activations through the MoE, running at TP=4 on a single node. The individual ingredients — A4W4 fused MoE kernels, collective-strategy gating fixes, a faster router, and calmer prefill scheduling — are each modest; compounded, they move the MI355X Kimi K2.5 frontier to **[FILL: headline claim vs. B200]**.

Just as importantly, the pace is visible in public: every step above corresponds to an upstream PR in [ROCm/aiter](https://github.com/ROCm/aiter) or a configuration change in [InferenceX](https://github.com/SemiAnalysisAI/InferenceX), landed and re-measured continuously. And more is coming on the same silicon: Two-Batch Overlap and DP Attention for Kimi K2.5 — the scheduling techniques we introduced for DeepSeek-V4 — and **ATOMmesh**, ATOM's distributed serving layer with prefill/decode disaggregation for multi-node deployments.

## Additional Resources

- [ATOM on GitHub](https://github.com/ROCm/atom) · [AITER on GitHub](https://github.com/ROCm/aiter)
- [InferenceX benchmark platform](https://inferencex.semianalysis.com/) · [InferenceX PR #2132](https://github.com/SemiAnalysisAI/InferenceX/pull/2132)
- Upstream kernel PRs: [aiter #3470](https://github.com/ROCm/aiter/pull/3470) (A4W4 MoE), [aiter #3832](https://github.com/ROCm/aiter/pull/3832) (RadeonFlow A4W4 MoE), [aiter #3466](https://github.com/ROCm/aiter/pull/3466) (TopK), [aiter #3458](https://github.com/ROCm/aiter/pull/3458) / [#3880](https://github.com/ROCm/aiter/pull/3880) (fused all-reduce)
- Related reading: [DP Attention and TBO for DeepSeek-V4 on MI355X](https://rocm.blogs.amd.com/software-tools-optimization/atom-optimiztion/README.html) · [Supercharge DeepSeek-R1 Inference on MI300X](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html)

---

*[FILL: standard ROCm blogs disclaimer / endnotes block — testing configuration, date of measurement, system details (CPU, ROCm version 7.2.4, ATOM 0.1.4, node config 8× MI355X), and the B200 comparison source & date.]*
