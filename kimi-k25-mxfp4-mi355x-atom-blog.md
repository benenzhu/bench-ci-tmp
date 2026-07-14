# Serving Kimi K2.5 in MXFP4 on AMD Instinct™ MI355X GPUs with ATOM

**Authors:** [FILL: author list — e.g. benenzhu, RadeonFlow Team, AMD contributors]
**Date:** [FILL: publication date]
**Tags:** AI/ML, LLM, Inference, Optimization, Performance

---

## Introduction

Trillion-parameter Mixture-of-Experts (MoE) models are no longer research curiosities — they are production workloads. Moonshot AI's Kimi K2.5 packs 1T total parameters with roughly 32B activated per token, routed across 384 experts (8 routed plus 1 shared per token) on top of Multi-head Latent Attention (MLA). Serving a model of this shape efficiently stresses every layer of the inference stack at once: low-precision GEMM throughput in the MoE experts, expert-routing overhead, inter-GPU communication, KV-cache capacity, and the scheduler that decides when prefill is allowed to interrupt decode.

In a [previous post](https://rocm.blogs.amd.com/software-tools-optimization/atom-optimiztion/README.html) we showed how ATOM, AMD's open inference engine, uses DP Attention scheduling and Two-Batch Overlap to deliver strong DeepSeek-V4 performance on AMD Instinct™ MI355X GPUs. In this post we turn to Kimi K2.5 and a different lever: end-to-end **MXFP4** serving. The MI355X's CDNA™ 4 architecture natively supports the OCP microscaling MXFP4 data type, and with the latest ATOM release and AITER (AI Tensor Engine for ROCm) kernels, the entire MoE path — activations and weights — now runs in 4-bit.

All results below are produced with [InferenceX](https://github.com/SemiAnalysisAI/InferenceX), SemiAnalysis's open, continuously-run inference benchmark, and are reproducible from the public configuration merged in [InferenceX PR #2132](https://github.com/SemiAnalysisAI/InferenceX/pull/2132). Accuracy is gated the same way for every submission: GSM8K exact-match must hold at every measured operating point, and the InferenceX compliance checklist forbids model-architecture changes, engine patching, or evaluation shortcuts — what is benchmarked is what you can `docker pull` today.

## At a Glance

- Kimi K2.5 (1T-parameter MoE) served in MXFP4 on a **single MI355X node at TP=4** — the 4-bit weights (~[FILL: ~5xx] GB) fit comfortably in 4 × 288 GB of HBM3E, leaving ample room for KV cache; one 8-GPU node hosts two independent TP4 replicas.
- Peak throughput of **[FILL: X,XXX] tok/s/GPU** on the 8k/1k workload and **[FILL: X,XXX] tok/s/GPU** on 1k/1k, versus the published single-node NVIDIA B200 (vLLM, NVFP4) figure of ~4,021 tok/s/GPU on 8k/1k — [FILL: ratio / positioning statement].
- A fully 4-bit MoE path: AITER MXFP4 (A4W4) fused MoE kernels for gfx950, MXFP4 intermediate activations, and INT4-compressed all-reduce via Quick Reduce.
- Accuracy preserved: GSM8K exact-match (strict) of **97.19%** at concurrency 64 and **97.27%** at concurrency 128, matching the FP8/BF16 reference within noise.
- Everything ships in the public `rocm/atom` Docker image — no private branches, no patches.

## Background: Kimi K2.5 Meets MI355X

Kimi K2.5 inherits the DeepSeek-style recipe — MLA attention plus a wide, sparsely-activated FFN — but at 1T scale with 384 experts. Two properties dominate its serving profile:

1. **It is memory-bound at the weights.** Even though only ~32B parameters activate per token, the full 1T parameters must be resident. At 4 bits per weight this is roughly [FILL: ~5xx] GB plus scales and metadata. On MI355X (288 GB HBM3E, 8 TB/s per GPU), TP=4 holds the model with headroom for large KV caches; competing 192 GB-class GPUs need TP=8 for the same model, spending twice the GPUs per replica.
2. **It is communication-sensitive.** With tensor-parallel MoE, every layer ends in an all-reduce. At interactive batch sizes these collectives sit on the critical path of each decode step, so shrinking or hiding them translates directly into tok/s/user.

ATOM ([ROCm/atom](https://github.com/ROCm/atom)) is AMD's lightweight, PyTorch-native LLM inference engine, and AITER ([ROCm/aiter](https://github.com/ROCm/aiter)) supplies its hand-tuned kernels. The work described here landed in ATOM 0.1.4 and the AITER kernels shipped in the `rocm/atom:rocm7.2.4_..._atom0.1.4` image, and was validated end-to-end through InferenceX's continuous benchmarking.

## Key Optimizations

The gains come from five threads of work, each visible as an upstream PR — plus one deliberate configuration decision (TP4-only). We describe them in the order a token experiences them.

### 1. MXFP4 (A4W4) fused MoE kernels for gfx950

The heart of Kimi K2.5 inference is the grouped GEMM inside the MoE block. Earlier MXFP4 support quantized weights only (W4A8/W4A16): activations were upcast, halving the arithmetic benefit of the format. The new AITER backends ([aiter #3470](https://github.com/ROCm/aiter/pull/3470), with a RadeonFlow-contributed variant in [aiter #3832](https://github.com/ROCm/aiter/pull/3832)) execute the full **A4W4** path on gfx950: both activations and weights enter the matrix units as MXFP4, using CDNA 4's native microscaling instructions with per-32-element E8M0 scales.

Two backends are provided because MoE GEMMs live in two regimes. At decode-time batch sizes the kernel is launched over many small expert problems and is bound by scheduling and memory; at prefill it approaches dense GEMM behavior. The dispatcher picks per-shape, so both ends of the Pareto curve benefit.

Setting `AITER_MXFP4_INTERMEDIATE=1` extends 4-bit to the **intermediate activations between the up- and down-projection** of each expert. The gate/up output is quantized to MXFP4 on the fly (fused into the activation kernel, so no extra pass over memory) before the down-projection consumes it. This cuts the intermediate tensor traffic — the largest activation in the model, `topk × tokens × 2 × moe_intermediate_dim` — by [FILL: 2×/4× vs previous format], which matters most at high concurrency where prefill and decode share bandwidth. GSM8K showed no measurable degradation from this step (see Accuracy below).

### 2. INT4-compressed all-reduce with Quick Reduce

With TP=4, every transformer layer performs an all-reduce over the residual-stream tensor. AITER's **Quick Reduce** implements the collective directly over XGMI links with optional on-the-wire compression. Setting

```bash
AITER_QUICK_REDUCE_QUANTIZATION=INT4
```

quantizes the payload to INT4 (with per-block scales) before it crosses the link and dequantizes on arrival — a ~4× reduction in bytes moved versus FP16. Because the all-reduce operates on the pre-normalization residual contribution, and each layer immediately renormalizes, the numerical impact is small; the accuracy gate confirms it. The wall-clock effect is largest exactly where users feel it: small-batch decode, where the collective is latency- rather than bandwidth-bound and every microsecond is visible in tok/s/user. On the 8k/1k workload this contributed [FILL: X% at conc 4 / X% at conc 64] end-to-end.

### 3. Single-stage fused all-reduce at production batch sizes

Quick Reduce has two internal strategies: a **one-stage** algorithm (each rank pulls and reduces peers' buffers directly — lowest latency, more redundant traffic) and a **two-stage** reduce-scatter + all-gather (less traffic, more synchronization). The crossover had been tuned for small messages, so the batch sizes that matter for serving — concurrency 32–64 — silently fell onto the slower two-stage path. [aiter #3458](https://github.com/ROCm/aiter/pull/3458) raises the one-stage limit to cover concurrency 64, and [aiter #3880](https://github.com/ROCm/aiter/pull/3880) fixes the gating condition so the decision is made on actual message bytes ([FILL: exact condition]). This is a two-line class of fix that is invisible in kernel microbenchmarks and worth [FILL: X%] in end-to-end throughput — precisely the kind of regression a continuous benchmark like InferenceX exists to catch.

### 4. Faster expert routing: the TopK kernel

Routing every token to 8 of 384 experts sounds negligible next to the GEMMs, but at decode batch sizes the router runs once per layer per step, and Kimi K2.5 has [FILL: N] MoE layers. The original top-k selection launched [FILL: multiple kernels / used a generic sort]; [aiter #3466](https://github.com/ROCm/aiter/pull/3466) replaces it with a fused single-kernel reduction specialized for K2.5's expert count and group-routing scheme, cutting router time by [FILL: X×]. Together with the MoE-metadata parallelization work for MLA models that landed alongside it, the non-GEMM overhead per decode step dropped from [FILL: X µs to Y µs].

### 5. Scheduling: `--scheduler-delay-factor 1`

ATOM, like vLLM, must decide when to admit new prefills into a busy decode loop. Admitting them greedily maximizes time-to-first-token at low load but causes **decode stalls** at high concurrency: a long prefill lands in the middle of a decode stream and every in-flight user sees a latency spike; worse, decode batches fragment and per-step efficiency drops.

`--scheduler-delay-factor 1` delays prefill admission by one previous-prompt-latency unit, letting the scheduler accumulate prefills and co-schedule them instead of dribbling them into the decode stream. On the fixed-sequence-length InferenceX workloads this smooths the concurrency-64/128 operating points — [FILL: X% throughput at conc 128, X% p99 TPOT improvement] — at negligible TTFT cost.

Operationally, `ATOM_DISABLE_MMAP=true` is also set in the new configuration: loading the ~[FILL: 5xx] GB checkpoint through direct reads rather than mmap-backed paging cuts model-load time [FILL: from X to Y minutes], which matters for benchmark automation and elastic production deployments alike, though it does not affect steady-state performance.

### Why TP4-only (and what happened to TP8)

The previous configuration swept both TP=4 and TP=8. The updated submission removes TP=8 because it was **consistently slower than TP=4** for Kimi K2.5 MXFP4 on MI355X, on both workloads and at every concurrency:

- TP=8 halves the per-GPU GEMM work. K2.5's expert matrices are narrow once sharded 8 ways; the MXFP4 kernels drop below their efficiency knee, so each GPU does half the work at well under half the time saved.
- The all-reduce doubles in participant count, and cross-node-free 8-GPU collectives still cost more than 4-GPU ones — growing exactly the term optimizations #2 and #3 shrink.
- 288 GB of HBM3E removes the usual reason to shard wider: TP=4 already fits the model plus [FILL: X] GB of KV cache per GPU.

The practical consequence: an 8-GPU MI355X node runs **two independent TP4 replicas**, each serving its own traffic. For throughput-oriented serving this is strictly better than one TP8 instance — and it is the configuration InferenceX now tracks.

## Performance Results

We benchmark with InferenceX's single-node fixed-sequence-length harness on two standard workloads — 1k/1k (ISL 1024 / OSL 1024) and 8k/1k (ISL 8192 / OSL 1024) — sweeping concurrency from 4 to 128 and plotting the resulting Pareto frontier of throughput (tok/s/GPU) versus interactivity (tok/s/user).

**Figure 1: [FILL: Pareto frontier, 8k/1k — MI355X ATOM MXFP4 (TP4) vs. NVIDIA B200 vLLM NVFP4 (single node)]**

**Figure 2: [FILL: Pareto frontier, 1k/1k — same systems]**

On the 8k/1k workload, MI355X with ATOM reaches **[FILL: X,XXX] tok/s/GPU** at concurrency 128 while sustaining **[FILL: XX] tok/s/user**, and **[FILL: XX] tok/s/user** at concurrency 4 on the interactive end. For reference, the published single-node B200 result on this workload (vLLM, NVFP4, TP[FILL]) is **~4,021 tok/s/GPU** ([InferenceX](https://inferencex.semianalysis.com/)); the best previously published MI355X number on the vLLM path was 2,687 tok/s/GPU. [FILL: one-sentence positioning — e.g. "The ATOM MXFP4 stack closes/reverses this gap, reaching XX% of / exceeding B200 single-node throughput while offering YY% more HBM per GPU."]

**Figure 3: [FILL: optimization waterfall — baseline ATOM 0.1.3 → +A4W4 MoE → +MXFP4 intermediate → +Quick Reduce INT4 → +AR 1-stage fix → +TopK → +scheduler, at conc 64, 8k/1k]**

Beyond the peak number, the shape of the frontier is the story: the communication work (optimizations #2–3) lifts the low-concurrency, interactivity-critical end, while the A4W4 MoE kernels and scheduler change lift the high-concurrency end. [FILL: 1–2 sentences on where the biggest deltas landed vs. the previous ATOM image, e.g. "Relative to the previous atom0.1.3 image, the new stack is X% faster at conc 4 and Y% faster at conc 128."]

Note also what the comparison does *not* include: rack-scale disaggregated serving (e.g. wide-EP on NVL72-class systems) is a different deployment class with its own InferenceX category; here we compare single-node, buy-it-today configurations.

## Accuracy Validation

Every InferenceX submission must pass its accuracy gate at the measured operating points, with the model architecture untouched and the engine unpatched. With the full 4-bit path enabled — MXFP4 weights, MXFP4 intermediate activations, and INT4-compressed all-reduce — `amd/Kimi-K2.5-MXFP4` scores:

| Concurrency | GSM8K exact-match (strict) |
|---|---|
| 64 | **97.19%** |
| 128 | **97.27%** |

This matches the reference within run-to-run noise ([FILL: reference number/format, e.g. FP8 baseline 97.2x%]), confirming that each of the low-precision steps above stays inside the model's numerical tolerance.

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

## Summary

Kimi K2.5 on MI355X is now a fully 4-bit serving stack: MXFP4 weights *and* activations through the MoE, 4-bit compressed collectives between GPUs, running at TP=4 on a single node with accuracy held at the FP8 reference level. The individual ingredients — A4W4 fused MoE kernels, Quick Reduce, collective-strategy gating fixes, a faster router, and calmer prefill scheduling — are each modest; compounded, they move the MI355X Kimi K2.5 frontier to **[FILL: headline claim vs. B200]**.

Just as importantly, the pace is visible in public: every step above corresponds to an upstream PR in [ROCm/aiter](https://github.com/ROCm/aiter) or a configuration change in [InferenceX](https://github.com/SemiAnalysisAI/InferenceX), landed, gated on accuracy, and re-measured continuously. More is coming — [FILL: teaser: e.g. TBO for K2.5, wide-EP multi-node, MTP/speculative decoding] — on the same silicon.

## Additional Resources

- [ATOM on GitHub](https://github.com/ROCm/atom) · [AITER on GitHub](https://github.com/ROCm/aiter)
- [InferenceX benchmark platform](https://inferencex.semianalysis.com/) · [InferenceX PR #2132](https://github.com/SemiAnalysisAI/InferenceX/pull/2132)
- Upstream kernel PRs: [aiter #3470](https://github.com/ROCm/aiter/pull/3470) (A4W4 MoE), [aiter #3832](https://github.com/ROCm/aiter/pull/3832) (RadeonFlow A4W4 MoE), [aiter #3466](https://github.com/ROCm/aiter/pull/3466) (TopK), [aiter #3458](https://github.com/ROCm/aiter/pull/3458) / [#3880](https://github.com/ROCm/aiter/pull/3880) (fused all-reduce)
- Related reading: [DP Attention and TBO for DeepSeek-V4 on MI355X](https://rocm.blogs.amd.com/software-tools-optimization/atom-optimiztion/README.html) · [Supercharge DeepSeek-R1 Inference on MI300X](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html)

---

*[FILL: standard ROCm blogs disclaimer / endnotes block — testing configuration, date of measurement, system details (CPU, ROCm version 7.2.4, ATOM 0.1.4, node config 8× MI355X), and the B200 comparison source & date.]*
