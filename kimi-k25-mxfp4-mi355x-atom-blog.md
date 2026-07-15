# Serving Kimi K2.5/K2.6/K2.7-Code in MXFP4 on AMD Instinct™ MI355X GPUs with ATOM

**Authors:** [@fty, @colorwinds, @jpy, @benenzhu, @felix(++), @lingpeng(++), @xun, @daniel, @George, @Sunpeng, @guru]
**Date:** [FILL: publication date]
**Tags:** AI/ML, LLM, Inference, Optimization, Performance

---

## Introduction

Serving Kimi K2.5/K2.6/K2.7-Code efficiently means keeping four things fast at once: the MLA attention kernels, the MoE GEMMs that dominate compute, the all-reduce collectives on every decode step's critical path, and the scheduler that decides when prefill is allowed to interrupt decode. Attention is already well covered by AITER's hand-tuned MLA kernels; this post is about pushing the other three on AMD Instinct™ MI355X GPUs with ATOM, running end-to-end in **MXFP4** — the OCP microscaling 4-bit format that CDNA™ 4 supports natively — far enough to beat single-node NVIDIA B200.

## At a Glance

- Kimi K2.5/K2.6/K2.7-Code (1T-parameter MoE) served in MXFP4 on a **single MI355X node at TP=4** — the 4-bit weights fit comfortably in 4 × 288 GB of HBM3E, and one 8-GPU node hosts two independent TP4 replicas.
- Peak throughput of **5,369.6 tok/s/GPU** on the 8k/1k workload ([FILL: 1k/1k number, or drop the 1k/1k mention]) — roughly **34% above single-node NVIDIA B200** (vLLM, NVFP4, ~4,021 tok/s/GPU on 8k/1k).
- Built together with the community: optimizations co-developed by AMD engineers and open-source contributors, merged upstream into AITER/ATOM/InferenceX, with **$300K** in AMD awards recognizing community work.

## Key Optimizations: From Kernel to Engine to Benchmark

Every gain in this post traveled the same three-stage, fully public pipeline: a kernel is optimized in **AITER** ([ROCm/aiter](https://github.com/ROCm/aiter), the AI Tensor Engine for ROCm), **ATOM** ([ROCm/atom](https://github.com/ROCm/atom), AMD's PyTorch-native inference engine) wires it into serving, and the configuration lands in **InferenceX**, where it is re-measured continuously. We walk through each stage.

### 1. AITER: kernel-level optimization

#### 1.1 Deeper-fused MXFP4 (A4W4) MoE backends for gfx950

The heart of Kimi K2 inference is the grouped GEMM inside the MoE block. AITER already ran this path in full A4W4 MXFP4 — activations and weights both entering the matrix units as 4-bit, using CDNA 4's native microscaling instructions with per-32-element E8M0 scales. The new backends ([aiter #3470](https://github.com/ROCm/aiter/pull/3470), with a RadeonFlow-contributed variant in [aiter #3832](https://github.com/ROCm/aiter/pull/3832)) rebuild the MoE path around deeper fusion and MoE-specific optimizations: [FILL: the concrete list — e.g. fused quantize/gather/scatter stages around the grouped GEMM, tiling tuned to K2's expert shapes, …]. Two backends exist because MoE GEMMs live in two regimes — many small expert problems at decode, near-dense GEMM at prefill — so both ends of the Pareto curve benefit. The fused activation kernel also quantizes the **intermediate activations between up- and down-projection** to MXFP4 on the fly, cutting traffic on the largest activation tensor in the model.

Several of these kernels are authored in **FlyDSL** ([ROCm/FlyDSL](https://github.com/ROCm/FlyDSL)), ROCm's Python-first, MLIR-native kernel DSL that lets developers express tiling, layouts, and data movement at a high level without giving up expert-level performance — a big part of why community contributors can iterate on AITER kernels this quickly ([FILL: link the specific FlyDSL-based kernel PRs]; see also the [FlyDSL introduction on ROCm Blogs](https://rocm.blogs.amd.com/software-tools-optimization/flydsl-python-native/README.html)).

#### 1.2 Single-stage fused all-reduce at production batch sizes

With TP=4, every transformer layer ends in an all-reduce that sits on the critical path of each decode step. AITER's Quick Reduce, the fused all-reduce ATOM uses over XGMI links, chooses between a low-latency **one-stage** algorithm and a lower-traffic **two-stage** one; the crossover had been tuned for small messages, so serving-scale batches (concurrency 32–64) silently fell onto the slower path. [aiter #3458](https://github.com/ROCm/aiter/pull/3458) raises the one-stage limit to cover concurrency 64 and [aiter #3880](https://github.com/ROCm/aiter/pull/3880) fixes the gating condition ([FILL: exact condition]) — a two-line class of fix that is invisible in kernel microbenchmarks and worth [FILL: X%] end-to-end.

#### 1.3 Faster expert routing

Routing every token to 8 of 384 experts runs once per layer per decode step. [aiter #3466](https://github.com/ROCm/aiter/pull/3466) replaces the original top-k selection with a fused single-kernel reduction specialized for K2's expert count and group-routing scheme, and together with MoE-metadata parallelization for MLA models, drops non-GEMM overhead per decode step from [FILL: X µs to Y µs].

### 2. ATOM: wiring the kernels into the engine

#### 2.1 Per-shape backend dispatch

Fast kernels only pay off if the engine picks them at the right moments. ATOM's dispatcher selects the MoE backend per shape, so the decode- and prefill-optimized A4W4 kernels each serve the regime they win in; `AITER_MXFP4_INTERMEDIATE=1` turns on the 4-bit intermediate-activation path.

#### 2.2 Prefill scheduling and parallelism

The engine-level tuning matters just as much. `--scheduler-delay-factor 1` changes when ATOM admits new prefills into a busy decode loop: instead of greedily injecting them — which fragments decode batches and spikes every in-flight user's latency — the scheduler briefly accumulates prefills and co-schedules them, smoothing the concurrency-64/128 operating points at negligible TTFT cost. And the parallelism choice is deliberate: TP=4 keeps expert GEMMs above the kernels' efficiency knee and collectives cheap, so the InferenceX submission runs TP4 only.

### 3. InferenceX: shipping it and measuring it in the open

The last mile is [InferenceX PR #2132](https://github.com/SemiAnalysisAI/InferenceX/pull/2132): bump the public `rocm/atom` image, set the two flags above, and let the harness sweep both workloads at concurrency 4–128. Because InferenceX re-runs continuously, this stage is also the safety net — regressions like the all-reduce gating miss are exactly what a continuously-run public benchmark exists to catch, and improvements show up on the dashboard within days of merging.

## Performance Results

We benchmark with InferenceX's single-node fixed-sequence-length harness on two standard workloads — 1k/1k (ISL 1024 / OSL 1024) and 8k/1k (ISL 8192 / OSL 1024) — sweeping concurrency from 4 to 128 and plotting the resulting Pareto frontier of throughput (tok/s/GPU) versus interactivity (tok/s/user).

<img width="3180" height="1754" alt="Pareto frontier, 8k/1k — MI355X ATOM MXFP4 (TP4) vs. NVIDIA B200 (single node)" src="https://github.com/user-attachments/assets/8d5b7f18-a589-4a18-b048-475de82bca5a" />

**Figure 1: Pareto frontier on the 8k/1k workload — MI355X ATOM MXFP4 (TP4) vs. NVIDIA B200 (single node).** *(data source: [InferenceX](https://inferencex.semianalysis.com/))*


On the 8k/1k workload, MI355X with ATOM reaches **5369.6 tok/s/GPU** at concurrency 128 while sustaining **19.2 tok/s/user**, and **116.4 tok/s/user** at concurrency 4 on the interactive end. The published single-node NVIDIA B200 result on this workload (vLLM, NVFP4) is **~4,021 tok/s/GPU** ([InferenceX](https://inferencex.semianalysis.com/)) — MI355X with ATOM beats it by roughly **34%**.

Note also what the comparison does *not* include: rack-scale disaggregated serving (e.g. wide-EP on NVL72-class systems) is a different deployment class with its own InferenceX category; here we compare single-node, buy-it-today configurations.

## How to Reproduce

To serve Kimi K2 models with ATOM on MI355X, follow the official recipe: [ATOM Kimi-K2 recipe — Launching Server](https://github.com/ROCm/ATOM/blob/main/recipes/Kimi-K2.md#launching-server). The exact benchmark configuration behind the results in this post is the InferenceX submission merged in [PR #2132](https://github.com/SemiAnalysisAI/InferenceX/pull/2132), and its results are directly comparable with the public dashboard at [inferencex.semianalysis.com](https://inferencex.semianalysis.com/).

## Built With the Developer Community

The optimizations above were co-developed in the open by AMD engineers and community developers. Many of the ideas originated with the community; kernels, reviews, and benchmarks flowed in both directions; and everything landed as public PRs across [AITER](https://github.com/ROCm/aiter), [ATOM](https://github.com/ROCm/atom), and [InferenceX](https://github.com/SemiAnalysisAI/InferenceX). The A4W4 MoE backend from the RadeonFlow team is one example, now shipping in the public `rocm/atom` image and selected by the dispatcher whenever it wins on a given shape. AMD recognizes and sponsors developers who move the needle like this: contributors to this work have received **$300K** in awards, and that program continues.

The loop is open to anyone. If you have a faster kernel, a better scheduling heuristic, or a sharper configuration: write it (FlyDSL makes the kernel side far more approachable), send a PR to [AITER](https://github.com/ROCm/aiter) or [ATOM](https://github.com/ROCm/atom), and let [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) measure it in the open. We are also upstreaming these optimizations into **vLLM** and **SGLang**, so the same gains will surface across those stacks over time. We want the ROCm software stack — kernels, serving engine, benchmark harness, all open source — to be iterated by the community that runs on it.

## Summary

Kimi K2.5/K2.6/K2.7-Code on MI355X is now a fully 4-bit serving stack: MXFP4 weights *and* activations through the MoE, running at TP=4 on a single node, at **5,369.6 tok/s/GPU on 8k/1k — roughly 34% above single-node NVIDIA B200**. The individual ingredients — A4W4 fused MoE kernels, collective-strategy gating fixes, a faster router, and calmer prefill scheduling — are each modest; compounded, and traveling the AITER → ATOM → InferenceX pipeline in public, they moved the frontier past B200 on the same silicon that started at [FILL: X,XXX] tok/s/GPU.

And more is coming: Two-Batch Overlap and DP Attention for the Kimi K2 family, **ATOMmesh** — ATOM's distributed serving layer with prefill/decode disaggregation for multi-node deployments — and the vLLM/SGLang upstreaming mentioned above.

## Additional Resources

- [ATOM on GitHub](https://github.com/ROCm/atom) · [AITER on GitHub](https://github.com/ROCm/aiter) · [FlyDSL on GitHub](https://github.com/ROCm/FlyDSL)
- [InferenceX benchmark platform](https://inferencex.semianalysis.com/) · [InferenceX PR #2132](https://github.com/SemiAnalysisAI/InferenceX/pull/2132)
- Upstream kernel PRs: [aiter #3470](https://github.com/ROCm/aiter/pull/3470) (A4W4 MoE), [aiter #3832](https://github.com/ROCm/aiter/pull/3832) (RadeonFlow A4W4 MoE), [aiter #3466](https://github.com/ROCm/aiter/pull/3466) (TopK), [aiter #3458](https://github.com/ROCm/aiter/pull/3458) / [#3880](https://github.com/ROCm/aiter/pull/3880) (fused all-reduce)
- Related reading: [FlyDSL: Expert GPU Kernel Development with a Python-Native DSL](https://rocm.blogs.amd.com/software-tools-optimization/flydsl-python-native/README.html) · [DP Attention and TBO for DeepSeek-V4 on MI355X](https://rocm.blogs.amd.com/software-tools-optimization/atom-optimiztion/README.html)

---

*[FILL: standard ROCm blogs disclaimer / endnotes block — testing configuration, date of measurement, system details (CPU, ROCm version, ATOM version, node config 8× MI355X), and the B200 comparison source & date.]*
