# Kimi K2.5 / K2.6 / K2.7-Code in MXFP4 on AMD Instinct™ MI355X GPUs with ATOM outperforms NVIDIA B200

**Authors:** [@fty, @colorwinds, @jpy, @benenzhu, @felix(++), @lingpeng(++), @bernard, @xun, @daniel, @George, @Sunpeng, @guru]
**Date:** [FILL: publication date]
**Tags:** AI/ML, LLM, Inference, Optimization, Performance

---

## Introduction

Serving Kimi K2.5/K2.6/K2.7-Code efficiently means keeping four things fast at once: the MLA attention kernels, the MoE GEMMs that dominate compute, the all-reduce collectives on every decode step's critical path, and the scheduler that decides when prefill is allowed to interrupt decode. Attention is already well covered by AITER's hand-tuned MLA kernels; this post is about pushing the other three on AMD Instinct™ MI355X GPUs with ATOM, running end-to-end in **MXFP4** — the OCP microscaling 4-bit format that CDNA™ 4 supports natively — far enough to beat single-node NVIDIA B200.

## At a Glance

- Kimi K2.5/K2.6/K2.7-Code (1T-parameter MoE) served in MXFP4 on a **single MI355X node at TP=4** — the 4-bit weights fit comfortably in 4 × 288 GB of HBM3E, and one 8-GPU node hosts two independent TP4 replicas.
- Peak throughput of **5,369.6 tok/s/GPU** on the 8k/1k workload, staying **ahead of single-node NVIDIA B200 across multiple concurrency levels**.
- Built together with the community: optimizations co-developed by AMD engineers and open-source contributors, merged upstream into AITER/ATOM/InferenceX through an AMD developer collaboration program.

## Key Optimizations: From Kernel to Engine to Benchmark

Every gain in this post traveled the same three-stage, fully public pipeline: a kernel is optimized in **AITER** ([ROCm/aiter](https://github.com/ROCm/aiter), the AI Tensor Engine for ROCm), **ATOM** ([ROCm/atom](https://github.com/ROCm/atom), AMD's PyTorch-native inference engine) wires it into serving, and the configuration lands in **InferenceX**, where it is re-measured continuously. We walk through each stage.

### 1. AITER: kernel-level optimization

#### 1.1 Deeper-fused MXFP4 (A4W4) MoE backends for gfx950

The heart of Kimi K2.7-Code inference — and of every model on the shared K2 backbone — is the grouped GEMM inside the MoE block. AITER already ran this path in full A4W4 MXFP4 — activations and weights both entering the matrix units as 4-bit, using CDNA 4's native microscaling instructions with per-32-element E8M0 scales. The new backend ([aiter #3832](https://github.com/ROCm/aiter/pull/3832), building on [#3470](https://github.com/ROCm/aiter/pull/3470)) rebuilds the MoE path around deeper fusion and Kimi-K2-specific tuning:

- **Two grouped GEMMs, one fused bridge between them.** The gate/up projection (`gemm1`) and the down projection (`gemm2`) run as two FlyDSL grouped GEMMs. Between them, a single kernel fuses the SwiGLU gate activation, the elementwise multiply, the MXFP4 quantization of the **intermediate activations**, *and* the E8M0 scale write in the sorted layout `gemm2` expects — so the largest activation tensor in the model is never written out in high precision or re-read for a separate quantization pass.
- **Inline activation quantization for decode.** For small-batch decode shapes, gemm1 converts incoming bf16 hidden states to MXFP4 in its prologue as they are gathered, keeping quantization directly on the GEMM's input path.
- **Sorting fused with the GEMM metadata.** An adaptive MoE-sorting backend emits the token-gather metadata (`m_indices` / reverse-sorted indices) the grouped GEMM consumes, in the same pass rather than as a separate pre-step.
- **Per-shape tuning for K2's experts.** Tile shapes and split-K are tuned per expert shape (`kimik2_fp4_tuned_fmoe.csv`), and the per-32-element scale amax is computed with hardware DPP instructions instead of a separate reduction.

Two GEMM configurations are kept because MoE GEMMs live in two regimes — many small expert problems at decode, near-dense GEMM at prefill — so both ends of the Pareto curve benefit.

These kernels are authored in **FlyDSL** ([ROCm/FlyDSL](https://github.com/ROCm/FlyDSL)), ROCm's Python-first, MLIR-native kernel DSL that lets developers express tiling, layouts, and data movement at a high level without giving up expert-level performance — a big part of why community contributors can iterate on AITER kernels this quickly (see the [FlyDSL introduction on ROCm Blogs](https://rocm.blogs.amd.com/software-tools-optimization/flydsl-python-native/README.html)).

#### 1.2 Single-stage fused all-reduce at production batch sizes

Every transformer layer ends in an all-reduce that sits on the critical path of each decode step. AITER's Quick Reduce, the fused all-reduce ATOM uses over XGMI links, chooses between a low-latency **one-stage** algorithm and a lower-traffic **two-stage** one. [aiter #3458](https://github.com/ROCm/aiter/pull/3458) and [aiter #3880](https://github.com/ROCm/aiter/pull/3880) introduced TP-aware gating, selecting the one-stage path through batch 32 at TP4 and batch 16 at TP8 before switching to two-stage for larger decode batches. This keeps communication latency low at common serving batch sizes without sacrificing efficiency as concurrency grows.

#### 1.3 Faster expert routing

Routing every token to 8 of 384 experts runs once per layer per decode step. [aiter #3466](https://github.com/ROCm/aiter/pull/3466) replaces the original top-k selection with a fused single-kernel reduction specialized for K2's expert count and group-routing scheme, trimming the per-step non-GEMM overhead that otherwise eats into decode throughput.

### 2. ATOM: wiring the kernels into the engine

#### 2.1 Per-shape backend dispatch

Fast kernels only pay off if the engine picks them at the right moments. ATOM's dispatcher selects the MoE backend per shape, so the decode- and prefill-optimized A4W4 kernels each serve the regime they win in; `AITER_MXFP4_INTERMEDIATE=1` turns on the 4-bit intermediate-activation path.

#### 2.2 Prefill scheduling and parallelism

The engine-level tuning matters just as much. `--scheduler-delay-factor 1` changes when ATOM admits new prefills into a busy decode loop: instead of greedily injecting them — which fragments decode batches and spikes every in-flight user's latency — the scheduler briefly accumulates prefills and co-schedules them, smoothing the concurrency-64/128 operating points at negligible TTFT cost. And the parallelism choice is deliberate: TP=4 keeps expert GEMMs above the kernels' efficiency knee and collectives cheap, so the InferenceX submission runs TP4 only.

### 3. InferenceX: shipping it and measuring it in the open

The last mile is [InferenceX PR #2132](https://github.com/SemiAnalysisAI/InferenceX/pull/2132): bump the public `rocm/atom` image, set the two flags above, and let the harness sweep both workloads at concurrency 4–128. Because InferenceX re-runs continuously, this stage is also the safety net — regressions like the all-reduce gating miss are exactly what a continuously-run public benchmark exists to catch, and improvements show up on the dashboard within days of merging.

## Performance Results

We benchmark with InferenceX's single-node fixed-sequence-length harness on the 8k/1k workload (ISL 8192 / OSL 1024), sweeping concurrency from 4 to 128 and plotting the resulting Pareto frontier of throughput (tok/s/GPU) versus interactivity (tok/s/user).

<img width="3180" height="1754" alt="Pareto frontier, 8k/1k — MI355X ATOM MXFP4 (TP4) vs. NVIDIA B200 (single node)" src="https://github.com/user-attachments/assets/8d5b7f18-a589-4a18-b048-475de82bca5a" />

**Figure 1: Pareto frontier on the 8k/1k workload — MI355X ATOM MXFP4 (TP4) vs. NVIDIA B200 (single node).** *(data source: [InferenceX](https://inferencex.semianalysis.com/))*


On the 8k/1k workload, MI355X with ATOM reaches **5369.6 tok/s/GPU** at concurrency 128 while sustaining **19.2 tok/s/user**, and **116.4 tok/s/user** at concurrency 4 on the interactive end. As Figure 1 shows, its Pareto frontier sits above single-node NVIDIA B200 (vLLM, NVFP4) across multiple concurrency levels — MI355X delivers more throughput at matched interactivity along the curve, not just at the peak.

Note also what the comparison does *not* include: rack-scale disaggregated serving (e.g. wide-EP on NVL72-class systems) is a different deployment class with its own InferenceX category; here we compare single-node, buy-it-today configurations.

## How to Reproduce

To serve Kimi K2 models with ATOM on MI355X, follow the official recipe: [ATOM Kimi-K2 recipe — Launching Server](https://github.com/ROCm/ATOM/blob/main/recipes/Kimi-K2.md#launching-server). The exact benchmark configuration behind the results in this post is the InferenceX submission merged in [PR #2132](https://github.com/SemiAnalysisAI/InferenceX/pull/2132), and its results are directly comparable with the public dashboard at [inferencex.semianalysis.com](https://inferencex.semianalysis.com/).

## Built With the Developer Community

The optimizations above were co-developed in the open by AMD engineers and community developers. Many of the ideas originated with the community; kernels, reviews, and benchmarks flowed in both directions; and everything landed as public PRs across [AITER](https://github.com/ROCm/aiter), [ATOM](https://github.com/ROCm/atom), and [InferenceX](https://github.com/SemiAnalysisAI/InferenceX). The A4W4 MoE backend from the RadeonFlow team is one example, now shipping in the public `rocm/atom` image and selected by the dispatcher whenever it wins on a given shape.

Much of this grew out of an earlier AMD collaboration program that invited developers to push MI355X inference and rewarded the teams that met its performance bar [FILL: award amount, e.g. "with up to $XXK per team"]. We plan to keep running programs like it — recognizing and sponsoring the developers who move the needle.

The loop is open to anyone. If you have a faster kernel, a better scheduling heuristic, or a sharper configuration: write it (FlyDSL makes the kernel side far more approachable), send a PR to [AITER](https://github.com/ROCm/aiter) or [ATOM](https://github.com/ROCm/atom), and let [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) measure it in the open. We are also upstreaming these optimizations into **vLLM** and **SGLang**, so the same gains will surface across those stacks over time. We want the ROCm software stack — kernels, serving engine, benchmark harness, all open source — to be iterated by the community that runs on it.

## Summary

Kimi K2.5/K2.6/K2.7-Code on MI355X is now a fully 4-bit serving stack: MXFP4 weights *and* activations through the MoE, running at TP=4 on a single node, at **5,369.6 tok/s/GPU on 8k/1k — ahead of single-node NVIDIA B200**. The individual ingredients — A4W4 fused MoE kernels, the collective-strategy gating fix, a faster router, and calmer prefill scheduling — are each modest; compounded, and traveling the AITER → ATOM → InferenceX pipeline in public, they move the frontier past B200.

And more is coming: Two-Batch Overlap and DP Attention for the Kimi K2 family, **ATOMmesh** — ATOM's distributed serving layer with prefill/decode disaggregation for multi-node deployments — and the vLLM/SGLang upstreaming mentioned above.

## Additional Resources

- [ATOM on GitHub](https://github.com/ROCm/atom) · [AITER on GitHub](https://github.com/ROCm/aiter) · [FlyDSL on GitHub](https://github.com/ROCm/FlyDSL)
- [InferenceX benchmark platform](https://inferencex.semianalysis.com/) · [InferenceX PR #2132](https://github.com/SemiAnalysisAI/InferenceX/pull/2132)
- Upstream kernel PRs: [aiter #3470](https://github.com/ROCm/aiter/pull/3470) (A4W4 MoE), [aiter #3832](https://github.com/ROCm/aiter/pull/3832) (RadeonFlow A4W4 MoE), [aiter #3466](https://github.com/ROCm/aiter/pull/3466) (TopK), [aiter #3458](https://github.com/ROCm/aiter/pull/3458) / [#3880](https://github.com/ROCm/aiter/pull/3880) (fused all-reduce)
- Related reading: [FlyDSL: Expert GPU Kernel Development with a Python-Native DSL](https://rocm.blogs.amd.com/software-tools-optimization/flydsl-python-native/README.html) · [DP Attention and TBO for DeepSeek-V4 on MI355X](https://rocm.blogs.amd.com/software-tools-optimization/atom-optimiztion/README.html)

---

*[FILL: standard ROCm blogs disclaimer / endnotes block — testing configuration, date of measurement, system details (CPU, ROCm version, ATOM version, node config 8× MI355X), and the B200 comparison source & date.]*
