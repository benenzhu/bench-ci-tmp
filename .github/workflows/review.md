# ROCm/aiter + ROCm/ATOM PR 进度汇总

更新时间：2026-06-10  
时间口径：GitHub UTC 时间  
范围：`ftyghome`、`ColorsWind`、`jpy794` 相关公开 PR / 公开 commit

## 总体结论

当前共查到 9 个相关 open PR：
- `ROCm/aiter`：6 个 open PR。
- `ROCm/ATOM`：3 个 open PR。

整体进度仍处于 review、性能验证、CI 修复阶段，尚无相关 PR 合入。当前最接近推进的是 aiter `#3465`，CI 状态最好但仍缺人工 approve；剩余工作最多的是 aiter `#3470` 及其 ATOM 联动 PR `#1057`，需要先修 CI、correctness review 点和端到端 benchmark。

## 跨仓库总表

| 仓库 | PR | 作者 / 贡献者 | 创建日期 | 当前状态 | Review / 评论进度 | 主要卡点 | 当前判断 |
|---|---|---|---|---|---|---|---|
| ROCm/aiter | [#3458](https://github.com/ROCm/aiter/pull/3458) | ftyghome | 2026-06-01 | Open | `TennyWang1223` 参与讨论；暂无 approve | 性能结论有分歧；作者已补 trace；存在 merge conflict | 还在性能确认阶段 |
| ROCm/aiter | [#3462](https://github.com/ROCm/aiter/pull/3462) | ftyghome | 2026-06-01 | Open | Copilot 自动 review 3 条；暂无人工 approve | DPP correctness / width 语义 / magic number；CI 不完整 | 初始 review 阶段 |
| ROCm/aiter | [#3465](https://github.com/ROCm/aiter/pull/3465) | ftyghome | 2026-06-01 | Open | 请求 `minmengdie` review；Copilot 4 条；暂无 approve | 等人工 review；需处理 launch/grid、env var、LDS sizing 等问题 | 最接近推进 |
| ROCm/aiter | [#3466](https://github.com/ROCm/aiter/pull/3466) | ftyghome | 2026-06-01 | Open | `valarLip` 要求补 full test results；Copilot 4 条 | 当前只覆盖 token <=256；需补更完整 benchmark | 等补测试数据 |
| ROCm/aiter | [#3468](https://github.com/ROCm/aiter/pull/3468) | ftyghome | 2026-06-01 | Open | Copilot 5 条；暂无人工 approve | 输入校验、dispatch、alignment、JIT build 配置问题 | 初始 review 阶段 |
| ROCm/aiter | [#3470](https://github.com/ROCm/aiter/pull/3470) | ColorsWind | 2026-06-01 | Open | `coderfeli`、`benenzhu`、`lalala-sh` 参与；暂无 approve | CI 红；correctness 风险；benchmark 口径仍需收敛；merge conflict | 活跃但剩余工作最多 |
| ROCm/ATOM | [#1017](https://github.com/ROCm/ATOM/pull/1017) | ftyghome；jpy794 贡献 4 commit | 2026-06-01 | Open | 请求 `ZhangLirong-amd`、`zejunchen-zejun`、`gbyu-amd` review；Copilot 8 条；暂无人工 approve | DPA+TBO 逻辑复杂；GPU keepalive 内存、buffer 上限、DP magic number 等 review 点；merge conflict | 需要补 e2e 验证和 reviewer 结论 |
| ROCm/ATOM | [#1018](https://github.com/ROCm/ATOM/pull/1018) | ftyghome | 2026-06-01 | Open | 请求 `XiaobingSuper`、`gbyu-amd` review；Copilot 2 条；暂无人工 approve | 依赖 aiter `#3468`；optional import / kv_buffer layout 待处理 | 依赖项未收敛 |
| ROCm/ATOM | [#1057](https://github.com/ROCm/ATOM/pull/1057) | ftyghome | 2026-06-03 | Open | Copilot 3 条；暂无人工 approve | 依赖 aiter `#3470`；hard-coded layout / `.data` mutation 风险；测试结果只引用 aiter `#3470` | 依赖项未收敛 |

## 按日期 Timeline

### 2026-06-01

- 09:51：`ftyghome` 提出 aiter [#3458](https://github.com/ROCm/aiter/pull/3458)：提高 all-reduce 1-stage fused path threshold。
- 11:36：`ftyghome` 提出 aiter [#3462](https://github.com/ROCm/aiter/pull/3462)：给 custom all-reduce 增加 DPP fast path。
- 12:54：`ftyghome` 提出 aiter [#3465](https://github.com/ROCm/aiter/pull/3465)：增加 MLA metadata parallel path。
- 13:21：`ftyghome` 提出 aiter [#3466](https://github.com/ROCm/aiter/pull/3466)：优化 topk kernel。
- 14:10：`ftyghome` 提出 ATOM [#1017](https://github.com/ROCm/ATOM/pull/1017)：Enable DPA + TBO Support for Kimi K2.5 Inference。该 PR 中 `jpy794` 是主要 commit author。
- 14:12：Copilot 对 ATOM `#1017` 给出 8 条 review comment，集中在 TBO keepalive GPU memory、MoE buffer 上限、DP magic number、文档一致性等问题。
- 14:29：`ftyghome` 提出 aiter [#3468](https://github.com/ROCm/aiter/pull/3468)：新增 MLA decode standalone reduce kernel。
- 14:56：`ftyghome` 提出 ATOM [#1018](https://github.com/ROCm/ATOM/pull/1018)：Add HIP MLA reduce kernel dispatch logic，依赖 aiter `#3468`。
- 15:45：`valarLip` 在 aiter `#3466` 评论，要求补 full test results，不要只给 token num <=256。
- 15:52：`ColorsWind` 提出 aiter [#3470](https://github.com/ROCm/aiter/pull/3470)：MXFP4 a4w4 MoE backend for gfx950。

### 2026-06-02

- 02:50：`benenzhu` 在 aiter `#3470` 提供 performance bench script 链接。
- 09:08：aiter `#3465` 的主要标准测试 / tuned op bench 完成并通过，CI 状态在当前相关 PR 中相对最好。

### 2026-06-03

- 03:58：`coderfeli` 在 aiter `#3470` 指出 benchmark 使用 `perf_counter` 可能计入 CPU overhead，建议按真实 e2e / cudagraph 口径评估。
- 08:05：`ftyghome` 在 aiter `#3470` 更新 cudagraph 版本 microbench，给出 `mxfp4_moe` vs FlyDSL 的新性能数据。
- 09:38：`lalala-sh` 在 aiter `#3470` 要求用 `fused_moe` interface 重新收集 FlyDSL 对比数据，使 benchmark 更接近实际调度路径。
- 09:58：`TennyWang1223` 在 aiter `#3458` 给出本地相反结果：fused 1-stage 在多个 shape 上比原 2-stage 慢，并要求确认测试环境。
- 11:07：`benenzhu` 继续在 aiter `#3470` 讨论 FlyDSL 通过 `moe tuner / fused_moe` 路径重新比较的问题。
- 13:15：`ftyghome` 提出 ATOM [#1057](https://github.com/ROCm/ATOM/pull/1057)：为 HIP fused_moe 增加 preshuffle，依赖 aiter `#3470`。
- 15:30：`ftyghome` 在 aiter `#3458` 回复：standalone microbench 能复现 regression，但实际 decode path 表现不同，并给出 MI350X VF / ROCm 7.2.3 环境信息。

### 2026-06-04

- 03:00：`TennyWang1223` 在 aiter `#3458` 要求上传 Kimi K2.5 MXFP4 decode path 的 torch profiler traces。
- 09:33：`ftyghome` 在 aiter `#3458` 上传 fused / non-fused、conc32 / conc64 的 profiler trace。

### 2026-06-05

- `coderfeli` 对 aiter `#3470` 给出多条 review comment，包括需要 enable flag、`A_scale` buffer bound 可能导致 silent wrong results、prefill hard limit 可能导致 wrong outputs。

### 2026-06-07

- `benenzhu` 在 aiter `#3470` 说明当前只覆盖 deepseek 和 kimi shapes；同日该 PR 仍有 CI failure。

## 分仓库进度

### ROCm/aiter

#### [ROCm/aiter#3458](https://github.com/ROCm/aiter/pull/3458): perf: raise fused ar 1stage limit to cover conc64

状态：Open，未合入。  
创建日期：2026-06-01。  
变更规模：1 commit，1 file，约 +1/-1。  
Review 进度：

- `TennyWang1223` 已人工参与讨论。
- 暂无 approve。
- 作者已按 reviewer 要求上传 profiler trace。

当前进展：

- PR 目标是把 all-reduce 1-stage fused path 的 gating 从 `total_bytes <= 128K` 提高到 1M。
- Reviewer 本地复现结果和作者结论相反：fused path 在多个 shape 上更慢。
- 作者确认 standalone microbench 也能看到 regression，但实际 decode path 上不同。

卡点：

- 需要基于 profiler trace 对 `conc32 / conc64` 的真实 decode path 给出结论。
- 存在 merge conflict。
- benchmark 目前不足以说服 reviewer。

建议补充：

- 补充 conc 维度：`4, 8, 16, 32, 64, 128`。
- 补充 TP 维度：至少 TP4 / TP8。
- 同时给 microbench 和 Kimi K2.5 serving e2e。
- 表格里明确列 `1-stage fused`、`2-stage original`、latency、bandwidth、speedup、regression case。

#### [ROCm/aiter#3462](https://github.com/ROCm/aiter/pull/3462): perf: ar add dpp fast path

状态：Open，未合入。  
创建日期：2026-06-01。  
变更规模：1 commit，1 file，约 +24/-1。  
Review 进度：

- Copilot 自动 review 3 条。
- 暂无人工 approve。

卡点：

- DPP fast path 忽略 `width` 参数，`width < warp size` 时可能跨 subgroup shuffle。
- DPP control constants 是 magic numbers，需要解释或封装。
- 当前测试数据非常薄，只给了 aggregate 指标。

建议补充：

- 补 correctness unit test：不同 `width`、不同 dtype、不同 TP / rank 数。
- 补 microbench：small message sizes / small token counts 下 DPP vs baseline。
- 补 e2e：Kimi K2.5 decode path 的实际收益，避免只给单个 aggregate number。

#### [ROCm/aiter#3465](https://github.com/ROCm/aiter/pull/3465): perf: add mla metadata parallel path

状态：Open，未合入。  
创建日期：2026-06-01。  
变更规模：1 commit，1 file，约 +362/-55。  
Review 进度：

- 已请求 `minmengdie` review。
- Copilot 自动 review 4 条。
- 暂无人工 approve。
- CI 相对最好：标准测试、OPUS、格式检查等多数通过，部分扩展测试 skipped。

当前进展：

- 作者给出的 microbench：concurrency 16 / 64 / 256 分别约 8.2x、4.3x、1.7x 加速。
- Kimi K2.5 E2E profile 中 concurrency 16 约 4.3x。

卡点：

- 需要人工 reviewer 给结论。
- 需要处理 Copilot 指出的 grid launch、env var hot path overhead、文档、LDS sizing 问题。

建议补充：

- 补 conc sweep：`1, 4, 8, 16, 32, 64, 128, 256`。
- 对每个 conc 给 serial vs parallel 的 latency 和 speedup。
- 增加 bit-identical metadata correctness 测试，并说明 gating 条件。
- 增加 Kimi K2.5 e2e decode profile，至少覆盖 conc16 / 64 / 128。

#### [ROCm/aiter#3466](https://github.com/ROCm/aiter/pull/3466): perf: optimize topk kernel

状态：Open，未合入。  
创建日期：2026-06-01。  
变更规模：1 commit，1 file，约 +12/-7。  
Review 进度：

- `valarLip` 评论要求补 full test results，不要只测 token num <=256。
- Copilot 自动 review 4 条。
- 暂无 approve。

当前进展：

- 作者给出 token 1 到 256 的 microbench，约 9% 到 11% 改善。
- Kimi K2.5 e2e conc64 从 `9172ns` 到 `8343ns`，约 1.1x。

卡点：

- 需要补 token >256 的 full test results。
- AMDGCN intrinsic 需要 portability guard 或 fallback。
- fast sigmoid 的数值差异需要说明和验证。

建议补充：

- token sweep 至少覆盖：`1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096`。
- 给出 topk result stability / routing accuracy：topk index match rate、score diff、e2e accuracy。
- 给出 Kimi K2.5 / DeepSeek e2e profile，覆盖 conc64 / 128 / 256。

#### [ROCm/aiter#3468](https://github.com/ROCm/aiter/pull/3468): Add mla decode standalone kernel

状态：Open，未合入。  
创建日期：2026-06-01。  
变更规模：1 commit，11 files，约 +940。  
Review 进度：

- Copilot 自动 review 5 条。
- 暂无人工 approve。

当前进展：

- 新增 standalone `mla_decode_reduce` module。
- 作者给出的结果：
  - Kimi K2.5 conc128 rocprof：stock `7997ns` 到新实现 `6176ns`，约 1.3x。
  - Microbench conc64：stock 约 4.7us 到新实现约 2.5us，约 1.9x。

卡点：

- host entrypoint 缺 tensor dtype / layout / size 校验。
- dispatcher 对 `num_splits == 16` 的 batch 组合支持不完整。
- vector store 有 alignment / aliasing 风险。
- `.cuh` 加入 JIT `srcs` 有 build 风险。
- ATOM dispatch PR `#1018` 依赖它。

建议补充：

- 单测：torch reference correctness，覆盖不同 `num_splits`、batch、dtype、layout。
- microbench：conc `16, 32, 64, 128, 256` 下 stock vs new reduce。
- e2e：ATOM / Kimi K2.5 中打开 dispatch 后的 TTFT / TPOT / ITL / E2EL。

#### [ROCm/aiter#3470](https://github.com/ROCm/aiter/pull/3470): [PERF] MXFP4 (a4w4) MoE backend for gfx950

状态：Open，未合入。  
创建日期：2026-06-01。  
Base branch：`dev/randomflow_pr`。  
变更规模：4 commits，25 files，约 +4647/-6。  
Review 进度：

- `coderfeli` 多次 review / comment。
- `benenzhu` 参与评论。
- `lalala-sh` 参与 benchmark 口径讨论。
- 暂无 approve。

当前进展：

- 新增 native pure-HIP MXFP4 MoE backend。
- 作者给出 MI350X / Kimi-K2.5 TP4 benchmark，声称全 token range 比 FlyDSL 快约 1.12x 到 1.80x。
- benchmark 口径已被 review 推进：需要用 cudagraph / `fused_moe` interface 收集更接近实际路径的数据。

卡点：

- CI 红：Black、Ruff、Windows Smoke、多个 `check-signal` failure。
- correctness 风险：
  - `A_scale` buffer-resource bound 对 BM=64 可能小 2 倍，导致 silent wrong results。
  - topk * tokens hard limit 可能导致大 prefill 输入 wrong outputs。
  - fp4 quant 对部分 real tests 可能不够，需要 flag 控制。
- 存在 merge conflict，主要在 `aiter/fused_moe.py`。
- ATOM `#1057` 依赖这个 PR。

建议补充：

- 先修 CI 和 correctness review 点。
- benchmark 需要两层：
  - op / unit 级：`mxfp4_moe` vs FlyDSL，M sweep，包含 latency、cos、accuracy、selected kernel。
  - e2e serving 级：Kimi K2.5 / DeepSeek，conc sweep，给 TTFT / TPOT / ITL / E2EL、tok/s/user、accuracy。
- 明确列出支持范围：gfx950、shape、M upper bound、topk * tokens limit、哪些模型/shape 已验证。

### ROCm/ATOM

#### [ROCm/ATOM#1017](https://github.com/ROCm/ATOM/pull/1017): Enable DPA + TBO Support for Kimi K2.5 Inference

状态：Open，未合入。  
创建日期：2026-06-01。  
变更规模：5 commits，6 files，约 +109/-31。  
贡献者：

- PR author：`ftyghome`。
- commit author：`jpy794` 贡献 4 个 commit，`Zesen Liu / ftyghome` 贡献 1 个 commit。

Review 进度：

- 已请求 `ZhangLirong-amd`、`zejunchen-zejun`、`gbyu-amd` review。
- Copilot 自动 review 8 条。
- 暂无人工 approve。

当前进展：

- 目标是让 Kimi K2.5 inference 支持 DPA + TBO 路径。
- 改动涉及 persistent MLA、DPA fallback 下 fused MoE、TBO tensor lifetime、prefill admission 和 scheduler。

卡点：

- 存在 merge conflict。
- Copilot 指出 `_TBO_KEEPALIVE` 可能无限增长并持有 GPU tensor。
- `moe_max_num_tokens * dp_size` 可能扩大 persistent buffer，带来 OOM 风险。
- `dp_size <= 8` 是 magic number，需要解释或配置化。
- 当前没有看到完整 CI/check results。

建议补充：

- E2E 必须补：DPA on/off、TBO on/off、DP size sweep，至少覆盖 DP2 / DP4 / DP8。
- 指标建议：throughput、TTFT、TPOT、ITL、E2EL、GPU memory peak、OOM / long-run stability。
- 加 long-run test，验证 `_TBO_KEEPALIVE` 不随 ubatch id 无界增长。

#### [ROCm/ATOM#1018](https://github.com/ROCm/ATOM/pull/1018): Add HIP MLA reduce kernel dispatch logic

状态：Open，未合入。  
创建日期：2026-06-01。  
变更规模：1 commit，2 files，约 +50/-20。  
Review 进度：

- 已请求 `XiaobingSuper`、`gbyu-amd` review。
- Copilot 自动 review 2 条。
- 暂无人工 approve。

当前进展：

- 该 PR 是 aiter `#3468` 的 ATOM dispatch 侧接入。
- PR 描述里直接引用了 aiter `#3468` 的性能结果。

卡点：

- 依赖 aiter `#3468` 合入或至少接口稳定。
- unconditional import 可能导致没有 `aiter` / `mla_v_up_proj` 的环境 import 失败。
- new fast path 和 baseline path 对 `kv_buffer` 的 shape / layout 处理不同，需要明确。
- 当前没有看到完整 CI/check results。

建议补充：

- 等 aiter `#3468` 的 kernel API 和 correctness 结论稳定后再推进。
- 补 ATOM e2e benchmark：同一模型、同一 conc 下打开/关闭 `mla_reduce_decode`。
- 增加 fallback test：缺少 aiter kernel 时 ATOM 仍可 import 并走原路径。

#### [ROCm/ATOM#1057](https://github.com/ROCm/ATOM/pull/1057): perf: add preshuffle logic for HIP fused_moe

状态：Open，未合入。  
创建日期：2026-06-03。  
变更规模：1 commit，1 file，约 +48/-22。  
Review 进度：

- Copilot 自动 review 3 条。
- 暂无人工 approve。

当前进展：

- 该 PR 是 aiter `#3470` 的 ATOM 侧接入：为 HIP `mxfp4_moe` backend 增加 model load 阶段 preshuffle。
- PR 的 Test Plan / Test Result 当前都引用 aiter `#3470`。

卡点：

- 依赖 aiter `#3470` 收敛。
- `is_guinterleave` 被 hard-code 为 `True`，可能破坏原有 runtime behavior。
- `.data` 读写 / assignment 绕过 autograd 语义和 Parameter hooks。
- 缺少 ATOM 自己的 e2e benchmark 和 fallback / correctness test。

建议补充：

- 补 model-load correctness：preshuffle 前后权重 layout、scale layout、`shuffle_kind` 是否一致。
- 补 e2e：ATOM 加载 Kimi / DeepSeek 后实际走 HIP backend 的 serving benchmark。
- 明确 fallback：没有 aiter `#3470` 或非 gfx950 时仍走旧路径。

## 建议补充的 PR 测试格式

建议让相关 PR 统一补一套类似 vLLM [#39080](https://github.com/vllm-project/vllm/pull/39080) 的测试结构：既有 op / microbench，也有端到端 serving benchmark 和 accuracy。

### PR 描述建议结构

1. Motivation：说明要解决的瓶颈和影响路径。
2. Technical Details：说明改动范围、开关、fallback、支持条件。
3. Unit / op microbench：
   - before / after 对比。
   - conc 或 token sweep。
   - latency、bandwidth、speedup。
   - correctness：torch reference、cos、topk match rate 或 bit-identical。
4. E2E benchmark：
   - 模型：Kimi K2.5 / DeepSeek / GPT-OSS 等实际服务模型。
   - 配置：GPU SKU、ROCm、TP、DP、dtype、batch、max model len、backend flags。
   - conc sweep：建议至少 `4, 8, 16, 32, 64, 128, 256, 512`，按 PR 适当裁剪。
   - 指标：TTFT、TPOT、ITL、E2EL、tok/s/user、overall throughput。
5. Accuracy：
   - gsm8k 或内部 accuracy suite。
   - main vs PR 对比。
   - 对 fp4 / approximate math / topk 这类 PR 必须给 accuracy 或 route stability。
6. Test command：
   - 给完整可复现命令。
   - 给 env var、commit SHA、tuned CSV、benchmark script 链接。

### 建议统一补的 conc 表格

| PR 类型 | 建议 conc / token 范围 | 必须补充的结果 |
|---|---|---|
| all-reduce / comm path | conc `4, 8, 16, 32, 64, 128, 256`；TP4 / TP8 | 1-stage vs 2-stage latency、bandwidth、speedup、regression point |
| MLA metadata / reduce | conc `16, 32, 64, 128, 256` | stock vs new latency、correctness、e2e decode profile |
| topk / routing | token `1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096` | latency、topk match、score diff、e2e accuracy |
| MXFP4 MoE | M sweep + serving conc sweep | mxfp4_moe vs FlyDSL latency、cos、accuracy、e2e TTFT/TPOT/ITL/E2EL |
| ATOM DPA/TBO | conc sweep + DP size sweep | throughput、latency、GPU memory peak、long-run stability |
