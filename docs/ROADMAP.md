# ROADMAP：按章节分批交付

**目标**：PRML（2006）14 章的原创中文、可执行、可复现实验教程，而非原书的翻译或再分发。

## 提交策略

- `main` 只容纳已验收成果；采用独立 `feat/stage-XX-*` 分支和 PR。
- 每批次以可独立学习、运行、验收的「节」为基本单元。
- 首批只是 Ch.1.1 + 质量基础设施，**不是**完成整本第一章。
- 任何章节不得以“有 Markdown + 代码文件”替代测试通过的证据。

## Stage 01 三批交付依赖

| 批次 | 内容 | 分支 | 主线发布门禁 |
| --- | --- | --- | --- |
| Batch 01 | §1.1 多项式曲线拟合 | `feat/stage-01-foundations` | PR #1 已审阅且 CI 成功，再合并 |
| Batch 02 | §1.2–1.3 概率与模型选择 | `feat/stage-01-probability-model-selection` | PR #2 基于 Batch 01，复查依赖与 CI |
| Batch 03 | §1.4–1.6 维度灾难、决策论、信息论 | `feat/stage-01-decision-information` | 本批新增三节完整验证、PNG/JSON、CI/审阅 |

上述三个 PR 采用 stacked 依赖结构；即使 Batch 03 的全部测试通过，也**不等于**此前两批已合并、Chapter 1 已正式发布。Stage 02 进入 Ch.2 时从已发布的主线创建新分支。

| 阶段 | PRML 范围 | 必须交付 |
|---|---|---|
| 00 | 目录与规范 | 官方来源、数学验收、实验模板、CI |
| 01 | Ch.1 | 曲线拟合、概率论、模型选择、维度灾难、决策论、信息论 |
| 02 | Ch.2 | Bernoulli/Beta、Dirichlet、Gaussian、指数族、KDE |
| 03 | Ch.3–4 | 最小二乘、贝叶斯回归、Logistic、Softmax、Laplace |
| 04 | Ch.5 | MLP、反向传播、梯度检查、正则化 |
| 05 | Ch.6–7 | 核函数、GP、SVM、RVM |
| 06 | Ch.8–9 | 条件独立、因子图、消息传递、GMM、EM |
| 07 | Ch.10–11 | ELBO、Mean-Field VI、MCMC、HMC |
| 08 | Ch.12–13 | PCA、PPCA、HMM、Kalman |
| 09 | Ch.14 | 模型平均、Boosting、条件混合/MoE |
| 10 | 跨章节 | 审计、知识索引、跨章节综合应用 |

**DoD**：原创数学教程（含假设、符号和逐步推导）+ 代码 + 两种独立正确性证据 + 失败/边界案例 + 3 种固定随机种子实验（确定性实验可例外）+ 可复现实验配置/指标 + CI 通过。

**明确不做**：不收录原书全文/插图/习题解答；不在最初阶段开发数据库、Web 后端、Agent；不为凑数量而同时实现多个 ML 框架。
