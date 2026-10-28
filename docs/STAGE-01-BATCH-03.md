# Stage 01 · Batch 03：PRML §1.4–1.6 独立验证记录

本 PR 建立在 Batch 02 的 `feat/stage-01-probability-model-selection` 上；不是 Stage 02（Ch.2）。**三层 stacked PR 的 HEAD 尚不等于 main**；只有完成审核并按序合并之后，才可在主线宣布第 1 章完成。

## 范围与论文对应

| 原书节号 | 独立教程 | 核心实现 | 解析证据 | 数值证据 |
| --- | --- | --- | --- | --- |
| §1.4 Curse of Dimensionality | `04-dimensionality.md` | `dimensionality.py` | 单位球 V1/V2/V3，球壳比率，`E||X−Y||²=d/6` | 固定 3 seed 立方体采样、距离 CV |
| §1.5 Decision Theory | `05-decision-theory.md` | `decision_theory.py` | 0–1 MAP、非对称代价 `1/6` 阈值 | 逐样本 Bayes 风险不大于 MAP、经验代价 |
| §1.6 Information Theory | `06-information-theory.md` | `information_theory.py` | `H=ln2`、交叉熵恒等式、独立/相关互信息 | 边界数值、不同对数底数与矩阵转置 |

## 实验合同

- 参数：维度 `[1,2,5,10,20,50,100]`，每维 4000 对独立均匀立方体样本；三种 seed `[0,1,2]`。
- 决策：每个 seed 20000 次；生成真实后验 `p1 ~ Uniform(0,1)` 和 `Y~Bernoulli(p1)`，使用损失矩阵 `[[0,5],[1,0]]`。
- 所有候选模型与损失矩阵先固定，不从测试标签中调参；决策结果与 MAP 共享同一组随机样本。
- 信息：独立二元均匀联合分布 `0.25` 和相关联合分布 `[[0.45,0.05],[0.05,0.45]]`。
- 机器可读报告包含 seed、Python/NumPy 版本、时间、概率模型、全部原始结果。
- 可视化使用 Matplotlib（**可选依赖**），输出三张原创图：球壳比例、距离 CV、两种决策成本。

## 工程验收

~~~bash
python -m pip install -e '.[dev,viz]'
ruff check .
python -m pytest -q
python -m nano_prml.ch01.experiment --output artifacts/ch01 --seeds 0 1 2
python -m nano_prml.ch01.probability_experiment --output artifacts/ch01/probability --seeds 0 1 2
python -m nano_prml.ch01.model_selection_experiment --output artifacts/ch01/model-selection --seeds 0 1 2
python -m nano_prml.ch01.theory_experiment --output artifacts/ch01/theory --seeds 0 1 2 --figures
~~~

- Python 3.10 和 3.12 两种 GitHub Actions 环境均应完成所有命令、产生 JSON 与 3 张 PNG，才能视为 CI 验收。
- pytest 必须覆盖解析、不变性、边界、重复 seed；拒绝 `NaN`、负概率、错误 shape 和不合法损失。
- 审核前 PR 保持 Draft；不应把 CI 通过直接等价为数学或版权人工审核通过。

## 已知限制

1. 高维实验是独立均匀 **立方体两点距离**，不是单位球或最近邻实验，不能替代一般性距离集中定理。
2. 决策实验输入是已知真实后验，不测试概率模型训练、偏移、校准或有限数据模型选择。
3. 信息论仅涵盖离散有限分布，不包含连续微分熵、估计器或神经网络信息瓶颈。
4. 没有复制 PRML 原始插图、段落或习题；教程须独立进行版权复核。

完成这批后可以规划 Stage 02 Ch.2，但必须先审阅/合并 Stage 01 依赖链。
