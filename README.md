# nano-prml

**从数学原理到可复现实验的中文概率机器学习教程**  
*Original executable companion to Christopher M. Bishop's Pattern Recognition and Machine Learning (2006).*

按 PRML 14 章构建原创中文教程：**问题 → 假设 → 推导 → NumPy 从零实现 → 独立验证 → 实验 → 习题**。不会复制原书正文、插图与习题解答。

## 快速开始

~~~bash
python -m pip install -e ".[dev,viz]"
ruff check .
python -m pytest -q
python -m nano_prml.ch01.experiment --output artifacts/ch01 --seeds 0 1 2
python -m nano_prml.ch01.probability_experiment --output artifacts/ch01/probability --seeds 0 1 2
python -m nano_prml.ch01.model_selection_experiment --output artifacts/ch01/model-selection --seeds 0 1 2
python -m nano_prml.ch01.theory_experiment --output artifacts/ch01/theory --seeds 0 1 2 --figures
~~~

## 从这里开始

- [1.1 多项式曲线拟合](docs/chapters/ch01/01-polynomial.md)：OLS、Ridge、Gaussian Bayesian Regression
- [1.2 概率论](docs/chapters/ch01/02-probability.md)：贝叶斯定理、联合/边缘概率、稳定后验
- [1.3 模型选择](docs/chapters/ch01/03-model-selection.md)：K-fold、LOOCV、避免测试泄漏
- [1.4 维度灾难](docs/chapters/ch01/04-dimensionality.md)：高维球体积、径向质量、距离集中
- [1.5 决策论](docs/chapters/ch01/05-decision-theory.md)：损失矩阵、最小贝叶斯风险、拒绝决策
- [1.6 信息论](docs/chapters/ch01/06-information-theory.md)：熵、交叉熵、KL、互信息
- [Stage 01 Batch 03 验收表](docs/STAGE-01-BATCH-03.md)
- [Stage 01 Batch 02 验收表](docs/STAGE-01-BATCH-02.md)
- [阶段计划与章节地图](docs/ROADMAP.md)
- [数学、实验与 CI 验收标准](docs/ACCEPTANCE.md)
- [教程创作规范](docs/AUTHORING.md)
- [原始资料和版权记录](docs/REFERENCES.md)
- [可复制的分阶段执行 Prompt](prompts/STAGE_EXECUTION.md)

`main` 只容纳经过验收的成果。Stage 01 三批分别覆盖 §1.1、§1.2–1.3、§1.4–1.6，采用 stacked PR。**所有六节在功能分支齐备不意味着已在主线发布**；CI 与人工 Review、按序合并后再宣布 Chapter 1 完成。Stage 02 (Ch.2) 不属于本批。

图表可视化需要可选依赖 `pip install -e '.[viz]'`，默认 NumPy 算法不依赖 Matplotlib。原始实验产物写入 `artifacts/ch01/`（通常不提交 Git）。

## 一手参考资料

- [PRML 作者官方资源与勘误](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
- [官方完整目录及第 8 章样章](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/Bishop-PRML-sample.pdf)
- [NumPy 数值线性代数接口](https://numpy.org/doc/stable/reference/routines.linalg.html)

本项目 MIT 许可仅适用于仓库的原创内容，原书和第三方材料版权归原权利人。
