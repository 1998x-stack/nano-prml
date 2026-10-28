# REFERENCES：第一手来源账本

| ID | 原始来源 | 使用范围 | 版权处理 |
|---|---|---|---|
| BOOK-001 | Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006 | 14 章章节主线、数学模型 | 只定位节号和核实数学，不复制原文 |
| OFFICIAL-001 | https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/ | 官方勘误、补充材料 | 只链接 |
| OFFICIAL-002 | https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/Bishop-PRML-sample.pdf | 官方完整目录、第 8 章样章 | 只链接 |
| NUMPY-001 | https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html | OLS 数值计算语义 | 官方 API 文档 |
| NUMPY-002 | https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html | 后验线性求解 | 官方 API 文档 |
| NUMPY-003 | https://numpy.org/doc/stable/reference/random/generator.html | NumPy 独立实验 RNG、seed 语义 | 官方 API 文档 |
| PEDAGOGY-001 | https://d2l.ai/ | 可执行教材的组织思想 | 不复制文字/图片/代码 |

**首批原创样例**：Ch.1 `1.1 Polynomial Curve Fitting`，以及增量 `1.2 Probability Theory` 的二元贝叶斯示例和 `1.3 Model Selection` 的 K-fold/LOOCV 实验。全部采用可重现合成数据，无外部下载或第三方数据授权问题。

后续引入原始论文时登记：作者、年份、DOI/arXiv、原文链接、验证的主张、文中使用位置、许可。

## Stage 01 · Batch 03 独立验证来源

| ID | 来源 | 本批用途 | 版权处理 |
| --- | --- | --- | --- |
| BOOK-001 | Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006, §§1.4–1.6 | 维度灾难、决策论、信息论节号与概念定位 | 不再分发任何原书正文、图表或习题 |
| NUMPY-RNG | https://numpy.org/doc/stable/reference/random/generator.html | 默认生成器与固定 seed 随机实验 | 官方 API 引用 |
| PYTHON-MATH | https://docs.python.org/3/library/math.html#math.lgamma | 高维球体积的 log-Gamma 求值 | 官方 API 引用 |
| MPL-001 | https://matplotlib.org/stable/users/index.html | 生成原创参数实验图片 | 官方 API 引用 |

所有图表使用程序生成的合成数据；这批未使用额外第三方数据集。概念严格停留在 PRML 原始第 1 章范围内。
