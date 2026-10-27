# 1.3 模型选择：为什么测试集不能参与调参

**定位**：Bishop, *Pattern Recognition and Machine Learning* (2006), Chapter 1, §1.3 Model Selection。本文为独立原创教学材料。

## 目标与先修

先修：§1.1 最小二乘多项式回归、训练误差与泛化误差、§1.2 概率基本术语。目标：说明训练误差不适合直接选择复杂度；实现不会泄漏测试集的 K 折交叉验证；对照可手算的 LOOCV oracle；评价运行复杂度和边界。

## 1. 为什么需要模型选择

数据集来自 `t=f(x)+ε`，其中噪声的单次实现未知。若增加多项式次数 `M`，训练平方误差通常不会上升，因为高阶模型的假设空间包含低阶模型；但未来样本误差未必降低。这产生模型复杂度与泛化之间的权衡。一个模型参数已用训练数据拟合，不能仅凭其训练误差选超参数。

区分三种角色：**训练**用于拟合权重；**验证**用于选择候选超参数；**最终独立测试**只评估已锁定流程的泛化表现。若重复查看测试集并据此调整候选、特征或 seed，测试集即发生泄漏。

## 2. K 折交叉验证的数学定义

令开发数据 `D={(x_i,t_i)}_{i=1}^N` 划分为互不相交的 `K` 个索引集合 `V_1,…,V_K`，其并集等于全集。第 k 次拟合只使用 `D\V_k`，得到 `w_{M,-k}`；再计算该折验证均方误差：

\[
E_k(M)=\frac1{|V_k|}\sum_{i\in V_k}
[t_i-y_M(x_i;w_{M,-k})]^2.
\]

当前项目的选择指标是各折**等权平均**：

\[
\mathrm{CV}(M)=\frac1K\sum_{k=1}^K E_k(M),\quad
M^*=\operatorname*{argmin}_{M\in\mathcal M}\mathrm{CV}(M).
\]

若折大小不相等，此值与按每个样本等权汇总的总体 MSE 不完全相同；本实现明确记录各折误差，且 K 折大小最多差 1。候选模型共享同一组折，以减小对比实验的划分噪声。出现**完全相等**的 CV 分数时按较小次数优先；极近但不相等的浮点数不视为平局。

## 3. 留一法（LOOCV）的可手算 Oracle

取常数模型 `y(x)=μ`，第 i 个样本被留出后的均值为

\[
\hat μ_{-i}=\frac{\sum_j t_j-t_i}{N-1},\quad
(t_i-\hat μ_{-i})^2=\left(\frac{N}{N-1}\right)^2(t_i-\bar t)^2.
\]

因此对 `degree=0, K=N` 的交叉验证，应有

\[
\mathrm{CV}_{LOO}(0)=\left(\frac{N}{N-1}\right)^2\cdot\frac1N\sum_i(t_i-\bar t)^2.
\]

它完全不依赖 sklearn 或其它现成 CV 库，是本节独立的数学正确性基准。有限样本下 `N≥2`。

## 4. 代码 API 和明确的数据边界

~~~python
import numpy as np
from nano_prml.ch01.model_selection import cross_validate_polynomials
from nano_prml.ch01.polynomial import fit_least_squares, predict

x_train = np.linspace(-1, 1, 60)
t_train = np.sin(np.pi*x_train)
selection = cross_validate_polynomials(
    x_train, t_train, degrees=(1, 3, 5, 9), n_folds=5, seed=0
)
# selection 中不可能访问未传入的 test 数据。
weights = fit_least_squares(x_train, t_train, selection.selected_degree)
# 到这里再锁定测试集并做最终报告；不要循环查看 test 后重新选模型。
~~~

`make_folds` 返回可枚举的每折验证索引，测试覆盖“无重复、无遗漏、同种子稳定”。`cross_validate_polynomials` 同时返回每个候选的五折 MSE、均值、标准差，以及被选中的模型次数。参考实现使用 Ch.1.1 的 NumPy OLS；不额外引入 sklearn。

## 5. 三种随机种子与评价协议

~~~bash
python -m nano_prml.ch01.model_selection_experiment --output artifacts/ch01/model-selection --seeds 0 1 2
~~~

生成机制：`x~Uniform[-1,1]`，`t=sin(πx)+ε`，`ε~N(0,0.1²)`；开发集 `N=60`、独立最终测试集 `N=300`。候选次数固定为 `[1,3,5,9]`，`K=5`。**先**完成交叉验证并确定次数，**再**生成和读取测试数据；测试指标仅记录被选中的模型以及事先指定的一阶基线。输出原始每折 CV 分数、最终测试 MSE、基线 MSE、配置、种子与版本。

对照并不意味着“CV 胜者的 test MSE 一定优于一阶基线”；任何过度宣称都是错误的。各候选训练 `K` 次并最终重训一次，计算开销约为 `(候选数 × K + 1)` 次求解，不包含数据预处理费用。

## 6. 独立验收和失败模式

- **解析证据**：`degree=0` 的 LOOCV 平方误差与上面的闭式恒等式一致。
- **结构性质**：每个样本恰好一次被留出，训练/验证在每折互斥；更换独立测试集内容不会改变只接收开发数据的选择 API。
- **可复现性**：同种子完全重复，变换种子验证各折变化；三个固定 seed 保存完整候选 CV 分数。
- **边界**：`K>N`、非法次数、重复次数、非有限输入、维度不匹配均显式失败。
- **统计局限**：单次交叉验证均值有估计方差，挑选 CV 最小值本身带选择偏差。对外估计模型选择流程的泛化，需要最终独立测试集或嵌套交叉验证；本阶段采用独立测试集。

## 7. 原创习题

1. 证明增加多项式最高次数后，未经正则化的最优训练残差平方和不会增加；说明为何这一结论不能推广到测试误差。
2. 手算 4 个样本的常数模型 LOOCV，再与公式和代码对比。
3. 将五折改为两折和留一折，统计三种 seed 下候选 MSE 的波动，并解释方差来源。
4. 人为构造数据泄漏：用测试误差选次数，说明该做法为什么会对最终泛化评估形成乐观偏差（不要把这种结果当作正式模型分数）。

**来源**：`docs/REFERENCES.md` 中 BOOK-001、OFFICIAL-002、NUMPY-001。
