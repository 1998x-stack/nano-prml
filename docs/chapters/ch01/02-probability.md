# 1.2 概率论：从条件概率到稳定的贝叶斯更新

**定位**：Bishop, *Pattern Recognition and Machine Learning* (2006), Chapter 1, §1.2 Probability Theory。本文为独立原创教学材料，不翻译原书正文、示例或图。

## 目标与先修

先修：事件与集合、有限求和、对数与指数、NumPy。学习后应能区分联合、边缘、条件概率，推导 Bayes 公式，用枚举校验结果，并解释为什么真实代码需要 log-space。

## 1. 有限离散概率空间与符号

定义离散潜在假设 `H∈{h_1,…,h_K}` 和观测 `D`。先验 `π_k=p(H=h_k)` 满足 `π_k≥0` 且 `Σ_k π_k=1`。模型给出似然 `L_k=p(D|H=h_k)`。只要 `p(D)>0`：

\[
p(H=h_k\mid D)=\frac{L_k\pi_k}{\sum_{j=1}^K L_j\pi_j}.
\]

分母 `p(D)=Σ_j p(D|h_j)p(h_j)` 是**证据**，不能忽略。若全部 `L_k π_k=0`，条件概率没有定义；程序必须报告异常，不能返回 NaN 或伪造一个均匀分布。

对联合分布表 `P_ij=p(X=x_i,Y=y_j)`，有边缘化 `p(X=x_i)=Σ_j P_ij`、`p(Y=y_j)=Σ_i P_ij`，以及在 `p(Y=y_j)>0` 时 `p(X=x_i|Y=y_j)=P_ij/Σ_r P_rj`。零概率条件事件不能正常条件化。

## 2. 原创案例：二元观测如何更新假设

设 `p(H=1)=0.1`，观测为正的条件概率是 `p(D=+|H=1)=0.8`，`p(D=+|H=0)=0.05`。则：

\[
p(D=+)=0.1\times0.8+0.9\times0.05=0.125,\quad
p(H=1\mid D=+)=\frac{0.08}{0.125}=0.64.
\]

`0.8` 是似然，不是后验。用四格联合分布表独立枚举，得到相同的 `0.64`。数值是**人为设计的教学参数**，不对应任何实际医疗或安全设备。

## 3. 从直接相乘到 log-sum-exp

多个观测下若条件独立，`p(D_1,…,D_n|H)=∏_t p(D_t|H)`。概率连乘会下溢，尤其当 `n` 大时。引入 `a_k=log π_k+log L_k`：

\[
\operatorname{LSE}(a)=m+\log\!\left(\sum_k e^{a_k-m}\right),\quad m=\max_k a_k.
\]

故 `p(H=h_k|D)=exp(a_k-LSE(a))`。我们允许 `log L_k=-∞` 表示不可能观测，以及 `π_k=0` 的不可能先验；但禁止 NaN、正无穷和所有假设联合权重为零。移去最大值后所有指数的自变量均不大于零，可以避免大正数溢出。

重要数值边界：即使归一化前的非负权重接近 `1e308`，先除以最大权重再求和可避免总权重溢出。表示范围以下的相对权重仍可能下溢为零，这是浮点精度限制。

## 4. 透明 NumPy 实现

~~~python
import numpy as np
from nano_prml.ch01.probability import (
    normalize_weights, bayes_posterior, binary_positive_posterior,
    joint_to_marginals, condition_on_column,
)

prior = [0.1, 0.9]
log_likelihood = np.log([0.8, 0.05])
post = bayes_posterior(prior, log_likelihood)
assert np.allclose(post, [0.64, 0.36])
assert np.isclose(binary_positive_posterior(0.1, 0.8, 0.05), 0.64)

joint = np.array([[0.08, 0.02], [0.045, 0.855]])
px, py = joint_to_marginals(joint)
assert np.allclose(condition_on_column(joint, 0), post)
assert np.isclose(py[0], 0.125)
~~~

源代码见 `src/nano_prml/ch01/probability.py`。输入为 `float64` 的有限非负权重、1D 对数似然或 2D 联合表；返回 NumPy 概率向量，错误输入显式失败。

## 5. 频率实验：固定机制、三个随机种子

~~~bash
python -m nano_prml.ch01.probability_experiment --output artifacts/ch01/probability --seeds 0 1 2
~~~

独立生成 `N=1000,10000,100000` 个样本的前缀，统计满足观测 `D=+` 的样本中 `H=1` 的频率，并与精确解析后验 `0.64` 对照。每个种子共享一个最大样本序列，保证不同 N 结果可配对追踪。输出 `metrics.json` 包含种子、生成参数、条件事件计数、频率、绝对误差、版本与运行时间。若某个前缀没有正观测，频率设为 JSON `null`，绝不编造估计。

随着样本数量提高，估计的波动**通常**变小，但单次随机轨迹的绝对误差不必逐点单调递减；不得使用一次实验“证明收敛速度必然单调”。

## 6. 双重独立验收与边界

- **解析 oracle**：四格联合表的精确条件概率 `0.64`、全概率公式、边缘化和归一化。
- **数值性质**：对假设标签作同一置换，后验按相同置换变化；极小 log-likelihood 不发生全零下溢；概率之和为一。
- **反例**：全零权重、零证据、负联合概率、错误维度、NaN、正无穷、零概率条件事件均抛异常。

## 7. 原创习题

1. 给定三种假设的先验 `[0.2,0.3,0.5]` 与观测似然 `[0.5,0.1,0.3]`，手算证据和后验，验证代码结果。
2. 证明同一个数据集下，所有似然同时乘以同一正数不会改变后验；解释为何这是模型无关的归一化性质。
3. 将所有 log-likelihood 减去 `10000`，验证 log-space 后验不变；再用直接 `exp` 展示下溢问题。
4. 修改实验的假设先验，解释为何“观测准确”不等于“观测后某假设一定成立”。

**来源**：`docs/REFERENCES.md` 中 BOOK-001、OFFICIAL-001/002；所有数值与练习均原创。
