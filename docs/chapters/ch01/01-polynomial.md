# 1.1 多项式曲线拟合：从过拟合到贝叶斯预测

**原书定位**：Bishop, PRML (2006), `1.1 Example: Polynomial Curve Fitting。本文为独立创作，不复制原书正文或插图。

## 目标与准备

读完本节，能够推导带 Gaussian 噪声的最小二乘问题、解释模型容量的影响、写出 Ridge 与 Bayesian Linear Regression 的求解式，并独立验证预测方差的两种来源。先修：向量与矩阵乘法、概率密度、Gaussian 分布、NumPy。

## 1. 问题和符号

给定训练数据 `x∈R^N` 和 `t∈R^N`，用次数为 `M` 的多项式近似未知信号：

$$
y(x,\mathbf w)=\sum_{j=0}^{M}w_jx^j,\qquad
t_n=y(x_n,\mathbf w)+\epsilon_n
$$

假设噪声相互独立且 `ε_n ~ N(0, β^{-1})`，其中 `β > 0` 是噪声精度。

引入设计矩阵 `Φ∈R^{N×(M+1)}`，第 `n` 行为 `[1,x_n,...,x_n^M]`。则 `t=Φw+ε`。模型虽然对输入 `x` 非线性，但对于系数 `w` 是线性的。

## 2. 推导：Gaussian 似然到最小二乘

独立噪声假设得到：

$$
\log p(\mathbf t\mid\mathbf w,\beta)
=-\frac N2\log(2\pi\beta^{-1})
-\frac\beta2\|\mathbf t-\Phi\mathbf w\|^2
$$

对 `w` 最大化对数似然等价于最小化残差平方和；求导并令其为零：

$$
\Phi^\top\Phi \mathbf w=\Phi^\top\mathbf t.
$$

注意正规方程**不保证** `ΦᵀΦ` 可逆。参考代码使用 `numpy.linalg.lstsq`，不会显式计算 Gram 矩阵的逆。

## 3. 推导：带有正则化的 Ridge

我们定义以下约定：**截距不参与惩罚**。

$$
\min_\mathbf w\|\Phi\mathbf w-\mathbf t\|^2
+\lambda\sum_{j=1}^{M}w_j^2,\qquad\lambda\ge0.
$$

令 `R=diag(0,1,...,1)`，则系数满足：

$$
(\Phi^\top\Phi+\lambda R)\mathbf w=\Phi^\top\mathbf t.
$$

代码通过增广最小二乘完成求解。第 4 节采用的 Bayesian 先验对**所有**参数（含截距）施加 Gaussian 先验，属于不同且明确声明的建模约定。

## 4. 推导：权重后验和预测方差

假设 `w~N(0,α^{-1}I)`，精度 `α>0`。Bayesian 后验满足：

$$
S_N^{-1}=\alpha I+\beta\Phi^\top\Phi,\quad
m_N=\beta S_N\Phi^\top\mathbf t.
$$

对新样本 `x_*`，特征向量为 `φ_*=[1,x_*,...,x_*^M]ᵀ`。函数预测：

$$
\mathbb E[f_*]=\phi_*^\top m_N,\quad
\operatorname{Var}(f_*)=\phi_*^\top S_N\phi_*.
$$

若预测**新的观测**而非无噪声函数，则：

$$
\operatorname{Var}(t_*)=\phi_*^\top S_N\phi_*+\beta^{-1}.
$$

这区分了参数不确定性与观测噪声。模型不满足假设时，上述方差不能简单当成一般意义的“真实置信度”。

## 5. 从零实现

~~~python
import numpy as np
from nano_prml.ch01.polynomial import (
    fit_least_squares, fit_ridge, fit_bayesian, predict, predict_bayesian
)

rng = np.random.default_rng(7)
x = np.linspace(0, 1, 12)
t = np.sin(2*np.pi*x) + rng.normal(0, 0.1, len(x))
ols = fit_least_squares(x, t, degree=3)
ridge = fit_ridge(x, t, degree=3, penalty=1e-3)
posterior = fit_bayesian(x, t, degree=3, alpha=1.0, beta=100.0)
queries = np.linspace(0, 1, 50)
mean, noisy_var = predict_bayesian(queries, posterior, include_noise=True)
assert np.all(noisy_var > 0)
~~~

实现参见 `src/nano_prml/ch01/polynomial.py`。默认用 float64 和有限实数一维输入，病态矩阵会影响模型系数与泛化；`lstsq` 返回的是求解器选择的最小范数解。

## 6. 可复现实验

~~~bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m nano_prml.ch01.experiment --output artifacts/ch01 --seeds 0 1 2
~~~

使用固定生成机制 `sin(2πx)+ε`（`ε~N(0,0.1²)`），训练数据为 12 个均匀位置，测试数据为 250 个独立生成点；对同一拆分分别拟合 1、3、9 阶多项式。

输出 `artifacts/ch01/metrics.json`，记录训练/测试 MSE、设计矩阵条件数、样本规模、Python/NumPy 版本、随机种子及时间。测试集仅用于评价固定候选模型，不用于选择超参数。

**观察要点**：高阶多项式在一些小样本设置中容易更复杂且数值病态，但没有“9 阶必然过拟合”的定理。记录实际结果而不是预先宣布胜者。

## 7. 验证与反例

- **解析 oracle**：无噪三次多项式恢复已知系数；零阶 Bayesian 与 Gaussian 共轭公式完全一致。
- **性质检验**：Ridge 截距免惩罚、prediction variance 非负、带噪方差=无噪方差+噪声方差。
- **数值验证**：MSE 梯度和中心有限差分一致；秩亏设计矩阵仍返回有限预测。
- **失败模式**：高阶 Vandermonde 条件数上升、训练误差与泛化误差背离、训练域外推、过强模型假设。

## 8. 原创习题

1. 从 Gaussian 似然推导正规方程，说明为何不建议直接显式求逆。
2. 用有限差分验证 `∇_w MSE=2Φᵀ(Φw-t)/N`，并研究步长过小造成的舍入误差。
3. 推导 degree=0 时带 Gaussian 先验的后验均值与方差。
4. 比较预测函数值的方差与预测未来观测的方差，并解释二者不同的来源。
5. 修改数据量与随机种子，观察 1/3/9 阶模型的训练与测试误差，禁止只保留“好看”的图。

**来源**：`docs/REFERENCES.md` 中 BOOK-001、OFFICIAL-002、NUMPY-001/002。
