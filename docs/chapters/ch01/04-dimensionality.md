# 1.4 维度灾难：空间体积、边界质量与距离集中

**原书定位**：Bishop (2006), PRML §1.4 *The Curse of Dimensionality*。本文为独立编写的补充教程，不是原书翻译。

## 目标与先修

解释什么是维度灾难，以及为何**样本数量、数据分布、几何结构、模型假设**都会改变结论。先修：概率分布、期望与方差、欧氏范数、Gamma 函数（只需知道 `Γ(n+1)=n!`）。

## 1. 单位球体积：维度不是体积的单调放大器

在 `d>=1` 维欧氏空间中，半径为 `r` 的球体积为

\[
V_d(r)=\frac{\pi^{d/2}}{\Gamma(d/2+1)}r^d.
\]

因为 `r^d` 的比例关系，半径为 `1−ε` 的内球占单位球体积 `(1−ε)^d`。故外侧径向厚度为 `ε` 的球壳体积分数为

\[
P(1-\epsilon<\|X\|\le1)=1-(1-\epsilon)^d,
\qquad X\sim\mathrm{Uniform}(\text{unit ball}).
\]

当 `0<ε<1` 固定时，该比例随 `d` 增大而趋近 1。这是**单位球均匀分布、径向厚度固定**下的结论，不代表一切高维数据都分布在边界。数值计算避免先算两个接近 1 的数再相减，代码使用 `-expm1(d*log1p(-ε))`。

球体积本身在较大维度会非常小，不能用普通浮点 `V_d` 等于 0 来推断理论体积为 0。提供 `log_unit_ball_volume` 用 `lgamma` 计算对数体积。

## 2. 立方体均匀样本：距离的精确二阶矩

取两点 `X,Y` 独立同分布于 `[0,1]^d` 的均匀分布。每个坐标独立且有 `Var(X_i)=1/12`，因此

\[
\mathbb{E}(X_i-Y_i)^2=2\operatorname{Var}(X_i)=\frac16,\quad
\mathbb{E}\|X-Y\|_2^2=\frac{d}{6}.
\]

注意 `E||X−Y||² = d/6` **不等于** `(E||X−Y||)²`。通过独立成对随机采样验证二阶矩，同时观察距离标准差与平均距离之比 `CV=SD(R)/E[R]` 随维度变化。不能将成对距离的集中趋势直接等同于最近邻距离集中；后者需要固定样本规模与独立的最近邻实验。

## 3. 解释与边界

- 对全空间光滑函数做网格离散时，每坐标 `m` 个格点共需 `m^d` 个格点，是典型指数成本；实际应用中的结构性先验、流形假设和稀疏性会改变样本需求。
- 高维小体积、质量向径向外壳集中、距离 CV 变化是**不同的数学命题**，不应混为一谈。
- 立方体模型的 `d/6` oracle 不适用于高斯分布、相关坐标、归一化球面数据。
- 实验独立地改变维度，不固定“单位球”和“单位立方体”为同一采样分布。

## 4. 代码、实验与图

~~~python
from nano_prml.ch01.dimensionality import (
    log_unit_ball_volume, unit_ball_shell_mass, pairwise_distance_stats
)

assert abs(unit_ball_shell_mass(2, 0.1) - 0.19) < 1e-14
stats = pairwise_distance_stats(10, 20000, seed=7)
print(stats.mean_squared_distance, stats.expected_squared_distance)
print(log_unit_ball_volume(1000))   # finite even when direct volume underflows
~~~

~~~bash
python -m nano_prml.ch01.theory_experiment --output artifacts/ch01/theory --seeds 0 1 2 --figures
~~~

输出 `metrics.json`、`shell_mass.png`、`distance_concentration.png`，包括每个维度的真实 Monte Carlo 指标和对应解析值；三种 seed 完整保存，不筛选最漂亮的曲线。上述命令需要 `pip install -e '.[dev,viz]'`。

## 5. 验证与原创练习

1. 证明 `V_1(1)=2`、`V_2(1)=π`、`V_3(1)=4π/3`，与代码进行 float64 对照。
2. 推导外侧球壳概率；解释小 `ε` 时直接相减可能丢失有效位。
3. 手动从 `Var(U)=1/12` 推出 `E||X−Y||²=d/6`；比较 1、10、100 维独立采样，记录 Monte Carlo 误差。
4. 在相同样本量下重复使用高度相关坐标，说明独立坐标假设失效时实验结论如何变化（**扩展习题，不在本批实现**）。

**来源**：PRML §1.4（`docs/REFERENCES.md` 中 BOOK-001）；几何恒等式和实验均为独立实现。
