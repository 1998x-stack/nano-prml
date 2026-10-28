# 1.6 信息论：熵、交叉熵、KL 散度与互信息

**原书定位**：Bishop (2006), PRML §1.6 *Information Theory*。本教程为原创重述与数值实现。

## 目标与先修

推导离散 Shannon 熵、交叉熵、KL 散度、互信息的关系，特别注意零概率与支持集不一致。先修：概率分布、对数、期望、Jensen 不等式的概念。

## 1. 离散熵

对 `C` 个类别上的分布 `P=(p_1,...,p_C)`，概率非负且求和为 1，定义

\[
H(P)=-\sum_{i=1}^{C}p_i\log p_i.
\]

约定 `0log0=0`（极限）。取自然对数单位是 **nat**，底数为 2 单位是 **bit**。公平硬币 `H([0.5,0.5])=ln2 nats = 1 bit`，确定性事件熵为 0。

## 2. 交叉熵与 KL 散度

对目标分布 `P` 和模型 `Q`，

\[
H(P,Q)=-\sum_i p_i\log q_i,
\qquad D_{\mathrm{KL}}(P\Vert Q)=\sum_i p_i\log\frac{p_i}{q_i}.
\]

代数展开得到 `H(P,Q)=H(P)+D_KL(P||Q)`。当存在 `p_i>0` 且 `q_i=0` 时，两者都应返回 `+inf`，不能用任意微小的 `epsilon` 暗改目标分布。`D_KL(P||Q)` 一般与 `D_KL(Q||P)` 不相等，不是度量距离。Gibbs 不等式表明它在合法分布下非负，且相等当且仅当两分布相同（在有限、共享支持集的常规意义下）。

KL 常见的数值验收包括：非负、同分布时为零、交叉熵恒等式、支持集不一致时为正无穷。实践中小于浮点容差的负数视为舍入误差归零，而不是理论负 KL。

## 3. 互信息

给定离散联合分布 `p(x,y)` 与边缘 `p(x),p(y)`，

\[
I(X;Y)=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)p(y)}
=D_{\mathrm{KL}}(p(X,Y)\Vert p(X)p(Y)).
\]

等价地 `I(X;Y)=H(X)+H(Y)−H(X,Y)`。当 `X` 和 `Y` 独立时互信息为 0；对于两个完全相同的公平比特，互信息为 `ln2 nats`。这里的实验只研究有限类别、**离散**概率；连续分布的微分熵和互信息估计是另外的问题。

## 4. 实现和数值稳定性

~~~python
from nano_prml.ch01.information_theory import (
    entropy, cross_entropy, kl_divergence, mutual_information
)

p, q = [0.3, 0.7], [0.8, 0.2]
assert abs(cross_entropy(p, q) - entropy(p) - kl_divergence(p, q)) < 1e-12
assert mutual_information([[0.25, 0.25], [0.25, 0.25]]) == 0
assert abs(entropy([0.5, 0.5], base=2) - 1) < 1e-12
~~~

实现仅对正 `p_i` 项取对数，因此允许零概率类别，但不会容忍输入为负数、NaN、未归一化数组或不合法的对数底数。默认返回 nats；`base=2` 返回 bits。

## 5. 数值验证与边界

- **解析 oracle**：公平比特熵为 `ln2`；独立二元联合表互信息为零；完全相关公平比特互信息为 `ln2`。
- **代数恒等式**：`H(P,Q)-H(P) = D_KL(P||Q)`；`I(X;Y) = H(X)+H(Y)-H(X,Y)`。
- **边界**：当目标有正质量而预测赋零概率时，交叉熵和 KL 必须等于 `+inf`，不能静默截断。
- **误差控制**：数值上近零的负 KL/MI 仅作为浮点舍入处理，而非允许真正的负理论值。

## 6. 原创练习

1. 手算 `P=(1/4,3/4)` 的熵，并换算到 bits。
2. 证明交叉熵和 KL 的代数关系，指出在 `Q_i=0<P_i` 时应如何定义结果。
3. 构造一个互信息为零的非均匀联合分布；与完全相关分布对照。
4. 比较 `D_KL(P||Q)` 与 `D_KL(Q||P)`；说明为何不能将其称为欧氏距离。

**来源**：PRML §1.6（`docs/REFERENCES.md` 中 BOOK-001）；推导、代码、测试均为独立原创。
