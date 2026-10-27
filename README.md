# nano-prml

**从原理到可复现实验的中文概率机器学习教程**  
*An original, testable companion to Christopher M. Bishop's Pattern Recognition and Machine Learning (2006).*

本项目以 PRML 的 **14 章**为知识地图，采用「问题 → 假设 → 推导 → NumPy 从零实现 → 独立验证 → 实验 → 习题」的章节结构；不复制原书正文、习题或插图。

## 项目状态

- `main`：稳定、已验收内容；分阶段通过分支和 Pull Request 提交。
- 第一期：工程基础、章节规范、Chapter 1.1 多项式曲线拟合的可运行示例。
- 后续阶段和门禁详见后续 PR 中的 `docs/ROADMAP.md`、`docs/ACCEPTANCE.md`。

## 权威资料

- [PRML 作者主页、勘误与习题资源](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
- [官方目录与第 8 章样章](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/Bishop-PRML-sample.pdf)
- [Dive into Deep Learning（仅借鉴交互式教程形式）](https://d2l.ai/)

## 内容原则

1. 数学公式注明符号、假设、维度和适用范围；纠错需记录依据。
2. 教学实现尽量透明，生产级数值计算避免显式矩阵求逆。
3. 每个实验固定种子、记录指标和软件版本，并在 CI 中校验。
4. 数学正确性、教学清晰性与性能宣称分别验收，不使用单次运行推断普遍结论。
5. 所有正文和图表须原创；参考他人成果时标明出处与许可。

## 许可证

首批教程代码与原创文本拟采用 MIT（后续 PR 提供 LICENSE）。第三方教材的版权不因此改变。
