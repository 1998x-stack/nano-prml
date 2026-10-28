# nano-prml

**从数学原理到可复现实验的中文概率机器学习教程**  
*Original executable companion to Christopher M. Bishop's Pattern Recognition and Machine Learning (2006).*

按 PRML 14 章构建原创中文教程：**问题 → 假设 → 推导 → NumPy 从零实现 → 独立验证 → 实验 → 习题**。不会复制原书正文、插图与习题解答。

## 快速开始

~~~bash
python -m pip install -e ".[dev]"
ruff check .
python -m pytest -q
python -m nano_prml.ch01.experiment --output artifacts/ch01 --seeds 0 1 2
~~~

## 从这里开始

- [第一节：多项式曲线拟合](docs/chapters/ch01/01-polynomial.md)
- [阶段计划与章节地图](docs/ROADMAP.md)
- [数学、实验与 CI 验收标准](docs/ACCEPTANCE.md)
- [教程创作规范](docs/AUTHORING.md)
- [原始资料和版权记录](docs/REFERENCES.md)
- [可复制的分阶段执行 Prompt](prompts/STAGE_EXECUTION.md)

当前 `main` 为稳定入口，首批课程和测试在 `feat/stage-01-foundations` PR 中待审阅；PR 通过后再合并。

## 一手参考资料

- [PRML 作者官方资源与勘误](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
- [官方完整目录及第 8 章样章](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/Bishop-PRML-sample.pdf)
- [Dive into Deep Learning](https://d2l.ai/) — 仅参考可执行教材格式。

本项目 MIT 许可仅适用于仓库的原创内容，原书和第三方材料版权归原权利人。
