# Stage 01 · Batch 02：PRML §1.2–1.3 交付与验收

本批为 `feat/stage-01-foundations` 的增量 stacked PR；目标分支在 PR #1 合并后再调整为 `main`，不可在主线重复覆盖相同文件。

## 范围

- §1.2：联合/条件/边缘概率、全概率与贝叶斯定理、log-sum-exp、零概率事件。
- §1.3：模型选择、K-fold、LOOCV 闭式基准、训练/验证/测试边界。
- 不预支 §1.4–1.6 或 Chapter 2；这两个主题仍属于 Stage 01。

## 数学/代码契约

| 条目 | 验证 | 门限 |
| --- | --- | --- |
| 贝叶斯结果 | 独立四格联合分布枚举 | 绝对误差 `<=1e-14` |
| log-space | 输入 `-10000` 对数似然 | 返回归一化有限后验 |
| 边缘与条件概率 | 两轴求和；条件零质量拒绝 | `allclose` + `ValueError` |
| K-fold 索引 | `N` 个样本一次且仅一次验证 | 严格集合相等 |
| LOOCV | 常数模型闭式恒等式 | 相对误差 `<=1e-12` |
| 数据泄漏 | 模型选择 API 无 test 参数 | 独立测试仅最终评估 |
| 重现性 | 0/1/2 三个随机种子 | 同 seed 测试严格相等 |
| 健壮性 | 非法维度、重复候选、非有限数字、全零证据 | 显式失败 |

## 运行命令

~~~bash
python -m pip install -e '.[dev]'
ruff check .
python -m pytest -q
python -m nano_prml.ch01.experiment --output artifacts/ch01 --seeds 0 1 2
python -m nano_prml.ch01.probability_experiment --output artifacts/ch01/probability --seeds 0 1 2
python -m nano_prml.ch01.model_selection_experiment --output artifacts/ch01/model-selection --seeds 0 1 2
~~~

`artifacts/ch01/probability/metrics.json` 与 `artifacts/ch01/model-selection/metrics.json` 属运行产物，不能把未经执行的预期数字写作结果。Stage 01 全章须在 §1.4–1.6 完成并经审核后才转 Stage 02。

## 质量门禁

必须通过 Python 3.10 和 3.12 的 GitHub Actions、检查 PR diff 与依赖、Review 后再合并。对于未通过或尚未执行的检查，PR 保持 Draft。不要在 CI 成功之前宣称仓库验收完成。
