# 实验记录与复现指南

本文档整理了在 Apple Silicon (M4 Pro) 上针对本项目进行的关键实验、复现脚本以及主要结论，方便研究与工程人员对照论文结果或进一步扩展。

## 1. 基础确定性验证

脚本：`benchmarks/determinism_benchmark.py`

- **目的**：比较原生 PyTorch 与批次不变算子在矩阵乘、`log_softmax`、`mean` 等操作下的数值差异。
- **方法**：对同一批数据分别以“单样本 + 整批”两种方式计算，统计最大绝对误差，并测量执行耗时。
- **结果摘要**：

  | op          | shape            | eager diff | det diff | eager ms | det ms |
  |-------------|------------------|-----------:|---------:|---------:|-------:|
  | mm          | (32, 256, 256)   | 1.53e-05   | 0.00e+00 | 39.18    | 3.79   |
  | mm          | (64, 512, 512)   | 3.43e-05   | 0.00e+00 | 1.29     | 1.03   |
  | mm          | (128, 768, 768)  | 7.63e-05   | 0.00e+00 | 18.32    | 2.21   |
  | log_softmax | (32, 512)        | 0.00e+00   | 0.00e+00 | 184.54   | 1.03   |
  | log_softmax | (64, 1024)       | 0.00e+00   | 0.00e+00 | 1.95     | 1.04   |
  | mean        | (16, 64, 128)    | 2.79e-09   | 0.00e+00 | 92.06    | 1.73   |
  | mean        | (32, 64, 256)    | 3.49e-09   | 0.00e+00 | 49.72    | 1.90   |

- **结论**：批次不变算子可完全消除批次相关的数值偏差；在需要精确重现的推理任务中可作为直接替代方案。

## 2. 随机种子敏感性研究

脚本：`benchmarks/seed_sensitivity.py`

- **目的**：考察不同随机种子对原生数值差异的影响，并验证 `BatchInvariantConfig(manual_seed=...)` 的运行时保护机制。
- **代表输出**：

  ```
  Seed sensitivity on mps
  seed    eager diff    det diff
     0      3.81e-06    0.00e+00
     1      3.81e-06    0.00e+00
     7      3.81e-06    0.00e+00
    42      5.72e-06    0.00e+00
  1337      5.72e-06    0.00e+00

  Manual-seed experiment:
    outside restored: True
    inside deterministic: True
  ```

- **结论**：原生 diff 随种子变化而波动，而批次不变模式保持 0。启用 `manual_seed` 时，进入上下文会设置固定种子并在退出时自动恢复先前状态。

## 3. 文本生成的一致性对照

- **验证思路**：
  1. 文本输出的一致性受采样策略主导（例如 `temperature=0.0`、固定 `seed`）。
  2. 批次不变算子解决的是底层数值层面的非确定性。
- **参考脚本**：
  - `examples/lm_studio_client.py`：调用 LM Studio Embeddings API，演示“单行 vs 整批”在批次不变模式下差异归零。
  - 可配合 LM Studio 的 `temperature=0.0`+固定 `seed` 配置，实现端到端可重现的输出。

## 4. 单元测试覆盖

命令：`pytest tests -q --override-ini addopts=''`

- 验证点包括：
  - 核心算子 (`mm/addmm/log_softmax/mean`) 数值匹配参考实现。
  - 批次不变性检查（单行与整批结果一致）。
  - `BatchInvariantConfig` 的 `force_cpu`、`precision`、`manual_seed`、`enforce_deterministic_algorithms` 行为及恢复机制。
  - 非法精度的容错处理。

## 5. 复现实验步骤总览

1. 准备环境：
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e .[full]
   ```
2. 单元测试：`pytest tests`
3. 数值&性能基准：`python benchmarks/determinism_benchmark.py`
4. 随机种子实验：`python benchmarks/seed_sensitivity.py`
5. （可选）LM Studio 集成：参照 `examples/` 并确保本地服务已开启

如需进一步对比更多算子或引入自定义工作负载，可参考上述脚本改写，或扩展 `BatchInvariantConfig` 以支持自定义策略。
