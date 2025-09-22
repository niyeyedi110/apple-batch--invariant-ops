# Batch Invariant Ops (Apple Silicon Edition)

面向 Apple Silicon (M 系列) 的批次不变 PyTorch 扩展。灵感来自 [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)，在无 CUDA/Triton 的条件下重现论文提出的确定性推理思路，并针对 MPS/CPU 场景完成工程化改造。

- **算子级劫持**：运行时用 `torch.library` 改写 `aten::mm`、`aten::addmm`、`aten::_log_softmax`、`aten::mean.dim`。
- **可配置精度**：默认 CPU `float64` 保证稳定舍入，也可选择 `float32` 或保留原设备。
- **运行时防护**：支持强制启用 PyTorch 确定性算法、固定随机种子并自动恢复。
- **MPS 友好**：自动避开 MPS 对 `float64` 的限制，可直接用于 LM Studio 等 Apple GPU 推理流程。

## 安装

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .          # 核心功能
pip install -e .[dev]     # 开发 + 测试（pytest、numpy）
pip install -e .[integration]  # LM Studio 集成示例需要 requests
```

依赖 `torch>=2.2`，推荐在 macOS 14+/Apple M 系列机器上使用。

## 快速上手

```python
import torch
from apple_batch_invariant_ops import set_batch_invariant_mode

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
a = torch.randn(128, 512, device=device)
b = torch.randn(512, 512, device=device)

with set_batch_invariant_mode():
    out_single = torch.mm(a[:1], b)
    out_full = torch.mm(a, b)[:1]

print("max diff", (out_single - out_full).abs().max().item())  # -> 0.0
```

### 高级配置

```python
from apple_batch_invariant_ops import BatchInvariantConfig, set_batch_invariant_mode

config = BatchInvariantConfig(
    precision=torch.float32,
    force_cpu=False,
    enforce_deterministic_algorithms=True,
    manual_seed=42,
)

with set_batch_invariant_mode(config=config):
    logits = torch.log_softmax(x, dim=-1)
```

## 实验结果（M4 Pro, macOS 15, PyTorch 2.8）

```
Batch-invariant benchmark on mps
op          shape               eager diff          det diff        eager ms            det ms
----------------------------------------------------------------------------------------------
mm          (32, 256, 256)        1.53e-05          0.00e+00           39.18              3.79
mm          (64, 512, 512)        3.43e-05          0.00e+00            1.29              1.03
mm          (128, 768, 768)       7.63e-05          0.00e+00           18.32              2.21
log_softmax (32, 512)             0.00e+00          0.00e+00          184.54              1.03
log_softmax (64, 1024)            0.00e+00          0.00e+00            1.95              1.04
mean        (16, 64, 128)         2.79e-09          0.00e+00           92.06              1.73
mean        (32, 64, 256)         3.49e-09          0.00e+00           49.72              1.90
```

- `eager diff`：原生算子在“单样本切片 vs 整批”两条路径下的最大误差。
- `det diff`：启用批次不变模式后误差降到 0。
- `det ms`：批次不变模式耗时，部分算子回落到 CPU `float64`，吞吐低于 MPS，但换来完全可复现的结果。

更多实验（随机种子敏感性、控制变量对比等）见 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)。

## LM Studio 集成示例

`examples/lm_studio_client.py` 演示如何在调用本地 LM Studio OpenAI 接口后，使用批次不变算子处理嵌入向量，消除批量差异。

```bash
export LM_STUDIO_URL=http://192.168.0.100:1234
export LM_STUDIO_MODEL=qwen3-30b-a3b-thinking-2507-mlx
python -m venv venv
source venv/bin/activate
pip install -e .[integration]
python examples/lm_studio_client.py
```

输出将比对“单行 vs 整批”的最大差异，批次不变模式下预期为 `0.00e+00`。

### 文本生成的确定性

- **采样策略是首要因素。** 在 LM Studio/OpenAI 接口中，将 `temperature=0.0`、禁用 `top_p`、固定 `seed` 后，可以获得逐字符一致的回复。
- **批次不变算子提供底层保障。** 数值实验显示矩阵运算的批次差异被抹平；配合确定性采样，可实现端到端可复现的文本生成。
- **控制变量实验结论：**
  - 仅使用确定性采样 → 文本一致；
  - 仅使用批次不变算子 → 文本仍受采样波动；
  - 两者结合 → 文本与数值双重一致，尤其适用于嵌入后处理、向量检索等场景。

## 测试与基准

```bash
pip install -e .[dev]
pytest tests -q --override-ini addopts=''        # 单元测试
python benchmarks/determinism_benchmark.py        # 数值 + 性能表格
python benchmarks/seed_sensitivity.py             # 随机种子敏感性分析
```

更多脚本与说明见 `benchmarks/` 与 `docs/` 目录。

## 设计与限制

- 目前仅覆盖推理常用的 `mm`、`addmm`、`log_softmax`、`mean`。
- 主要面向推理/分析工作负载；若用于训练，请评估 CPU 回落的性能成本。
- MPS 暂不支持 `float64`，框架会自动回退到 CPU；如需更高性能，可选 `precision=torch.float32` 并在关键步骤使用批次不变模式。
- 批次不变算子解决的是**数值层面的非确定性**；要得到完全一致的文本输出，仍需结合确定性采样策略。

## 引用与致谢

- [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- PyTorch `torch.library` 的设计，使得算子级重载成为可能。

如在研究或产品中使用本项目，欢迎引用“Batch Invariant Ops (Apple Silicon Edition)”。
