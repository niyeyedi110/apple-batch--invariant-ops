# Batch Invariant Ops (Apple Silicon Edition)

本项目复现并改写了 [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) 中的核心思想，针对 Apple Silicon (M 系列) 环境给出无 CUDA、无 Triton 依赖的批次不变算子实现。所有算子均通过将计算转移到 CPU 上的高精度 (float64) 运算来保持运算顺序的确定性，并在执行结束后自动还原到原始设备/精度，因此可以无缝与 MPS 后端的推理代码协同使用。

## 背景概述

原论文指出：在 GPU 上进行矩阵乘、归一化等操作时，线程分块策略会随批次大小改变，导致数值舍入顺序发生变化，进而产生看似“随机”的推理结果。论文方案通过定制 Triton kernel 强制固定分块顺序以保证批次不变性。

在 Apple Silicon 环境中暂无法使用 Triton/CUDA，因此这里采用以下策略实现同样的目标：

1. **算子劫持**：利用 `torch.library` 在运行时注册 Apple 端算子实现，取代 PyTorch 默认的 `aten::mm`/`aten::addmm`/`aten::_log_softmax`/`aten::mean.dim`。
2. **高精度 CPU 计算**：在劫持后的算子内部，将输入张量复制到 CPU，转换为 `float64`，执行确定性的 CPU 运算，再转换回原始设备与数据类型。
3. **面向推理**：方案专注推理工作负载（无梯度需求），适合需要在 MPS/CPU 上稳定复现 LLM 推理结果的场景。

虽然这种方法较原始 GPU 持久化 kernel 更慢，但在 Apple 芯片上可立即使用，同时可作为日后官方 Apple GPU 支持到来前的过渡解。

## 安装

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .
```

要求 `torch>=2.2` 且运行于 macOS arm64。

## 快速开始

```python
import torch
from apple_batch_invariant_ops import set_batch_invariant_mode

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

B, D = 128, 512
a = torch.linspace(-50, 50, B * D, device=device).reshape(B, D)
b = torch.linspace(-10, 10, D * D, device=device).reshape(D, D)

with set_batch_invariant_mode():
    # 批次=1 与 批次=全部 得到完全一致的数值
    out_single = torch.mm(a[:1], b)
    out_batch = torch.mm(a, b)[:1]
    diff = (out_single - out_batch).abs().max()
    print("difference", diff.item())  # 始终为 0
```

`set_batch_invariant_mode()` 会在进入上下文时劫持算子，退出时还原。

## 运行测试

```bash
pip install -e .[dev]
pytest
```

测试用例会自动选择 `mps` 或 `cpu` 设备，验证劫持后的算子在批次变化情况下依然给出完全一致的结果。

## 限制与后续工作

- 目前实现将所有运算迁移到 CPU 上，性能低于 GPU/MPS 原生算子，更适合强调确定性的场景。
- 仅覆盖 `mm`/`addmm`/`log_softmax`/`mean`，可按需扩展更多算子。
- 若未来 Triton 或 PyTorch 在 Apple GPU 上提供可控的线程调度接口，可将本项目替换为真正的 GPU 持久化 kernel 实现。

## 致谢

- [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) 原作者提供的方法论基础。
