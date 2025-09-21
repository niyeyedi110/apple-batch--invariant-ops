import contextlib
from typing import Optional, Sequence

import torch

__all__ = [
    "set_batch_invariant_mode",
    "enable_batch_invariant_mode",
    "disable_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "mm_batch_invariant",
    "addmm_batch_invariant",
    "log_softmax_batch_invariant",
    "mean_batch_invariant",
]


def _to_cpu_fp64(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor to CPU float64 with autograd support preserved."""
    if tensor.device.type != "cpu":
        tensor = tensor.to("cpu")
    if tensor.dtype != torch.float64:
        tensor = tensor.to(torch.float64)
    return tensor


def _restore_device_dtype(
    tensor: torch.Tensor, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if device.type != "cpu":
        tensor = tensor.to(device)
    return tensor


def mm_batch_invariant(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    a_cpu = _to_cpu_fp64(mat1)
    b_cpu = _to_cpu_fp64(mat2)
    result_cpu = torch.einsum("ik,kj->ij", a_cpu, b_cpu)
    return _restore_device_dtype(result_cpu, device=mat1.device, dtype=mat1.dtype)


def addmm_batch_invariant(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    mm_cpu = torch.einsum("ik,kj->ij", _to_cpu_fp64(mat1), _to_cpu_fp64(mat2))
    if alpha != 1.0:
        mm_cpu = mm_cpu.mul(alpha)
    if beta != 0:
        input_cpu = _to_cpu_fp64(input)
        if beta != 1.0:
            input_cpu = input_cpu.mul(beta)
        mm_cpu = mm_cpu.add(input_cpu)
    return _restore_device_dtype(mm_cpu, device=input.device, dtype=input.dtype)


def log_softmax_batch_invariant(
    input: torch.Tensor,
    dim: int,
    _half_to_float: bool = False,
) -> torch.Tensor:
    if _half_to_float:
        raise ValueError("half_to_float is not supported in batch-invariant mode")
    dim = dim if dim >= 0 else input.ndim + dim
    data = _to_cpu_fp64(input)
    shifted = data - data.max(dim=dim, keepdim=True).values
    logsum = torch.logsumexp(shifted, dim=dim, keepdim=True)
    result_cpu = shifted - logsum
    return _restore_device_dtype(result_cpu, device=input.device, dtype=input.dtype)


def mean_batch_invariant(
    input: torch.Tensor,
    dim: Sequence[int],
    keepdim: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    dims = tuple(sorted(d if d >= 0 else input.ndim + d for d in dim))
    data = _to_cpu_fp64(input)
    target_dtype = dtype if dtype is not None else input.dtype
    if data.dtype != target_dtype:
        data = data.to(target_dtype)
    sum_cpu = data.sum(dim=dims, keepdim=keepdim)
    count = 1
    for d in dims:
        count *= input.shape[d]
    result_cpu = sum_cpu / count
    return _restore_device_dtype(result_cpu, device=input.device, dtype=target_dtype)


_batch_invariant_MODE = False
_batch_invariant_LIB: Optional[torch.library.Library] = None


def is_batch_invariant_mode_enabled() -> bool:
    return _batch_invariant_MODE


def enable_batch_invariant_mode() -> None:
    global _batch_invariant_MODE, _batch_invariant_LIB
    if _batch_invariant_MODE:
        return
    _batch_invariant_MODE = True
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")
    for backend in ("CPU", "MPS"):
        if backend == "MPS" and not torch.backends.mps.is_available():
            continue
        _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, backend)
        _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, backend)
        _batch_invariant_LIB.impl("aten::_log_softmax", log_softmax_batch_invariant, backend)
        _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, backend)


def disable_batch_invariant_mode() -> None:
    global _batch_invariant_MODE, _batch_invariant_LIB
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()
    _batch_invariant_MODE = False
    _batch_invariant_LIB = None


@contextlib.contextmanager
def set_batch_invariant_mode(enabled: bool = True):
    global _batch_invariant_MODE
    previously_enabled = _batch_invariant_MODE
    disable_batch_invariant_mode()
    if enabled:
        enable_batch_invariant_mode()
    try:
        yield
    finally:
        disable_batch_invariant_mode()
        if previously_enabled:
            enable_batch_invariant_mode()
