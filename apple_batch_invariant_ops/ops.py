import contextlib
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

__all__ = [
    "BatchInvariantConfig",
    "get_active_config",
    "set_batch_invariant_mode",
    "enable_batch_invariant_mode",
    "disable_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "mm_batch_invariant",
    "addmm_batch_invariant",
    "log_softmax_batch_invariant",
    "mean_batch_invariant",
]


_SUPPORTED_PRECISIONS = {torch.float32, torch.float64}


@dataclass(frozen=True)
class BatchInvariantConfig:
    """Runtime options that control how batch-invariant kernels operate."""

    precision: torch.dtype = torch.float64
    force_cpu: bool = True
    enforce_deterministic_algorithms: bool = True
    manual_seed: Optional[int] = None

    def validate(self) -> None:
        if self.precision not in _SUPPORTED_PRECISIONS:
            raise ValueError(
                f"Unsupported precision {self.precision}; supported: {_SUPPORTED_PRECISIONS}."
            )


_DEFAULT_CONFIG = BatchInvariantConfig()
_active_config: BatchInvariantConfig = _DEFAULT_CONFIG
_batch_invariant_MODE = False
_batch_invariant_LIB: Optional[torch.library.Library] = None
_previous_deterministic_setting: Optional[bool] = None
_previous_cpu_rng_state: Optional[torch.Tensor] = None
_previous_mps_rng_state: Optional[torch.Tensor] = None


def get_active_config() -> BatchInvariantConfig:
    return _active_config


def _supports_dtype_on_device(device: torch.device, dtype: torch.dtype) -> bool:
    if device.type == "mps" and dtype == torch.float64:
        return False
    if device.type == "cpu" and dtype == torch.float16:
        return False
    return True


def _resolve_compute_location(tensor: torch.Tensor) -> Tuple[torch.device, torch.dtype]:
    config = _active_config
    requested_device = torch.device("cpu") if config.force_cpu else tensor.device
    requested_dtype = config.precision
    if not _supports_dtype_on_device(requested_device, requested_dtype):
        requested_device = torch.device("cpu")
        requested_dtype = torch.float64
    return requested_device, requested_dtype


def _clone_to_compute_space(tensor: torch.Tensor) -> torch.Tensor:
    device, dtype = _resolve_compute_location(tensor)
    result = tensor
    if tensor.device != device:
        result = result.to(device=device, copy=True)
    else:
        result = result.clone()
    if result.dtype != dtype:
        result = result.to(dtype)
    return result


def _restore_device_dtype(
    tensor: torch.Tensor, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if tensor.device != device:
        tensor = tensor.to(device)
    return tensor


def mm_batch_invariant(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    a_compute = _clone_to_compute_space(mat1)
    b_compute = _clone_to_compute_space(mat2)
    result = torch.einsum("ik,kj->ij", a_compute, b_compute)
    out_dtype = torch.result_type(mat1, mat2)
    return _restore_device_dtype(result, device=mat1.device, dtype=out_dtype)


def addmm_batch_invariant(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    result = torch.einsum("ik,kj->ij", _clone_to_compute_space(mat1), _clone_to_compute_space(mat2))
    if alpha != 1.0:
        result = result.mul(alpha)
    if beta != 0:
        addend = _clone_to_compute_space(input)
        if beta != 1.0:
            addend = addend.mul(beta)
        result = result.add(addend)
    return _restore_device_dtype(result, device=input.device, dtype=input.dtype)


def log_softmax_batch_invariant(
    input: torch.Tensor,
    dim: int,
    _half_to_float: bool = False,
) -> torch.Tensor:
    if _half_to_float:
        raise ValueError("half_to_float is not supported in batch-invariant mode")
    dim = dim if dim >= 0 else input.ndim + dim
    data = _clone_to_compute_space(input)
    shifted = data - data.max(dim=dim, keepdim=True).values
    logits = shifted - torch.logsumexp(shifted, dim=dim, keepdim=True)
    return _restore_device_dtype(logits, device=input.device, dtype=input.dtype)


def mean_batch_invariant(
    input: torch.Tensor,
    dim: Sequence[int],
    keepdim: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    dims = tuple(sorted(d if d >= 0 else input.ndim + d for d in dim))
    data = _clone_to_compute_space(input)
    if dtype is not None and data.dtype != dtype:
        data = data.to(dtype)
    count = 1
    for d in dims:
        count *= input.shape[d]
    total = data.sum(dim=dims, keepdim=keepdim)
    mean = total / count
    target_dtype = dtype if dtype is not None else input.dtype
    return _restore_device_dtype(mean, device=input.device, dtype=target_dtype)


def is_batch_invariant_mode_enabled() -> bool:
    return _batch_invariant_MODE


def _capture_rng_state() -> None:
    global _previous_cpu_rng_state, _previous_mps_rng_state
    _previous_cpu_rng_state = torch.get_rng_state()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        _previous_mps_rng_state = torch.mps.get_rng_state()
    else:
        _previous_mps_rng_state = None


def _restore_rng_state() -> None:
    if _previous_cpu_rng_state is not None:
        torch.set_rng_state(_previous_cpu_rng_state)
    if _previous_mps_rng_state is not None and hasattr(torch.mps, "set_rng_state"):
        torch.mps.set_rng_state(_previous_mps_rng_state)


def _apply_runtime_guards(config: BatchInvariantConfig) -> None:
    global _previous_deterministic_setting, _previous_cpu_rng_state, _previous_mps_rng_state
    if config.enforce_deterministic_algorithms:
        _previous_deterministic_setting = torch.are_deterministic_algorithms_enabled()
        if not _previous_deterministic_setting:
            torch.use_deterministic_algorithms(True)
    else:
        _previous_deterministic_setting = None
    if config.manual_seed is not None:
        _capture_rng_state()
        torch.manual_seed(config.manual_seed)
        if torch.backends.mps.is_available() and hasattr(torch.mps, "manual_seed"):
            torch.mps.manual_seed(config.manual_seed)
    else:
        _previous_cpu_rng_state = None
        _previous_mps_rng_state = None


def _restore_runtime_guards(config: BatchInvariantConfig) -> None:
    global _previous_deterministic_setting, _previous_cpu_rng_state, _previous_mps_rng_state
    if config.manual_seed is not None:
        _restore_rng_state()
    if _previous_deterministic_setting is not None:
        torch.use_deterministic_algorithms(_previous_deterministic_setting)
    _previous_deterministic_setting = None
    _previous_cpu_rng_state = None
    _previous_mps_rng_state = None


def enable_batch_invariant_mode(config: Optional[BatchInvariantConfig] = None) -> None:
    global _batch_invariant_MODE, _batch_invariant_LIB, _active_config
    config = config or _DEFAULT_CONFIG
    config.validate()
    if _batch_invariant_MODE and config == _active_config:
        return
    if _batch_invariant_MODE:
        disable_batch_invariant_mode()
    _apply_runtime_guards(config)
    _active_config = config
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
    global _batch_invariant_MODE, _batch_invariant_LIB, _active_config
    if not _batch_invariant_MODE:
        return
    config = _active_config
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()
    _batch_invariant_MODE = False
    _batch_invariant_LIB = None
    _restore_runtime_guards(config)
    _active_config = _DEFAULT_CONFIG


@contextlib.contextmanager
def set_batch_invariant_mode(
    enabled: bool = True, config: Optional[BatchInvariantConfig] = None
):
    previous_state = is_batch_invariant_mode_enabled()
    previous_config = get_active_config()
    if enabled:
        enable_batch_invariant_mode(config)
    else:
        disable_batch_invariant_mode()
    try:
        yield
    finally:
        disable_batch_invariant_mode()
        if previous_state:
            enable_batch_invariant_mode(previous_config)
