"""Apple Silicon-friendly batch invariant operations."""

from .ops import (
    BatchInvariantConfig,
    get_active_config,
    set_batch_invariant_mode,
    enable_batch_invariant_mode,
    disable_batch_invariant_mode,
    is_batch_invariant_mode_enabled,
    mm_batch_invariant,
    addmm_batch_invariant,
    log_softmax_batch_invariant,
    mean_batch_invariant,
)

__version__ = "0.2.0"

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
