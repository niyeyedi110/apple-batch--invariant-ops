"""Apple Silicon-friendly batch invariant operations."""

from .ops import (
    set_batch_invariant_mode,
    enable_batch_invariant_mode,
    disable_batch_invariant_mode,
    is_batch_invariant_mode_enabled,
    mm_batch_invariant,
    addmm_batch_invariant,
    log_softmax_batch_invariant,
    mean_batch_invariant,
)

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
