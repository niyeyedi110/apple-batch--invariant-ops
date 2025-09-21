"""Study how random seeds influence numerical stability and how the library reacts."""

from __future__ import annotations

from typing import Iterable

import torch

from apple_batch_invariant_ops import BatchInvariantConfig, set_batch_invariant_mode


def _mm_diff(seed: int, device: torch.device) -> tuple[float, float]:
    torch.manual_seed(seed)
    a = torch.randn(64, 64, device=device)
    b = torch.randn(64, 64, device=device)

    eager_single = torch.mm(a[:1], b)
    eager_batch = torch.mm(a, b)[:1]
    eager_diff = (eager_single - eager_batch).abs().max().item()

    with set_batch_invariant_mode():
        det_single = torch.mm(a[:1], b)
        det_batch = torch.mm(a, b)[:1]
    det_diff = (det_single - det_batch).abs().max().item()
    return eager_diff, det_diff


def _manual_seed_demo(device: torch.device) -> None:
    torch.manual_seed(1234)
    baseline = torch.rand(4, device=device)

    config = BatchInvariantConfig(manual_seed=42)
    with set_batch_invariant_mode(config=config):
        inside = torch.rand(4, device=device)

    torch.manual_seed(1234)
    after = torch.rand(4, device=device)

    torch.manual_seed(42)
    expected = torch.rand(4, device=device)

    print("Manual-seed experiment:")
    print("  outside restored:", torch.allclose(baseline, after))
    print("  inside deterministic:", torch.allclose(inside, expected))


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    seeds: Iterable[int] = [0, 1, 7, 42, 1337]
    print("Seed sensitivity on", device)
    print("seed    eager diff    det diff")
    for seed in seeds:
        eager, det = _mm_diff(seed, device)
        print(f"{seed:>4}    {eager:>10.2e}    {det:>8.2e}")
    print()
    _manual_seed_demo(device)


if __name__ == "__main__":
    main()
