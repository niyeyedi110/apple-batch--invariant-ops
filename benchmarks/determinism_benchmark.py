"""Compare deterministic behaviour and runtime with and without batch-invariant ops."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch

from apple_batch_invariant_ops import BatchInvariantConfig, set_batch_invariant_mode


@dataclass
class BenchmarkResult:
    name: str
    shape: Tuple[int, ...]
    diff_eager: float
    diff_batch_invariant: float
    time_eager_ms: float
    time_batch_invariant_ms: float


def _sync_device(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(fn, device: torch.device) -> float:
    _sync_device(device)
    start = time.perf_counter()
    fn()
    _sync_device(device)
    return (time.perf_counter() - start) * 1000.0


def _mm_case(batch: int, dim: int, device: torch.device) -> BenchmarkResult:
    torch.manual_seed(0)
    a = torch.randn(batch, dim, device=device, dtype=torch.float32)
    b = torch.randn(dim, dim, device=device, dtype=torch.float32)

    def single_eager():
        return torch.mm(a[:1], b)

    def batch_eager():
        return torch.mm(a, b)[:1]

    def batch_invariant():
        with set_batch_invariant_mode():
            return torch.mm(a, b)[:1]

    eager_time = _time_call(batch_eager, device)
    eager_diff = (single_eager() - batch_eager()).abs().max().item()

    det_time = _time_call(batch_invariant, device)
    with set_batch_invariant_mode():
        det_single = torch.mm(a[:1], b)
        det_batch = torch.mm(a, b)[:1]
    det_diff = (det_single - det_batch).abs().max().item()

    return BenchmarkResult(
        name="mm",
        shape=(batch, dim, dim),
        diff_eager=eager_diff,
        diff_batch_invariant=det_diff,
        time_eager_ms=eager_time,
        time_batch_invariant_ms=det_time,
    )


def _log_softmax_case(batch: int, dim: int, device: torch.device) -> BenchmarkResult:
    torch.manual_seed(1)
    x = torch.randn(batch, dim, device=device, dtype=torch.float32)

    def eager_batched():
        return torch.log_softmax(x, dim=-1)

    def eager_single_rows():
        rows = [torch.log_softmax(x[i : i + 1], dim=-1).squeeze(0) for i in range(batch)]
        return torch.stack(rows)

    def invariant_batched():
        with set_batch_invariant_mode():
            return torch.log_softmax(x, dim=-1)

    eager_time = _time_call(eager_batched, device)
    eager_diff = (eager_single_rows() - eager_batched()).abs().max().item()

    det_time = _time_call(invariant_batched, device)
    with set_batch_invariant_mode():
        det_rows = [torch.log_softmax(x[i : i + 1], dim=-1).squeeze(0) for i in range(batch)]
        det_diff = (torch.stack(det_rows) - torch.log_softmax(x, dim=-1)).abs().max().item()

    return BenchmarkResult(
        name="log_softmax",
        shape=(batch, dim),
        diff_eager=eager_diff,
        diff_batch_invariant=det_diff,
        time_eager_ms=eager_time,
        time_batch_invariant_ms=det_time,
    )


def _mean_case(batch: int, feature: int, seq: int, device: torch.device) -> BenchmarkResult:
    torch.manual_seed(2)
    x = torch.randn(batch, feature, seq, device=device, dtype=torch.float32)

    def eager_batched():
        return x.mean(dim=(-2, -1))

    def eager_single():
        rows = [x[i].mean(dim=(-2, -1)) for i in range(batch)]
        return torch.stack(rows)

    def invariant_batched():
        with set_batch_invariant_mode():
            return x.mean(dim=(-2, -1))

    eager_time = _time_call(eager_batched, device)
    eager_diff = (eager_single() - eager_batched()).abs().max().item()

    det_time = _time_call(invariant_batched, device)
    with set_batch_invariant_mode():
        det_rows = [x[i].mean(dim=(-2, -1)) for i in range(batch)]
        det_diff = (torch.stack(det_rows) - x.mean(dim=(-2, -1))).abs().max().item()

    return BenchmarkResult(
        name="mean",
        shape=(batch, feature, seq),
        diff_eager=eager_diff,
        diff_batch_invariant=det_diff,
        time_eager_ms=eager_time,
        time_batch_invariant_ms=det_time,
    )


def run(device: torch.device) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    for batch, dim in [(32, 256), (64, 512), (128, 768)]:
        results.append(_mm_case(batch, dim, device))
    for batch, dim in [(32, 512), (64, 1024)]:
        results.append(_log_softmax_case(batch, dim, device))
    for batch, feature, seq in [(16, 64, 128), (32, 64, 256)]:
        results.append(_mean_case(batch, feature, seq, device))
    return results


def _format_row(res: BenchmarkResult) -> str:
    return (
        f"{res.name:<12}{str(res.shape):<18}"
        f"{res.diff_eager:>12.2e}{res.diff_batch_invariant:>18.2e}"
        f"{res.time_eager_ms:>16.2f}{res.time_batch_invariant_ms:>18.2f}"
    )


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    BatchInvariantConfig().validate()
    heading = (
        "op".ljust(12)
        + "shape".ljust(18)
        + "eager diff".rjust(12)
        + "det diff".rjust(18)
        + "eager ms".rjust(16)
        + "det ms".rjust(18)
    )
    print("Batch-invariant benchmark on", device)
    print(heading)
    print("-" * len(heading))
    for res in run(device):
        print(_format_row(res))


if __name__ == "__main__":
    main()
