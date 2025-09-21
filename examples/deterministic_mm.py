"""Minimal演示：在 Apple Silicon 上获得批次不变的结果。"""

import torch

from apple_batch_invariant_ops import set_batch_invariant_mode


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    B, D = 64, 128
    a = torch.linspace(-5, 5, B * D, device=device, dtype=torch.float32).reshape(B, D)
    b = torch.linspace(-2, 2, D * D, device=device, dtype=torch.float32).reshape(D, D)

    with set_batch_invariant_mode():
        out_single = torch.mm(a[:1], b)
        out_batch = torch.mm(a, b)[:1]

    diff = (out_single - out_batch).abs().max().item()
    print(f"Max difference between batch sizes: {diff}")


if __name__ == "__main__":
    main()
