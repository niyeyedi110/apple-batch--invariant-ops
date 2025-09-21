"""Demonstrate deterministic post-processing with LM Studio embeddings."""

from __future__ import annotations

import os
from typing import List

import requests
import torch

from apple_batch_invariant_ops import BatchInvariantConfig, set_batch_invariant_mode

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234")
LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", "qwen3-30b-a3b-thinking-2507-mlx")


def _fetch_embeddings(texts: List[str]) -> torch.Tensor:
    response = requests.post(
        f"{LM_STUDIO_URL}/v1/embeddings",
        json={"model": LM_STUDIO_MODEL, "input": texts},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    vectors = [torch.tensor(item["embedding"], dtype=torch.float32) for item in payload["data"]]
    return torch.stack(vectors)


def main() -> None:
    texts = [
        "深度学习的基本原理是什么？",
        "请用一句话概括机器学习",
        "列举自然语言处理的常见任务",
    ]

    print(f"Connecting to LM Studio at {LM_STUDIO_URL} (model={LM_STUDIO_MODEL})")
    embeddings = _fetch_embeddings(texts)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    embeddings = embeddings.to(device)

    baseline_single = torch.mm(embeddings[:1], embeddings.T)
    baseline_batch = torch.mm(embeddings, embeddings.T)[:1]
    baseline_diff = (baseline_single - baseline_batch).abs().max().item()

    with set_batch_invariant_mode(BatchInvariantConfig(manual_seed=42)):
        invariant_single = torch.mm(embeddings[:1], embeddings.T)
        invariant_batch = torch.mm(embeddings, embeddings.T)[:1]
    invariant_diff = (invariant_single - invariant_batch).abs().max().item()

    print("Baseline single-row vs full-batch diff:", f"{baseline_diff:.2e}")
    print("Batch-invariant single-row vs full-batch diff:", f"{invariant_diff:.2e}")
    print("
Example deterministic similarity vector:")
    print(invariant_single)


if __name__ == "__main__":
    main()
