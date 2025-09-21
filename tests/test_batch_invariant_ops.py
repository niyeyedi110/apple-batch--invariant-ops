import pytest
import torch
import torch.nn.functional as F

from apple_batch_invariant_ops import (
    BatchInvariantConfig,
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
    get_active_config,
    is_batch_invariant_mode_enabled,
    set_batch_invariant_mode,
)


def _device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _to_cpu64(tensor: torch.Tensor) -> torch.Tensor:
    cpu_tensor = tensor.to("cpu")
    return cpu_tensor.to(torch.float64)


def _restore(tensor: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    out = tensor.to(like.dtype)
    if like.device.type != "cpu":
        out = out.to(like.device)
    return out


def test_mm_matches_reference():
    device = _device()
    torch.manual_seed(0)
    a = torch.randn(8, 16, device=device, dtype=torch.float32)
    b = torch.randn(16, 32, device=device, dtype=torch.float32)

    with set_batch_invariant_mode():
        actual = torch.mm(a, b)

    reference = _restore(_to_cpu64(a) @ _to_cpu64(b), like=a.new_empty(8, 32))
    assert torch.allclose(actual, reference, atol=0, rtol=0)


def test_mm_batch_invariance():
    device = _device()
    a = torch.linspace(-5, 5, 512, device=device, dtype=torch.float32).reshape(16, 32)
    b = torch.linspace(-2, 2, 1024, device=device, dtype=torch.float32).reshape(32, 32)

    with set_batch_invariant_mode():
        single = torch.mm(a[:1], b)
        batched = torch.mm(a, b)[:1]
    assert torch.equal(single, batched)


def test_log_softmax_matches_reference():
    device = _device()
    torch.manual_seed(1)
    x = torch.randn(4, 5, device=device, dtype=torch.float32)

    with set_batch_invariant_mode():
        actual = F.log_softmax(x, dim=-1)

    ref = _restore(torch.log_softmax(_to_cpu64(x), dim=-1), like=x)
    assert torch.allclose(actual, ref, atol=1e-6, rtol=0)


def test_mean_matches_reference():
    device = _device()
    torch.manual_seed(2)
    x = torch.randn(3, 4, 5, device=device, dtype=torch.float32)

    with set_batch_invariant_mode():
        actual = x.mean(dim=(-2, -1), keepdim=False)

    ref = _restore(_to_cpu64(x).mean(dim=(-2, -1)), like=x)
    assert torch.allclose(actual, ref, atol=1e-6, rtol=0)


def test_mean_dtype_override():
    device = _device()
    x = torch.randn(2, 3, device=device, dtype=torch.float32)

    with set_batch_invariant_mode():
        actual = torch.mean(x, dim=1, dtype=torch.float32)

    ref = _restore(_to_cpu64(x).mean(dim=1), like=x)
    assert torch.allclose(actual, ref, atol=1e-6, rtol=0)


def test_manual_seed_restored():
    torch.manual_seed(1234)
    baseline = torch.rand(4)

    config = BatchInvariantConfig(manual_seed=42)
    with set_batch_invariant_mode(config=config):
        inside = torch.rand(4)

    torch.manual_seed(1234)
    after = torch.rand(4)
    assert torch.allclose(after, baseline)

    torch.manual_seed(42)
    expected = torch.rand(4)
    assert torch.allclose(inside.cpu(), expected)


def test_force_cpu_false_precision_float32_matches_eager():
    device = _device()
    a = torch.randn(6, 6, device=device, dtype=torch.float32)
    b = torch.randn(6, 6, device=device, dtype=torch.float32)
    config = BatchInvariantConfig(force_cpu=False, precision=torch.float32)

    with set_batch_invariant_mode(config=config):
        actual = torch.mm(a, b)

    reference = torch.mm(a, b)
    assert torch.allclose(actual, reference, atol=0, rtol=0)


def test_context_disable_keeps_global_state():
    disable_batch_invariant_mode()
    assert not is_batch_invariant_mode_enabled()
    with set_batch_invariant_mode(enabled=False):
        assert not is_batch_invariant_mode_enabled()
    assert not is_batch_invariant_mode_enabled()


def test_enable_disable_cycle_restores_previous_config():
    disable_batch_invariant_mode()
    base_config = BatchInvariantConfig(force_cpu=True, precision=torch.float64)
    enable_batch_invariant_mode(base_config)
    assert is_batch_invariant_mode_enabled()
    assert get_active_config() == base_config

    override_config = BatchInvariantConfig(force_cpu=False, precision=torch.float32)
    with set_batch_invariant_mode(config=override_config):
        assert get_active_config() == override_config
        _ = torch.mm(torch.randn(2, 2), torch.randn(2, 2))

    assert get_active_config() == base_config
    disable_batch_invariant_mode()
    assert not is_batch_invariant_mode_enabled()


def test_invalid_precision_rejected():
    config = BatchInvariantConfig(precision=torch.bfloat16)
    disable_batch_invariant_mode()
    with pytest.raises(ValueError):
        enable_batch_invariant_mode(config)
