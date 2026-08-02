import pytest
import torch

from model.ReKTP.qrnn_scan import qrnn_pool


def _sequential_pool(z, f):
    """Direct transcription of the ForgetMult recurrence."""
    cells = []
    prev = torch.zeros_like(z[:, :1])
    for t in range(z.size(1)):
        prev = f[:, t : t + 1] * z[:, t : t + 1] + (1 - f[:, t : t + 1]) * prev
        cells.append(prev)
    return torch.cat(cells, dim=1)


def _random_example(batch=3, length=9, dim=5):
    torch.manual_seed(23)
    z = torch.randn(batch, length, dim)
    f = torch.sigmoid(torch.randn(batch, length, dim))
    return z, f


def test_qrnn_pool_matches_sequential_recurrence():
    z, f = _random_example()
    actual = qrnn_pool(z, f)
    expected = _sequential_pool(z, f)
    torch.testing.assert_close(actual, expected)


def test_qrnn_pool_gradients_match_sequential_recurrence():
    z, f = _random_example()
    z.requires_grad_(True)
    f.requires_grad_(True)
    qrnn_pool(z, f).square().mean().backward()
    z_grad, f_grad = z.grad.clone(), f.grad.clone()

    z_ref, f_ref = _random_example()
    z_ref.requires_grad_(True)
    f_ref.requires_grad_(True)
    _sequential_pool(z_ref, f_ref).square().mean().backward()

    torch.testing.assert_close(z_grad, z_ref.grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(f_grad, f_ref.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("length", [1, 2, 3, 7, 16, 33])
def test_qrnn_pool_handles_short_and_non_power_of_two_lengths(length):
    z = torch.randn(2, length, 4)
    f = torch.sigmoid(torch.randn(2, length, 4))
    expected = _sequential_pool(z, f)
    torch.testing.assert_close(qrnn_pool(z, f), expected)


def test_qrnn_pool_zero_length_returns_zeros():
    z = torch.randn(2, 0, 4)
    f = torch.sigmoid(torch.randn(2, 0, 4))
    out = qrnn_pool(z, f)
    assert out.shape == z.shape
    assert (out == 0).all()


def test_qrnn_pool_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="matching"):
        qrnn_pool(torch.randn(2, 3, 4), torch.randn(2, 3, 5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qrnn_pool_triton_matches_cpu_and_autograd_on_cuda():
    torch.manual_seed(7)
    z = torch.randn(8, 64, 32, device="cuda")
    f = torch.sigmoid(torch.randn(8, 64, 32, device="cuda"))

    expected = _sequential_pool(z, f)
    actual = qrnn_pool(z, f)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    z.requires_grad_(True)
    f.requires_grad_(True)
    qrnn_pool(z, f).square().mean().backward()
    z_grad, f_grad = z.grad.clone(), f.grad.clone()

    z_ref = z.detach().requires_grad_(True)
    f_ref = f.detach().requires_grad_(True)
    _sequential_pool(z_ref, f_ref).square().mean().backward()

    torch.testing.assert_close(z_grad, z_ref.grad, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(f_grad, f_ref.grad, atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qrnn_pool_triton_handles_bf16():
    z = torch.randn(4, 16, 8, device="cuda").bfloat16()
    f = torch.sigmoid(torch.randn(4, 16, 8, device="cuda")).bfloat16()
    out = qrnn_pool(z, f)
    assert out.dtype == z.dtype
    assert torch.isfinite(out).all()
