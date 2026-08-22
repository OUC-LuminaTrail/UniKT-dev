"""CPU serial fallback of the segmented scalar affine exclusive scan.

The Triton kernels require CUDA; CPU inputs route through the pure PyTorch
serial scan. These tests pin its semantics against a straightforward serial
reference on CPU, so they run without a GPU.
"""

import pytest
import torch

from model.AxisKT.triton_scan import (
    _serial_scalar_affine_exclusive_scan,
    segmented_scalar_affine_exclusive_scan,
)


def _scalar_reference(matrix, bias, segment_ids, valid_mask, initial_state):
    """Serial reference matching the Triton kernel semantics exactly.

    Per-position initial states through the exclusive carry; the carry resets
    at segment heads and invalid positions act as the identity and output
    zero. Pure PyTorch ops, so autograd covers the backward pass.
    """
    batch, length, heads, _, _ = matrix.shape
    result = torch.zeros_like(bias)
    for b in range(batch):
        carry_a = torch.ones(heads, 1, device=matrix.device, dtype=matrix.dtype)
        carry_b = torch.zeros(heads, 1, device=matrix.device, dtype=matrix.dtype)
        prev_valid = False
        prev_seg = -1
        for position in range(length):
            valid = bool(valid_mask[b, position])
            segment = int(segment_ids[b, position])
            is_head = valid and (position == 0 or not prev_valid or segment != prev_seg)
            if is_head:
                carry_a = torch.ones(heads, 1, device=matrix.device, dtype=matrix.dtype)
                carry_b = torch.zeros(
                    heads, 1, device=matrix.device, dtype=matrix.dtype
                )
            if valid:
                result[b, position] = carry_a * initial_state[b, position] + carry_b
                a = matrix[b, position].squeeze(-1)  # [heads, 1]
                carry_a = a * carry_a
                carry_b = a * carry_b + bias[b, position]
            prev_valid = valid
            prev_seg = segment
    return result


def _example(*, seed: int = 11, device=None):
    torch.manual_seed(seed)
    segment_ids = torch.tensor(
        [[0, 0, 0, 2, 2, 5, 5, 5, 9]], dtype=torch.long, device=device
    )
    valid_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool, device=device
    )
    matrix = 0.9 + 0.1 * torch.randn(1, 9, 3, 1, 1, device=device)
    bias = torch.randn(1, 9, 3, 1, device=device)
    initial = torch.randn(1, 9, 3, 1, device=device)
    initial[:, 1:3] = initial[:, :1]
    initial[:, 4:5] = initial[:, 3:4]
    initial[:, 6:8] = initial[:, 5:6]
    return matrix, bias, segment_ids, valid_mask, initial


def test_serial_fallback_matches_reference():
    matrix, bias, segment_ids, valid_mask, initial = _example()
    actual = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    expected = _scalar_reference(matrix, bias, segment_ids, valid_mask, initial)
    torch.testing.assert_close(actual, expected)


def test_serial_fallback_gradients_match_reference():
    matrix, bias, segment_ids, valid_mask, initial = _example()
    weights = torch.randn_like(bias)

    fallback_matrix = matrix.clone().requires_grad_()
    fallback_bias = bias.clone().requires_grad_()
    fallback_initial = initial.clone().requires_grad_()
    fallback = segmented_scalar_affine_exclusive_scan(
        fallback_matrix,
        fallback_bias,
        segment_ids,
        valid_mask,
        fallback_initial,
    )
    fallback_gradients = torch.autograd.grad(
        (fallback * weights).sum(),
        (fallback_matrix, fallback_bias, fallback_initial),
    )

    serial_matrix = matrix.clone().requires_grad_()
    serial_bias = bias.clone().requires_grad_()
    serial_initial = initial.clone().requires_grad_()
    serial = _scalar_reference(
        serial_matrix, serial_bias, segment_ids, valid_mask, serial_initial
    )
    serial_gradients = torch.autograd.grad(
        (serial * weights).sum(), (serial_matrix, serial_bias, serial_initial)
    )

    torch.testing.assert_close(fallback, serial)
    for fallback_gradient, serial_gradient in zip(
        fallback_gradients, serial_gradients, strict=True
    ):
        torch.testing.assert_close(fallback_gradient, serial_gradient)


def test_serial_fallback_uses_per_position_initial_state():
    # Vary the initial state within segments so the per-position semantics are
    # exercised, not hidden behind segment-constant values.
    matrix, bias, segment_ids, valid_mask, initial = _example()
    initial = initial + 0.3 * torch.randn_like(initial)

    actual = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    expected = _scalar_reference(matrix, bias, segment_ids, valid_mask, initial)
    torch.testing.assert_close(actual, expected)


def test_serial_fallback_does_not_cross_segment_boundaries():
    matrix, bias, segment_ids, valid_mask, initial = _example()
    baseline = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    changed_bias = bias.clone()
    changed_bias[:, :3] += 100.0
    changed = segmented_scalar_affine_exclusive_scan(
        matrix, changed_bias, segment_ids, valid_mask, initial
    )
    torch.testing.assert_close(changed[:, 3:], baseline[:, 3:])


def test_serial_fallback_is_exclusive():
    matrix, bias, segment_ids, valid_mask, initial = _example()
    baseline = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    changed_matrix = matrix.clone()
    changed_bias = bias.clone()
    changed_matrix[:, 1] = -7.0
    changed_bias[:, 1] = 99.0
    changed = segmented_scalar_affine_exclusive_scan(
        changed_matrix, changed_bias, segment_ids, valid_mask, initial
    )
    # The exclusive output at position 1 predates its own transition.
    torch.testing.assert_close(changed[:, 1], baseline[:, 1])
    assert not torch.allclose(changed[:, 2], baseline[:, 2])


def test_serial_fallback_accepts_empty_sequences():
    matrix = torch.zeros(1, 0, 3, 1, 1)
    bias = torch.zeros(1, 0, 3, 1)
    segment_ids = torch.zeros(1, 0, dtype=torch.long)
    valid_mask = torch.zeros(1, 0, dtype=torch.bool)
    initial = torch.zeros(1, 0, 3, 1)

    result = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    assert result.shape == initial.shape


def test_serial_fallback_all_invalid_positions_output_zero():
    matrix = torch.randn(1, 3, 2, 1, 1)
    bias = torch.randn(1, 3, 2, 1)
    segment_ids = torch.zeros(1, 3, dtype=torch.long)
    valid_mask = torch.zeros(1, 3, dtype=torch.bool)
    initial = torch.randn(1, 3, 2, 1)

    result = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    torch.testing.assert_close(result, torch.zeros_like(initial))


def test_non_scalar_matrix_is_rejected():
    matrix = torch.zeros(1, 2, 3, 2, 2)
    bias = torch.zeros(1, 2, 3, 2)
    segment_ids = torch.zeros(1, 2, dtype=torch.long)
    valid_mask = torch.ones(1, 2, dtype=torch.bool)
    initial = torch.zeros(1, 2, 3, 2)

    with pytest.raises(ValueError, match=r"\[B, N, H, 1, 1\]"):
        segmented_scalar_affine_exclusive_scan(
            matrix, bias, segment_ids, valid_mask, initial
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA for the Triton kernel"
)
def test_serial_fallback_matches_triton():
    device = torch.device("cuda")
    matrix, bias, segment_ids, valid_mask, initial = _example(device=device)
    weights = torch.randn_like(initial)

    # Forward: the fallback matches the per-position initial-state semantics
    # even beyond Triton's assumptions (segment-constant init).
    varied = initial + 0.3 * torch.randn_like(initial)
    triton_fwd = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, varied
    )
    serial_fwd = _serial_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, varied
    )
    torch.testing.assert_close(triton_fwd, serial_fwd, atol=1e-5, rtol=1e-4)

    # Gradients: Triton's d matrix / d bias match autograd only while
    # ``initial_state`` is constant within a segment (its documented
    # assumption); the fallback has no such restriction, so compare under it.
    triton_matrix = matrix.clone().requires_grad_()
    triton_bias = bias.clone().requires_grad_()
    triton_initial = initial.clone().requires_grad_()
    triton_result = segmented_scalar_affine_exclusive_scan(
        triton_matrix,
        triton_bias,
        segment_ids,
        valid_mask,
        triton_initial,
    )
    triton_gradients = torch.autograd.grad(
        (triton_result * weights).sum(),
        (triton_matrix, triton_bias, triton_initial),
    )

    serial_matrix = matrix.clone().requires_grad_()
    serial_bias = bias.clone().requires_grad_()
    serial_initial = initial.clone().requires_grad_()
    serial_result = _serial_scalar_affine_exclusive_scan(
        serial_matrix, serial_bias, segment_ids, valid_mask, serial_initial
    )
    serial_gradients = torch.autograd.grad(
        (serial_result * weights).sum(),
        (serial_matrix, serial_bias, serial_initial),
    )

    torch.testing.assert_close(triton_result, serial_result, atol=1e-5, rtol=1e-4)
    for triton_gradient, serial_gradient in zip(
        triton_gradients, serial_gradients, strict=True
    ):
        torch.testing.assert_close(
            triton_gradient, serial_gradient, atol=1e-5, rtol=1e-4
        )
