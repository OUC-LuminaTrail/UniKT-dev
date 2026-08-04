"""Serial fallback of the segmented block-affine exclusive scan.

The Triton kernels hard-code 2x2 blocks; any other block size (ablation:
1x1 uncoupled, 3x3 wider coupling) and CPU inputs route through the pure
PyTorch serial scan. These tests pin its semantics against a straightforward
serial reference, on CPU so they run without a GPU.
"""

import pytest
import torch

from model.ReKTP.triton_scan import (
    _serial_block_affine_exclusive_scan,
    segmented_block_affine_exclusive_scan,
)


def _serial_reference(matrix, bias, segment_ids, valid_mask, initial_state):
    """Serial reference matching the Triton kernel for constant per-segment init.

    Only valid when ``initial_state`` is constant within each segment (ReKTP's
    usage, since the initial state derives from the segment's skill id).
    """
    result = torch.zeros_like(bias)
    for batch in range(matrix.size(0)):
        states = {}
        for position in range(matrix.size(1)):
            if not valid_mask[batch, position]:
                continue
            segment = int(segment_ids[batch, position])
            state = states.get(segment, initial_state[batch, position])
            result[batch, position] = state
            states[segment] = (matrix[batch, position] @ state.unsqueeze(-1)).squeeze(
                -1
            ) + bias[batch, position]
    return result


def _serial_reference_per_position_init(
    matrix, bias, segment_ids, valid_mask, initial_state
):
    """Serial reference matching the Triton kernel for any initial_state.

    Every position reads its own ``initial_state`` through the exclusive
    carry, exactly like ``_fwd_kernel``'s ``carry @ init_n + carry_bias``; the
    carry resets at segment heads and invalid positions act as the identity
    and output zero.
    """
    batch, length, heads, block, _ = matrix.shape
    result = torch.zeros_like(bias)
    identity = (
        torch.eye(block, dtype=matrix.dtype)
        .view(1, block, block)
        .expand(heads, block, block)
    )
    for b in range(batch):
        carry = identity
        carry_bias = torch.zeros(heads, block, dtype=matrix.dtype)
        prev_valid = False
        prev_seg = -1
        for position in range(length):
            if not valid_mask[b, position]:
                prev_valid = False
                continue
            segment = int(segment_ids[b, position])
            is_head = (position == 0) or (not prev_valid) or (segment != prev_seg)
            if is_head:
                carry = identity
                carry_bias = torch.zeros(heads, block, dtype=matrix.dtype)
            result[b, position] = (
                torch.einsum("hij,hj->hi", carry, initial_state[b, position])
                + carry_bias
            )
            carry = torch.einsum("hij,hjk->hik", matrix[b, position], carry)
            carry_bias = (
                torch.einsum("hij,hj->hi", matrix[b, position], carry_bias)
                + bias[b, position]
            )
            prev_valid = True
            prev_seg = segment
    return result


def _block_example(block_size: int, *, seed: int = 11, device=None):
    torch.manual_seed(seed)
    segment_ids = torch.tensor(
        [[0, 0, 0, 2, 2, 5, 5, 5, 9]], dtype=torch.long, device=device
    )
    valid_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool, device=device
    )
    identity = torch.eye(block_size, device=device).view(
        1, 1, 1, block_size, block_size
    )
    matrix = identity + 0.1 * torch.randn(
        1, 9, 3, block_size, block_size, device=device
    )
    bias = torch.randn(1, 9, 3, block_size, device=device)
    initial = torch.randn(1, 9, 3, block_size, device=device)
    initial[:, 1:3] = initial[:, :1]
    initial[:, 4:5] = initial[:, 3:4]
    initial[:, 6:8] = initial[:, 5:6]
    return matrix, bias, segment_ids, valid_mask, initial


@pytest.mark.parametrize("block_size", [1, 2, 3])
def test_serial_fallback_matches_reference(block_size):
    matrix, bias, segment_ids, valid_mask, initial = _block_example(block_size)
    actual = segmented_block_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    expected = _serial_reference(matrix, bias, segment_ids, valid_mask, initial)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("block_size", [1, 2, 3])
def test_serial_fallback_gradients_match_reference(block_size):
    matrix, bias, segment_ids, valid_mask, initial = _block_example(block_size)
    weights = torch.randn_like(bias)

    fallback_matrix = matrix.clone().requires_grad_()
    fallback_bias = bias.clone().requires_grad_()
    fallback = segmented_block_affine_exclusive_scan(
        fallback_matrix,
        fallback_bias,
        segment_ids,
        valid_mask,
        initial,
    )
    fallback_gradients = torch.autograd.grad(
        (fallback * weights).sum(), (fallback_matrix, fallback_bias)
    )

    serial_matrix = matrix.clone().requires_grad_()
    serial_bias = bias.clone().requires_grad_()
    serial = _serial_reference(
        serial_matrix,
        serial_bias,
        segment_ids,
        valid_mask,
        initial,
    )
    serial_gradients = torch.autograd.grad(
        (serial * weights).sum(), (serial_matrix, serial_bias)
    )

    torch.testing.assert_close(fallback, serial)
    for fallback_gradient, serial_gradient in zip(
        fallback_gradients, serial_gradients, strict=True
    ):
        torch.testing.assert_close(fallback_gradient, serial_gradient)


@pytest.mark.parametrize("block_size", [1, 3])
def test_serial_fallback_does_not_cross_segment_boundaries(block_size):
    matrix, bias, segment_ids, valid_mask, initial = _block_example(block_size)
    baseline = segmented_block_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    changed_bias = bias.clone()
    changed_bias[:, :3] += 100.0
    changed = segmented_block_affine_exclusive_scan(
        matrix, changed_bias, segment_ids, valid_mask, initial
    )
    torch.testing.assert_close(changed[:, 3:], baseline[:, 3:])


@pytest.mark.parametrize("block_size", [1, 3])
def test_serial_fallback_is_exclusive(block_size):
    matrix, bias, segment_ids, valid_mask, initial = _block_example(block_size)
    baseline = segmented_block_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    changed_matrix = matrix.clone()
    changed_bias = bias.clone()
    changed_matrix[:, 1] = torch.eye(block_size) * 2.0
    changed_bias[:, 1] = 99.0
    changed = segmented_block_affine_exclusive_scan(
        changed_matrix, changed_bias, segment_ids, valid_mask, initial
    )
    # The exclusive output at position 1 predates its own transition.
    torch.testing.assert_close(changed[:, 1], baseline[:, 1])
    assert not torch.allclose(changed[:, 2], baseline[:, 2])


def test_serial_fallback_accepts_empty_sequences():
    matrix = torch.zeros(1, 0, 3, 2, 2)
    bias = torch.zeros(1, 0, 3, 2)
    segment_ids = torch.zeros(1, 0, dtype=torch.long)
    valid_mask = torch.zeros(1, 0, dtype=torch.bool)
    initial = torch.zeros(1, 0, 3, 2)

    result = segmented_block_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    assert result.shape == initial.shape


def test_serial_fallback_initial_state_gradients_match_reference():
    # ``initial_state`` is per-position in ReKTP (derived from the skill
    # embedding); the Triton backward recovers ``prefix_i^T @ g_i`` per
    # position, so the fallback must flow each position's gradient through
    # every later output in its segment.
    matrix, bias, segment_ids, valid_mask, initial = _block_example(2)
    # Vary the initial state within segments so the per-position semantics
    # are exercised, not hidden behind segment-constant values.
    initial = initial + 0.3 * torch.randn_like(initial)
    weights = torch.randn_like(initial)

    fallback_initial = initial.clone().requires_grad_()
    fallback = segmented_block_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, fallback_initial
    )
    fallback_gradient = torch.autograd.grad(
        (fallback * weights).sum(), fallback_initial
    )[0]

    serial_initial = initial.clone().requires_grad_()
    serial = _serial_reference_per_position_init(
        matrix, bias, segment_ids, valid_mask, serial_initial
    )
    serial_gradient = torch.autograd.grad((serial * weights).sum(), serial_initial)[0]

    torch.testing.assert_close(fallback, serial)
    torch.testing.assert_close(fallback_gradient, serial_gradient)


def test_serial_fallback_all_invalid_positions_output_zero():
    matrix = torch.randn(1, 3, 2, 2, 2)
    bias = torch.randn(1, 3, 2, 2)
    segment_ids = torch.zeros(1, 3, dtype=torch.long)
    valid_mask = torch.zeros(1, 3, dtype=torch.bool)
    initial = torch.randn(1, 3, 2, 2)

    result = segmented_block_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    torch.testing.assert_close(result, torch.zeros_like(initial))


def test_serial_fallback_supports_a_single_full_width_block():
    # One 16x16 block (num_state_blocks == 1): the far end of the ablation
    # range, exercising the full coupling within the whole state vector.
    block_size = 16
    matrix = torch.eye(block_size).view(1, 1, 1, block_size, block_size).expand(
        1, 2, 1, block_size, block_size
    ).clone() + 0.01 * torch.randn(1, 2, 1, block_size, block_size)
    bias = torch.randn(1, 2, 1, block_size)
    segment_ids = torch.zeros(1, 2, dtype=torch.long)
    valid_mask = torch.ones(1, 2, dtype=torch.bool)
    initial = torch.randn(1, 2, 1, block_size)

    result = segmented_block_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    expected = _serial_reference_per_position_init(
        matrix, bias, segment_ids, valid_mask, initial
    )
    torch.testing.assert_close(result, expected)


def test_non_square_matrix_is_rejected():
    matrix = torch.zeros(1, 2, 3, 2, 4)
    bias = torch.zeros(1, 2, 3, 2)
    segment_ids = torch.zeros(1, 2, dtype=torch.long)
    valid_mask = torch.ones(1, 2, dtype=torch.bool)
    initial = torch.zeros(1, 2, 3, 2)

    with pytest.raises(ValueError, match=r"\[B, N, H, S, S\]"):
        segmented_block_affine_exclusive_scan(
            matrix, bias, segment_ids, valid_mask, initial
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA for the Triton kernel"
)
def test_serial_fallback_matches_triton_at_block_size_2():
    device = torch.device("cuda")
    matrix, bias, segment_ids, valid_mask, initial = _block_example(2, device=device)
    weights = torch.randn_like(initial)

    # Forward: the fallback matches the per-position initial-state semantics
    # even beyond Triton's assumptions (segment-constant init).
    varied = initial + 0.3 * torch.randn_like(initial)
    triton_fwd = segmented_block_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, varied
    )
    serial_fwd = _serial_block_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, varied
    )
    torch.testing.assert_close(triton_fwd, serial_fwd, atol=1e-5, rtol=1e-4)

    # Gradients: Triton's d matrix / d bias match autograd only while
    # ``initial_state`` is constant within a segment (its documented
    # assumption); the fallback has no such restriction, so compare under it.
    triton_matrix = matrix.clone().requires_grad_()
    triton_bias = bias.clone().requires_grad_()
    triton_initial = initial.clone().requires_grad_()
    triton_result = segmented_block_affine_exclusive_scan(
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
    serial_result = _serial_block_affine_exclusive_scan(
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
