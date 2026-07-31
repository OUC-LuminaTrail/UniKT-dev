"""Parallel segmented scans for KC-specific affine state transitions."""

import torch


def segmented_affine_exclusive_scan(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    segment_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Apply an exclusive affine scan independently inside each segment.

    Each valid position represents ``h_after = alpha * h_before + beta``.
    Inputs must already be packed so positions belonging to the same segment
    are contiguous. The Hillis-Steele doubling schedule exposes logarithmic
    sequential depth and uses only regular PyTorch tensor operations.

    Args:
        alpha: Multiplicative transitions with shape ``[B, N, D]``.
        beta: Additive transitions with shape ``[B, N, D]``.
        segment_ids: Segment identifier per position, shape ``[B, N]``.
        valid_mask: Valid occurrence mask, shape ``[B, N]``.
        initial_state: Initial state for the position's segment, ``[B, N, D]``.

    Returns:
        State immediately before each transition, shape ``[B, N, D]``.
    """
    if alpha.shape != beta.shape or alpha.shape != initial_state.shape:
        raise ValueError("alpha, beta, and initial_state must have the same shape")
    if segment_ids.shape != alpha.shape[:2] or valid_mask.shape != alpha.shape[:2]:
        raise ValueError(
            "segment_ids and valid_mask must match alpha's first dimensions"
        )

    valid_mask = valid_mask.bool()
    valid_3d = valid_mask.unsqueeze(-1)
    prefix_alpha = torch.where(valid_3d, alpha, torch.ones_like(alpha))
    prefix_beta = torch.where(valid_3d, beta, torch.zeros_like(beta))

    length = alpha.size(1)
    offset = 1
    while offset < length:
        current_alpha = prefix_alpha
        current_beta = prefix_beta

        previous_alpha = torch.ones_like(current_alpha)
        previous_beta = torch.zeros_like(current_beta)
        previous_alpha[:, offset:] = current_alpha[:, :-offset]
        previous_beta[:, offset:] = current_beta[:, :-offset]

        same_segment = torch.zeros_like(valid_mask)
        same_segment[:, offset:] = (
            valid_mask[:, offset:]
            & valid_mask[:, :-offset]
            & (segment_ids[:, offset:] == segment_ids[:, :-offset])
        )
        combine = same_segment.unsqueeze(-1)

        composed_alpha = current_alpha * previous_alpha
        composed_beta = current_alpha * previous_beta + current_beta
        prefix_alpha = torch.where(combine, composed_alpha, current_alpha)
        prefix_beta = torch.where(combine, composed_beta, current_beta)
        offset *= 2

    previous_prefix_alpha = torch.ones_like(prefix_alpha)
    previous_prefix_beta = torch.zeros_like(prefix_beta)
    previous_prefix_alpha[:, 1:] = prefix_alpha[:, :-1]
    previous_prefix_beta[:, 1:] = prefix_beta[:, :-1]

    has_predecessor = torch.zeros_like(valid_mask)
    has_predecessor[:, 1:] = (
        valid_mask[:, 1:]
        & valid_mask[:, :-1]
        & (segment_ids[:, 1:] == segment_ids[:, :-1])
    )
    scanned_state = previous_prefix_alpha * initial_state + previous_prefix_beta
    state = torch.where(has_predecessor.unsqueeze(-1), scanned_state, initial_state)
    return torch.where(valid_3d, state, torch.zeros_like(state))


def segmented_block_affine_exclusive_scan(
    matrix: torch.Tensor,
    bias: torch.Tensor,
    segment_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Apply an exclusive block-affine scan inside contiguous segments.

    Each valid position represents ``h_after = matrix @ h_before + bias``.
    ReKTP uses fixed 2x2 feature blocks.

    On CUDA the scan runs a fused, differentiable Triton kernel; on CPU it uses
    the original PyTorch work-efficient recursive pair scan (logarithmic depth).

    Args:
        matrix: Block matrices with shape ``[B, N, H, 2, 2]``.
        bias: Block biases with shape ``[B, N, H, 2]``.
        segment_ids: Segment identifier per position, shape ``[B, N]``.
        valid_mask: Valid occurrence mask, shape ``[B, N]``.
        initial_state: Initial block states with shape ``[B, N, H, 2]``.

    Returns:
        State immediately before each transition, shape ``[B, N, H, 2]``.
    """
    if matrix.ndim != 5 or matrix.shape[-2:] != (2, 2):
        raise ValueError("matrix must have shape [B, N, H, 2, 2]")
    expected_vector_shape = matrix.shape[:-1]
    if bias.shape != expected_vector_shape:
        raise ValueError("bias must match matrix shape without its last dimension")
    if initial_state.shape != expected_vector_shape:
        raise ValueError(
            "initial_state must match matrix shape without its last dimension"
        )
    if segment_ids.shape != matrix.shape[:2] or valid_mask.shape != matrix.shape[:2]:
        raise ValueError(
            "segment_ids and valid_mask must match matrix's first dimensions"
        )
    if matrix.size(1) == 0:
        return torch.zeros_like(initial_state)

    # CUDA uses the fused differentiable Triton kernel; CPU stays on PyTorch.
    if matrix.is_cuda:
        try:
            from model.ReKTP.triton_scan import (
                triton_segmented_block_affine_exclusive_scan,
            )

            return triton_segmented_block_affine_exclusive_scan(
                matrix, bias, segment_ids, valid_mask, initial_state
            )
        except ImportError:
            pass  # Fall back to PyTorch when triton is unavailable.
    return _segmented_block_affine_exclusive_scan_py(
        matrix, bias, segment_ids, valid_mask, initial_state
    )


def _segmented_block_affine_exclusive_scan_py(
    matrix: torch.Tensor,
    bias: torch.Tensor,
    segment_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference implementation (work-efficient recursive pair scan)."""
    valid_mask = valid_mask.bool()
    head_mask = torch.zeros_like(valid_mask)
    head_mask[:, 0] = valid_mask[:, 0]
    head_mask[:, 1:] = valid_mask[:, 1:] & (
        ~valid_mask[:, :-1] | (segment_ids[:, 1:] != segment_ids[:, :-1])
    )

    # A column-vector transform h' = A h + b is stored in row-vector form as
    # [[A.T], [b.T]]. This lets one 3x2 @ 2x2 product compose both A and b.
    operator = torch.cat((matrix.transpose(-1, -2), bias.unsqueeze(-2)), dim=-2)
    identity = _row_affine_identity(operator)
    operator = torch.where(valid_mask[:, :, None, None, None], operator, identity)
    prefix, _ = _work_efficient_segmented_exclusive_scan(operator, head_mask)

    scanned_state = (
        torch.matmul(initial_state.unsqueeze(-2), prefix[..., :2, :]).squeeze(-2)
        + prefix[..., 2, :]
    )
    has_predecessor = valid_mask & ~head_mask
    state = torch.where(has_predecessor[:, :, None, None], scanned_state, initial_state)
    return torch.where(valid_mask[:, :, None, None], state, torch.zeros_like(state))


def _row_affine_identity(operator: torch.Tensor) -> torch.Tensor:
    identity = operator.new_tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    return identity.view(1, 1, 1, 3, 2).expand_as(operator)


def _compose_row_affine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compose row-vector affine operators with ``left`` applied first."""
    left_00 = left[..., 0, 0]
    left_01 = left[..., 0, 1]
    left_10 = left[..., 1, 0]
    left_11 = left[..., 1, 1]
    left_b0 = left[..., 2, 0]
    left_b1 = left[..., 2, 1]

    right_00 = right[..., 0, 0]
    right_01 = right[..., 0, 1]
    right_10 = right[..., 1, 0]
    right_11 = right[..., 1, 1]
    right_b0 = right[..., 2, 0]
    right_b1 = right[..., 2, 1]

    row00 = left_00 * right_00 + left_01 * right_10
    row01 = left_00 * right_01 + left_01 * right_11
    row10 = left_10 * right_00 + left_11 * right_10
    row11 = left_10 * right_01 + left_11 * right_11
    bias0 = left_b0 * right_00 + left_b1 * right_10 + right_b0
    bias1 = left_b0 * right_01 + left_b1 * right_11 + right_b1

    return torch.stack(
        (row00, row01, row10, row11, bias0, bias1),
        dim=-1,
    ).reshape(*left.shape[:-2], 3, 2)


def _combine_segmented(
    left_operator: torch.Tensor,
    left_head: torch.Tensor,
    right_operator: torch.Tensor,
    right_head: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    composed = _compose_row_affine(left_operator, right_operator)
    operator = torch.where(right_head[:, :, None, None, None], right_operator, composed)
    return operator, left_head | right_head


def _work_efficient_segmented_exclusive_scan(
    operator: torch.Tensor,
    head_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exclusive prefixes with O(N) work and O(log N) depth."""
    length = operator.size(1)
    if length == 1:
        return _row_affine_identity(operator), torch.zeros_like(head_mask)

    paired_length = length // 2
    paired_operator, paired_head = _combine_segmented(
        operator[:, : 2 * paired_length : 2],
        head_mask[:, : 2 * paired_length : 2],
        operator[:, 1 : 2 * paired_length : 2],
        head_mask[:, 1 : 2 * paired_length : 2],
    )
    pair_prefix, pair_prefix_head = _work_efficient_segmented_exclusive_scan(
        paired_operator, paired_head
    )

    odd_prefix, odd_prefix_head = _combine_segmented(
        pair_prefix,
        pair_prefix_head,
        operator[:, : 2 * paired_length : 2],
        head_mask[:, : 2 * paired_length : 2],
    )
    prefix = torch.stack((pair_prefix, odd_prefix), dim=2).flatten(1, 2)
    prefix_head = torch.stack((pair_prefix_head, odd_prefix_head), dim=2).flatten(1, 2)

    if length % 2:
        final_prefix, final_prefix_head = _combine_segmented(
            pair_prefix[:, -1:],
            pair_prefix_head[:, -1:],
            paired_operator[:, -1:],
            paired_head[:, -1:],
        )
        prefix = torch.cat((prefix, final_prefix), dim=1)
        prefix_head = torch.cat((prefix_head, final_prefix_head), dim=1)
    return prefix, prefix_head


__all__ = [
    "segmented_affine_exclusive_scan",
    "segmented_block_affine_exclusive_scan",
]
