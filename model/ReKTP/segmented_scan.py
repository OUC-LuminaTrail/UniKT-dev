"""Parallel segmented scans for KC-specific affine state transitions."""

import torch


def _block_matvec(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.matmul(matrix, vector.unsqueeze(-1)).squeeze(-1)


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
    ``matrix`` may contain any square block size; ReKTP uses 2x2 blocks.
    Feature blocks are independent, while their affine transitions compose in
    logarithmic sequential depth with a Hillis-Steele doubling schedule.

    Args:
        matrix: Block matrices with shape ``[B, N, H, P, P]``.
        bias: Block biases with shape ``[B, N, H, P]``.
        segment_ids: Segment identifier per position, shape ``[B, N]``.
        valid_mask: Valid occurrence mask, shape ``[B, N]``.
        initial_state: Initial block states with shape ``[B, N, H, P]``.

    Returns:
        State immediately before each transition, shape ``[B, N, H, P]``.
    """
    if matrix.ndim != 5 or matrix.size(-1) != matrix.size(-2):
        raise ValueError("matrix must have shape [B, N, H, P, P]")
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

    valid_mask = valid_mask.bool()
    valid_matrix = valid_mask[:, :, None, None, None]
    valid_vector = valid_mask[:, :, None, None]
    block_size = matrix.size(-1)
    identity = torch.eye(block_size, device=matrix.device, dtype=matrix.dtype)
    identity = identity.view(1, 1, 1, block_size, block_size)
    identity = identity.expand_as(matrix)

    prefix_matrix = torch.where(valid_matrix, matrix, identity)
    prefix_bias = torch.where(valid_vector, bias, torch.zeros_like(bias))

    length = matrix.size(1)
    offset = 1
    while offset < length:
        current_matrix = prefix_matrix
        current_bias = prefix_bias

        previous_matrix = identity.clone()
        previous_bias = torch.zeros_like(current_bias)
        previous_matrix[:, offset:] = current_matrix[:, :-offset]
        previous_bias[:, offset:] = current_bias[:, :-offset]

        same_segment = torch.zeros_like(valid_mask)
        same_segment[:, offset:] = (
            valid_mask[:, offset:]
            & valid_mask[:, :-offset]
            & (segment_ids[:, offset:] == segment_ids[:, :-offset])
        )
        combine_matrix = same_segment[:, :, None, None, None]
        combine_vector = same_segment[:, :, None, None]

        composed_matrix = torch.matmul(current_matrix, previous_matrix)
        composed_bias = _block_matvec(current_matrix, previous_bias) + current_bias
        prefix_matrix = torch.where(combine_matrix, composed_matrix, current_matrix)
        prefix_bias = torch.where(combine_vector, composed_bias, current_bias)
        offset *= 2

    previous_prefix_matrix = identity.clone()
    previous_prefix_bias = torch.zeros_like(prefix_bias)
    previous_prefix_matrix[:, 1:] = prefix_matrix[:, :-1]
    previous_prefix_bias[:, 1:] = prefix_bias[:, :-1]

    has_predecessor = torch.zeros_like(valid_mask)
    has_predecessor[:, 1:] = (
        valid_mask[:, 1:]
        & valid_mask[:, :-1]
        & (segment_ids[:, 1:] == segment_ids[:, :-1])
    )
    scanned_state = (
        _block_matvec(previous_prefix_matrix, initial_state) + previous_prefix_bias
    )
    state = torch.where(has_predecessor[:, :, None, None], scanned_state, initial_state)
    return torch.where(valid_vector, state, torch.zeros_like(state))


__all__ = [
    "segmented_affine_exclusive_scan",
    "segmented_block_affine_exclusive_scan",
]
