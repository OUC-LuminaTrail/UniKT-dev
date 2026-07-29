"""Parallel segmented scans for question- and KC-specific state transitions."""

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


def segmented_normalized_evidence_exclusive_scan(
    decay: torch.Tensor,
    write_mass: torch.Tensor,
    candidate: torch.Tensor,
    segment_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    prior_state: torch.Tensor,
    prior_mass: float = 1.0,
) -> torch.Tensor:
    """Read a bounded state from parallel scans of evidence and evidence mass.

    Each transition updates decayed evidence ``E`` and its non-negative mass ``M``::

        E_after = decay * E_before + write_mass * candidate
        M_after = decay * M_before + write_mass

    The returned state is exclusive of the current write. Its current gap decay
    is applied before combining historical evidence with a fixed prior::

        state = (prior_mass * prior_state + decay * E_before)
                / (prior_mass + decay * M_before)

    Evidence and mass are concatenated so both recurrences share one segmented
    affine scan. With bounded priors/candidates and non-negative masses, the
    normalized state remains inside their elementwise convex hull.
    """
    if not (
        decay.shape
        == write_mass.shape
        == candidate.shape
        == prior_state.shape
    ):
        raise ValueError(
            "decay, write_mass, candidate, and prior_state must have the same shape"
        )
    if prior_mass <= 0.0:
        raise ValueError("prior_mass must be positive")

    valid_3d = valid_mask.bool().unsqueeze(-1)
    decay = torch.where(valid_3d, decay, torch.ones_like(decay))
    write_mass = torch.where(valid_3d, write_mass, torch.zeros_like(write_mass))

    transition_alpha = torch.cat([decay, decay], dim=-1)
    transition_beta = torch.cat(
        [write_mass * candidate, write_mass],
        dim=-1,
    )
    pre_transition = segmented_affine_exclusive_scan(
        transition_alpha,
        transition_beta,
        segment_ids,
        valid_mask,
        torch.zeros_like(transition_alpha),
    )
    evidence_before, mass_before = pre_transition.chunk(2, dim=-1)
    evidence_for_read = decay * evidence_before
    mass_for_read = decay * mass_before
    state = (prior_mass * prior_state + evidence_for_read) / (
        prior_mass + mass_for_read
    )
    return torch.where(valid_3d, state, torch.zeros_like(state))


__all__ = [
    "segmented_affine_exclusive_scan",
    "segmented_normalized_evidence_exclusive_scan",
]
