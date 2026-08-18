"""Fused packed-space readout and KC pooling for inference.

Replaces the aten ``scatter_reduce`` chain (segment softmax with amax/sum
scatters, weighted sum scatter, and the event-embedding pooling) with one
Triton kernel per (batch, position) program. Each program gathers the K
occurrences of its position through the inverse of the KC packing order, so
the per-occurrence score/weight tensors are never materialized.

Positions whose occurrences are all invalid pool to zero here, while the aten
chain leaves the unpooled sum of invalid-slot embeddings there; the difference
never reaches the logits because masked positions are zeroed downstream.

Inference-only: the training path keeps the aten implementation, so gradient
semantics are untouched.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _readout_pool_kernel(
    state_ptr,
    skemb_ptr,
    skchg_ptr,
    qproj_ptr,
    valid_ptr,
    inv_ptr,
    out_ptr,
    pooled_sk_ptr,
    pooled_ch_ptr,
    counts_ptr,
    wl_ptr,
    ws_ptr,
    bias,
    S,
    K,
    P,
    H,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One program per (batch row, position): fused segment-softmax readout
    plus mean pooling of the skill/change embeddings."""
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    k_offs = tl.arange(0, BLOCK_K)
    h_offs = tl.arange(0, BLOCK_H)
    k_mask = k_offs < K
    h_mask = h_offs < H

    # Packed index of each (position, slot) occurrence and its validity. An
    # invalid slot's index may fall outside the trimmed packed width, so it is
    # clamped before any load; the load mask keeps its values unused.
    flat = pid_b * S * K + pid_s * K + k_offs
    kv = tl.load(valid_ptr + flat, mask=k_mask, other=0) != 0
    j = tl.load(inv_ptr + flat, mask=k_mask, other=0).to(tl.int32)
    j = tl.where(kv & (j < P), j, 0)

    offs = pid_b * P * H + j[:, None] * H + h_offs[None, :]
    m2 = (k_mask & kv)[:, None] & h_mask[None, :]
    state = tl.load(state_ptr + offs, mask=m2, other=0.0)
    sk = tl.load(skemb_ptr + offs, mask=m2, other=0.0)
    ch = tl.load(skchg_ptr + offs, mask=m2, other=0.0)

    wl = tl.load(wl_ptr + h_offs, mask=h_mask, other=0.0)
    ws = tl.load(ws_ptr + h_offs, mask=h_mask, other=0.0)
    qp = tl.load(qproj_ptr + pid_b * S + pid_s)
    score = tl.sum(state * wl[None, :], 1) + tl.sum(sk * ws[None, :], 1) + qp + bias
    score = tl.where(kv, score, float("-inf"))

    m = tl.max(score, 0)
    e = tl.exp(score - m)
    e = tl.where(kv, e, 0.0)
    z = tl.sum(e, 0)
    w = e / tl.where(z > 0, z, 1.0)

    out = tl.sum(w[:, None] * state, 0)
    count = tl.sum(kv.to(tl.float32), 0)
    denom = tl.maximum(count, 1.0)
    pooled_sk = tl.sum(tl.where(kv[:, None], sk, 0.0), 0) / denom
    pooled_ch = tl.sum(tl.where(kv[:, None], ch, 0.0), 0) / denom

    o_off = (pid_b * S + pid_s) * H + h_offs
    tl.store(out_ptr + o_off, out, mask=h_mask)
    tl.store(pooled_sk_ptr + o_off, pooled_sk, mask=h_mask)
    tl.store(pooled_ch_ptr + o_off, pooled_ch, mask=h_mask)
    tl.store(counts_ptr + pid_b * S + pid_s, count)


def fused_readout_pool(
    packed_state: torch.Tensor,
    packed_skill_embedding: torch.Tensor,
    packed_skill_change: torch.Tensor,
    question_proj: torch.Tensor,
    slot_valid: torch.Tensor,
    kc_inverse: torch.Tensor,
    max_skills: int,
    weight: torch.Tensor,
    bias: float,
) -> tuple[torch.Tensor, ...]:
    """Run the fused readout + pooling kernel.

    Args:
        packed_state: Scanned per-occurrence states, [B, P, H].
        packed_skill_embedding: Skill embeddings at the packed slots, [B, P, H].
        packed_skill_change: Change embeddings at the packed slots, [B, P, H].
        question_proj: Per-position question score projection, [B, S].
        slot_valid: Occurrence validity over the flat [B, S*K] slot domain.
        kc_inverse: Inverse of the packing permutation over the same flat
            domain: flat slot -> packed index, [B, S*K]. Entries pointing
            past the trimmed packed width belong to invalid slots and are
            clamped away by the kernel.
        max_skills: K, the flattened KC width per position.
        weight: Row vector of ``local_readout``, [1, 3H]; its first H entries
            multiply the state and the next H the skill embedding.
        bias: The scalar readout bias.

    Returns:
        (readout [B, S, H], pooled_skill [B, S, H], pooled_change [B, S, H],
        counts [B, S]).
    """
    batch, packed_len, hidden = packed_state.shape
    seq_len = question_proj.shape[1]
    device = packed_state.device

    if kc_inverse is None or slot_valid is None:
        raise ValueError("kc_inverse and slot_valid must be precomputed")
    if slot_valid.ndim != 2 or slot_valid.shape[1] != seq_len * max_skills:
        raise ValueError("slot_valid must have shape [B, S*K]")

    # One pooled allocation for the three [B, S, H] outputs; each slice is
    # contiguous so the kernel's compact [B, S, H] offsets stay valid.
    n_out = batch * seq_len * hidden
    pool = torch.empty(3 * n_out, device=device)
    readout = pool[:n_out].view(batch, seq_len, hidden)
    pooled_skill = pool[n_out : 2 * n_out].view(batch, seq_len, hidden)
    pooled_change = pool[2 * n_out :].view(batch, seq_len, hidden)
    counts = torch.empty(batch, seq_len, device=device)

    _readout_pool_kernel[(batch, seq_len)](
        packed_state,
        packed_skill_embedding,
        packed_skill_change,
        question_proj,
        slot_valid,
        kc_inverse,
        readout,
        pooled_skill,
        pooled_change,
        counts,
        weight[:, :hidden].reshape(-1),
        weight[:, hidden : 2 * hidden].reshape(-1),
        float(bias),
        seq_len,
        max_skills,
        packed_len,
        hidden,
        BLOCK_K=triton.next_power_of_2(max_skills),
        BLOCK_H=triton.next_power_of_2(hidden),
        num_warps=4,
    )
    return readout, pooled_skill, pooled_change, counts


__all__ = ["fused_readout_pool"]
