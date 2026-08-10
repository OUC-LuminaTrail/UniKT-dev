"""Scan kernels for the segmented scalar affine exclusive scan.

Each position carries a scalar operator ``a`` and bias ``b``; the scan
returns the state before that operator is applied, resetting at segment
boundaries.

- Forward (Triton, 1x1 blocks): one program per batch row, vectorised over
  the hidden channels, running a sequential exclusive scan with segment
  resets along the sequence.
- Backward (Triton): an adjoint reverse scan over the recurrence
  ``h_{i+1} = a_i h_i + b_i`` (``a_i = g_i + a_i * a_{i+1}``), plus a forward
  prefix scan to recover ``d init``.

``d matrix`` / ``d bias`` match PyTorch autograd only while ``initial_state``
is constant within a segment, which holds because it is derived from the
segment id. ``d init`` is exact per position (``prefix_i * g_i``) regardless.

CPU inputs fall back to a pure-PyTorch serial scan with identical semantics
and full autograd coverage.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_scalar(
    matrix_ptr,
    bias_ptr,
    seg_ptr,
    valid_ptr,
    init_ptr,
    out_ptr,
    N,
    H,
    stride_mb,
    stride_mn,
    stride_vb,
    stride_vn,
    BLOCK_H: tl.constexpr,
):
    """Forward segmented scalar affine exclusive scan for one batch row.

    1x1 specialization of ``_fwd_kernel_scalar``: the carry is a scalar pair
    ``(c_a, c_b)`` per hidden channel, the output is ``c_a * init + c_b``, and
    the carry advances by composing the position operator ``(a, b)`` after it
    (``c_a <- a * c_a``, ``c_b <- a * c_b + b``).
    """
    pid_b = tl.program_id(0)
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # carry = identity affine operator (a=1, b=0).
    ca = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
    cb = tl.zeros([BLOCK_H], dtype=tl.float32)

    prev_valid = tl.zeros([], dtype=tl.int32)
    prev_seg = tl.full([], -1, dtype=tl.int32)

    m_row = matrix_ptr + pid_b * stride_mb
    b_row = bias_ptr + pid_b * stride_vb
    i_row = init_ptr + pid_b * stride_vb
    o_row = out_ptr + pid_b * stride_vb
    s_row = seg_ptr + pid_b * N
    v_row = valid_ptr + pid_b * N

    for n in range(N):
        seg_n = tl.load(s_row + n).to(tl.int32)
        valid_n = tl.load(v_row + n).to(tl.int32)
        vn = valid_n != 0

        na = tl.load(m_row + n * stride_mn + h_offs, mask=h_mask, other=0.0).to(
            tl.float32
        )
        nb = tl.load(b_row + n * stride_vn + h_offs, mask=h_mask, other=0.0).to(
            tl.float32
        )
        ni = tl.load(i_row + n * stride_vn + h_offs, mask=h_mask, other=0.0).to(
            tl.float32
        )

        # Invalid positions act as the identity operator.
        na = tl.where(vn, na, 1.0)
        nb = tl.where(vn, nb, 0.0)

        # Segment head: valid, and either row start, prior invalid, or new id.
        is_head = vn & ((n == 0) | (prev_valid == 0) | (seg_n != prev_seg))
        ca = tl.where(is_head, 1.0, ca)
        cb = tl.where(is_head, 0.0, cb)

        # Output = carry applied to init, zeroed at invalid positions.
        o = ca * ni + cb
        o = tl.where(vn, o, 0.0)
        tl.store(o_row + n * stride_vn + h_offs, o, mask=h_mask)

        # Advance carry = position operator composed after carry.
        nca = na * ca
        ncb = na * cb + nb
        ca = nca
        cb = ncb

        prev_valid = valid_n
        prev_seg = seg_n


@triton.jit
def _bwd_adj_kernel_scalar(
    dmat_ptr,
    dbias_ptr,
    g_ptr,
    h_ptr,
    mat_ptr,
    seg_ptr,
    valid_ptr,
    N,
    H,
    stride_mb,
    stride_mn,
    stride_vb,
    stride_vn,
    BLOCK_H: tl.constexpr,
):
    """Adjoint reverse scan producing ``d matrix`` and ``d bias``.

    Scalar specialization of ``_bwd_adj_kernel``: for the recurrence
    ``h_{i+1} = a_i h_i + b_i`` the adjoint is ``a_i = g_i + a_i * a_{i+1}``,
    ``dA_i = a_{i+1} * h_i`` and ``db_i = a_{i+1}``, nonzero only where the
    operator is used, that is where ``i+1`` shares the segment. ``h`` is the
    forward output.
    """
    pid_b = tl.program_id(0)
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    adj = tl.zeros([BLOCK_H], dtype=tl.float32)
    next_valid = tl.zeros([], dtype=tl.int32)
    next_seg = tl.full([], -1, dtype=tl.int32)

    dm_row = dmat_ptr + pid_b * stride_mb
    db_row = dbias_ptr + pid_b * stride_vb
    g_row = g_ptr + pid_b * stride_vb
    h_row = h_ptr + pid_b * stride_vb
    m_row = mat_ptr + pid_b * stride_mb
    s_row = seg_ptr + pid_b * N
    v_row = valid_ptr + pid_b * N

    for n in range(N):
        i = N - 1 - n
        seg_i = tl.load(s_row + i).to(tl.int32)
        valid_i = tl.load(v_row + i).to(tl.int32)
        vi = valid_i != 0

        ai = tl.load(m_row + i * stride_mn + h_offs, mask=h_mask, other=0.0).to(
            tl.float32
        )
        hi = tl.load(h_row + i * stride_vn + h_offs, mask=h_mask, other=0.0).to(
            tl.float32
        )
        gi = tl.load(g_row + i * stride_vn + h_offs, mask=h_mask, other=0.0).to(
            tl.float32
        )

        same_next = vi & (next_valid != 0) & (seg_i == next_seg)
        ae = tl.where(same_next, adj, 0.0)

        # adj_i = g_i + a_i * adj_eff
        ai_adj = gi + ai * ae

        # dA_i = adj_eff * h_i, db_i = adj_eff, nonzero only where op is used.
        da = tl.where(same_next, ae * hi, 0.0)
        dbo = tl.where(same_next, ae, 0.0)

        # Zero the gradient and cut the adjoint chain at invalid positions.
        da = tl.where(vi, da, 0.0)
        dbo = tl.where(vi, dbo, 0.0)
        adj = tl.where(vi, ai_adj, 0.0)

        tl.store(dm_row + i * stride_mn + h_offs, da, mask=h_mask)
        tl.store(db_row + i * stride_vn + h_offs, dbo, mask=h_mask)

        next_valid = valid_i
        next_seg = seg_i


@triton.jit
def _bwd_dinit_kernel_scalar(
    dinit_ptr,
    g_ptr,
    mat_ptr,
    seg_ptr,
    valid_ptr,
    N,
    H,
    stride_mb,
    stride_mn,
    stride_vb,
    stride_vn,
    BLOCK_H: tl.constexpr,
):
    """Recompute exclusive prefix scalars to recover ``d init``.

    Scalar specialization of ``_bwd_dinit_kernel``: yields ``prefix_i * g_i``,
    matching the per-position use of ``initial_state`` rather than only using
    it at segment heads.
    """
    pid_b = tl.program_id(0)
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # carry = scalar part of the exclusive prefix, starting from the identity.
    c = tl.full([BLOCK_H], 1.0, dtype=tl.float32)

    prev_valid = tl.zeros([], dtype=tl.int32)
    prev_seg = tl.full([], -1, dtype=tl.int32)

    di_row = dinit_ptr + pid_b * stride_vb
    g_row = g_ptr + pid_b * stride_vb
    m_row = mat_ptr + pid_b * stride_mb
    s_row = seg_ptr + pid_b * N
    v_row = valid_ptr + pid_b * N

    for n in range(N):
        seg_n = tl.load(s_row + n).to(tl.int32)
        valid_n = tl.load(v_row + n).to(tl.int32)
        vn = valid_n != 0

        a = tl.load(m_row + n * stride_mn + h_offs, mask=h_mask, other=0.0).to(
            tl.float32
        )
        g = tl.load(g_row + n * stride_vn + h_offs, mask=h_mask, other=0.0).to(
            tl.float32
        )

        is_head = vn & ((n == 0) | (prev_valid == 0) | (seg_n != prev_seg))
        c = tl.where(is_head, 1.0, c)

        # dinit_n = c_n * g_n
        di = c * g
        di = tl.where(vn, di, 0.0)
        tl.store(di_row + n * stride_vn + h_offs, di, mask=h_mask)

        # Advance carry = a * c, using the identity at invalid positions.
        a_eff = tl.where(vn, a, 1.0)
        c = a_eff * c

        prev_valid = valid_n
        prev_seg = seg_n


class _SegmentedScalarAffineExclusiveScan(torch.autograd.Function):
    """Differentiable segmented scalar affine exclusive scan (1x1 blocks)."""

    @staticmethod
    def forward(ctx, matrix, bias, segment_ids, valid_mask, initial_state):
        if matrix.ndim != 5 or matrix.shape[-2:] != (1, 1):
            raise ValueError("matrix must have shape [B, N, H, 1, 1]")
        if matrix.size(1) == 0:
            ctx.empty = True
            ctx.save_for_backward(matrix, bias, segment_ids, valid_mask, initial_state)
            return torch.zeros_like(initial_state)
        ctx.empty = False

        matrix = matrix.contiguous()
        bias = bias.contiguous()
        initial_state = initial_state.contiguous()
        segment_ids = segment_ids.contiguous()
        valid_mask = valid_mask.bool().contiguous()

        batch, length, heads, _, _ = matrix.shape
        out = torch.empty_like(initial_state)
        block_h = triton.next_power_of_2(heads)
        ctx.block_h = block_h
        ctx.length = length
        ctx.heads = heads

        _fwd_kernel_scalar[(batch,)](
            matrix,
            bias,
            segment_ids,
            valid_mask,
            initial_state,
            out,
            length,
            heads,
            matrix.stride(0),
            matrix.stride(1),
            initial_state.stride(0),
            initial_state.stride(1),
            BLOCK_H=block_h,
        )
        ctx.save_for_backward(matrix, bias, segment_ids, valid_mask, initial_state, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.empty:
            return torch.zeros_like(ctx.saved_tensors[0]), None, None, None, None
        matrix, bias, segment_ids, valid_mask, initial_state, out = ctx.saved_tensors
        grad_out = grad_out.contiguous()

        batch = matrix.shape[0]
        dmat = torch.zeros_like(matrix)
        dbias = torch.zeros_like(bias)
        dinit = torch.zeros_like(initial_state)
        length = ctx.length
        heads = ctx.heads
        block_h = ctx.block_h

        _bwd_adj_kernel_scalar[(batch,)](
            dmat,
            dbias,
            grad_out,
            out,
            matrix,
            segment_ids,
            valid_mask,
            length,
            heads,
            matrix.stride(0),
            matrix.stride(1),
            initial_state.stride(0),
            initial_state.stride(1),
            BLOCK_H=block_h,
        )
        _bwd_dinit_kernel_scalar[(batch,)](
            dinit,
            grad_out,
            matrix,
            segment_ids,
            valid_mask,
            length,
            heads,
            matrix.stride(0),
            matrix.stride(1),
            initial_state.stride(0),
            initial_state.stride(1),
            BLOCK_H=block_h,
        )
        return dmat, dbias, None, None, dinit


def _serial_scalar_affine_exclusive_scan(
    matrix: torch.Tensor,
    bias: torch.Tensor,
    segment_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Serial scan mirroring the Triton kernel, for any device.

    Pure PyTorch ops, so autograd covers the backward pass. Semantics match
    ``_fwd_kernel_scalar``: output is the carry composed over the segment so
    far, applied to the per-position initial state; the carry resets to the
    identity at segment heads and invalid positions act as the identity and
    output zero.
    """
    batch, length, heads, _, _ = matrix.shape
    matrix = matrix[..., 0, 0]  # [B, N, H]
    bias = bias[..., 0]
    initial_state = initial_state[..., 0]
    carry_a = torch.ones(batch, heads, device=matrix.device, dtype=matrix.dtype)
    carry_b = torch.zeros(batch, heads, device=matrix.device, dtype=matrix.dtype)
    prev_valid = torch.zeros(batch, dtype=torch.bool, device=matrix.device)
    prev_seg = torch.full((batch,), -1, dtype=segment_ids.dtype, device=matrix.device)
    outputs = []
    for n in range(length):
        op = matrix[:, n]
        op_bias = bias[:, n]
        init_n = initial_state[:, n]
        valid_n = valid_mask[:, n]
        seg_n = segment_ids[:, n]

        # Segment head: valid, and either row start, prior invalid, or new id.
        # Reset happens before the output: a segment head emits its own
        # initial state (carry = identity), matching the Triton kernel.
        is_head = valid_n & ((n == 0) | ~prev_valid | (seg_n != prev_seg))
        carry_a = torch.where(is_head[:, None], torch.ones_like(carry_a), carry_a)
        carry_b = torch.where(is_head[:, None], torch.zeros_like(carry_b), carry_b)

        # Output = carry applied to init, zeroed at invalid positions.
        state = carry_a * init_n + carry_b
        outputs.append(torch.where(valid_n[:, None], state, torch.zeros_like(state)))

        # Advance carry = position operator composed after carry; invalid acts
        # as the identity.
        op_eff = torch.where(valid_n[:, None], op, torch.ones_like(op))
        op_bias_eff = torch.where(
            valid_n[:, None], op_bias, torch.zeros_like(op_bias)
        )
        carry_a = op_eff * carry_a
        carry_b = op_eff * carry_b + op_bias_eff
        prev_valid = valid_n
        prev_seg = seg_n
    return torch.stack(outputs, dim=1).unsqueeze(-1)


def segmented_scalar_affine_exclusive_scan(
    matrix: torch.Tensor,
    bias: torch.Tensor,
    segment_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Apply an exclusive scalar affine scan inside contiguous segments.

    Each valid position represents ``h_after = a * h_before + b`` with 1x1
    blocks (``matrix`` has shape ``[B, N, H, 1, 1]``). Block size 1 on CUDA
    runs the fused differentiable Triton kernel; CPU inputs fall back to the
    serial PyTorch scan with identical semantics.

    Args:
        matrix: Scalar operators with shape ``[B, N, H, 1, 1]``.
        bias: Scalar biases with shape ``[B, N, H, 1]``.
        segment_ids: Segment identifier per position, shape ``[B, N]``.
        valid_mask: Valid occurrence mask, shape ``[B, N]``.
        initial_state: Initial states with shape ``[B, N, H, 1]``.

    Returns:
        State immediately before each transition, shape ``[B, N, H, 1]``.
    """
    if matrix.ndim != 5 or matrix.shape[-2:] != (1, 1):
        raise ValueError("matrix must have shape [B, N, H, 1, 1]")
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
    if matrix.is_cuda:
        return _SegmentedScalarAffineExclusiveScan.apply(
            matrix, bias, segment_ids, valid_mask, initial_state
        )
    return _serial_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial_state
    )


__all__ = ["segmented_scalar_affine_exclusive_scan"]
