"""Triton kernels for the segmented block-affine exclusive scan.

Each position carries a fixed 2x2 block operator ``A`` and bias ``b``; the scan
returns the state before that operator is applied, resetting at segment
boundaries.

- Forward: one program per batch row, vectorised over ``hidden_block``, running
  a sequential exclusive scan with segment resets along the sequence.
- Backward: an adjoint reverse scan over the recurrence
  ``h_{i+1} = A_i h_i + b_i``, plus a forward prefix scan to recover ``d init``.

``d matrix`` / ``d bias`` match PyTorch autograd only while ``initial_state`` is
constant within a segment, which holds because it is derived from the segment id.
``d init`` is exact per position (``prefix_i^T @ g_i``) regardless.

CUDA only; CPU uses the PyTorch implementation in ``segmented_scan.py``.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
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
    """Forward segmented block-affine exclusive scan for one batch row.

    ``carry`` holds the exclusive prefix operator; it resets to the identity at
    each segment head. Invalid positions act as the identity and output zero.
    """
    pid_b = tl.program_id(0)
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # carry = identity affine operator (A=I, b=0).
    ca00 = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
    ca01 = tl.zeros([BLOCK_H], dtype=tl.float32)
    ca10 = tl.zeros([BLOCK_H], dtype=tl.float32)
    ca11 = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
    cb0 = tl.zeros([BLOCK_H], dtype=tl.float32)
    cb1 = tl.zeros([BLOCK_H], dtype=tl.float32)

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

        m_base = m_row + n * stride_mn + h_offs * 4
        na00 = tl.load(m_base + 0, mask=h_mask, other=0.0).to(tl.float32)
        na01 = tl.load(m_base + 1, mask=h_mask, other=0.0).to(tl.float32)
        na10 = tl.load(m_base + 2, mask=h_mask, other=0.0).to(tl.float32)
        na11 = tl.load(m_base + 3, mask=h_mask, other=0.0).to(tl.float32)

        b_base = b_row + n * stride_vn + h_offs * 2
        nb0 = tl.load(b_base + 0, mask=h_mask, other=0.0).to(tl.float32)
        nb1 = tl.load(b_base + 1, mask=h_mask, other=0.0).to(tl.float32)

        i_base = i_row + n * stride_vn + h_offs * 2
        ni0 = tl.load(i_base + 0, mask=h_mask, other=0.0).to(tl.float32)
        ni1 = tl.load(i_base + 1, mask=h_mask, other=0.0).to(tl.float32)

        # Invalid positions act as the identity operator.
        na00 = tl.where(vn, na00, 1.0)
        na01 = tl.where(vn, na01, 0.0)
        na10 = tl.where(vn, na10, 0.0)
        na11 = tl.where(vn, na11, 1.0)
        nb0 = tl.where(vn, nb0, 0.0)
        nb1 = tl.where(vn, nb1, 0.0)

        # Segment head: valid, and either row start, prior invalid, or new id.
        is_head = vn & ((n == 0) | (prev_valid == 0) | (seg_n != prev_seg))
        ca00 = tl.where(is_head, 1.0, ca00)
        ca01 = tl.where(is_head, 0.0, ca01)
        ca10 = tl.where(is_head, 0.0, ca10)
        ca11 = tl.where(is_head, 1.0, ca11)
        cb0 = tl.where(is_head, 0.0, cb0)
        cb1 = tl.where(is_head, 0.0, cb1)

        # Output = carry @ init, zeroed at invalid positions.
        o0 = ca00 * ni0 + ca01 * ni1 + cb0
        o1 = ca10 * ni0 + ca11 * ni1 + cb1
        o0 = tl.where(vn, o0, 0.0)
        o1 = tl.where(vn, o1, 0.0)
        o_base = o_row + n * stride_vn + h_offs * 2
        tl.store(o_base + 0, o0, mask=h_mask)
        tl.store(o_base + 1, o1, mask=h_mask)

        # Advance carry = op composed after carry.
        nca00 = na00 * ca00 + na01 * ca10
        nca01 = na00 * ca01 + na01 * ca11
        nca10 = na10 * ca00 + na11 * ca10
        nca11 = na10 * ca01 + na11 * ca11
        ncb0 = na00 * cb0 + na01 * cb1 + nb0
        ncb1 = na10 * cb0 + na11 * cb1 + nb1
        ca00 = nca00
        ca01 = nca01
        ca10 = nca10
        ca11 = nca11
        cb0 = ncb0
        cb1 = ncb1

        prev_valid = valid_n
        prev_seg = seg_n


@triton.jit
def _bwd_adj_kernel(
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

    For the recurrence ``h_{i+1} = A_i h_i + b_i`` the adjoint is
    ``a_i = g_i + A_i^T a_{i+1}``, scanned backwards. ``dA_i = a_{i+1} h_i^T``
    and ``db_i = a_{i+1}`` are nonzero only where the operator is used, that is
    where ``i+1`` shares the segment. ``h`` is the forward output.
    """
    pid_b = tl.program_id(0)
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    a0 = tl.zeros([BLOCK_H], dtype=tl.float32)
    a1 = tl.zeros([BLOCK_H], dtype=tl.float32)
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

        m_base = m_row + i * stride_mn + h_offs * 4
        a00 = tl.load(m_base + 0, mask=h_mask, other=0.0).to(tl.float32)
        a01 = tl.load(m_base + 1, mask=h_mask, other=0.0).to(tl.float32)
        a10 = tl.load(m_base + 2, mask=h_mask, other=0.0).to(tl.float32)
        a11 = tl.load(m_base + 3, mask=h_mask, other=0.0).to(tl.float32)

        h_base = h_row + i * stride_vn + h_offs * 2
        h0 = tl.load(h_base + 0, mask=h_mask, other=0.0).to(tl.float32)
        h1 = tl.load(h_base + 1, mask=h_mask, other=0.0).to(tl.float32)

        g_base = g_row + i * stride_vn + h_offs * 2
        g0 = tl.load(g_base + 0, mask=h_mask, other=0.0).to(tl.float32)
        g1 = tl.load(g_base + 1, mask=h_mask, other=0.0).to(tl.float32)

        same_next = vi & (next_valid != 0) & (seg_i == next_seg)
        ae0 = tl.where(same_next, a0, 0.0)
        ae1 = tl.where(same_next, a1, 0.0)

        # a_i = g_i + A_i^T @ a_eff
        ai0 = g0 + a00 * ae0 + a10 * ae1
        ai1 = g1 + a01 * ae0 + a11 * ae1

        # dA_i = a_eff (x) h_i, db_i = a_eff, nonzero only where op is used.
        da00 = tl.where(same_next, ae0 * h0, 0.0)
        da01 = tl.where(same_next, ae0 * h1, 0.0)
        da10 = tl.where(same_next, ae1 * h0, 0.0)
        da11 = tl.where(same_next, ae1 * h1, 0.0)
        dbo0 = tl.where(same_next, ae0, 0.0)
        dbo1 = tl.where(same_next, ae1, 0.0)

        # Zero the gradient and cut the adjoint chain at invalid positions.
        da00 = tl.where(vi, da00, 0.0)
        da01 = tl.where(vi, da01, 0.0)
        da10 = tl.where(vi, da10, 0.0)
        da11 = tl.where(vi, da11, 0.0)
        dbo0 = tl.where(vi, dbo0, 0.0)
        dbo1 = tl.where(vi, dbo1, 0.0)
        a0 = tl.where(vi, ai0, 0.0)
        a1 = tl.where(vi, ai1, 0.0)

        dm_base = dm_row + i * stride_mn + h_offs * 4
        tl.store(dm_base + 0, da00, mask=h_mask)
        tl.store(dm_base + 1, da01, mask=h_mask)
        tl.store(dm_base + 2, da10, mask=h_mask)
        tl.store(dm_base + 3, da11, mask=h_mask)
        db_base = db_row + i * stride_vn + h_offs * 2
        tl.store(db_base + 0, dbo0, mask=h_mask)
        tl.store(db_base + 1, dbo1, mask=h_mask)

        next_valid = valid_i
        next_seg = seg_i


@triton.jit
def _bwd_dinit_kernel(
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
    """Recompute exclusive prefix matrices to recover ``d init``.

    Yields ``prefix_i^T @ g_i``, matching the per-position use of
    ``initial_state`` rather than only using it at segment heads.
    """
    pid_b = tl.program_id(0)
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # carry = matrix part of the exclusive prefix, starting from the identity.
    c00 = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
    c01 = tl.zeros([BLOCK_H], dtype=tl.float32)
    c10 = tl.zeros([BLOCK_H], dtype=tl.float32)
    c11 = tl.full([BLOCK_H], 1.0, dtype=tl.float32)

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

        m_base = m_row + n * stride_mn + h_offs * 4
        a00 = tl.load(m_base + 0, mask=h_mask, other=0.0).to(tl.float32)
        a01 = tl.load(m_base + 1, mask=h_mask, other=0.0).to(tl.float32)
        a10 = tl.load(m_base + 2, mask=h_mask, other=0.0).to(tl.float32)
        a11 = tl.load(m_base + 3, mask=h_mask, other=0.0).to(tl.float32)

        g_base = g_row + n * stride_vn + h_offs * 2
        g0 = tl.load(g_base + 0, mask=h_mask, other=0.0).to(tl.float32)
        g1 = tl.load(g_base + 1, mask=h_mask, other=0.0).to(tl.float32)

        is_head = vn & ((n == 0) | (prev_valid == 0) | (seg_n != prev_seg))
        c00 = tl.where(is_head, 1.0, c00)
        c01 = tl.where(is_head, 0.0, c01)
        c10 = tl.where(is_head, 0.0, c10)
        c11 = tl.where(is_head, 1.0, c11)

        # dinit_n = carry^T @ g_n
        di0 = c00 * g0 + c10 * g1
        di1 = c01 * g0 + c11 * g1
        di0 = tl.where(vn, di0, 0.0)
        di1 = tl.where(vn, di1, 0.0)
        di_base = di_row + n * stride_vn + h_offs * 2
        tl.store(di_base + 0, di0, mask=h_mask)
        tl.store(di_base + 1, di1, mask=h_mask)

        # Advance carry = A_n @ carry, using the identity at invalid positions.
        a00e = tl.where(vn, a00, 1.0)
        a01e = tl.where(vn, a01, 0.0)
        a10e = tl.where(vn, a10, 0.0)
        a11e = tl.where(vn, a11, 1.0)
        nc00 = a00e * c00 + a01e * c10
        nc01 = a00e * c01 + a01e * c11
        nc10 = a10e * c00 + a11e * c10
        nc11 = a10e * c01 + a11e * c11
        c00 = nc00
        c01 = nc01
        c10 = nc10
        c11 = nc11

        prev_valid = valid_n
        prev_seg = seg_n


class _SegmentedBlockAffineExclusiveScan(torch.autograd.Function):
    """Differentiable segmented block-affine exclusive scan."""

    @staticmethod
    def forward(ctx, matrix, bias, segment_ids, valid_mask, initial_state):
        if matrix.ndim != 5 or matrix.shape[-2:] != (2, 2):
            raise ValueError("matrix must have shape [B, N, H, 2, 2]")
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

        _fwd_kernel[(batch,)](
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
        if not grad_out.is_cuda:
            return None, None, None, None, None

        batch = matrix.shape[0]
        dmat = torch.zeros_like(matrix)
        dbias = torch.zeros_like(bias)
        dinit = torch.zeros_like(initial_state)
        length = ctx.length
        heads = ctx.heads
        block_h = ctx.block_h

        _bwd_adj_kernel[(batch,)](
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
        _bwd_dinit_kernel[(batch,)](
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


def triton_segmented_block_affine_exclusive_scan(
    matrix: torch.Tensor,
    bias: torch.Tensor,
    segment_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Differentiable Triton scan matching the ``segmented_scan`` interface.

    Args:
        matrix: Block operators with shape ``[B, N, H, 2, 2]``.
        bias: Block biases with shape ``[B, N, H, 2]``.
        segment_ids: Segment identifier per position, shape ``[B, N]``.
        valid_mask: Valid occurrence mask, shape ``[B, N]``.
        initial_state: Initial block states with shape ``[B, N, H, 2]``.

    Returns:
        State immediately before each transition, shape ``[B, N, H, 2]``.
    """
    return _SegmentedBlockAffineExclusiveScan.apply(
        matrix, bias, segment_ids, valid_mask, initial_state
    )


__all__ = ["triton_segmented_block_affine_exclusive_scan"]
