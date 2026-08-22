"""Scan kernels for the segmented scalar affine exclusive scan.

Each position carries a scalar operator ``a`` and bias ``b``; the scan
returns the state before that operator is applied, resetting at segment
boundaries.

Two CUDA paths share the same semantics:

- Single-program kernels: one program per batch row, vectorised over the
  hidden channels, running a sequential exclusive scan along the sequence.
  Used for short sequences where kernel-launch latency dominates.
- Chunked kernels: the sequence is split into ``BLOCK_N`` chunks. Pass 1
  scans each (batch, chunk) tile in parallel for the tile-local end carry
  and any-head flag; pass 2 propagates carries across chunks (a tiny
  sequential scan of length ``num_chunks``); pass 3 rescans each tile,
  composing the incoming carry ``P`` into positions whose segment run
  crosses the tile start. Parallelism grows from ``B`` to
  ``B * ceil(N / BLOCK_N)`` programs.

All three passes are parameterised by ``MODE`` so the backward kernels map
onto the same affine scan: the adjoint reverse scan is the scan of
``(a_eff, g)`` in reversed order where ``a_eff = 0`` cuts the chain (with a
zero initial state the exclusive output is the adjoint itself), and
``d init`` is the multiplicative special case (zero bias, ``init = g``).

``d matrix`` / ``d bias`` match PyTorch autograd only while ``initial_state``
is constant within a segment, which holds because it is derived from the
segment id. ``d init`` is exact per position (``prefix_i * g_i``) regardless.

CPU inputs fall back to a pure-PyTorch serial scan with identical semantics
and full autograd coverage.
"""

import torch
import triton
import triton.language as tl

# Below this length the chunked pipeline cannot amortise its launch overhead;
# the single-program kernels also preserve the exact bitwise outputs that the
# golden tests anchor on.
_CHUNKED_MIN_LENGTH = 128


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

    1x1 specialization: the carry is a scalar pair ``(c_a, c_b)`` per hidden
    channel, the output is ``c_a * init + c_b``, and the carry advances by
    composing the position operator ``(a, b)`` after it (``c_a <- a * c_a``,
    ``c_b <- a * c_b + b``).
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

    Scalar specialization: for the recurrence ``h_{i+1} = a_i h_i + b_i`` the
    adjoint is ``a_i = g_i + a_i * a_{i+1}``, ``dA_i = a_{i+1} * h_i`` and
    ``db_i = a_{i+1}``, nonzero only where the operator is used, that is
    where ``i+1`` shares the segment. ``h`` is the forward output.
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

    Scalar specialization: yields ``prefix_i * g_i``, matching the
    per-position use of ``initial_state`` rather than only using it at
    segment heads.
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


# ---------------------------------------------------------------------------
# Chunked parallel kernels. MODE: 0 = forward, 1 = adjoint reverse, 2 = dinit.
# Per-tile [BLOCK_N] sequential steps over [BLOCK_H] channels; MODE 1 walks
# the sequence in reverse. All row tensors are contiguous with row stride
# ``stride_row`` and step ``H``; segment/valid are [B, N].
# ---------------------------------------------------------------------------


@triton.jit
def _chunk_pass1_kernel(
    matrix_ptr,
    bias_ptr,
    g_ptr,
    seg_ptr,
    valid_ptr,
    fa_ptr,
    fb_ptr,
    hh_ptr,
    N,
    H,
    stride_row,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MODE: tl.constexpr,
):
    """Tile-local scan: end carry plus an any-head flag per tile.

    The carry resets to the identity at segment heads exactly like the
    single-program kernel, so ``fa/fb`` is the operator composed from the
    last head (or tile start) to the tile end. MODE 1 encodes the chain cut
    directly in ``a_eff = 0`` (no head flag needed); MODE 2 skips the bias
    entirely (multiplicative scan).
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    cs = pid_c * BLOCK_N
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    ca = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
    cb = tl.zeros([BLOCK_H], dtype=tl.float32)
    hh = tl.zeros([], dtype=tl.int32)

    m_row = matrix_ptr + pid_b * stride_row
    b_row = bias_ptr + pid_b * stride_row
    g_row = g_ptr + pid_b * stride_row
    s_row = seg_ptr + pid_b * N
    val_row = valid_ptr + pid_b * N

    if MODE == 1:
        # Reversed order: the "previous" position of the tile start is the
        # original successor N-cs (out of range at the sequence end).
        pv = tl.load(val_row + N - cs, mask=cs > 0, other=0).to(tl.int32)
        ps = tl.load(s_row + N - cs, mask=cs > 0, other=-1).to(tl.int32)
    else:
        pv = tl.load(val_row + cs - 1, mask=cs > 0, other=0).to(tl.int32)
        ps = tl.load(s_row + cs - 1, mask=cs > 0, other=-1).to(tl.int32)

    for n in range(BLOCK_N):
        pos = cs + n
        i = N - 1 - pos if MODE == 1 else pos
        inb = pos < N

        seg_n = tl.load(s_row + i, mask=inb, other=-1).to(tl.int32)
        valid_n = tl.load(val_row + i, mask=inb, other=0).to(tl.int32)
        vn = (valid_n != 0) & inb

        a = tl.load(m_row + i * H + h_offs, mask=h_mask & inb, other=1.0).to(tl.float32)

        if MODE == 1:
            same = vn & (pv != 0) & (seg_n == ps)
            g = tl.load(g_row + i * H + h_offs, mask=h_mask & inb, other=0.0).to(
                tl.float32
            )
            a_eff = tl.where(same, a, 0.0)
            b_eff = tl.where(vn, g, 0.0)
        elif MODE == 0:
            b = tl.load(b_row + i * H + h_offs, mask=h_mask & inb, other=0.0).to(
                tl.float32
            )
            # Segment head: valid, and row start, prior invalid, or new id.
            is_head = vn & ((pos == 0) | (pv == 0) | (seg_n != ps))
            ca = tl.where(is_head, 1.0, ca)
            cb = tl.where(is_head, 0.0, cb)
            a_eff = tl.where(vn, a, 1.0)
            b_eff = tl.where(vn, b, 0.0)
            hh = tl.where(is_head, 1, hh)
        else:
            is_head = vn & ((pos == 0) | (pv == 0) | (seg_n != ps))
            ca = tl.where(is_head, 1.0, ca)
            a_eff = tl.where(vn, a, 1.0)
            b_eff = tl.zeros([BLOCK_H], dtype=tl.float32)
            hh = tl.where(is_head, 1, hh)

        ca = a_eff * ca
        if MODE != 2:
            cb = a_eff * cb + b_eff

        pv = valid_n
        ps = seg_n

    base = (pid_b * tl.num_programs(1) + pid_c) * H
    tl.store(fa_ptr + base + h_offs, ca, mask=h_mask)
    if MODE != 2:
        tl.store(fb_ptr + base + h_offs, cb, mask=h_mask)
    if MODE != 1:
        tl.store(hh_ptr + pid_b * tl.num_programs(1) + pid_c, hh)


@triton.jit
def _chunk_pass2_kernel(
    fa_ptr,
    fb_ptr,
    hh_ptr,
    pa_ptr,
    pb_ptr,
    C,
    H,
    BLOCK_H: tl.constexpr,
    MODE: tl.constexpr,
):
    """Carry-in prefix across chunks: ``P_{k+1} = tail_k`` when tile ``k``
    contains a head, else ``tail_k`` composed after ``P_k``."""
    pid_b = tl.program_id(0)
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    pa = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
    pb = tl.zeros([BLOCK_H], dtype=tl.float32)
    fbase = pid_b * C * H

    for k in range(C):
        off = fbase + k * H + h_offs
        tl.store(pa_ptr + off, pa, mask=h_mask)
        fa = tl.load(fa_ptr + off, mask=h_mask, other=1.0)
        if MODE != 2:
            tl.store(pb_ptr + off, pb, mask=h_mask)
            fb = tl.load(fb_ptr + off, mask=h_mask, other=0.0)
        if MODE == 1:
            cut = tl.zeros([], dtype=tl.int32)
        else:
            cut = tl.load(hh_ptr + pid_b * C + k)

        pa = tl.where(cut != 0, fa, fa * pa)
        if MODE != 2:
            pb = tl.where(cut != 0, fb, fa * pb + fb)


@triton.jit
def _chunk_pass3_fwd_kernel(
    matrix_ptr,
    bias_ptr,
    init_ptr,
    seg_ptr,
    valid_ptr,
    pa_ptr,
    pb_ptr,
    out_ptr,
    N,
    H,
    stride_row,
    C,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Rescan a tile emitting the corrected exclusive outputs.

    A position whose segment run crosses the tile start composes the
    incoming carry ``P`` before its tile-local carry:
    ``lca * (Pa * init + Pb) + lcb``; any head at or before the position
    severs the incoming carry and the tile-local value stands alone.
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    cs = pid_c * BLOCK_N
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    coff = (pid_b * C + pid_c) * H + h_offs
    pa = tl.load(pa_ptr + coff, mask=h_mask, other=1.0)
    pb = tl.load(pb_ptr + coff, mask=h_mask, other=0.0)

    m_row = matrix_ptr + pid_b * stride_row
    b_row = bias_ptr + pid_b * stride_row
    i_row = init_ptr + pid_b * stride_row
    o_row = out_ptr + pid_b * stride_row
    s_row = seg_ptr + pid_b * N
    val_row = valid_ptr + pid_b * N

    pv = tl.load(val_row + cs - 1, mask=cs > 0, other=0).to(tl.int32)
    ps = tl.load(s_row + cs - 1, mask=cs > 0, other=-1).to(tl.int32)

    ca = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
    cb = tl.zeros([BLOCK_H], dtype=tl.float32)
    seen = tl.zeros([], dtype=tl.int1)

    for n in range(BLOCK_N):
        pos = cs + n
        inb = pos < N

        seg_n = tl.load(s_row + pos, mask=inb, other=-1).to(tl.int32)
        valid_n = tl.load(val_row + pos, mask=inb, other=0).to(tl.int32)
        vn = (valid_n != 0) & inb

        is_head = vn & ((pos == 0) | (pv == 0) | (seg_n != ps))
        ca = tl.where(is_head, 1.0, ca)
        cb = tl.where(is_head, 0.0, cb)

        ni = tl.load(i_row + pos * H + h_offs, mask=h_mask & inb, other=0.0).to(
            tl.float32
        )
        needs = ~(seen | is_head)
        o = tl.where(needs, ca * (pa * ni + pb) + cb, ca * ni + cb)
        o = tl.where(vn, o, 0.0)
        tl.store(o_row + pos * H + h_offs, o, mask=h_mask & inb)

        a = tl.load(m_row + pos * H + h_offs, mask=h_mask & inb, other=1.0).to(
            tl.float32
        )
        b = tl.load(b_row + pos * H + h_offs, mask=h_mask & inb, other=0.0).to(
            tl.float32
        )
        a_eff = tl.where(vn, a, 1.0)
        b_eff = tl.where(vn, b, 0.0)
        cb = a_eff * cb + b_eff
        ca = a_eff * ca

        seen = seen | is_head
        pv = valid_n
        ps = seg_n


@triton.jit
def _chunk_pass3_adj_kernel(
    matrix_ptr,
    g_ptr,
    h_ptr,
    seg_ptr,
    valid_ptr,
    pb_ptr,
    dmat_ptr,
    dbias_ptr,
    N,
    H,
    stride_row,
    C,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Reverse rescan of a tile emitting ``d matrix`` / ``d bias``.

    The exclusive carry is the adjoint of everything strictly after the
    position; tiles whose chain continues across the (right) boundary add
    the incoming ``P.b`` component. ``same`` at the position itself gates
    whether the adjoint is used at all.
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    cs = pid_c * BLOCK_N
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    coff = (pid_b * C + pid_c) * H + h_offs
    pb = tl.load(pb_ptr + coff, mask=h_mask, other=0.0)

    m_row = matrix_ptr + pid_b * stride_row
    dm_row = dmat_ptr + pid_b * stride_row
    g_row = g_ptr + pid_b * stride_row
    h_row = h_ptr + pid_b * stride_row
    db_row = dbias_ptr + pid_b * stride_row
    s_row = seg_ptr + pid_b * N
    val_row = valid_ptr + pid_b * N

    # Successor of the tile start (original position N-1-cs): the boundary
    # "previous" in reversed order. Out of range at the sequence end.
    nv = tl.load(val_row + N - cs, mask=cs > 0, other=0).to(tl.int32)
    ns = tl.load(s_row + N - cs, mask=cs > 0, other=-1).to(tl.int32)

    ca = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
    cb = tl.zeros([BLOCK_H], dtype=tl.float32)
    seen = tl.zeros([], dtype=tl.int1)

    for n in range(BLOCK_N):
        pos = cs + n
        i = N - 1 - pos
        inb = pos < N

        seg_i = tl.load(s_row + i, mask=inb, other=0).to(tl.int32)
        valid_i = tl.load(val_row + i, mask=inb, other=0).to(tl.int32)
        vi = (valid_i != 0) & inb

        same = vi & (nv != 0) & (seg_i == ns)

        ae = tl.where(~seen, ca * pb + cb, cb)
        ae = tl.where(same, ae, 0.0)
        hv = tl.load(h_row + i * H + h_offs, mask=h_mask & inb, other=0.0).to(
            tl.float32
        )
        da = tl.where(vi, ae * hv, 0.0)
        dbo = tl.where(vi, ae, 0.0)
        tl.store(dm_row + i * H + h_offs, da, mask=h_mask & inb)
        tl.store(db_row + i * H + h_offs, dbo, mask=h_mask & inb)

        a = tl.load(m_row + i * H + h_offs, mask=h_mask & inb, other=0.0).to(tl.float32)
        g = tl.load(g_row + i * H + h_offs, mask=h_mask & inb, other=0.0).to(tl.float32)
        a_eff = tl.where(same, a, 0.0)
        b_eff = tl.where(vi, g, 0.0)
        cb = a_eff * cb + b_eff
        ca = a_eff * ca

        seen = seen | (~same)
        nv = valid_i
        ns = seg_i


@triton.jit
def _chunk_pass3_dinit_kernel(
    matrix_ptr,
    g_ptr,
    seg_ptr,
    valid_ptr,
    pa_ptr,
    dinit_ptr,
    N,
    H,
    stride_row,
    C,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Rescan a tile emitting ``d init = prefix * g`` (multiplicative)."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    cs = pid_c * BLOCK_N
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    coff = (pid_b * C + pid_c) * H + h_offs
    pa = tl.load(pa_ptr + coff, mask=h_mask, other=1.0)

    m_row = matrix_ptr + pid_b * stride_row
    g_row = g_ptr + pid_b * stride_row
    di_row = dinit_ptr + pid_b * stride_row
    s_row = seg_ptr + pid_b * N
    val_row = valid_ptr + pid_b * N

    pv = tl.load(val_row + cs - 1, mask=cs > 0, other=0).to(tl.int32)
    ps = tl.load(s_row + cs - 1, mask=cs > 0, other=-1).to(tl.int32)

    ca = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
    seen = tl.zeros([], dtype=tl.int1)

    for n in range(BLOCK_N):
        pos = cs + n
        inb = pos < N

        seg_n = tl.load(s_row + pos, mask=inb, other=-1).to(tl.int32)
        valid_n = tl.load(val_row + pos, mask=inb, other=0).to(tl.int32)
        vn = (valid_n != 0) & inb

        is_head = vn & ((pos == 0) | (pv == 0) | (seg_n != ps))
        ca = tl.where(is_head, 1.0, ca)

        g = tl.load(g_row + pos * H + h_offs, mask=h_mask & inb, other=0.0).to(
            tl.float32
        )
        needs = ~(seen | is_head)
        di = tl.where(needs, ca * pa * g, ca * g)
        di = tl.where(vn, di, 0.0)
        tl.store(di_row + pos * H + h_offs, di, mask=h_mask & inb)

        a = tl.load(m_row + pos * H + h_offs, mask=h_mask & inb, other=1.0).to(
            tl.float32
        )
        a_eff = tl.where(vn, a, 1.0)
        ca = a_eff * ca

        seen = seen | is_head
        pv = valid_n
        ps = seg_n


# ---------------------------------------------------------------------------
# Host-side orchestration: config autotuning keyed on device and shape.
# ---------------------------------------------------------------------------

# Autotune candidates: ``None`` selects the legacy single-program kernels,
# otherwise (BLOCK_N, num_warps). Older architectures keep smaller tiles.
_CHUNK_CANDIDATES_NEW = [
    None,
    (32, 4),
    (64, 4),
    (64, 8),
    (128, 4),
    (128, 8),
]
_CHUNK_CANDIDATES_OLD = [None, (16, 2), (32, 2), (32, 4), (64, 4)]

_chunk_cfg_cache: dict = {}


def _chunk_candidates():
    major, _ = torch.cuda.get_device_capability()
    return _CHUNK_CANDIDATES_NEW if major >= 9 else _CHUNK_CANDIDATES_OLD


def _launch_legacy(mode, tensors, extras):
    """Dispatch the single-program kernels (one program per batch row)."""
    matrix, bias, seg, valid, out1, out2 = tensors
    init = extras.get("init", bias)
    g = extras.get("g", bias)
    h = extras.get("h", bias)
    batch, length, heads, _, _ = matrix.shape
    block_h = triton.next_power_of_2(heads)
    if mode == 1:
        _bwd_adj_kernel_scalar[(batch,)](
            out1,
            out2,
            g,
            h,
            matrix,
            seg,
            valid,
            length,
            heads,
            matrix.stride(0),
            matrix.stride(1),
            bias.stride(0),
            bias.stride(1),
            BLOCK_H=block_h,
        )
    elif mode == 2:
        _bwd_dinit_kernel_scalar[(batch,)](
            out1,
            g,
            matrix,
            seg,
            valid,
            length,
            heads,
            matrix.stride(0),
            matrix.stride(1),
            bias.stride(0),
            bias.stride(1),
            BLOCK_H=block_h,
        )
    else:
        _fwd_kernel_scalar[(batch,)](
            matrix,
            bias,
            seg,
            valid,
            init,
            out1,
            length,
            heads,
            matrix.stride(0),
            matrix.stride(1),
            bias.stride(0),
            bias.stride(1),
            BLOCK_H=block_h,
        )


def _launch_chunked(mode, tensors, extras, block_n, num_warps):
    """Dispatch the three-pass pipeline for a MODE.

    ``tensors`` is ``(matrix, bias, seg, valid, out1, out2)``; ``extras``
    carries the mode-specific side tensors (``init`` / ``g`` / ``h``).
    """
    matrix, bias, seg, valid, out1, out2 = tensors
    batch, length, heads = matrix.shape[0], matrix.shape[1], matrix.shape[2]
    n_chunks = triton.cdiv(length, block_n)
    block_h = triton.next_power_of_2(heads)
    stride_row = heads * length

    # One pooled allocation for the per-chunk carry buffers.
    n_f = batch * n_chunks * heads
    slots = 4 if mode != 2 else 2
    pool = torch.empty(slots * n_f, device=matrix.device)
    fa = pool[:n_f]
    pa = pool[n_f : 2 * n_f]
    fb = pool[2 * n_f : 3 * n_f] if mode != 2 else fa
    pb = pool[3 * n_f :] if mode != 2 else pa
    hh = torch.empty(batch * n_chunks, device=matrix.device, dtype=torch.int32)

    g = extras.get("g", bias)
    h = extras.get("h", bias)

    _chunk_pass1_kernel[(batch, n_chunks)](
        matrix,
        bias,
        g,
        seg,
        valid,
        fa,
        fb,
        hh,
        length,
        heads,
        stride_row,
        BLOCK_N=block_n,
        BLOCK_H=block_h,
        MODE=mode,
        num_warps=num_warps,
    )
    _chunk_pass2_kernel[(batch,)](
        fa,
        fb,
        hh,
        pa,
        pb,
        n_chunks,
        heads,
        BLOCK_H=block_h,
        MODE=mode,
        num_warps=4,
    )
    if mode == 1:
        _chunk_pass3_adj_kernel[(batch, n_chunks)](
            matrix,
            g,
            h,
            seg,
            valid,
            pb,
            out1,
            out2,
            length,
            heads,
            stride_row,
            n_chunks,
            BLOCK_N=block_n,
            BLOCK_H=block_h,
            num_warps=num_warps,
        )
    elif mode == 2:
        _chunk_pass3_dinit_kernel[(batch, n_chunks)](
            matrix,
            g,
            seg,
            valid,
            pa,
            out1,
            length,
            heads,
            stride_row,
            n_chunks,
            BLOCK_N=block_n,
            BLOCK_H=block_h,
            num_warps=num_warps,
        )
    else:
        init = extras["init"]
        _chunk_pass3_fwd_kernel[(batch, n_chunks)](
            matrix,
            bias,
            init,
            seg,
            valid,
            pa,
            pb,
            out1,
            length,
            heads,
            stride_row,
            n_chunks,
            BLOCK_N=block_n,
            BLOCK_H=block_h,
            num_warps=num_warps,
        )


def _launch_scan(mode, tensors, extras, config):
    if config is None:
        _launch_legacy(mode, tensors, extras)
    else:
        _launch_chunked(mode, tensors, extras, config[0], config[1])


def _pick_chunk_config(mode, tensors, extras, length, heads):
    """Autotune the launch config once per (device, mode, batch, length, heads).

    ``None`` (legacy single-program) is a candidate alongside the chunked
    tile shapes. Batch and length are bucketed (power-of-two length,
    16-wide batch) because the packed length varies per training batch; the
    chosen tile shape stays valid across a bucket.
    """
    key = (
        torch.cuda.current_device(),
        torch.cuda.get_device_capability(),
        mode,
        tensors[0].shape[0] // 16,
        triton.next_power_of_2(length),
        heads,
    )
    if key in _chunk_cfg_cache:
        return _chunk_cfg_cache[key]
    # Raise the GPU clocks before timing so candidate order cannot bias the
    # ranking.
    warm = torch.empty(256 * 1024 * 1024 // 4, device="cuda")
    for _ in range(40):
        warm.mul_(1.0000001)
    torch.cuda.synchronize()
    best, best_time = None, float("inf")
    for cand in _chunk_candidates():
        try:
            t = _time_config(mode, tensors, extras, cand)
        except Exception:
            continue
        if t < best_time:
            best, best_time = cand, t
    _chunk_cfg_cache[key] = best
    return best


def _time_config(mode, tensors, extras, config, iters=30):
    """Amortised wall time per scan run for a candidate config.

    Iterations are enqueued back-to-back so launch latency overlaps with GPU
    execution; a per-iteration sync would over-penalise the multi-kernel
    chunked path. Best of two rounds guards against clock-speed noise.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    warm = torch.empty(256 * 1024 * 1024 // 4, device="cuda")
    best = float("inf")
    for _ in range(2):
        for _ in range(4):
            warm.mul_(1.0000001)
        for _ in range(3):
            _launch_scan(mode, tensors, extras, config)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            _launch_scan(mode, tensors, extras, config)
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters)
    return best


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

        length, heads = matrix.shape[1], matrix.shape[2]
        out = torch.empty_like(initial_state)
        ctx.length = length
        ctx.heads = heads

        tensors = (matrix, bias, segment_ids, valid_mask, out, out)
        extras = {"init": initial_state}
        if length >= _CHUNKED_MIN_LENGTH:
            config = _pick_chunk_config(0, tensors, extras, length, heads)
            _launch_scan(0, tensors, extras, config)
        else:
            _launch_legacy(0, tensors, extras)
        ctx.save_for_backward(matrix, bias, segment_ids, valid_mask, initial_state, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.empty:
            return torch.zeros_like(ctx.saved_tensors[0]), None, None, None, None
        matrix, bias, segment_ids, valid_mask, initial_state, out = ctx.saved_tensors
        grad_out = grad_out.contiguous()

        length = ctx.length
        heads = ctx.heads
        dmat = torch.empty_like(matrix)
        dbias = torch.empty_like(bias)
        dinit = torch.empty_like(initial_state)

        adj_tensors = (matrix, bias, segment_ids, valid_mask, dmat, dbias)
        adj_extras = {"g": grad_out, "h": out}
        dinit_tensors = (matrix, bias, segment_ids, valid_mask, dinit, dinit)
        dinit_extras = {"g": grad_out}
        if length >= _CHUNKED_MIN_LENGTH:
            config = _pick_chunk_config(1, adj_tensors, adj_extras, length, heads)
            _launch_scan(1, adj_tensors, adj_extras, config)
            config2 = _pick_chunk_config(2, dinit_tensors, dinit_extras, length, heads)
            _launch_scan(2, dinit_tensors, dinit_extras, config2)
        else:
            _launch_legacy(1, adj_tensors, adj_extras)
            _launch_legacy(2, dinit_tensors, dinit_extras)
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
        op_bias_eff = torch.where(valid_n[:, None], op_bias, torch.zeros_like(op_bias))
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
    runs the fused differentiable Triton kernel (chunked for long sequences);
    CPU inputs fall back to the serial PyTorch scan with identical semantics.

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
