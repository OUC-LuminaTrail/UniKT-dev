"""Triton kernels for the QRNN pooling recurrence (ForgetMult).

The QRNN's recurrent pooling is the per-channel scalar affine scan
``c_t = f_t * z_t + (1 - f_t) * c_{t-1}`` with ``c_{-1} = 0``. Unlike the
segmented block-affine scan, it has no segments, no initial states, and no
valid mask: padding is trailing and the recurrence is causal, so padding
cannot influence earlier valid positions.

- Forward: one program per batch row, vectorised over ``hidden``, running a
  sequential scan along the sequence. Each position is two FMAs.
- Backward: an adjoint reverse scan over ``q_t = g_t + (1 - f_{t+1}) q_{t+1}``
  with ``d f_t = q_t (z_t - c_{t-1})`` and ``d z_t = q_t f_t``.

CUDA only; CPU uses the log-depth scan in ``qrnn_scan.py``.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    z_ptr,
    f_ptr,
    out_ptr,
    N,
    H,
    stride_b,
    BLOCK_H: tl.constexpr,
):
    """Forward QRNN pooling scan for one batch row (vectorised over hidden)."""
    pid_b = tl.program_id(0)
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # carry = c_{t-1}, starting from c_{-1} = 0.
    carry = tl.zeros([BLOCK_H], dtype=tl.float32)

    z_row = z_ptr + pid_b * stride_b
    f_row = f_ptr + pid_b * stride_b
    o_row = out_ptr + pid_b * stride_b

    for n in range(N):
        zn = tl.load(z_row + n * H + h_offs, mask=h_mask, other=0.0).to(tl.float32)
        fn = tl.load(f_row + n * H + h_offs, mask=h_mask, other=0.0).to(tl.float32)
        carry = fn * zn + (1.0 - fn) * carry
        tl.store(o_row + n * H + h_offs, carry, mask=h_mask)


@triton.jit
def _bwd_kernel(
    g_ptr,
    z_ptr,
    f_ptr,
    c_ptr,
    dz_ptr,
    df_ptr,
    N,
    H,
    stride_b,
    BLOCK_H: tl.constexpr,
):
    """Backward adjoint reverse scan for the QRNN pooling.

    For ``c_t = a_t c_{t-1} + b_t`` with ``a_t = 1 - f_t`` and
    ``b_t = f_t z_t``, the adjoint is ``q_t = g_t + a_{t+1} q_{t+1}`` with
    ``d a_t = q_t c_{t-1}`` and ``d b_t = q_t``, so
    ``d f_t = q_t (z_t - c_{t-1})`` and ``d z_t = q_t f_t``. The previous cell
    state ``c_{t-1}`` is carried from the saved forward output.
    """
    pid_b = tl.program_id(0)
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    q = tl.zeros([BLOCK_H], dtype=tl.float32)
    next_f = tl.zeros([BLOCK_H], dtype=tl.float32)

    g_row = g_ptr + pid_b * stride_b
    z_row = z_ptr + pid_b * stride_b
    f_row = f_ptr + pid_b * stride_b
    c_row = c_ptr + pid_b * stride_b
    dz_row = dz_ptr + pid_b * stride_b
    df_row = df_ptr + pid_b * stride_b

    for n in range(N):
        i = N - 1 - n
        gi = tl.load(g_row + i * H + h_offs, mask=h_mask, other=0.0).to(tl.float32)
        zi = tl.load(z_row + i * H + h_offs, mask=h_mask, other=0.0).to(tl.float32)
        fi = tl.load(f_row + i * H + h_offs, mask=h_mask, other=0.0).to(tl.float32)
        # Previous cell state c_{i-1}, zero at the sequence start.
        cm1 = tl.load(
            c_row + (i - 1) * H + h_offs,
            mask=(i > 0) & h_mask,
            other=0.0,
        ).to(tl.float32)

        # q_i = g_i + (1 - f_{i+1}) q_{i+1}; at i = N-1 the term vanishes.
        q = gi + (1.0 - next_f) * q
        next_f = fi
        dz = q * fi
        df = q * (zi - cm1)

        tl.store(dz_row + i * H + h_offs, dz, mask=h_mask)
        tl.store(df_row + i * H + h_offs, df, mask=h_mask)


class _QRNNPool(torch.autograd.Function):
    """Differentiable QRNN pooling scan (ForgetMult recurrence)."""

    @staticmethod
    def forward(ctx, z, f):
        if z.shape != f.shape or z.ndim != 3:
            raise ValueError("z and f must have matching [batch, seq_len, dim] shape")
        if z.size(1) == 0:
            ctx.empty = True
            ctx.save_for_backward(f)
            return torch.zeros_like(z)

        z = z.contiguous()
        f = f.contiguous()
        batch, length, dim = z.shape
        out = torch.empty_like(z)
        block_h = triton.next_power_of_2(dim)
        ctx.block_h = block_h
        ctx.length = length
        ctx.dim = dim

        _fwd_kernel[(batch,)](
            z,
            f,
            out,
            length,
            dim,
            z.stride(0),
            BLOCK_H=block_h,
        )
        ctx.empty = False
        ctx.save_for_backward(z, f, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.empty:
            return torch.zeros_like(ctx.saved_tensors[0]), None
        z, f, out = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        if not grad_out.is_cuda:
            return None, None

        batch = z.shape[0]
        length = ctx.length
        dim = ctx.dim
        block_h = ctx.block_h
        dz = torch.zeros_like(z)
        df = torch.zeros_like(f)

        _bwd_kernel[(batch,)](
            grad_out,
            z,
            f,
            out,
            dz,
            df,
            length,
            dim,
            z.stride(0),
            BLOCK_H=block_h,
        )
        return dz, df


def triton_qrnn_pool(z: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Differentiable Triton QRNN pooling scan.

    Args:
        z: Candidate values with shape ``[B, N, H]`` (already tanh-activated).
        f: Forget gates with shape ``[B, N, H]``, assumed in ``[0, 1]``.

    Returns:
        Cell states ``c`` with shape ``[B, N, H]``, where
        ``c_t = f_t * z_t + (1 - f_t) * c_{t-1}`` and ``c_{-1} = 0``.
    """
    return _QRNNPool.apply(z, f)


__all__ = ["triton_qrnn_pool"]
