"""Triton kernels for the QRNN pooling recurrence (ForgetMult).

The QRNN's recurrent pooling is the per-channel scalar affine scan
``c_t = sigmoid(f_t) * tanh(z_t) + (1 - sigmoid(f_t)) * c_{t-1}`` with
``c_{-1} = 0``. The ``tanh``/``sigmoid`` activations are fused into the kernels,
so callers pass the raw candidate ``z`` and pre-gate ``f`` straight from the
gated linear layer -- this saves two elementwise passes plus their temporary
tensors and shortens the launch chain of the enclosing block.

- Forward: one program per batch row, vectorised over ``hidden``, running a
  sequential scan with the activation folded into each step.
- Backward: an adjoint reverse scan that recomputes the activations and applies
  the tanh/sigmoid chain rule, returning gradients w.r.t. the raw ``z``/``f``.

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
    """Forward QRNN pooling scan for one batch row, fusing tanh/sigmoid.

    ``z`` and ``f`` are raw (pre-activation); the recurrence is
    ``c_t = sigmoid(f_t) * tanh(z_t) + (1 - sigmoid(f_t)) * c_{t-1}``.
    """
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
        sf = tl.sigmoid(fn)
        # tl has no tanh; use the identity tanh(x) = 2*sigmoid(2x) - 1.
        tz = 2.0 * tl.sigmoid(2.0 * zn) - 1.0
        carry = sf * tz + (1.0 - sf) * carry
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
    """Backward adjoint reverse scan, fusing the tanh/sigmoid chain rule.

    With ``tz = tanh(z)``, ``sf = sigmoid(f)``, ``a = 1 - sf``, ``b = sf * tz``
    and the adjoint ``q_t = g_t + a_{t+1} q_{t+1}``, the gradients back at the
    raw inputs are ``dL/dz_t = q_t * sf * (1 - tz^2)`` and
    ``dL/df_t = q_t * (tz - c_{t-1}) * sf * (1 - sf)``. The activations are
    recomputed from the saved raw ``z``/``f``; ``c_{t-1}`` comes from the saved
    forward output.
    """
    pid_b = tl.program_id(0)
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    q = tl.zeros([BLOCK_H], dtype=tl.float32)
    # sf at i+1, carried across iterations; 0 at i = N-1 (term vanishes anyway).
    next_sf = tl.zeros([BLOCK_H], dtype=tl.float32)

    g_row = g_ptr + pid_b * stride_b
    z_row = z_ptr + pid_b * stride_b
    f_row = f_ptr + pid_b * stride_b
    c_row = c_ptr + pid_b * stride_b
    dz_row = dz_ptr + pid_b * stride_b
    df_row = df_ptr + pid_b * stride_b

    for n in range(N):
        i = N - 1 - n
        gi = tl.load(g_row + i * H + h_offs, mask=h_mask, other=0.0).to(tl.float32)
        zr = tl.load(z_row + i * H + h_offs, mask=h_mask, other=0.0).to(tl.float32)
        fr = tl.load(f_row + i * H + h_offs, mask=h_mask, other=0.0).to(tl.float32)
        # Previous cell state c_{i-1}, zero at the sequence start.
        cm1 = tl.load(
            c_row + (i - 1) * H + h_offs,
            mask=(i > 0) & h_mask,
            other=0.0,
        ).to(tl.float32)

        sf = tl.sigmoid(fr)
        # tl has no tanh; use tanh(x) = 2*sigmoid(2x) - 1.
        tz = 2.0 * tl.sigmoid(2.0 * zr) - 1.0

        # q_i = g_i + (1 - sf_{i+1}) q_{i+1}; at i = N-1 the term vanishes.
        q = gi + (1.0 - next_sf) * q
        next_sf = sf

        # Chain rule back to the raw inputs z, f.
        dz = q * sf * (1.0 - tz * tz)
        df = q * (tz - cm1) * sf * (1.0 - sf)

        tl.store(dz_row + i * H + h_offs, dz, mask=h_mask)
        tl.store(df_row + i * H + h_offs, df, mask=h_mask)


class _QRNNPool(torch.autograd.Function):
    """Differentiable QRNN pooling scan over raw (pre-activation) z and f."""

    @staticmethod
    def forward(ctx, z, f):
        if z.shape != f.shape or z.ndim != 3:
            raise ValueError("z and f must have matching [batch, seq_len, dim] shape")
        if z.size(1) == 0:
            ctx.empty = True
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
            zg = torch.zeros_like(grad_out)
            return zg, zg.clone()
        z, f, out = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        if not grad_out.is_cuda:
            return None, None

        batch = z.shape[0]
        length = ctx.length
        dim = ctx.dim
        block_h = ctx.block_h
        dz = torch.empty_like(z)
        df = torch.empty_like(f)

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
    """Differentiable Triton QRNN pooling scan over raw (pre-activation) inputs.

    Args:
        z: Raw candidate values with shape ``[B, N, H]`` (the kernel applies
            ``tanh`` internally).
        f: Raw forget gates with shape ``[B, N, H]`` (the kernel applies
            ``sigmoid`` internally).

    Returns:
        Cell states ``c`` with shape ``[B, N, H]``, where
        ``c_t = sigmoid(f_t) * tanh(z_t) + (1 - sigmoid(f_t)) * c_{t-1}`` and
        ``c_{-1} = 0``.
    """
    return _QRNNPool.apply(z, f)


__all__ = ["triton_qrnn_pool"]
