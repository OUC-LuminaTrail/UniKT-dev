"""QRNN pooling scan (the ForgetMult recurrence) with CUDA/CPU dispatch.

The QRNN's recurrent pooling is the per-channel scalar affine scan
``c_t = f_t * z_t + (1 - f_t) * c_{t-1}`` with ``c_{-1} = 0``. Each position
is the affine map ``c -> (1 - f_t) c + f_t z_t``; composing adjacent maps with
``(a1, b1) then (a2, b2) -> (a1 * a2, a2 * b1 + b2)`` turns the sequential
recurrence into an associative prefix scan over the time axis.

On CUDA the scan runs a fused, differentiable Triton kernel; on CPU it uses a
work-efficient Hillis-Steele scan (logarithmic depth), which is autograd-
managed and matches the sequential recurrence up to float association order.
"""

import torch


def _pool_py(z: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Inclusive scalar affine scan via Hillis-Steele (log depth).

    ``z`` and ``f`` are raw (pre-activation); ``tanh``/``sigmoid`` are applied
    here so the CPU path matches the fused Triton kernel's contract.
    """
    tz = torch.tanh(z)
    sf = torch.sigmoid(f)
    a = 1.0 - sf
    b = sf * tz
    n = b.size(1)
    offset = 1
    while offset < n:
        # Combine the map at t-offset (earlier) after the map at t (later).
        new_a = a[:, :-offset] * a[:, offset:]
        new_b = a[:, offset:] * b[:, :-offset] + b[:, offset:]
        a = torch.cat((a[:, :offset], new_a), dim=1)
        b = torch.cat((b[:, :offset], new_b), dim=1)
        offset *= 2
    # c_{-1} = 0, so the total additive term of the prefix composition is c.
    return b


def qrnn_pool(z: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Apply the QRNN pooling recurrence ``c_t = f_t * z_t + (1 - f_t) * c_{t-1}``.

    Args:
        z: Raw candidate values with shape ``[B, N, H]`` (``tanh`` is applied
            inside the scan).
        f: Raw forget gates with shape ``[B, N, H]`` (``sigmoid`` is applied
            inside the scan).

    Returns:
        Cell states ``c`` with shape ``[B, N, H]``, starting from ``c_{-1} = 0``.
    """
    if z.shape != f.shape or z.ndim != 3:
        raise ValueError("z and f must have matching [batch, seq_len, dim] shape")
    if z.size(1) == 0:
        return torch.zeros_like(z)
    if z.is_cuda:
        try:
            from model.ReKTP.triton_qrnn_scan import triton_qrnn_pool

            return triton_qrnn_pool(z, f)
        except ImportError:
            pass  # Fall back to PyTorch when triton is unavailable.
    return _pool_py(z, f)


__all__ = ["qrnn_pool"]
