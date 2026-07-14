"""Register FLOP formulas for ops FlopCounterMode does not cover by default.

Importing this module (via :mod:`utils.efficiency.measures` package init) registers
a formula for ``aten::_cudnn_rnn`` so cuDNN-fused LSTM/GRU/RNN forwards are counted
under the same cuDNN-enabled config used for latency and memory, instead of
requiring cuDNN to be disabled during FLOP measurement. Convention follows
FlopCounterMode built-ins: 1 FMA = 2 FLOPs (m x n x 2 x k).
"""

import torch
from torch.utils.flop_counter import flop_registry, register_flop_formula

_CUDNN_RNN = torch.ops.aten._cudnn_rnn

if _CUDNN_RNN not in flop_registry:

    @register_flop_formula(_CUDNN_RNN)
    def _cudnn_rnn_flop(
        input_size,
        weight_sizes,
        weight_stride0,
        weight_buf_size,
        hx_size,
        cx_size,
        mode,
        hidden_size,
        proj_size,
        num_layers,
        batch_first,
        dropout,
        train,
        bidirectional,
        batch_sizes,
        dropout_state_size,
        **kwargs,
    ) -> int:
        """Compute forward FLOPs of a cuDNN-fused RNN (mode: 0/1=RNN, 2=LSTM, 3=GRU).

        Tensor args arrive as ``torch.Size`` via FlopCounterMode's shape wrapper;
        scalar args arrive unchanged. Positional order matches the
        ``aten::_cudnn_rnn`` schema.
        """
        # Gates per cell update per cuDNN mode code: RNN=1, LSTM=4, GRU=3.
        gate_mult = {0: 1, 1: 1, 2: 4, 3: 3}.get(mode, 4)
        if batch_first:
            batch, seq, input_dim = input_size
        else:
            seq, batch, input_dim = input_size
        num_directions = 2 if bidirectional else 1

        total = 0
        for layer in range(num_layers):
            layer_dim = input_dim if layer == 0 else hidden_size * num_directions
            for _ in range(num_directions):
                # input->hidden and hidden->hidden matmuls, one per gate.
                total += 2 * batch * seq * gate_mult * layer_dim * hidden_size
                total += 2 * batch * seq * gate_mult * hidden_size * hidden_size
        return total
