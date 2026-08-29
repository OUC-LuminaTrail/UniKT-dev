"""Shared measurement primitives reused across efficiency stages."""

try:
    from . import custom_formulas  # noqa: F401  — registers aten::_cudnn_rnn FLOP formula
except ModuleNotFoundError:  # torch < 2.1 has no torch.utils.flop_counter
    pass
