"""Shared measurement primitives reused across efficiency stages."""

import contextlib

# torch < 2.1 has no torch.utils.flop_counter
with contextlib.suppress(ModuleNotFoundError):
    from . import (
        custom_formulas,  # noqa: F401  — registers aten::_cudnn_rnn FLOP formula
    )
