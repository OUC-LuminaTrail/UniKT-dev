"""Runtime instance holder built per run.

Scalar configuration lives in :class:`~utils.config.run_config.RunConfig` and
is read directly off ``self.run_config``; only the non-serializable runtime
objects (model, optimizer, data, collators) a trainer constructs from
``rc`` + ``data_src`` are held here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class RuntimeComponents:
    """Per-run runtime objects returned by :meth:`BaseTrainer.build_components`.

    Scalar knobs (epochs, batch_size, device, early-stopping, compile, logging
    flags) are intentionally absent — read them from ``self.run_config`` so the
    config tree stays the single source of truth.
    """

    model: Any = None
    optimizer: Any = None
    loss_fn: Any = None
    lr_scheduler: Any = None
    train_data: Any = None
    val_data: Any = None
    test_data: Any = None
    collate_fn: Callable | None = None
    val_collate_fn: Callable | None = None
    test_collate_fn: Callable | None = None
    max_clip_grad_norm: float | None = None


__all__ = ["RuntimeComponents"]
