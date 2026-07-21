"""Metric plug-in interface: ``Metric`` ABC + ``MetricContext``.

Each metric is a class subclassing :class:`Metric`, registered via
``@register_metric("name")`` and auto-discovered (see package ``__init__``).
A metric decides for itself how to handle train/val mode (raw fields on
``ctx``) versus test/group mode (``ctx.groups``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class MetricContext:
    """Inputs for metric computation, abstracting train/val vs test/group.

    In train/val mode the per-instance raw fields (``y_pred``/``y_score``/
    ``y_prob``) are populated and ``groups`` is ``None``. In test/group mode
    ``groups`` maps each fusion name to ``(group_label, group_score)`` and
    ``y_label`` holds the per-group label (float64); the raw per-instance
    fields are ``None``.

    Attributes:
        phase: Phase name (``"train"`` / ``"val"`` / ``"test"``).
        y_label: Ground-truth labels — per-group (float64) in test/group mode,
            otherwise raw per-instance.
        y_pred: Binary predictions (train/val only).
        y_score: Ranking scores (train/val only).
        y_prob: Predicted probabilities (train/val only).
        groups: ``{fusion: (group_label, group_score)}`` (test/group only).
    """

    phase: str
    y_label: np.ndarray
    y_pred: np.ndarray | None = None
    y_score: np.ndarray | None = None
    y_prob: np.ndarray | None = None
    groups: dict[str, tuple[np.ndarray, np.ndarray]] | None = None


class Metric(ABC):
    """A pluggable evaluation metric.

    Subclasses register via ``@register_metric("name")`` and implement
    :meth:`compute`, returning ``{metric_key: value}``. In test/group mode a
    single metric typically emits one key per fusion (e.g. ``mean_acc``,
    ``vote_acc``, ``all_acc``).
    """

    @abstractmethod
    def compute(self, ctx: MetricContext) -> dict[str, float]:
        """Compute and return metric values for the given context.

        Args:
            ctx: The computation context (see :class:`MetricContext`).

        Returns:
            A dict of metric values; may contain multiple keys in
            test/group mode (one per fusion).
        """
