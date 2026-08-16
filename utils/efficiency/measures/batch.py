"""Batch-level helpers: shape probing, valid-interaction counting, device moves."""

import torch

from ..device import synchronize


def batch_size_of(batch) -> int:
    """Number of rows (student sequences B) in a batch."""
    first = _first_tensor(batch)
    return int(first.size(0)) if first is not None and first.dim() >= 1 else 0


def _first_tensor(batch) -> torch.Tensor | None:
    """First tensor found in a dict/tuple/list batch (depth 1)."""
    if isinstance(batch, dict):
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v
    elif isinstance(batch, (tuple, list)):
        for v in batch:
            if isinstance(v, torch.Tensor):
                return v
    return None


def _count_scored(forward, device) -> int:
    """``numel`` of the forward's aligned 1D ``y_label``, synchronized."""
    with torch.inference_mode():
        out = forward()
        n = int(out["y_label"].numel())
    synchronize(device)
    return n


def count_valid_interactions(target, sample_batch) -> int:
    """Valid interactions per forward pass that participate in the loss.

    Throughput denominator: runs one forward via the target, takes ``numel`` of
    the aligned+masked 1D ``y_label`` — the interactions retained by
    ``_extract_valid_predictions`` after the adjacent-pair mask, i.e. the samples
    ``_compute_loss`` actually consumes.
    """
    return _count_scored(lambda: target.forward(sample_batch), target.device)


def count_test_predictions(target, sample_batch) -> int:
    """Scored predictions per test forward pass.

    Same contract as :func:`count_valid_interactions` but through
    ``test_forward_pass``. For windowlate data this is far smaller than the
    training count — each window scores only its final position — so the two
    must never share a denominator.
    """
    return _count_scored(lambda: target.test_forward(sample_batch), target.device)


def to_device(batch, device: torch.device):
    """Recursively move batch tensors to device (tuple/list/dict aware)."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(b, device) for b in batch)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch
