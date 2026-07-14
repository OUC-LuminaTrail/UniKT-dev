"""Shared pseudo train-step used by the training and trace stages."""


def run_train_step(target, batch) -> None:
    """Run one training step via the :class:`BenchmarkTarget`.

    Delegates to ``target.compute_train_step`` — the same computation the real
    training loop performs (``BaseTrainer.compute_train_step``) — so the
    benchmark never mirrors the step by hand and the two cannot drift apart.
    """
    target.compute_train_step(batch)
