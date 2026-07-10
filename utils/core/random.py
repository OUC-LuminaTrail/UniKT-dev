"""Random seed setting module.

Provides a unified random seed setting function to ensure experiment
reproducibility.
"""


def seed_everything(seed: int | None, deterministic: bool = True) -> int | None:
    """Set random seeds to ensure reproducible results.

    This function seeds Python's ``random``, NumPy, and PyTorch, and
    optionally enables deterministic mode.

    Args:
        seed: Random seed value. If ``None``, no seed is set.
        deterministic: Whether to enable deterministic mode. Enabling this
            degrades performance but guarantees fully reproducible results.
            Defaults to ``True``.

    Returns:
        The seed value that was set, or ``None`` if ``seed`` was ``None``.

    Example:
        >>> seed_everything(42)
        >>> seed_everything(None)
    """
    import os
    import random

    import numpy as np
    import torch

    if seed is None:
        return None

    # Only affects child processes started after this point (e.g. DataLoader
    # workers); the current interpreter's hash randomization is fixed at startup.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Seed individual libraries
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Polars global seed (shuffle / sample ordering)
    try:
        import polars as pl

        pl.set_random_seed(seed)
    except ImportError:
        pass

    # Seed CUDA-related seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Enable deterministic mode
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch 1.8+ supports use_deterministic_algorithms
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                # Older PyTorch versions do not support warn_only
                torch.use_deterministic_algorithms(True)

        # Set CUDA workspace configuration
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return seed
