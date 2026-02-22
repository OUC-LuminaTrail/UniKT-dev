"""Ablation study framework package.

Provides minimal infrastructure for creating model variants through subclassing.
"""

from utils.ablation.config import AblationConfig, AblationStudyConfig
from utils.ablation.config_loader import load_config
from utils.ablation.runner import AblationRunner

__all__ = [
    "AblationConfig",
    "AblationStudyConfig",
    "load_config",
    "AblationRunner",
]
