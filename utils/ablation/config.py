"""Ablation configuration dataclasses.

Minimal configuration schema for ablation studies.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AblationConfig:
    """Single ablation configuration.

    Attributes:
        name: Experiment name for this ablation
        variant: Model variant name (must be registered in TRAINERS)
        description: Optional description of what this ablation tests
        params: Override params specific to this ablation (merged with shared_params)
    """

    name: str
    variant: str
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationStudyConfig:
    """Batch ablation study configuration.

    Note: dataset must be provided via command line argument, not from config file.

    Attributes:
        study_name: Name of the ablation study
        base_model: Base model name (e.g., "HGIKT")
        dataset: Dataset name (e.g., "assistments09"), provided from command line
        shared_params: Parameters shared across all ablations
        ablations: List of ablation configurations to run
    """

    study_name: str
    base_model: str
    dataset: str  # Provided from command line, not from config file
    shared_params: dict[str, Any] = field(default_factory=dict)
    ablations: list[AblationConfig] = field(default_factory=list)
