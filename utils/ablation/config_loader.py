"""Load ablation configs from JSON."""

import json
from pathlib import Path

from utils.ablation.config import AblationConfig, AblationStudyConfig


def load_config(config_path: str) -> AblationStudyConfig:
    """Load ablation study config from JSON.

    Args:
        config_path: Path to JSON config file

    Returns:
        AblationStudyConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is not valid JSON
        KeyError: If required fields are missing

    Example:
        >>> config = load_config("configs/ablation/hgikt_study.json")
        >>> print(config.study_name)
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file) as f:
        data = json.load(f)

    # Validate required fields
    required_fields = ["study_name", "base_model", "dataset"]
    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        raise KeyError(f"Missing required fields: {missing_fields}")

    # Parse ablation configs
    ablations = []
    for abl_data in data.get("ablations", []):
        if "name" not in abl_data or "variant" not in abl_data:
            raise KeyError("Each ablation must have 'name' and 'variant' fields")
        ablations.append(
            AblationConfig(
                name=abl_data["name"],
                variant=abl_data["variant"],
                description=abl_data.get("description", ""),
                params=abl_data.get("params", {}),
            )
        )

    return AblationStudyConfig(
        study_name=data["study_name"],
        base_model=data["base_model"],
        dataset=data["dataset"],
        shared_params=data.get("shared_params", {}),
        ablations=ablations,
    )
