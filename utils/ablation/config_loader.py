"""Load ablation configs from yaml."""

from pathlib import Path

import yaml

from utils.ablation.config import AblationConfig, AblationStudyConfig


def load_config(config_path: str, dataset: str, fold: int = 0) -> AblationStudyConfig:
    """Load ablation study config from a yaml file.

    Dataset and fold must be provided as parameters, not from the config file.
    Config file should contain study_name, base_model, shared_params, and ablations.

    Args:
        config_path: Path to yaml config file
        dataset: Dataset name (must be provided from command line)
        fold: Fold index for K-Fold cross-validation (default: 0)

    Returns:
        AblationStudyConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If required fields are missing
        ValueError: If dataset is not provided

    Example:
        >>> config = load_config("configs/ablation/hgikt_study.yaml", dataset="assistments09", fold=0)
        >>> print(config.study_name)
        >>> print(config.dataset)  # returns "assistments09"
        >>> print(config.fold)  # returns 0
    """
    if not dataset:
        raise ValueError("Dataset must be provided from command line")

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}

    # Validate required fields
    required_fields = ["study_name", "base_model"]
    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        raise KeyError(f"Missing required fields: {missing_fields}")

    # Parse shared params and merge with command line parameters
    shared_params = data.get("shared_params", {}).copy()

    # Set dataset and fold from command line
    shared_params["dataset"] = dataset
    shared_params["fold"] = fold

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
        dataset=dataset,
        fold=fold,
        shared_params=shared_params,
        ablations=ablations,
    )
