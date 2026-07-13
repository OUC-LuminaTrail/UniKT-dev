"""RunConfig yaml archive: lossless save/load for reproducibility.

The config tree serializes to ``run_config.yaml``; runtime-derived metadata
(param count, optimizer, device info) that does not belong in config goes to
a sidecar ``run_metadata.yaml``.

Load re-validates against the concrete model's structured schema, so a stale
archive is caught at load time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .run_config import build_run_config_schema


def save_run_config_archive(
    rc: Any,
    log_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path | None]:
    """Write ``rc`` (an OmegaConf DictConfig) to ``<log_dir>/run_config.yaml``.

    Args:
        rc: RunConfig as an OmegaConf ``DictConfig``.
        log_dir: Output directory (created if missing).
        metadata: Optional runtime-derived metadata written to a sidecar
            ``run_metadata.yaml`` (e.g. total_params, optimizer, device info).

    Returns:
        ``(config_path, metadata_path_or_none)``.
    """
    out = Path(log_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "run_config.yaml"
    OmegaConf.save(rc, config_path)

    metadata_path: Path | None = None
    if metadata:
        metadata_path = out / "run_metadata.yaml"
        metadata_path.write_text(
            OmegaConf.to_yaml(OmegaConf.create(metadata)), encoding="utf-8"
        )
    return config_path, metadata_path


def load_run_config_archive(yaml_path: str | Path) -> Any:
    """Load a ``run_config.yaml`` and revalidate against its model's schema.

    The model name is read from ``experiment.model_name``; the merge validates
    that every archived node/field still exists in the current schema.

    Returns:
        RunConfig as an OmegaConf ``DictConfig``.
    """
    yaml_path = Path(yaml_path)
    archived = OmegaConf.load(yaml_path)
    model_name = archived.experiment.model_name
    schema = OmegaConf.structured(build_run_config_schema(model_name))
    return OmegaConf.merge(schema, archived)


def load_run_metadata(log_dir: str | Path) -> dict[str, Any]:
    """Read the sidecar ``run_metadata.yaml`` if present, else return ``{}``."""
    path = Path(log_dir) / "run_metadata.yaml"
    if not path.exists():
        return {}
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


__all__ = [
    "load_run_config_archive",
    "load_run_metadata",
    "save_run_config_archive",
]
