"""Serialize Optuna search configuration to YAML for the ``optuna_search.py`` CLI.

The web launch form collects optuna knobs (n_trials, sampler, pruner, ...) as a
flat dict; this writes them to a per-task YAML consumed by ``optuna_search.py
--optuna_config``. Only ``OptunaConfig``-compatible keys are kept so
``load_optuna_config`` (which passes the dict straight to the dataclass
constructor) never raises on an unexpected key.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Keys accepted by OptunaConfig (utils/optuna_utils/config.py). ``directions`` and
# ``save_dir`` are intentionally excluded — optuna_search.py overrides both at
# runtime (direction from ``--metric``, save_dir from the experiment dir), so
# persisting them would be misleading. ``db_url`` is left at its null default so
# the study database is derived from save_dir (i.e. ``<output_dir>/study.db``).
_OPTUNA_CONFIG_KEYS = (
    "sampler",
    "sampler_kwargs",
    "seed",
    "pruner",
    "pruner_kwargs",
    "n_trials",
    "n_jobs",
    "timeout",
    "study_name",
    "verbose",
)
# Dict-typed knobs: OptunaConfig declares them with default_factory=dict, so a
# None would bypass the factory and later break ``.copy()`` in get_sampler /
# get_pruner. Coerce None -> {} for these.
_DICT_KEYS = ("sampler_kwargs", "pruner_kwargs")


def write_optuna_config(config: dict, path: Path) -> str:
    """Write an optuna search config dict to *path* as YAML.

    Args:
        config: Flat optuna knobs from the launch form. Unknown keys (e.g.
            ``metric``, which is a CLI flag rather than an OptunaConfig field)
            are dropped so the YAML round-trips through ``OptunaConfig(**yaml)``.
        path: Destination YAML path (parents created as needed).

    Returns:
        The string path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    filtered: dict = {}
    for key in _OPTUNA_CONFIG_KEYS:
        if key not in config:
            continue
        value = config[key]
        if key in _DICT_KEYS and value is None:
            value = {}
        filtered[key] = value
    with open(path, "w") as f:
        yaml.safe_dump(filtered, f, sort_keys=False)
    return str(path)
