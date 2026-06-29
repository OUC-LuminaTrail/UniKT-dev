# Hyperparameter Search

Automated hyperparameter optimization with Optuna.

## Overview

```mermaid
flowchart LR
    A[Load Search Space] --> B[Sample Parameters]
    B --> C[Train]
    C --> D[Evaluate]
    D --> E[Record]
    E --> B
```

## Quick Start

```bash
# Run with default config (100 trials from optuna_config.json)
python optuna_search.py -m HGIKT -d assistments09
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-m, --model` | Required | Model name |
| `-d, --dataset` | Required | Dataset name |
| `--optuna_config` | `./configs/optuna/optuna_config.json` | Optuna config path |
| `--param_space` | `./configs/optuna/param_space_<model>.json` | Parameter space path |
| `--metric` | auc | Target metric (auc/acc/rmse/loss) |

## Optuna Configuration

Configure search settings in `configs/optuna/optuna_config.json`:

```json
{
  "sampler": "tpe",
  "sampler_kwargs": {
    "seed": 42,
    "n_startup_trials": 10
  },
  "pruner": "median",
  "pruner_kwargs": {
    "n_startup_trials": 5,
    "n_warmup_steps": 10
  },
  "n_trials": 100,
  "n_jobs": 1,
  "timeout": null,
  "directions": ["maximize"],
  "study_name": "gikt_hyperparameter_search",
  "db_url": null,
  "save_dir": "./optuna_results",
  "verbose": 1
}
```

**Configuration Options:**

| Option | Description |
|--------|-------------|
| `n_trials` | Number of trials to run |
| `n_jobs` | Parallel jobs (1 = sequential) |
| `timeout` | Timeout in seconds (null = no limit) |
| `directions` | Optimization direction (`maximize` or `minimize`) |
| `db_url` | Database URL for persistence (null = in-memory) |

## Search Space Configuration

Search spaces are defined in `configs/optuna/param_space_<model>.json` as an **array**:

```json
[
  {
    "name": "lr",
    "type": "float",
    "low": 0.0001,
    "high": 0.01,
    "log": true,
    "default": 0.001
  },
  {
    "name": "hidden_dim",
    "type": "int",
    "low": 64,
    "high": 256,
    "log": true,
    "default": 100
  },
  {
    "name": "batch_size",
    "type": "categorical",
    "choices": [32, 64, 128, 256],
    "default": 128
  }
]
```

**Distribution Types:**

| Type | Required Fields | Optional Fields | Description |
|------|-----------------|-----------------|-------------|
| `float` | `low`, `high` | `log` | Float parameter |
| `int` | `low`, `high` | `log` | Integer parameter |
| `categorical` | `choices` | - | Categorical choice |

**Note:** Each parameter must have a `name` and `default` field.

## Output

Results are saved to `runs/hyperparam_search/<study_name>_<timestamp>/`:

```
runs/hyperparam_search/gikt_hyperparameter_search_20240403-120000/
├── best_params.json           # Best parameters
├── trials_history_gikt.csv    # All trials history
└── study.log                  # Search log
```

## Parallel Search

Run multiple workers sharing the same database:

```bash
# Terminal 1 - with SQLite storage
python optuna_search.py -m HGIKT -d assistments09 \
    --optuna_config configs/optuna/optuna_config_db.json

# Terminal 2 - same command
python optuna_search.py -m HGIKT -d assistments09 \
    --optuna_config configs/optuna/optuna_config_db.json
```

**Note:** For parallel search, set `db_url` in config:

```json
{
  "db_url": "sqlite:///optuna.db"
}
```

## Visualization

```python
import optuna

study = optuna.load_study(
    study_name="gikt_hyperparameter_search",
    storage="sqlite:///optuna.db"
)

# Optimization history
optuna.visualization.plot_optimization_history(study)

# Parameter importance
optuna.visualization.plot_param_importances(study)

# Parallel coordinate
optuna.visualization.plot_parallel_coordinate(study)
```

## Best Practices

1. **Start Broad**: Use wide search ranges initially
2. **Iterate**: Narrow ranges based on importance analysis
3. **Parallelize**: Use SQLite storage for multi-worker search
4. **Monitor**: Check intermediate results via SwanLab
5. **Prune**: Enable pruner to stop unpromising trials early

## Related Docs

- [Training](training.md) - Manual training
- [Ablation Study](ablation_study.md) - Component analysis