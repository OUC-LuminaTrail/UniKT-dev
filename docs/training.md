# Training

Train knowledge tracing models with the fluent trainer API.

## Prerequisites

Before training, ensure your environment is properly configured. See [Quick Start](quick_start.md) for installation options:

- **Pixi** (recommended): `pixi shell` for GPU or `pixi shell -e cpu` for CPU
- **Automated Conda**: `./scripts/setup_env.sh` for auto-configuration
- **Manual Conda**: Follow the guide in [Quick Start](quick_start.md#option-3-manual-conda-setup)

## Training Flow

```mermaid
flowchart LR
    A[Initialize Model] --> B[Build Trainer]
    B --> C[Train Loop]
    C --> D[Save Checkpoint]
```

## Quick Start

### Basic Training

```bash
# Train GIKT on ASSISTments 2009
python train.py -m GIKT -d assistments09
```

### K-Fold Cross-Validation

```bash
# Single fold
python train.py -m GIKT -d assistments09 --fold 0

# All folds
for i in {0..4}; do
    python train.py -m GIKT -d assistments09 --fold $i
done
```

### With Early Stopping

```bash
python train.py -m GIKT -d assistments09 \
    --es_patience 10 \
    --es_monitor auc \
    --es_mode max
```

## Parameters

### Common Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-m, --model` | Required | Model name |
| `-d, --dataset` | Required | Dataset name |
| `--fold` | 0 | K-fold index |
| `--seed` | 42 | Random seed |
| `--device` | auto | Device (cuda/cpu, auto-detect) |

### Early Stopping

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--es_patience` | 10 | Patience epochs (set to 0 to disable) |
| `--es_monitor` | auc | Metric to monitor |
| `--es_mode` | max | `max` or `min` |
| `--es_min_delta` | 0.0 | Minimum improvement |
| `--es_restore_best` | False | Restore best weights when stopped |

### Advanced Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_grad_norm` | None | Max gradient norm for clipping (model-specific, e.g., DyGKT uses 10.0) |
| `--skip_test` | False | Skip test evaluation after training |
| `--deterministic` | True | Enable deterministic algorithms |

## Output

Training results are saved to `runs/<type>/<run_id>/`:

```
runs/normal/GIKT_assistments09_20240403-120000_fold0_bs128/
├── best_model.pth          # Best model checkpoint
├── last_checkpoint.pth     # Last checkpoint
├── hyperparameters.json    # Hyperparameter config
├── training.log            # Training logs
└── case_analysis/          # Case analysis (optional)
```

### SwanLab Integration

Metrics are automatically logged to SwanLab:

```bash
# First-time login
swanlab login
```

Tracked metrics:
- Loss (train/val)
- Accuracy (ACC)
- AUC score
- Learning rate
- GPU utilization

## Model-Specific Parameters

Different models have additional parameters. View all options:

```bash
python train.py -m GIKT -h
```

Examples:

```bash
# GIKT-specific
python train.py -m GIKT -d assistments09 \
    --hidden_dim 256 \
    --n_layers 2 \
    --heads 4 \
    --dropout 0.1

# HGIKT-specific
python train.py -m HGIKT -d assistments09 \
    --hidden_dim 128 \
    --n_hop 4
```

## Advanced Usage

### Custom Trainer

```python
from utils.training import BaseTrainer

class MyTrainer(BaseTrainer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.model = MyModel(config)

    def forward_pass(self, batch):
        logits = self.model(batch)
        loss = self.loss_fn(logits, batch['response'].float())
        return loss, logits
```

### Gradient Clipping

```bash
python train.py -m GIKT -d assistments09 --max_grad_norm 1.0
```

### Skip Test Evaluation

```bash
python train.py -m GIKT -d assistments09 --skip_test
```

## Related Docs

- [Quick Start](quick_start.md) - Environment setup
- [Hyperparameter Search](hyperparameter_search.md) - Optuna optimization
- [Architecture](architecture.md) - Framework design