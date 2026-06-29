# Quick Start

Get up and running with kt-exp in minutes.

## Requirements

- **OS**: Linux
- **Python**: 3.10+
- **CUDA**: 11.7+ (for GPU)

> [!IMPORTANT]  
> We have tested on Ubuntu 24.04 with CUDA 12.8 and Python 3.12. And Ubuntu 22.04 is not supported due to glibc version issues.

## Installation

### Option 1: Pixi (Recommended)

```bash
# GPU environment (CUDA 12.8, Python 3.12, PyTorch 2.10)
pixi shell

# CPU environment
pixi shell -e cpu

# DHG environment (for HGIKT model)
pixi shell -e dhg-gpu

# Mamba environment (for Mamba-based models)
pixi shell -e mamba
```

### Option 2: Automated Conda Setup

Use the provided setup script for automatic environment configuration:

```bash
# Auto-detect GPU and create environment
./scripts/setup_env.sh

# Force CPU environment
./scripts/setup_env.sh --cpu

# Specify environment name
./scripts/setup_env.sh -n myenv

# Non-interactive mode
./scripts/setup_env.sh --yes
```

### Option 3: Manual Conda Setup

<details>
<summary>Click to expand manual installation guide</summary>

#### GPU Environment (CUDA 12.8)

```bash
# Create environment
conda create -n ktexp python=3.12
conda activate ktexp

# Install PyTorch with CUDA 12.8
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# Install PyTorch Geometric dependencies
pip install pyg_lib==0.6.0 torch-scatter==2.1.2 torch-geometric==2.7.0 \
    -f https://data.pyg.org/whl/torch-2.10.0+cu128.html

# Install core dependencies
conda install -c conda-forge optuna scikit-learn pandas pyarrow python-dotenv ruff pytest polars seaborn matplotlib -y
pip install swanlab
```

#### CPU Environment

```bash
# Create environment
conda create -n ktexp python=3.12
conda activate ktexp

# Install PyTorch (CPU)
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric dependencies
pip install pyg_lib==0.6.0 torch-scatter==2.1.2 torch-geometric==2.7.0 \
    -f https://data.pyg.org/whl/torch-2.10.0+cpu.html

# Install core dependencies
conda install -c conda-forge optuna scikit-learn pandas pyarrow python-dotenv ruff pytest polars seaborn matplotlib -y
pip install swanlab
```

#### DHG Environment (for HGIKT)

```bash
# Create environment (requires Python 3.10)
conda create -n ktexp-dhg python=3.10
conda activate ktexp-dhg

# Install PyTorch with CUDA 11.7
pip install torch==1.13.1 --index-url https://download.pytorch.org/whl/cu117

# Install PyTorch Geometric dependencies
pip install "pyg_lib>=0.4.0,<0.5" torch-scatter==2.1.1 "torch-geometric>=2.7.0,<3" \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

# Install DHG and other dependencies
pip install "dhg==0.9.*"
conda install -c conda-forge optuna scikit-learn pandas pyarrow python-dotenv ruff pytest polars seaborn matplotlib -y
pip install swanlab
```

</details>

## Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `LARK_WEBHOOK_URL` | Lark bot webhook for notifications |
| `SWANLAB_WORKSPACE` | SwanLab workspace name |
| `SWANLAB_MODE` | `cloud` (upload) or `local` |

## Workflow Overview

```mermaid
flowchart LR
    A[Download Data] --> B[Process Data]
    B --> C[Train Model]
    C --> D[Evaluate]
```

The typical workflow consists of:
1. **Download** raw dataset from source
2. **Process** data into standardized format with K-fold splits
3. **Train** a knowledge tracing model
4. **Evaluate** model performance

## Minimal Example

```bash
# 1. Download dataset
python data_process.py download -d assistments09

# 2. Process data
python data_process.py process -d assistments09

# 3. Train model
python train.py -m GIKT -d assistments09

# 4. (Optional) Login to SwanLab for experiment tracking
swanlab login
```

## Output

Experiments are saved to `runs/normal/<model>_<dataset>_<timestamp>/`:

```
runs/normal/GIKT_assistments09_20240403-120000_fold0_bs128/
├── best_model.pth          # Best model checkpoint
├── hyperparameters.json    # Hyperparameter config
└── training.log            # Training logs
```

## Next Steps

- [Data Processing](data_processing.md) - Download and preprocess datasets
- [Training](training.md) - Training configuration and K-fold validation
- [Architecture](architecture.md) - Framework design and extension
