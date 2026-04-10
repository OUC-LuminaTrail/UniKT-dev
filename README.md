<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/KT--Exp-Knowledge%20Tracing-blue?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=black">
    <img src="https://img.shields.io/badge/KT--Exp-Knowledge%20Tracing-blue?style=for-the-badge&logo=pytorch&logoColor=white" alt="KT-Exp">
  </picture>
</p>

<h3 align="center">A Modular Knowledge Tracing Framework</h3>

<p align="center">
  <a href="https://www.python.org/">
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white">
  </a>
  <a href="https://pytorch.org/">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square&logo=pytorch&logoColor=white">
  </a>
  <a href="https://pixi.sh/">
    <img alt="Pixi" src="https://img.shields.io/badge/Pixi-Ready-orange?style=flat-square">
  </a>
</p>

<p align="center">
  <a href="https://github.com/OUC-LuminaTrail/kt-exp-graph/blob/main/i18n/README_zh-CN.md">简体中文</a> | <b>English</b>
</p>

<p align="center">
  <a href="#-highlights">Highlights</a> •
  <a href="#-supported-models">Models</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-documentation">Documentation</a> •
</p>

---

## 🔥 Highlights

- **KT Models**: DKT, AKT, GIKT, HGIKT, SGKT, SQGKT, DyGKT, SimpleKT, and more
- **Multi-Dataset Support**: ASSISTments (2009/2012/2017), EdNet-KT1
- **Optuna Integration**: Automated hyperparameter search with parallel support
- **Ablation Framework**: Built-in tools for component-wise analysis
- **Experiment Tracking**: SwanLab integration for metrics visualization
- **Modular Design**: Registry-based architecture for easy extension

## 🤖 Supported Models

| Model | Paper | Code |
|-------|-------|------|
| **DKT** | [Deep Knowledge Tracing (WWW 2015)](https://dl.acm.org/doi/10.1145/2736277.2741782) | [GitHub](https://github.com/chrispiech/DeepKnowledgeTracing) |
| **AKT** | [Context-Aware Attentive Knowledge Tracing (KDD 2020)](https://dl.acm.org/doi/10.1145/3394486.3403282) | [GitHub](https://github.com/arghosh/AKT) |
| **GKT** | [Graph-based Knowledge Tracing (ICONIP 2019)](https://link.springer.com/chapter/10.1007/978-3-030-36708-4_12) | [GitHub](https://github.com/jhljx/GKT) |
| **GIKT** | [A Graph-based Interaction Model for KT (ECML-PKDD 2020)](https://link.springer.com/chapter/10.1007/978-3-030-67664-3_17) | [GitHub](https://github.com/ApexEDM/GIKT) |
| **SGKT** | [Session Graph-based Knowledge Tracing (ESA 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0957417422009770) | [GitHub](https://github.com/CCNUZFW/SGKT) |
| **SQGKT** | [Student-Question Interaction Graph-based KT (ESA 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0957417425027915) | [GitHub](https://github.com/Yingying933/SQGKT) |
| **DyGKT** | [Dynamic Graph Learning for Knowledge Tracing (CIKM 2024)](https://dl.acm.org/doi/10.1145/3627673.3679842) | [GitHub](https://github.com/PengLinzhi/DyGKT) |
| **SimpleKT** | [A Simple But Tough-to-Beat Baseline (ICLR 2023)](https://openreview.net/forum?id=V34evfF6ch) | [GitHub](https://github.com/pykt-team/pykt-toolkit) |

## 📦 Installation

### Option 1: Using Pixi (Recommended)

```bash
# Clone the repository
git clone https://github.com/OUC-LuminaTrail/kt-exp-graph.git
cd kt-exp-graph

# Activate GPU environment (CUDA 12.8, Python 3.12, PyTorch 2.10)
pixi shell

# Or use CPU environment
pixi shell -e cpu

# Or use DHG environment (for HGIKT model)
pixi shell -e dhg-gpu

# Or use Mamba environment (for Mamba-based models)
pixi shell -e mamba
```

### Option 2: Automated Conda Setup

Use the provided setup script for automatic environment configuration:

```bash
# Clone the repository
git clone https://github.com/OUC-LuminaTrail/kt-exp-graph.git
cd kt-exp-graph

# Auto-detect GPU and create environment (recommended)
./scripts/setup_env.sh

# Specify environment name
./scripts/setup_env.sh -n myenv

# Force CPU environment
./scripts/setup_env.sh --cpu

# Force GPU environment
./scripts/setup_env.sh --gpu

# Non-interactive mode (use existing env if present)
./scripts/setup_env.sh --yes

# Force recreate environment
./scripts/setup_env.sh --force
```

**Script Options:**

| Option | Description |
|--------|-------------|
| `-n, --env-name NAME` | Conda environment name (default: kt-exp) |
| `--feature FEATURE` | Install feature: gpu, cpu, dhg-gpu, dhg-cpu |
| `--cpu` | Force CPU installation |
| `--gpu` | Force GPU installation |
| `--force` | Remove and recreate existing environment |
| `--yes` | Non-interactive mode |

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

## 🚀 Quick Start

### 1. Data Preparation

```bash
# Download and process ASSISTments 2009 dataset
python data_process.py download -d assistments09
python data_process.py process -d assistments09
```

### 2. Train a Model

```bash
# Train GIKT model
python train.py -m GIKT -d assistments09

# Train with K-fold cross-validation
python train.py -m GIKT -d assistments09 --fold 0
```

### 3. Hyperparameter Search

```bash
# Optuna search with 50 trials
python optuna_search.py -m GIKT -d assistments09 --n_trials 50
```

### 4. Ablation Study

```bash
python ablation_study.py --config configs/ablation/hgikt_study.json -d assistments09
```

## 📊 Supported Datasets

| Dataset | Source |
|---------|--------|
| ASSISTments 2009 | [ASSISTmentsData](https://sites.google.com/site/assistmentsdata/datasets/2009-2010-assistment-data) |
| ASSISTments 2012 | [ASSISTmentsData](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect) |
| ASSISTments 2017 | [ASSISTmentsData](https://sites.google.com/site/assistmentsdata/datasets/2017-assistments-data) |
| EdNet-KT1 | [GitHub](https://github.com/riiid/ednet) |

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Quick Start](https://github.com/OUC-LuminaTrail/kt-exp-graph/blob/main/docs/quick_start.md) | Environment setup and minimal example |
| [Data Processing](https://github.com/OUC-LuminaTrail/kt-exp-graph/blob/main/docs/data_processing.md) | Download, cleaning, and K-fold splitting |
| [Training](https://github.com/OUC-LuminaTrail/kt-exp-graph/blob/main/docs/training.md) | Training pipeline, K-fold, early stopping |
| [Hyperparameter Search](https://github.com/OUC-LuminaTrail/kt-exp-graph/blob/main/docs/hyperparameter_search.md) | Optuna configuration and visualization |
| [Ablation Study](https://github.com/OUC-LuminaTrail/kt-exp-graph/blob/main/docs/ablation_study.md) | Component analysis framework |
| [Case Analysis](https://github.com/OUC-LuminaTrail/kt-exp-graph/blob/main/docs/case_analysis.md) | Inference, user selection, visualization |
| [Architecture](https://github.com/OUC-LuminaTrail/kt-exp-graph/blob/main/docs/architecture.md) | Framework design and extension guide |

## 📁 Project Structure

```
kt-exp-graph/
├── configs/           # Configuration files
│   ├── ablation/      # Ablation configs
│   └── optuna/        # Optuna search spaces
├── data/              # Processed datasets
├── docs/              # Documentation
├── model/             # Model implementations
│   ├── GIKT/
│   ├── HGIKT/
│   ├── SGKT/
│   └── layers/        # Shared components
├── runs/              # Experiment outputs
├── utils/             # Framework utilities
│   ├── training/      # Training infrastructure
│   ├── config/        # Configuration management
│   └── data_process/  # Data processing tools
├── train.py           # Training entry point
├── data_process.py    # Data processing entry
├── optuna_search.py   # Hyperparameter search
└── ablation_study.py  # Ablation experiments
```

## 🤝 Acknowledgments

This framework builds upon the excellent work of:

- **[pyKT](https://github.com/pykt-team/pykt-toolkit)** - Reference for AKT, SimpleKT, DKT, GKT
- **[pyedmine](https://github.com/ZhijieXiong/pyedmine)** - Reference for GIKTEdmine, SQGKT
