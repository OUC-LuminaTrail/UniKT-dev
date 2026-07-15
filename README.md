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
  <a href="https://github.com/szhhwh/UniKT/blob/main/i18n/README_zh-CN.md">简体中文</a> | <b>English</b>
</p>

<p align="center">
  <a href="#-highlights">Highlights</a> •
  <a href="#-supported-models">Models</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-documentation">Documentation</a>
</p>

---

## 🔥 Highlights

- **KT Models**: DKT, AKT, GKT, GIKT, HGIKT, SGKT, SQGKT, DyGKT, SimpleKT, DKVMN, DTransformer, HawkesKT, LBKT, QIKT, ReKT, RobustKT, UKT, ABKT, BDGKT, ClusterKT, and more
- **Multi-Dataset Support**: ASSISTments (2009/2012/2017), EdNet-KT1
- **Optuna Integration**: Automated hyperparameter search with parallel support
- **Experiment Tracking**: SwanLab integration for metrics visualization
- **Modular Design**: Registry-based architecture for easy extension

## 🤖 Supported Models

| Model | Paper | Code |
|-------|-------|------|
| **DKT** | [Deep Knowledge Tracing (NeurIPS 2015)](https://papers.nips.cc/paper/5654-deep-knowledge-tracing) | [GitHub](https://github.com/chrispiech/DeepKnowledgeTracing) |
| **AKT** | [Context-Aware Attentive Knowledge Tracing (KDD 2020)](https://dl.acm.org/doi/10.1145/3394486.3403282) | [GitHub](https://github.com/arghosh/AKT) |
| **GKT** | [Graph-based Knowledge Tracing (WI 2019)](https://dl.acm.org/doi/10.1145/3350546.3352513) | [GitHub](https://github.com/jhljx/GKT) |
| **GIKT** | [A Graph-based Interaction Model for KT (ECML-PKDD 2020)](https://arxiv.org/abs/2009.05991) | [GitHub](https://github.com/ApexEDM/GIKT) |
| **SGKT** | [Session Graph-based Knowledge Tracing (ESA 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0957417422009770) | [GitHub](https://github.com/CCNUZFW/SGKT) |
| **SQGKT** | [Student-Question Interaction Graph-based KT (ESA 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0957417425027915) | [GitHub](https://github.com/Yingying933/SQGKT) |
| **DyGKT** | [Dynamic Graph Learning for Knowledge Tracing (KDD 2024)](https://dl.acm.org/doi/10.1145/3637528.3671773) | [GitHub](https://github.com/PengLinzhi/DyGKT) |
| **SimpleKT** | [A Simple But Tough-to-Beat Baseline (ICLR 2023)](https://openreview.net/forum?id=9HiGqC9C-KA) | [pyKT](https://github.com/pykt-team/pykt-toolkit) |
| **DKVMN** | [Dynamic Key-Value Memory Networks for Knowledge Tracing (WWW 2017)](https://dl.acm.org/doi/10.1145/3038912.3052580) | [GitHub](https://github.com/jennyzhang0215/DKVMN) |
| **DTransformer** | [Tracing Knowledge Instead of Patterns: Stable KT with Diagnostic Transformer (WWW 2023)](https://dl.acm.org/doi/10.1145/3543507.3583255) | [GitHub](https://github.com/yxonic/DTransformer) |
| **HawkesKT** | [Temporal Cross-Effects in Knowledge Tracing (WSDM 2021)](https://dl.acm.org/doi/10.1145/3437963.3441802) | [GitHub](https://github.com/THUwangcy/HawkesKT) |
| **LBKT** | [Learning Behavior-oriented Knowledge Tracing (KDD 2023)](https://dl.acm.org/doi/10.1145/3580305.3599407) | [GitHub](https://github.com/bigdata-ustc/EduKTM) |
| **QIKT** | [Improving Interpretability of Deep Sequential KT with Question-centric Cognitive Representations (AAAI 2023)](https://ojs.aaai.org/index.php/AAAI/article/view/26661) | [pyKT](https://github.com/pykt-team/pykt-toolkit) |
| **ReKT** | [Revisiting Knowledge Tracing: A Simple and Powerful Model (ACM MM 2024)](https://openreview.net/forum?id=GYomxff6HZ) | [GitHub](https://github.com/lilstrawberry/ReKT) |
| **RobustKT** | [Enhancing Knowledge Tracing through Decoupling Cognitive Pattern from Error-Prone Data (WWW 2025)](https://dl.acm.org/doi/10.1145/3696410.3714486) | [pyKT](https://github.com/pykt-team/pykt-toolkit) |
| **UKT** | [Uncertainty-aware Knowledge Tracing (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/35007) | [GitHub](https://github.com/UncertaintyForKnowledgeTracing/UKT) |
| **ABKT** | [Ability Boosted Knowledge Tracing (Information Sciences 2022)](https://www.sciencedirect.com/science/article/pii/S0020025522001876) | [GitHub](https://github.com/ccnu-mathits/ABKT) |
| **BDGKT** | [Bidirectional Dynamic Graph Knowledge Tracing (Knowledge-Based Systems 2026)](https://doi.org/10.1016/j.knosys.2026.115532) | [GitHub](https://github.com/Oia-10/BDGKT) |
| **ClusterKT** | [Cluster-driven Knowledge Tracing: Joint Learning-Forgetting Effects Modeling via State Dependency (ESA 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0957417425022389) | [GitHub](https://github.com/Lzhenghua/ClusterKT) |

## 📦 Installation

### Option 1: Using Pixi (Recommended)

```bash
# Clone the repository
git clone https://github.com/szhhwh/UniKT.git
cd UniKT

# Activate GPU environment (CUDA 12.8, Python 3.12, PyTorch 2.10)
pixi shell

# Or use CPU environment
pixi shell -e cpu

# Or use DHG environment (for HGIKT model)
pixi shell -e dhg-gpu

# Or use Mamba environment (for Mamba-based models)
pixi shell -e mamba
```

### Option 2: Manual Conda Setup

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

## 📊 Supported Datasets

| Dataset | Source |
|---------|--------|
| ASSISTments 2009 | [ASSISTmentsData](https://www.etrialstestbed.org/data-sets) |
| ASSISTments 2012 | [ASSISTmentsData](https://www.etrialstestbed.org/data-sets) |
| ASSISTments 2015 | [ASSISTmentsData](https://www.etrialstestbed.org/data-sets) |
| ASSISTments 2017 | [ASSISTmentsData](https://www.etrialstestbed.org/data-sets) |
| EdNet-KT1 | [GitHub](https://github.com/riiid/ednet) |
| Junyi 2015 | [Junyi Academy](https://pslcdatashop.web.cmu.edu/Files?datasetId=1198) |
| Slepemapy | [SLEP](https://www.fi.muni.cz/adaptivelearning/) |

## 📚 Documentation

Full documentation is available at **[UniKT](https://unikt.lionhao.top/)**.

## 📁 Project Structure

```
UniKT/
├── configs/           # Configuration files
│   └── optuna/        # Optuna search spaces
├── data/              # Processed datasets
├── docs/              # Documentation site
├── model/             # Model implementations
│   ├── DKT/
│   └── layers/        # Shared components
├── runs/              # Experiment outputs
├── utils/             # Framework utilities
│   ├── training/      # Training infrastructure
│   ├── config/        # Configuration management
│   └── data_process/  # Data processing tools
├── web/               # Web management panel
├── train.py           # Training entry point
├── data_process.py    # Data processing entry
└── optuna_search.py   # Hyperparameter search
```

## 🤝 Acknowledgments

This framework builds upon the excellent work of:

- **[pyKT](https://github.com/pykt-team/pykt-toolkit)** - Reference for AKT, SimpleKT, DKT, GKT
