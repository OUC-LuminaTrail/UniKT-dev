<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/KT--Exp-%E7%9F%A5%E8%AF%86%E8%BF%BD%E8%B8%AA-blue?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=black">
    <img src="https://img.shields.io/badge/KT--Exp-%E7%9F%A5%E8%AF%86%E8%BF%BD%E8%B8%AA-blue?style=for-the-badge&logo=pytorch&logoColor=white" alt="KT-Exp">
  </picture>
</p>

<h3 align="center">模块化知识追踪实验框架</h3>

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

---

<p align="center">
  <b>简体中文</b> | <a href="https://github.com/OUC-LuminaTrail/kt-exp-graph/blob/main/README.md">English</a>
</p>

<p align="center">
  <a href="#-核心特性">核心特性</a> •
  <a href="#-支持的模型">模型</a> •
  <a href="#-安装">安装</a> •
  <a href="#-快速开始">快速开始</a> •
  <a href="#-文档">文档</a> •
</p>

---

## 🔥 核心特性

- **KT 模型**：DKT、AKT、GIKT、HGIKT、SGKT、SQGKT、DyGKT、SimpleKT 等
- **多数据集支持**：ASSISTments（2009/2012/2017）、EdNet-KT1
- **Optuna 集成**：自动化超参数搜索，支持并行优化
- **消融实验框架**：内置组件级分析工具
- **实验追踪**：SwanLab 集成，实时可视化训练指标
- **模块化设计**：注册表架构，易于扩展新模型

## 🤖 支持的模型

| 模型简写 | 模型论文链接 | 模型代码仓库链接 |
|------|------|------|
| **DKT** | [Deep Knowledge Tracing (WWW 2015)](https://dl.acm.org/doi/10.1145/2736277.2741782) | [GitHub](https://github.com/chrispiech/DeepKnowledgeTracing) |
| **AKT** | [Context-Aware Attentive Knowledge Tracing (KDD 2020)](https://dl.acm.org/doi/10.1145/3394486.3403282) | [GitHub](https://github.com/arghosh/AKT) |
| **GKT** | [Graph-based Knowledge Tracing (ICONIP 2019)](https://link.springer.com/chapter/10.1007/978-3-030-36708-4_12) | [GitHub](https://github.com/jhljx/GKT) |
| **GIKT** | [A Graph-based Interaction Model for KT (ECML-PKDD 2020)](https://link.springer.com/chapter/10.1007/978-3-030-67664-3_17) | [GitHub](https://github.com/ApexEDM/GIKT) |
| **SGKT** | [Session Graph-based Knowledge Tracing (ESA 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0957417422009770) | [GitHub](https://github.com/CCNUZFW/SGKT) |
| **SQGKT** | [Student-Question Interaction Graph-based KT (ESA 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0957417425027915) | [GitHub](https://github.com/Yingying933/SQGKT) |
| **DyGKT** | [Dynamic Graph Learning for Knowledge Tracing (CIKM 2024)](https://dl.acm.org/doi/10.1145/3627673.3679842) | [GitHub](https://github.com/PengLinzhi/DyGKT) |
| **SimpleKT** | [A Simple But Tough-to-Beat Baseline (ICLR 2023)](https://openreview.net/forum?id=V34evfF6ch) | [GitHub](https://github.com/pykt-team/pykt-toolkit) |

## 📦 安装

### 方式一：使用 Pixi（推荐）

```bash
# 克隆仓库
git clone https://github.com/OUC-LuminaTrail/kt-exp-graph.git
cd kt-exp-graph

# 激活 GPU 环境（CUDA 12.8, Python 3.12, PyTorch 2.10）
pixi shell

# 或使用 CPU 环境
pixi shell -e cpu

# 或使用 DHG 环境（用于 HGIKT 模型）
pixi shell -e dhg-gpu

# 或使用 Mamba 环境（用于基于 Mamba 的模型）
pixi shell -e mamba
```

### 方式二：自动配置 Conda 环境

使用提供的脚本自动配置 Conda 环境：

```bash
# 克隆仓库
git clone https://github.com/OUC-LuminaTrail/kt-exp-graph.git
cd kt-exp-graph

# 自动检测 GPU 并创建环境（推荐）
./scripts/setup_env.sh

# 指定环境名称
./scripts/setup_env.sh -n myenv

# 强制使用 CPU 环境
./scripts/setup_env.sh --cpu

# 强制使用 GPU 环境
./scripts/setup_env.sh --gpu

# 非交互模式（如环境已存在则使用现有环境）
./scripts/setup_env.sh --yes

# 强制重建环境
./scripts/setup_env.sh --force
```

**脚本选项：**

| 选项 | 说明 |
|------|------|
| `-n, --env-name NAME` | Conda 环境名称（默认：kt-exp） |
| `--feature FEATURE` | 安装特性：gpu, cpu, dhg-gpu, dhg-cpu |
| `--cpu` | 强制 CPU 安装 |
| `--gpu` | 强制 GPU 安装 |
| `--force` | 删除并重建已有环境 |
| `--yes` | 非交互模式 |

### 方式三：手动配置 Conda 环境

<details>
<summary>点击展开手动安装指南</summary>

#### GPU 环境（CUDA 12.8）

```bash
# 创建环境
conda create -n ktexp python=3.12
conda activate ktexp

# 安装 PyTorch（CUDA 12.8）
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# 安装 PyTorch Geometric 依赖
pip install pyg_lib==0.6.0 torch-scatter==2.1.2 torch-geometric==2.7.0 \
    -f https://data.pyg.org/whl/torch-2.10.0+cu128.html

# 安装核心依赖
conda install -c conda-forge optuna scikit-learn pandas pyarrow python-dotenv ruff pytest polars seaborn matplotlib -y
pip install swanlab
```

#### CPU 环境

```bash
# 创建环境
conda create -n ktexp python=3.12
conda activate ktexp

# 安装 PyTorch（CPU）
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# 安装 PyTorch Geometric 依赖
pip install pyg_lib==0.6.0 torch-scatter==2.1.2 torch-geometric==2.7.0 \
    -f https://data.pyg.org/whl/torch-2.10.0+cpu.html

# 安装核心依赖
conda install -c conda-forge optuna scikit-learn pandas pyarrow python-dotenv ruff pytest polars seaborn matplotlib -y
pip install swanlab
```

#### DHG 环境（用于 HGIKT 模型）

```bash
# 创建环境（需要 Python 3.10）
conda create -n ktexp-dhg python=3.10
conda activate ktexp-dhg

# 安装 PyTorch（CUDA 11.7）
pip install torch==1.13.1 --index-url https://download.pytorch.org/whl/cu117

# 安装 PyTorch Geometric 依赖
pip install "pyg_lib>=0.4.0,<0.5" torch-scatter==2.1.1 "torch-geometric>=2.7.0,<3" \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

# 安装 DHG 和其他依赖
pip install "dhg==0.9.*"
conda install -c conda-forge optuna scikit-learn pandas pyarrow python-dotenv ruff pytest polars seaborn matplotlib -y
pip install swanlab
```

</details>

## 🚀 快速开始

### 1. 数据准备

```bash
# 下载并处理 ASSISTments 2009 数据集
python data_process.py download -d assistments09
python data_process.py process -d assistments09
```

### 2. 训练模型

```bash
# 训练 GIKT 模型
python train.py -m GIKT -d assistments09

# K 折交叉验证
python train.py -m GIKT -d assistments09 --fold 0
```

### 3. 超参数搜索

```bash
# Optuna 搜索 50 轮
python optuna_search.py -m GIKT -d assistments09 --n_trials 50
```

### 4. 消融实验

```bash
python ablation_study.py --config configs/ablation/hgikt_study.json -d assistments09
```

## 📊 支持的数据集

| 数据集 | 数据来源 |
|--------|----------|
| ASSISTments 2009 | [ASSISTmentsData](https://sites.google.com/site/assistmentsdata/datasets/2009-2010-assistment-data) |
| ASSISTments 2012 | [ASSISTmentsData](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect) |
| ASSISTments 2017 | [ASSISTmentsData](https://sites.google.com/site/assistmentsdata/datasets/2017-assistments-data) |
| EdNet-KT1 | [GitHub](https://github.com/riiid/ednet) |

## 📚 文档

| 文档 | 内容 |
|------|------|
| [快速开始](docs/quick_start.md) | 环境配置与最小示例 |
| [数据处理](docs/data_processing.md) | 下载、清洗、K 折划分 |
| [模型训练](docs/training.md) | 训练流程、K 折验证、早停机制 |
| [超参搜索](docs/hyperparameter_search.md) | Optuna 配置与可视化 |
| [消融实验](docs/ablation_study.md) | 组件分析框架 |
| [案例分析](docs/case_analysis.md) | 推理、用户筛选、可视化 |
| [架构设计](docs/architecture.md) | 框架设计与扩展指南 |

## 📁 项目结构

```
kt-exp-graph/
├── configs/           # 配置文件
│   ├── ablation/      # 消融实验配置
│   └── optuna/        # Optuna 搜索空间
├── data/              # 处理后的数据集
├── docs/              # 项目文档
├── model/             # 模型实现
│   ├── GIKT/
│   ├── HGIKT/
│   ├── SGKT/
│   └── layers/        # 共享组件
├── runs/              # 实验输出
├── utils/             # 框架工具
│   ├── training/      # 训练基础设施
│   ├── config/        # 配置管理
│   └── data_process/  # 数据处理工具
├── train.py           # 训练入口
├── data_process.py    # 数据处理入口
├── optuna_search.py   # 超参搜索
└── ablation_study.py  # 消融实验
```

## 🤝 致谢

本框架参考了以下优秀开源项目：

- **[pyKT](https://github.com/pykt-team/pykt-toolkit)** - AKT、SimpleKT、DKT、GKT 实现参考
- **[pyedmine](https://github.com/ZhijieXiong/pyedmine)** - GIKTEdmine、SQGKT 实现参考
