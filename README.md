# KT-GNN 实验模型（PyG 框架）

## 项目依赖

- Python：3.13
- torch：2.8.0
- torch_geometric：2.7.0
- pyg-lib：0.5.0
- scikit-learn：1.7.2
- pandas：2.2.3
- pyarrow：22.0.0
- swanlab: 0.7.2
- python-dotenv: 1.2.1

### 环境安装

```bash
# CPU only
conda install scikit-learn pandas pyarrow numpy python=3.13 -c conda-forge -y
uv pip install swanlab python-dotenv
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install torch_geometric pyg-lib -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

# GPU (CUDA 12.9)
conda install scikit-learn pandas pyarrow numpy python=3.13 -c conda-forge -y
uv pip install swanlab python-dotenv
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
uv pip install torch_geometric pyg-lib -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
```

### 配置说明

本项目使用 `.env` 文件管理环境变量配置（如飞书通知）。

1. 复制示例配置文件：
   ```bash
   cp .env.example .env
   ```

2. 修改 `.env` 文件中的配置项：
   - `LARK_WEBHOOK_URL`: 飞书机器人 Webhook 地址
   - `LARK_SECRET`: 飞书机器人签名密钥

## 项目结构

```
kt-exp-graph/
├── data/                           # 数据目录
│   ├── assistments09/
│   ├── assistments12/
│   ├── assistments15/
│   ├── assistments17/
│   └── EdNet/
├── docs/                               # 文档
│   ├── early_stopping.md               # 早停机制说明
│   ├── train_gikt.md                   # GIKT 训练指南
│   └── train_sqgkt.md                  # SQGKT 训练指南
├── model/                              # 模型实现
│   ├── GIKT/
│   │   ├── GIKT_data.py                # 数据加载
│   │   ├── GIKT_model.py               # 模型定义
│   │   └── GIKT_trainer.py             # 训练逻辑
│   └── SQGKT/
│       ├── SQGKT_data.py               # 数据加载
│       ├── SQGKT_model.py              # 模型定义
│       └── SQGKT_trainer.py            # 训练逻辑
├── utility/                            # 工具模块
│   ├── early_stopping.py               # 通用早停器
│   ├── hyperparam_manager.py           # 超参数管理
│   ├── net_trainer.py                  # 基础训练器
│   └── data_process/                   # 数据处理模块
├── runs/                               # 实验日志目录
├── data_process.py                     # 数据预处理脚本
├── train_gikt.py                       # GIKT 训练脚本
└── train_sqgkt.py                      # SQGKT 训练脚本
```

## 快速开始

### 1. 数据预处理

首先需要对原始数据集进行预处理。预处理包括数据下载（可选）和数据清洗、序列构建及K折交叉验证划分。

#### 下载数据 (可选)

如果本地没有原始数据，可以使用 `download` 命令下载：

```bash
# 下载 ASSISTments2009 数据集
python data_process.py download -d assistments09
```

**下载参数说明：**
- `-d, --dataset`: 选择数据集（必需，可选值：`assistments09`, `assistments12`, `assistments17`, `ednet_kt1`）
- `--data_base_path`: 数据基础路径（默认：`./data`）
- `--data_url`: 自定义下载链接（可选）

#### 处理数据

使用 `process` 命令进行数据清洗和格式化：

```bash
# 预处理 ASSISTments2009 数据集
python data_process.py process -d assistments09

# 预处理其他数据集
python data_process.py process -d assistments12
python data_process.py process -d assistments17
python data_process.py process -d ednet_kt1
```

**处理参数说明：**

- `-d, --dataset`: 选择数据集（必需）
- `--data_base_path`: 数据基础路径（默认：`./data`）
- `--min_seq_len`: 最小序列长度（默认：10）
- `--max_seq_len`: 最大序列长度（默认：200）
- `--kfold`: K折交叉验证的折数（默认：5，>=2 启用交叉验证）
- `--seed`: 随机种子（默认：42）

**示例：自定义参数预处理**

```bash
python data_process.py process \
    -d assistments09 \
    --data_base_path ./data \
    --min_seq_len 5 \
    --max_seq_len 100 \
    --kfold 5 \
    --seed 42
```

### 2. 模型训练

各模型的训练说明在 `docs/` 目录下：

- GIKT 训练指南：参见 `docs/train_gikt.md`
- SQGKT 训练指南：参见 `docs/train_sqgkt.md`

### 复现性设置

训练脚本中的 `--seed` 参数现在由 `utility/net_trainer.py` 统一处理。`Trainer` 会调用 `seed_everything` 同步设置 Python、NumPy 与 PyTorch 的随机状态，并强制启用确定性的 cuDNN 配置，从而保证数据划分、图构建和模型训练全过程具有可复现性。

### 3. 查看训练结果

训练过程中的指标会自动上传到 SwanLab，可以在 SwanLab 官网查看训练曲线、损失、准确率、AUC等指标。

如果是首次使用，需要登录 SwanLab：

```bash
swanlab login
```

## 输出说明

### SwanLab 日志

训练过程中的指标、超参数和系统信息会自动记录到 SwanLab 实验中。
本地日志文件保存在 `swanlog/` 目录下（默认）。

### 超参数配置

每次训练的超参数会自动保存到 `runs/<timestamp>/hyperparameters.json`，方便实验管理和结果复现。
