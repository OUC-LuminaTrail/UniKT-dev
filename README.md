# KT-GNN 实验模型（PyG 框架）

## 项目依赖

- Python：3.13
- torch：2.8.0
- torch_geometric：2.7.0
- pyg-lib：0.5.0
- scikit-learn：1.7.2
- pandas：2.2.3
- pyarrow：22.0.0
- tensorboard：2.20.0

### 环境安装

```bash
# CPU only
conda install scikit-learn jupyterlab pandas pyarrow numpy tensorboard python=3.13 -c conda-forge -y
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install torch_geometric pyg-lib -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

# GPU (CUDA 12.9)
conda install scikit-learn jupyterlab pandas pyarrow numpy tensorboard python=3.13 -c conda-forge -y
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
uv pip install torch_geometric pyg-lib -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
```

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
├── runs/                               # TensorBoard 日志目录
├── data_process.py                     # 数据预处理脚本
├── train_gikt.py                       # GIKT 训练脚本
└── train_sqgkt.py                      # SQGKT 训练脚本
```

## 快速开始

### 1. 数据预处理

首先需要对原始数据集进行预处理。预处理包括数据清洗、序列构建和K折交叉验证划分。

```bash
# 预处理 ASSISTments2009 数据集
python data_process.py -d assistments09

# 预处理其他数据集
python data_process.py -d assistments12
python data_process.py -d assistments17
python data_process.py -d ednet_kt1
```

**数据预处理参数说明：**

- `-d, --dataset`: 选择数据集（必需）
  - 可选值：`assistments09`, `assistments12`, `assistments17`, `ednet_kt1`
- `--data_base_path`: 数据基础路径（默认：`./data`）
- `--min_seq_len`: 最小序列长度（默认：10）
- `--max_seq_len`: 最大序列长度（默认：200）
- `--kfold`: K折交叉验证的折数（默认：5）
- `--seed`: 随机种子（默认：42）
- `--download`: 是否下载数据集（可选）

**示例：自定义参数预处理**

```bash
python data_process.py \
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

### 3. 查看训练结果

训练过程中的指标会自动保存到 `runs/` 目录，可以使用 TensorBoard 进行可视化。

```bash
# 启动 TensorBoard
tensorboard --logdir runs

# 指定端口
tensorboard --logdir runs --port 6006
```

然后在浏览器中访问 `http://localhost:6006` 查看训练曲线、损失、准确率、AUC等指标。

## 输出说明

### TensorBoard 日志

在 `runs/` 目录下，每次训练会创建一个以时间戳命名的子目录，包含：
- `events.out.tfevents.*`: TensorBoard 事件文件
- `hyperparameters.json`: 超参数配置文件

### 超参数配置

每次训练的超参数会自动保存到 `runs/<timestamp>/hyperparameters.json`，方便实验管理和结果复现。
