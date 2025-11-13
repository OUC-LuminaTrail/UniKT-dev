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
├── model/                          # 模型实现
│   └── GIKT/                       # GIKT 模型
│       ├── GIKT_data.py           # 数据加载
│       ├── GIKT_model.py          # 模型定义
│       └── GIKT_trainer.py        # 训练逻辑
├── utility/                        # 工具模块
│   ├── net_trainer.py             # 基础训练器
│   ├── hyperparam_manager.py      # 超参数管理
│   └── data_process/              # 数据处理模块
│       ├── assist09.py            # ASSISTments2009 预处理
│       ├── assist12.py            # ASSISTments2012 预处理
│       ├── assist17.py            # ASSISTments2017 预处理
│       ├── ednet_kt1.py           # EdNet KT1 预处理
│       └── data_utility.py        # 通用数据处理函数
├── runs/                           # TensorBoard 日志目录
├── data_process.py                 # 数据预处理脚本
└── train_gikt.py                   # GIKT 模型训练脚本
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

### 2. 训练 GIKT 模型

数据预处理完成后，即可开始训练模型。

**基础训练命令：**

```bash
# 在 ASSISTments2009 数据集上训练
python train_gikt.py -d assistments09

# 在其他数据集上训练
python train_gikt.py -d assistments12
python train_gikt.py -d assistments17
python train_gikt.py -d ednet_kt1
```

**完整训练参数：**

```bash
python train_gikt.py \
    -d assistments09 \
    --hidden_dim 100 \
    --embedding_dim 100 \
    --lstm_layers 2 \
    --dropout 0.4 \
    --n_hop 3 \
    --history_neighbour 5 \
    --att_bound 0.2 \
    --epochs 150 \
    --batch_size 128 \
    --lr 0.001 \
    --weight_decay 1e-4 \
    --fold 0 \
    --seed 42
```

**训练参数说明：**

*模型参数：*
- `--hidden_dim`: 隐藏层维度（默认：100）
- `--embedding_dim`: 嵌入层维度（默认：100）
- `--lstm_layers`: LSTM 层数（默认：2）
- `--dropout`: Dropout 概率（默认：0.4）
- `--n_hop`: GNN 跳数（默认：3）
- `--history_neighbour`: 考虑的邻居数量（默认：5）
- `--att_bound`: 注意力边界值（默认：0.2）

*数据参数：*
- `-d, --dataset`: 数据集名称（必需）
- `--data_base_path`: 数据文件路径（默认：`./data`）
- `--fold`: K折交叉验证的折索引（默认：0）

*训练参数：*
- `--epochs`: 训练轮数（默认：150）
- `--batch_size`: 批大小（默认：128）
- `--lr`: 学习率（默认：0.001）
- `--lr_decay`: 学习率衰减因子（可选）
- `--weight_decay`: 权重衰减（L2正则化）（默认：1e-4）

*其他参数：*
- `--seed`: 随机种子（默认：42）
- `--device`: 设备（`cuda` 或 `cpu`，默认自动检测）

### 3. 查看训练结果

训练过程中的指标会自动保存到 `runs/` 目录，可以使用 TensorBoard 进行可视化。

```bash
# 启动 TensorBoard
tensorboard --logdir runs

# 指定端口
tensorboard --logdir runs --port 6006
```

然后在浏览器中访问 `http://localhost:6006` 查看训练曲线、损失、准确率、AUC等指标。

### 4. K折交叉验证

如果需要进行完整的K折交叉验证评估，可以依次训练所有折：

```bash
# 训练第0折到第4折（假设使用5折交叉验证）
for fold in {0..4}; do
    python train_gikt.py -d assistments09 --fold $fold
done
```

## 输出说明

### 训练日志

训练过程中会在终端输出：
- 每个 epoch 的训练损失、准确率和 AUC
- 验证集的损失、准确率和 AUC
- 当前最佳验证集性能

### TensorBoard 日志

在 `runs/` 目录下，每次训练会创建一个以时间戳命名的子目录，包含：
- `events.out.tfevents.*`: TensorBoard 事件文件
- `hyperparameters.json`: 超参数配置文件

### 超参数配置

每次训练的超参数会自动保存到 `runs/<timestamp>/hyperparameters.json`，方便实验管理和结果复现。
