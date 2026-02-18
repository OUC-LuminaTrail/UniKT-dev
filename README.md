# KT-GNN 实验框架

## 项目依赖

- Python：3.10
- torch：1.13.1
- torch_geometric：2.7.0
- dhg：0.9.5
- pyg-lib：0.4.0
- scikit-learn：1.7.2
- pandas：2.3.3
- pyarrow：22.0.0
- swanlab: 0.7.2
- python-dotenv: 1.2.1

## 运行环境配置

> 该框架仅支持 Linux 系统。

### 使用 `pixi`（推荐）

本项目推荐使用 [pixi](https://pixi.sh/) 管理依赖，它可以自动处理 CUDA 版本和 Python 环境。

```bash
# 安装环境并进入默认 GPU 环境
pixi shell

# 如果没有 GPU，请进入 CPU 环境
pixi shell -e cpu

# 退出环境
exit
```

### 使用 `Anaconda`

若习惯使用 `conda`，我们也提供了自动化配置脚本：

```bash
# 赋予脚本执行权限
chmod +x ./scripts/setup_env.sh

# 运行配置脚本
./scripts/setup_env.sh
```

该脚本会创建一个名为 `ktexp` 的环境并安装所有必要的依赖。


### 手动配置运行环境

```sh
conda create -n ktexp python=3.10

# CPU
uv pip install torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install torch_geometric pyg-lib -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
uv pip install dhg optuna pandas pyarrow swanlab python-dotenv  

# CUDA>=11.7
uv pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
uv pip install torch_geometric pyg-lib -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
uv pip install dhg optuna pandas pyarrow swanlab python-dotenv
```

### 配置说明

本项目使用 `.env` 文件管理环境变量配置。

1. 复制示例配置文件：
   ```bash
   cp .env.example .env
   ```

2. 修改 `.env` 文件中的配置项：
   - `LARK_WEBHOOK_URL`: 飞书机器人 Webhook 地址。
   - `SWANLAB_WORKSPACE`: SwanLab 项目空间名称。
   - `SWANLAB_MODE`: 设置为 `cloud`（上传）或 `local`（仅本地日志）。

## 项目结构

```
kt-exp-graph/
├── configs/                           # 配置与参数空间
│   ├── ablation/                      # 消融实验配置
│   │   ├── gikt_ablation.json
│   │   ├── hgikt_ablation.json
│   │   └── sqgkt_ablation.json
│   └── optuna/                        # Optuna 搜索配置
│       ├── optuna_config.json
│       ├── param_space_gikt.json
│       └── param_space_hgikt.json
├── data/                              # 数据目录
│   ├── assistments09/
│   ├── assistments12/
│   ├── assistments17/
│   └── ednet_kt1/
├── model/                             # 模型实现
│   ├── ABKT/
│   ├── DGEKT/
│   ├── GIKT/
│   ├── HGIKT/
│   ├── SGKT/
│   ├── SQGKT/
│   └── layers/                        # 共享模型组件
│       ├── components.py
│       └── __init__.py
├── runs/                              # 实验运行日志与检查点
├── swanlog/                           # SwanLab 本地日志
├── utils/                             # 工具模块
│   ├── ablation/                      # 消融实验工具
│   ├── config/                        # 配置管理
│   │   ├── data_config.py            # 数据加载器配置
│   │   ├── param_config.py            # 参数配置
│   │   └── training_config.py        # 训练配置（含早停）
│   ├── core/                          # 核心工具
│   │   ├── logger.py
│   │   ├── random.py
│   │   └── registry.py
│   ├── data_process/                  # 数据处理工具
│   │   ├── data_source.py
│   │   ├── assist09.py
│   │   ├── assist12.py
│   │   ├── assist17.py
│   │   └── ednet_kt1.py
│   ├── optuna_utils/                  # Optuna 工具
│   ├── training/                      # 训练核心逻辑
│   │   ├── base_trainer.py            # 基础训练器
│   │   ├── callbacks.py
│   │   ├── checkpoint.py
│   │   ├── metrics.py
│   │   └── multi_trainer.py
│   ├── experiment_manager.py
│   ├── hyperparam_manager.py
│   └── net_data.py
├── scripts/
│   ├── run_kfold.sh                   # K 折训练脚本
│   └── setup_env.sh                   # 环境配置脚本
├── ablation_study.py                  # 消融实验脚本
├── data_process.py                    # 数据预处理脚本（下载/清洗/划分）
├── optuna_search.py                   # 超参数搜索脚本
└── train.py                           # 模型训练脚本
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

使用 `train.py` 脚本进行模型训练：

**训练示例：**

```bash
# 训练 GIKT 模型
python train.py -m GIKT -d assistments09

# 训练 HGIKT 模型
python train.py -m HGIKT -d assistments09

# 训练 SQGKT 模型
python train.py -m SQGKT -d assistments09

# 训练 SGKT 模型
python train.py -m SGKT -d assistments09

# 训练 ABKT 模型
python train.py -m ABKT -d assistments09

# K折交叉验证训练
for fold in {0..4}; do
  python train.py -m GIKT -d assistments09 --fold $fold
done

# 使用早停机制训练
python train.py -m GIKT -d assistments09 --es_patience 10 --es_monitor auc --es_mode max
```

各模型的详细训练说明在 `docs/` 目录下。

### 3. 消融实验研究

消融实验通过配置文件定义不同的策略（如置零特征、禁用模块等）：

```bash
# 运行 GIKT 的消融实验
python ablation_study.py --model GIKT --dataset assistments09 --config configs/ablation/gikt_ablation.json
```

生成的实验结果会记录在 SwanLab 中，方便对比各消融版本的性能。

### 4. 超参数自动化搜索 (Optuna)

利用 Optuna 自动寻找最优超参数：

```bash
# 启动 50 轮超参搜索
python optuna_search.py -m GIKT -d assistments09 --n_trials 50
```

搜索的空间由 `configs/optuna/param_space_<model>.json` 定义。

### 5. 查看训练结果

训练过程中的指标会自动上传到 SwanLab，可以在 SwanLab 官网查看训练曲线、损失、准确率、AUC等指标。

如果是首次使用，需要登录 SwanLab：

```bash
swanlab login
```

### 6. 案例分析工具

案例分析工具用于对训练好的模型进行推理分析，支持用户筛选和可视化。

#### 工作流程

案例分析分为三个步骤：

1. **推理**：加载训练好的模型，对测试数据进行推理并保存预测结果
2. **筛选**：根据用户行为特征筛选感兴趣的用户
3. **可视化**：为选定用户生成热力图可视化

所有结果自动保存在 `<run_dir>/case_analysis/` 目录下。

#### Step 1: 运行推理

```bash
# 基本用法（自动从 run_dir 加载模型和超参数）
python case_analysis.py inference \
    --run_dir runs/normal/GIKT_assistments09_20260217-144913_fold0_bs128
```

**推理参数说明：**

| 参数 | 说明 |
|------|------|
| `--run_dir` | 训练运行目录（必需），自动查找 `best_model.pth` 和 `hyperparameters.json` |
| `--hyperparams` | 超参数 JSON 文件路径（可选，默认从 run_dir 自动检测） |
| `--data_base_path` | 数据基础路径（默认：`./data`） |

**输出文件（保存在 `<run_dir>/case_analysis/`）：**
- `predictions.parquet`：所有预测结果
- `user_summaries.parquet`：用户级别的统计指标

#### Step 2: 筛选用户

```bash
# 筛选 10 个多样化的用户
python case_analysis.py select \
    --run_dir runs/normal/GIKT_assistments09_20260217-144913_fold0_bs128 \
    --strategy diverse \
    --num_users 10

# 筛选错误率最高的用户
python case_analysis.py select \
    --run_dir runs/normal/GIKT_assistments09_20260217-144913_fold0_bs128 \
    --strategy extreme \
    --num_users 5
```

**筛选参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--run_dir` | 训练运行目录（必需） | - |
| `--strategy` | 筛选策略：`diverse`（多样化）、`extreme`（极端错误）、`random`（随机） | `diverse` |
| `--num_users` | 最大用户数 | 10 |
| `--min_seq_len` | 最小序列长度 | 20 |
| `--max_seq_len` | 最大序列长度 | 200 |
| `--min_error` | 最小错误率 | 0.1 |
| `--max_error` | 最大错误率 | 0.9 |

**输出文件（保存在 `<run_dir>/case_analysis/<strategy>/`）：**
- `selected_users.json`：选中用户的详细信息

#### Step 3: 生成可视化

```bash
# 使用策略名称
python case_analysis.py plot \
    --run_dir runs/normal/GIKT_assistments09_20260217-144913_fold0_bs128 \
    --selected_users diverse
```

**可视化参数说明：**

| 参数 | 说明 |
|------|------|
| `--run_dir` | 训练运行目录（必需） |
| `--selected_users` | 策略名称（diverse/extreme/random）或 selected_users.json 路径 |

**输出文件（保存在 `<run_dir>/case_analysis/<strategy>/figures/`）：**
- `user_{id}_heatmap.png`：每个用户的热力图

## 输出说明

### SwanLab 日志

训练过程中的指标、超参数和系统信息会自动记录到 SwanLab 实验中。
本地日志文件保存在 `swanlog/` 目录下（默认）。

### 超参数配置

每次训练的超参数会自动保存到 `runs/<timestamp>/hyperparameters.json`，方便实验管理和结果复现。
