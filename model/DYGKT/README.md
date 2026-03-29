# DYGKT 模型

## 简介

DYGKT (Dynamic Graph-based Knowledge Tracing) 是一个基于动态图的知识追踪模型，通过建模用户和问题的历史交互序列，捕捉用户知识状态和问题难度的动态变化。

## 核心特点

- **双时间衰减机制**：区分短期（24小时内）和长期（超过24小时）记忆
- **用户-问题双向建模**：同时追踪用户状态和问题特征的演化
- **GRU 序列更新**：动态更新节点表示
- **时间编码**：显式建模时间间隔对学习效果的影响

## 模型架构

```
输入数据
  ├─ 用户序列 [B, S]
  ├─ 问题序列 [B, S]
  ├─ 回答序列 [B, S]
  ├─ 时间序列 [B, S]
  └─ 掩码 [B, S]
         ↓
    Embedding层
         ↓
  ┌──────┴──────┐
  │             │
用户GRU      问题GRU + 时间双衰减编码
  │             │
  └──────┬──────┘
         ↓
    特征融合
         ↓
    全连接层
         ↓
  预测 logits [B, S]
```

## 使用方法

### 1. 基本训练

```bash
# 使用默认参数训练
python train.py -m DYGKT --dataset ASSISTments12 --fold 0

# 指定超参数
python train.py -m DYGKT \
  --dataset ASSISTments12 \
  --fold 0 \
  --epochs 100 \
  --batch_size 2000 \
  --learning_rate 0.0005 \
  --embedding_dim 128 \
  --hidden_dim 128 \
  --dim_time 16 \
  --dropout 0.1
```

### 2. 使用专用脚本

```bash
# 运行 DYGKT 专用训练脚本
python scripts/train_dygkt.py --dataset ASSISTments12 --fold 0
```

### 3. GPU 加速

```bash
# 使用 GPU
python train.py -m DYGKT --dataset ASSISTments12 --fold 0 --device cuda

# 指定 GPU 设备
CUDA_VISIBLE_DEVICES=0 python train.py -m DYGKT --dataset ASSISTments12 --fold 0
```

### 4. 性能分析与缓存

```bash
# 训练时对前 50 个 batch 做瓶颈分析（日志会输出 wait/compute 比例）
python train.py -m DYGKT --dataset ASSISTments12 --fold 0 --profile_batches 50

# 禁用数据缓存（强制重建预处理）
python train.py -m DYGKT --dataset ASSISTments12 --fold 0 --no_cache

# 指定缓存目录
python train.py -m DYGKT --dataset ASSISTments12 --fold 0 --cache_dir ./cache/dygkt_assist12

# 若历史缓存来自旧版本实现，建议首次运行强制失效缓存
python train.py -m DYGKT --dataset ASSISTments12 --fold 0 --cache_version 2

# 调整 DataLoader 并行度（仅影响吞吐，不改变预测逻辑）
python train.py -m DYGKT --dataset ASSISTments12 --fold 0 --loader_num_workers 8 --loader_prefetch_factor 4

# 单独加速 VAL/TEST（默认会自动使用 2x train batch 做评估）
python train.py -m DYGKT --dataset ASSISTments12 --fold 0 --eval_batch_size 4000 --eval_loader_num_workers 8

# 使用专用 profiling 脚本
python scripts/profile_dygkt.py --dataset ASSISTments12 --fold 0 --profile_batches 50
```

## 超参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embedding_dim` | 128 | 嵌入维度 |
| `hidden_dim` | 128 | 隐藏层维度 |
| `dim_time` | 16 | 时间编码维度 |
| `dropout` | 0.1 | Dropout 概率 |
| `epochs` | 100 | 训练轮数 |
| `batch_size` | 2000 | 批次大小 |
| `learning_rate` | 0.0005 | 学习率 |
| `weight_decay` | 1e-4 | 权重衰减（L2正则化） |
| `lr_decay` | None | 学习率衰减因子 |
| `no_cache` | False | 禁用 DYGKT 数据缓存 |
| `cache_dir` | ./cache/dygkt | DYGKT 缓存目录（默认） |
| `cache_version` | 2 | 缓存版本号，用于手动失效旧缓存 |
| `profile_batches` | 0 | 对前 N 个训练 batch 做计时分析 |
| `loader_num_workers` | -1 | DataLoader 进程数（-1 自动） |
| `loader_prefetch_factor` | 2 | 每个 worker 预取批次数（仅 num_workers>0） |
| `loader_persistent_workers` | True | 是否复用 DataLoader workers |
| `eval_batch_size` | 0 | 验证/测试批次大小（0 表示自动=2x训练批次） |
| `eval_loader_num_workers` | -1 | 验证/测试 DataLoader 进程数（-1 跟随训练） |

## 数据格式

DYGKT 需要以下数据：

- **用户序列**：每个样本对应一个用户的答题序列
- **问题序列**：用户回答的问题 ID 序列
- **回答序列**：0/1 表示错误/正确
- **时间序列**：每次交互的时间戳（秒）
- **掩码**：标识有效位置

## 模型文件

- `DYGKT_model.py`: 模型定义
  - `TimeDualDecayEncoder`: 时间双衰减编码器
  - `DyKT_Seq`: 动态序列更新模块
  - `DYGKT`: 主模型类
  
- `DYGKT_trainer.py`: 训练器
  - `DYGKTTrainer`: 训练逻辑
  - `DYGKTModelParams`: 参数配置
  
- `DYGKT_data.py`: 数据处理
  - `DYGKTDataset`: 数据集类
  - `DYGKTModelData`: 数据准备类

## 实现细节

### TimeDualDecayEncoder

时间编码器使用两个独立的线性层处理短期和长期时间间隔：

```python
# 24小时内的时间差使用短期衰减
short_term = w_short(time_diff * (time_diff <= 86400))

# 超过24小时的时间差使用长期衰减
long_term = w_long(time_diff * (time_diff > 86400))

# 融合输出
output = w_o(short_term + long_term)
```

### 动态图更新

用户和问题的状态通过 GRU 动态更新：

```python
# 用户状态更新
user_emb, _ = gru4user(exercise_embedding)

# 问题状态更新
que_emb, _ = gru4que(exercise_embedding)
```

## 与原始实现的差异

本实现基于 pyedmine 框架中的 DYGKT，并进行了以下适配：

1. **接口适配**：继承 `nn.Module`，实现 `forward` 方法
2. **训练器适配**：使用 `BaseTrainer` 基类
3. **数据处理**：适配 kt-exp-graph 的数据格式
4. **时间戳生成**：支持无时间戳数据集的模拟时间生成

## 参考文献

```
原始论文：待补充
原始代码：/home/lian/pyedmine/edmine/model/non_sequential_kt_model/DyGKT.py
```

## 维护者

- 迁移自 pyedmine 框架
- 适配到 kt-exp-graph 框架
- 日期：2026-03-28
