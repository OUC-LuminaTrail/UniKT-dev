# 训练与评估

使用训练器 API 训练知识追踪模型。

## 前置条件

训练前，请确保环境已正确配置。安装选项请参阅[快速上手](../getting-started/quick-start.md)：

- **Pixi**（推荐）：GPU 使用 ``pixi shell``，CPU 使用 ``pixi shell -e cpu``，Mamba 模型使用 ``pixi shell -e mamba``
- **自动 Conda**：使用 ``./scripts/setup_env.sh`` 自动配置
- **手动 Conda**：按照[环境配置](../getting-started/setup.md)中的指南操作

## 训练流程

```{mermaid}
flowchart LR
 A[初始化模型] --> B[构建训练器]
 B --> C[训练循环]
 C --> D[保存检查点]
```

## 快速上手

### 基本训练

```bash
# 在 ASSISTments 2009 上训练 GIKT
python train.py -m GIKT -d assistments09
```

### K 折交叉验证

```bash
# 单个折
python train.py -m GIKT -d assistments09 --fold 0

# 使用内置脚本运行训练
bash scripts/run_kfold.sh GIKT "0 1 2 3 4" -d assistments09

# 所有折
for i in {0..4}; do
 python train.py -m GIKT -d assistments09 --fold $i
done
```

### 使用早停

```bash
python train.py -m GIKT -d assistments09 \
 --es_patience 10 \
 --es_monitor auc \
 --es_mode max
```

## 参数说明

### 通用选项

| 参数 | 默认值 | 描述 |
| --- | --- | --- |
| ``-m, --model`` | 必填 | 模型名称 |
| ``-d, --dataset`` | 必填 | 数据集名称 |
| ``--fold`` | 0 | K 折索引 |
| ``--seed`` | 42 | 随机种子 |
| ``--device`` | 自动检测 | 设备（cuda/cpu，自动检测） |


### 早停

| 参数 | 默认值 | 描述 |
| --- | --- | --- |
| ``--es_patience`` | 10 | 耐心轮数（设为 0 禁用） |
| ``--es_monitor`` | auc | 监控指标 |
| ``--es_mode`` | max | ``max`` 或 ``min`` |
| ``--es_min_delta`` | 0.0 | 最小改进量 |
| ``--es_restore_best`` | False | 停止时恢复最佳权重 |


### 高级选项

| 参数 | 默认值 | 描述 |
| --- | --- | --- |
| ``--skip_test`` | False | 训练后跳过测试评估 |
| ``--no_deterministic`` | False | 关闭确定性算法（默认开启） |


:::{warning}
``--max_grad_norm`` 不是通用参数，仅部分模型（KQN、DeepIRT、MCKT）在自己的参数配置中注册。查看模型特定参数使用 ``python train.py -m <model> -h``。
:::

## 输出

训练结果保存在 ``runs/<type>/<run_id>/``：

```
runs/normal/GIKT_assistments09_20240403-120000_fold0_bs128/
├── best_model.pth # 最佳模型检查点
├── last_checkpoint.pth # 最后检查点
├── hyperparameters.json # 超参数配置
├── metrics_train.csv # 训练指标
├── metrics_val.csv # 验证指标
└── metrics_test.csv # 测试指标
```

### SwanLab 集成

指标自动记录到 SwanLab：

```bash
# 首次登录
swanlab login
```

跟踪的指标：

- 损失（训练/验证）
- 准确率（ACC）
- AUC 分数
- 学习率
- GPU 利用率

## 模型特定参数

不同模型有额外参数。查看所有选项：

```bash
python train.py -m GIKT -h
```

示例：

```bash
# GIKT 特定参数
python train.py -m GIKT -d assistments09 \
 --hidden_dim 256 \
 --n_layers 2 \
 --heads 4 \
 --dropout 0.1

# HDHKT 特定参数
python train.py -m HDHKT -d assistments09 \
 --hidden_dim 128 \
 --n_hop 4
```

## 高级用法

### 自定义训练器

```python
from utils.training import BaseTrainer


class MyTrainer(BaseTrainer):
 def __init__(self, config, **kwargs):
 super.__init__(**kwargs)
 self.model = MyModel(config)

 def forward_pass(self, batch) -> dict:
 logits = self.model(batch)
 y_hat = logits.squeeze(-1)
 y_label = batch["response"].float
 return {
 "y_hat": y_hat,
 "y_label": y_label,
 "y_predict": (y_hat >= 0.0).float,
 "y_score": y_hat,
 "y_prob": torch.sigmoid(y_hat),
 }
```

### 梯度裁剪

```bash
python train.py -m GIKT -d assistments09 --max_grad_norm 1.0
```

### 跳过测试评估

```bash
python train.py -m GIKT -d assistments09 --skip_test
```
