# 自定义模型

在 UniKT 中添加自定义知识追踪模型，遵循"定义 → 注册 → 配置"三步法。本文以真实模型 GIKT（Yang et al., ECML-PKDD 2020）的实现为例展开。

## 概述

```{mermaid}
flowchart LR
 A[定义模型] --> B[注册模型]
 B --> C[编写训练器]
 C --> D[CLI 训练]
```

UniKT 的模型系统由三个核心文件组成，以 GIKT 为例：

```
model/GIKT/
├── GIKT_model.py # 模型定义（nn.Module）
├── GIKT_data.py # 数据准备（继承 QuestionModelData）
└── GIKT_trainer.py # 训练器 + 参数注册（继承 BaseTrainer）
```

## 第一步：定义模型

模型是一个标准的 ``torch.nn.Module`` 子类。需要实现 ``forward`` 方法，接收训练器传入的 batch 数据，返回预测 logits。

```python
# model/MyModel/MyModel_model.py
import torch
import torch.nn as nn


class MyModel(nn.Module):
 """自定义知识追踪模型。"""

 def __init__(self, args, data_metadata):
 super.__init__
 self.num_skills = data_metadata["num_skills"]
 self.num_questions = data_metadata["num_questions"]
 self.embedding_dim = args.embedding_dim

 # 共享嵌入表：[0, num_skills) 技能，剩余为题目标识
 self.feature_embedding = nn.Embedding(
 self.num_skills + self.num_questions + 2,
 self.embedding_dim,
 )

 # 模型特有层
 self.lstm = nn.LSTM(self.embedding_dim, self.embedding_dim, batch_first=True)
 self.output = nn.Linear(self.embedding_dim, 1)

 def forward(self, user_sequence, user_response, user_mask, **kwargs):
 """前向传播，返回 [B, S-1] 形状的 logits。"""
 # 构建输入：题目嵌入 + 作答标记嵌入
 question_emb = self.feature_embedding(user_sequence[:, :-1])
 answer_emb = self.feature_embedding(
 user_response[:, :-1] + self.num_skills + self.num_questions
 )
 inputs = question_emb + answer_emb

 lstm_out, _ = self.lstm(inputs)
 logits = self.output(lstm_out).squeeze(-1)
 return logits
```

**关键约定**：

- 构造函数接收 ``args``（命令行参数 Namespace）和 ``data_metadata``（数据集元信息 dict，包含 ``num_skills``、``num_questions`` 等）。
- ``forward`` 输出形状为 ``[B, S-1]``（next-item 预测：t 时刻预测 t+1 时刻的正确性）。
- 可使用 ``kwargs`` 接收训练器额外传入的数据（如图数据、邻居索引等）。

## 第二步：注册模型参数与训练器

UniKT 使用装饰器实现按需加载。需要在 ``GIKT_trainer.py`` 中同时注册两样东西：

(21-注册模型参数)=

### 2.1 注册模型参数

通过 ``@register_model_params("NAME")`` 装饰器，定义模型的命令行参数：

```python
# model/MyModel/MyModel_trainer.py
from utils.config import BaseParamConfig, register_model_params


@register_model_params("MyModel")
class MyModelParams(BaseParamConfig):
 def define_params(self) -> tuple[str, dict]:
 return "MyModel Parameters", {
 "embedding_dim": {
 "type": int,
 "default": 100,
 "short": "ed",
 "help": "嵌入维度 (default: 100)",
 },
 "hidden_dim": {
 "type": int,
 "default": 200,
 "help": "隐藏层维度 (default: 200)",
 },
 "learning_rate": {
 "type": float,
 "default": 0.001,
 "short": "lr",
 "help": "学习率 (default: 0.001)",
 },
 "epochs": {
 "type": int,
 "default": 100,
 "short": "ep",
 "help": "训练轮数 (default: 100)",
 },
 "batch_size": {
 "type": int,
 "default": 32,
 "short": "bs",
 "help": "批次大小 (default: 32)",
 },
 }
```

``define_params`` 返回 ``(分组名, 参数字典)``。参数字典的每个键对应一个参数，值包含：

| 字段 | 说明 |
| --- | --- |
| ``type`` | 参数类型（``int``、``float``、``str``、``list``） |
| ``default`` | 默认值 |
| ``short`` | 短参数名（可选） |
| ``help`` | 帮助文本 |
| ``nargs`` | 参数个数（``list`` 类型使用，如 ``"?"``） |


``bool`` 类型参数会自动转换为 ``store_true``/``store_false``。

(22-注册训练器)=

### 2.2 注册训练器

通过 ``@register_trainer("NAME")`` 装饰器：

```python
# model/MyModel/MyModel_trainer.py (续)
from utils.core import register_trainer
from utils.training import BaseTrainer
from utils.config import EarlyStoppingConfig
import torch


@register_trainer("MyModel")
class MyModelTrainer(BaseTrainer):
 """自定义模型训练器。"""

 def __init__(self, args=None, data_src=None, exp_manager=None):
 # 1. 准备数据
 from model.MyModel.MyModel_data import MyModelData
 from model.MyModel.MyModel_model import MyModel

 model_data = MyModelData(data_src)
 train_data, val_data, test_data = model_data.prepare_data(args)

 # 2. 初始化模型
 model = MyModel(args=args, data_metadata=data_src.get_metadata)
 super.__init__(model) # 必须传入模型

 # 3. 配置优化器
 optimizer = torch.optim.Adam(
 model.parameters,
 lr=args.learning_rate,
 weight_decay=getattr(args, "weight_decay", 0.0),
 )

 # 4. 组装链式构建器
 self.with_training(
 epochs=args.epochs,
 seed=args.seed,
 device=args.device,
 ).with_data(
 train_data=train_data,
 val_data=val_data,
 test_data=test_data,
 batch_size=args.batch_size,
 ).with_optimization(
 optimizer=optimizer,
 loss_fn=torch.nn.BCEWithLogitsLoss,
 ).with_experiment(
 exp_manager=exp_manager,
 hyperparams=args,
 model_name="MyModel",
 dataset_name=args.dataset,
 ).build

 def forward_pass(self, batch_data):
 """模型前向传播，返回标准化的预测输出。"""
 sequence = batch_data["sequence"]
 response = batch_data["response"]
 mask = batch_data["mask"]

 y_hat_full = self._pad_to_full_sequence(
 self.model(
 user_sequence=sequence,
 user_response=response,
 user_mask=mask,
 )
 )
 y_hat, y_label, _ = self._extract_valid_predictions(y_hat_full, response, mask)

 return {
 "y_hat": y_hat,
 "y_label": y_label,
 "y_predict": self._generate_binary_predictions(y_hat, threshold=0.0),
 "y_score": y_hat,
 "y_prob": torch.sigmoid(y_hat),
 }
```

**``__init__`` 职责**：

1. 准备数据（调用对应的 ``ModelData.prepare_data(args)``）
2. 初始化 ``nn.Module`` 并传给 ``super.__init__(model)``
3. 配置优化器（Adam / SGD / AdamW）
4. 通过链式构建器 ``.with_*.build`` 完成装配

**``forward_pass`` 必须返回的键**：

| 键 | 类型 | 描述 |
| --- | --- | --- |
| ``y_hat`` | ``Tensor[B, S-1]`` | 原始 logits |
| ``y_label`` | ``Tensor[B, S-1]`` | 真实标签（0/1） |
| ``y_predict`` | ``Tensor[B, S-1]`` | 二值预测 |
| ``y_score`` | ``Tensor[B, S-1]`` | 用于 AUC 的分数 |
| ``y_prob`` | ``Tensor[B, S-1]`` | 用于校准的概率 |


基类提供了辅助方法：

| 方法 | 用途 |
| --- | --- |
| ``_pad_to_full_sequence(y_hat)`` | 将 ``[B, S-1]`` 补零为 ``[B, S]`` |
| ``_extract_valid_predictions(...)`` | next-item 对齐：``y_hat[t]`` 预测 ``response[t+1]`` |
| ``_generate_binary_predictions(y_hat)`` | logits → 0/1 |
| ``_move_tensor_to_device(tensor)`` | 自动移动到正确设备 |


## 第三步：自定义损失函数

自定义损失函数在训练器链式构建时通过 ``with_optimization(loss_fn=...)`` 传入，可以是任何 PyTorch ``nn.Module`` 或可调用对象。训练循环调用时传入 ``(y_hat, y_label)`` 两个张量。

### 方式一：标准 PyTorch 损失

UniKT 训练器直接兼容所有 PyTorch 损失函数：

```python
import torch.nn as nn

loss_fn = nn.BCEWithLogitsLoss
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
loss_fn = nn.MSELoss
```

### 方式二：自定义损失模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
 """Focal Loss for binary classification.

 FL = -α_t * (1-p_t)^γ * log(p_t)
 减少易分类样本的损失权重，聚焦于难分类样本。
 """

 def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
 super.__init__
 self.alpha = alpha
 self.gamma = gamma

 def forward(self, y_hat, y_label):
 bce_loss = F.binary_cross_entropy_with_logits(y_hat, y_label, reduction="none")
 p_t = torch.sigmoid(y_hat)
 p_t = y_label * p_t + (1.0 - y_label) * (1.0 - p_t)
 focal_weight = (1.0 - p_t) ** self.gamma
 alpha_weight = y_label * self.alpha + (1.0 - y_label) * (1.0 - self.alpha)
 return (alpha_weight * focal_weight * bce_loss).mean
```

### 方式三：带权重的组合损失

```python
class CombinedLoss(nn.Module):
 """BCE + 正则化的组合损失。"""

 def __init__(self, lambda_reg: float = 0.01):
 super.__init__
 self.lambda_reg = lambda_reg
 self.bce = nn.BCEWithLogitsLoss

 def forward(self, y_hat, y_label, model=None):
 loss = self.bce(y_hat, y_label)
 if model is not None:
 l2_reg = 0.0
 for param in model.parameters:
 l2_reg += torch.norm(param, p=2)
 loss += self.lambda_reg * l2_reg
 return loss
```

**损失函数签名约定**：

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| ``y_hat`` | ``Tensor[N]`` | 模型输出的 logits（已展平为一维） |
| ``y_label`` | ``Tensor[N]`` | 真实标签（0/1，已展平为一维） |


训练器内部会自动完成 next-item 对齐和展平，传入损失函数的 ``y_hat`` 和 ``y_label`` 已经是一维张量。

### 命令行控制损失函数

如果想从命令行切换损失函数，可在参数配置中添加：

```python
@register_model_params("MyModel")
class MyModelParams(BaseParamConfig):
 def define_params(self) -> tuple[str, dict]:
 return "MyModel Parameters", {
 "loss_type": {
 "type": str,
 "default": "bce",
 "choices": ["bce", "focal", "combined"],
 "help": "损失函数类型",
 },
 "focal_alpha": {
 "type": float,
 "default": 0.25,
 },
 "focal_gamma": {
 "type": float,
 "default": 2.0,
 },
 }
```

```python
def _build_loss(self, args):
 if args.loss_type == "focal":
 return FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
 elif args.loss_type == "combined":
 return CombinedLoss(lambda_reg=0.01)
 else:
 return nn.BCEWithLogitsLoss


loss_fn = self._build_loss(args)
```

```bash
# 使用 Focal Loss
python train.py -m MyModel -d assistments09 --loss_type focal --focal_gamma 2.0
```

### 常用损失函数速查

| 场景 | 推荐损失 | 说明 |
| --- | --- | --- |
| 二分类正确性预测 | ``BCEWithLogitsLoss`` | 默认，最常用 |
| 类别不平衡 | ``BCEWithLogitsLoss(pos_weight=...)`` | 对少类样本加权 |
| 难样本挖掘 | ``FocalLoss`` | 聚焦难分类样本 |
| 排名学习 | ``MarginRankingLoss`` | 正确作答预测应高于错误作答 |
| 知识状态回归 | ``MSELoss`` 或 ``SmoothL1Loss`` | 预测连续掌握度 |


## 第四步：自定义训练流程

通过继承 ``BaseTrainer`` 并重写特定方法，可以自定义训练行为。

### 可重写的方法

| 方法 | 作用域 | 默认行为 | 典型重写场景 |
| --- | --- | --- | --- |
| ``forward_pass(batch)`` | 每个 batch | 抽象方法 | 模型特有的前向传播 |
| ``test_forward_pass(batch)`` | 测试 batch | 调用 ``forward_pass`` | 测试时提取额外信息 |
| ``_compute_loss(y_hat, y_label)`` | 每个 batch | ``loss_fn(y_hat, y_label)`` | 自定义损失计算 |
| ``_run_train_batch(batch, idx)`` | 训练 batch | 标准训练 step | 梯度累积、AMP |
| ``_run_eval_batch(batch, idx)`` | 验证 batch | 标准评估 step | 自定义评估逻辑 |
| ``_extract_valid_predictions(...)`` | 每个 batch | next-item 对齐 | 自定义对齐方式 |
| ``_process_epoch(epoch, is_train)`` | 每个 epoch | 遍历 DataLoader | 完全自定义 epoch 逻辑 |


### 示例：重写 \_compute_loss

```python
class MyTrainer(BaseTrainer):
 def _compute_loss(self, y_hat, y_label):
 base_loss = self._optimization_config.loss_fn(y_hat, y_label)
 if hasattr(self.model, "knowledge_state"):
 ks_smooth = torch.mean(torch.diff(self.model.knowledge_state, dim=0) ** 2)
 return base_loss + 0.01 * ks_smooth
 return base_loss
```

### 使用 MultiTrainer 多阶段训练

对于多阶段训练场景（如预训练 + 微调），继承 ``MultiTrainer``：

```python
from utils.training import MultiTrainer


@register_trainer("MyMultiStage")
class MyMultiStageTrainer(MultiTrainer):
 def __init__(self, *, device="cuda", seed=42, deterministic=True, **kwargs):
 super.__init__(device=device, seed=seed, deterministic=deterministic)

 def build_stages(self):
 return [
 StageConfig("pretrain", self._build_pretrain_stage),
 StageConfig("finetune", self._build_finetune_stage),
 ]

 def _build_pretrain_stage(self):
 model = MyModel(self.args, self.data_metadata)
 for param in model.lstm.parameters:
 param.requires_grad = False

 optimizer = torch.optim.Adam(
 filter(lambda p: p.requires_grad, model.parameters), lr=0.001
 )
 return StageComponents(
 model=model,
 optimizer=optimizer,
 loss_fn=torch.nn.BCEWithLogitsLoss,
 train_data=train_data,
 val_data=val_data,
 epochs=50,
 )

 def _build_finetune_stage(self):
 for param in self.model.parameters:
 param.requires_grad = True

 optimizer = torch.optim.Adam(self.model.parameters, lr=0.0001)
 return StageComponents(
 model=self.model,
 optimizer=optimizer,
 loss_fn=torch.nn.BCEWithLogitsLoss,
 train_data=train_data,
 val_data=val_data,
 epochs=30,
 )

 def forward_pass(self, batch_data):
 if self._current_stage == "pretrain":
 ...
 else:
 ...
 return {
 "y_hat": y_hat,
 "y_label": y_label,
 "y_predict": (y_hat >= 0.0).float,
 "y_score": y_hat,
 "y_prob": torch.sigmoid(y_hat),
 }
```

### 使用回调系统

回调在训练循环的关键节点自动触发，让你无需修改训练逻辑即可扩展行为。

**预定义回调**：

| 回调 | 触发时机 | 功能 |
| --- | --- | --- |
| ``CheckpointCallback`` | ``on_phase_end`` | 保存最佳模型 |
| ``EarlyStoppingCallback`` | ``on_phase_end(phase=val)`` | 早停判断 |
| ``MemoryCleanupCallback`` | ``on_phase_end`` | 定期清理 CUDA 缓存 |
| ``TestEvaluationCallback`` | ``on_train_end`` | 训练结束后评估测试集 |
| ``FunctionCallback`` | 全部钩子 | 函数式回调包装器 |


**自定义回调**：

```python
from utils.training import Callback


class LearningRateMonitor(Callback):
 def on_epoch_begin(self, epoch: int, **kwargs):
 trainer = kwargs.get("trainer")
 if trainer is None or trainer.opt is None:
 return
 current_lr = trainer.opt.param_groups[0]["lr"]
 logger.info(f"Epoch {epoch}: lr = {current_lr:.6f}")
```

**Callback 钩子一览**：

| 钩子 | 调用时机 |
| --- | --- |
| ``on_train_begin(epochs)`` | 训练开始前 |
| ``on_train_end`` | 训练结束后 |
| ``on_epoch_begin(epoch)`` | 每个 epoch 开始 |
| ``on_epoch_end(epoch, train_loss, val_loss)`` | 每个 epoch 结束 |
| ``on_phase_begin(epoch, phase)`` | 每个阶段开始 |
| ``on_phase_end(epoch, phase, loss, metrics)`` | 每个阶段结束 |
| ``on_batch_begin(epoch, batch_idx, phase)`` | 每个 batch 开始 |
| ``on_batch_end(epoch, batch_idx, phase, loss)`` | 每个 batch 结束 |
| ``should_stop`` | 每次循环检查 |


**在训练器中注册回调**：

```python
trainer.with_callbacks(
    callbacks=[
        CheckpointCallback(checkpoint_manager=checkpoint_mgr),
        EarlyStoppingCallback(early_stopping=early_stopping_obj),
        LearningRateMonitor,
    ],
)
```

## 第五步：使用模型

完成定义和注册后，在命令行直接使用：

```bash
# 基本训练
python train.py -m MyModel -d assistments09

# 带自定义参数
python train.py -m MyModel -d assistments09 \
 --embedding_dim 256 \
 --hidden_dim 512 \
 --epochs 150 \
 --batch_size 64

# 查看所有可用参数
python train.py -m MyModel -h
```

查看已注册的模型列表：

```python
from utils.core import TRAINERS

print(TRAINERS.keys)
```

## 完整示例：GIKT 模型目录结构

```
model/MyModel/
├── MyModel_model.py # nn.Module 模型定义
├── MyModel_data.py # 数据准备，继承 QuestionModelData 或 SkillModelData
└── MyModel_trainer.py # 训练器 + 参数配置（包含 @register_trainer）
```

**数据准备层选择**：

| 基类 | 数据粒度 | 适用场景 |
| --- | --- | --- |
| ``QuestionModelData`` | 一次交互 = 一个时间步 | 题目级预测 |
| ``SkillModelData`` | 一个技能 = 一个时间步 | 技能/概念级预测 |


大多数知识追踪模型使用 ``QuestionModelData``。

## 注意事项

- ``forward`` 输出形状必须为 ``[B, S-1]``，即 next-item 预测格式（t 时刻预测 t+1 时刻）。
- 构造函数签名必须为 ``__init__(self, args, data_metadata, **kwargs)`` 或兼容形式。
- 训练器中 ``self.model`` 在 ``super.__init__(model)`` 后赋值，不要在此之前访问。
- ``with_*`` 链式调用顺序不重要，但所有四个方法（``with_training``、``with_data``、``with_optimization``、``with_experiment``）必须在 ``build`` 之前调用，否则会抛出 ``ValueError``。
- 损失函数的输入已由训练器完成 next-item 对齐和掩码过滤，``y_hat`` 和 ``y_label`` 是展平后的一维张量。
- ``forward_pass`` 必须返回包含 ``y_hat``、``y_label``、``y_predict`` 的 dict。
- 回调方法中通过 ``kwargs.get("trainer")`` 获取训练器实例。
