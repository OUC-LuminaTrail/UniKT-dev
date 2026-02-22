# 消融实验 (Ablation Study)

消融实验用于分析模型各组件的作用，通过系统性地移除或替换模型组件来评估其对性能的影响。

## 设计思路

本框架采用**模型子类化**的方式创建消融变体：

1. **模型子类化** - 继承基模型类，重写 `__init__` 和/或 `forward` 方法
2. **数据子类化**（可选） - 继承基模型的数据类，重写 `prepare_data` 方法修改图结构
3. **训练器子类化** - 每个变体有自己的训练器，使用变体模型和数据类
4. **配置驱动** - JSON 配置文件定义批量实验

## 架构概览

```
用户创建:
├── 变体模型类 (继承基模型)
│   └── 重写 __init__ 和/或 forward
├── 变体数据类 (可选，继承基数据)
│   └── 重写 prepare_data 修改图结构
└── 变体训练器类 (继承基训练器)
    └── 使用变体模型和数据类
```

## 创建消融变体的完整流程

### 示例：为 HGIKT 创建消融变体

假设我们要测试 HGIKT 中超图分支的作用，创建一个不含超图的变体。

**目录结构**: 变体文件放在 `model/<ModelName>/variants/` 目录下

```
model/
└── HGIKT/
    ├── HGIKT_data.py
    ├── HGIKT_model.py
    ├── HGIKT_trainer.py
    ├── __init__.py
    └── variants/
        ├── __init__.py
        ├── hgikt_no_hypergraph.py
        ├── hgikt_no_hypergraph_trainer.py
        ├── hgikt_no_template_edges.py
        ├── hgikt_no_template_edges_data.py
        └── hgikt_no_template_edges_trainer.py
```

#### 步骤 1: 创建变体模型类

**文件**: `model/HGIKT/variants/hgikt_no_hypergraph.py`

```python
"""HGIKT variant without hypergraph branch."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear

from model.layers import GeneralInteraction, HistoryRecap
from utils.core import register_model


@register_model("HGIKT_NoHypergraph")
class HGIKT_NoHypergraph(nn.Module):
    """HGIKT variant with hypergraph branch disabled."""

    def __init__(
        self,
        args: Any,
        data_metadata: dict[str, Any],
        hetero_metadata: tuple[list[str], list[tuple[str, str, str]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.args = args
        self.data_metadata = data_metadata

        # 模型参数
        self.hidden_dim = args.hidden_dim
        self.lstm_layers = args.lstm_layers
        self.dropout = args.dropout

        # Embedding 层（与原模型相同）
        self.question_embedding = nn.Embedding(
            num_embeddings=data_metadata["num_questions"],
            embedding_dim=self.hidden_dim,
        )
        # ... 其他 embedding 层

        # 异构图模块（保持不变）
        self.hetero_conv = HeteroGNN(...)

        # === 超图模块已禁用 ===
        # 不初始化 self.hgnn_conv

        # === 融合模块已禁用 ===
        # 不初始化 self.fuse

        # 其他模块保持不变
        self.fc_exercise = Linear(...)
        self.lstm = nn.LSTM(...)
        self.history_review = HistoryRecap(...)
        self.general_interaction = GeneralInteraction(...)

    def forward(
        self,
        user_sequence: torch.Tensor,
        user_response: torch.Tensor,
        user_mask: torch.Tensor,
        hetero_graph: Any,
        hypergraph: Any,  # 未使用但保留接口兼容性
        question_skill_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播，跳过超图计算。"""
        B, _ = user_sequence.size()

        # Answers embedding
        answers_embedding = self.answer_embedding(user_response)

        # === 修改: 跳过超图卷积 ===
        # 原代码: question_hyper_conv = self.hgnn_conv(...)
        # 新代码: 使用零张量（因为跳过了融合，实际不会用到）

        # 异构图卷积（不变）
        conv = self.hetero_conv(...)
        question_hetero_conv = conv["question"]
        skill_hetero_conv = conv["skill"]

        # === 修改: 跳过融合，仅使用异构图 ===
        # 原代码: question_conv_fused = self.fuse(question_hetero_conv, question_hyper_conv)
        # 新代码: question_conv_fused = question_hetero_conv
        question_conv_fused = question_hetero_conv

        # 其余前向传播代码与原模型相同
        question_embedding_sequence = question_conv_fused[user_sequence]
        # ... 继续原模型的逻辑

        return logits
```

**关键点:**
- 使用 `@register_model` 装饰器注册模型
- 继承 `nn.Module`（或继承原模型类）
- 重写 `forward` 方法显式控制执行流程
- 保留接口兼容性（参数签名相同）

#### 步骤 2: 创建变体训练器类

**文件**: `model/HGIKT/variants/hgikt_no_hypergraph_trainer.py`

```python
"""Trainer for HGIKT_NoHypergraph variant."""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


# 定义模型参数（与原模型相同）
@register_model_params("HGIKT_NoHypergraph")
class HGIKTNoHypergraphModelParams(BaseParamConfig):
    """HGIKT_NoHypergraph model parameters."""

    def define_params(self) -> tuple[str, dict]:
        group_name = "HGIKT_NoHypergraph Parameters"
        params = {
            "hidden_dim": {"type": int, "default": 250, ...},
            "n_hop": {"type": int, "default": 4, ...},
            # ... 其他参数
        }
        return group_name, params


# 注册训练器
@TRAINERS.register("HGIKT_NoHypergraph")
class HGIKTNoHypergraphTrainer(BaseTrainer):
    """Trainer for HGIKT without hypergraph."""

    def __init__(
        self,
        args: Any = None,
        data_src: Any = None,
        exp_manager: Any = None,
    ) -> None:
        # 1. 准备数据（与原模型相同）
        from model.HGIKT import HGIKTModelData

        model_data = HGIKTModelData(data_src)
        data_dict = model_data.prepare_data(args)

        train_dataset = data_dict["train_dataset"]
        val_dataset = data_dict["val_dataset"]
        self.hypergraph = data_dict["skill_hypergraph"]
        self.hetero_graph = data_dict["hetero_graph"]
        self.question_skill_matrix = data_dict["question_skill_matrix"]

        # 2. 初始化变体模型
        from variants.hgikt_no_hypergraph import HGIKT_NoHypergraph

        logger.info("Initializing HGIKT_NoHypergraph model...")
        model = HGIKT_NoHypergraph(
            args, data_src.get_metadata(), self.hetero_graph.metadata()
        )

        # 3. 调用父类构造函数
        super().__init__(model)

        # 4-7. 配置优化器、损失函数、学习率调度器、早停等
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        # ... 其他配置

        self.with_training(...).with_data(...).with_optimization(...).with_experiment(...).build()

        # 8. 将静态数据移动到设备
        self.hetero_graph = self.hetero_graph.to(self.device_)
        self.hypergraph = self.hypergraph.to(self.device_)
        self.question_skill_matrix = self.question_skill_matrix.to(self.device_)

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """前向传播，与原训练器相同。"""
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        y_hat_full = self.model(
            sequence, response, mask,
            self.hetero_graph, self.hypergraph, self.question_skill_matrix,
        )

        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, skip_first=True
        )

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.0)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": torch.sigmoid(y_hat),
        }
```

**关键点:**
- 使用 `@TRAINERS.register` 装饰器注册训练器
- 使用 `@register_model_params` 定义模型参数
- 导入并使用变体模型类
- 其他逻辑与原训练器相同

#### 步骤 3: 注册变体

**文件**: `model/HGIKT/variants/__init__.py`

```python
"""HGIKT ablation variants package."""

# 导入所有变体模块以触发注册
import model.HGIKT.variants.hgikt_no_hypergraph  # noqa: F401
import model.HGIKT.variants.hgikt_no_hypergraph_trainer  # noqa: F401

# 添加更多变体...
# import model.HGIKT.variants.my_variant  # noqa: F401
# import model.HGIKT.variants.my_variant_trainer  # noqa: F401

__all__ = []
```

**文件**: `model/HGIKT/__init__.py`

```python
from .HGIKT_data import HGIKTModelData
from .HGIKT_model import HGIKT
from .HGIKT_trainer import HGIKTModelParams, HGIKTTrainer

# 导入变体以触发注册
import model.HGIKT.variants  # noqa: F401

__all__ = ["HGIKTModelData", "HGIKT", "HGIKTTrainer", "HGIKTModelParams"]
```

**重要**: 必须导入变体模块，否则装饰器不会执行，变体不会被注册。

### 修改数据准备（可选）

如果消融需要修改图结构（例如移除特定边），创建数据子类：

**文件**: `model/HGIKT/variants/hgikt_no_template_edges_data.py`

```python
"""Data preparation for HGIKT without template edges."""

from typing import Any

from model.HGIKT.HGIKT_data import HGIKTDataset, HGIKTModelData
from typing_extensions import override

from utils.core import get_logger

logger = get_logger(__name__)


class HGIKTNoTemplateEdgesData(HGIKTModelData):
    """数据准备，不包含模板边。"""

    @override
    def prepare_data(self, args):
        """准备 HGIKT 数据，不包含模板边。"""
        # ... 前面的逻辑与原类相同

        # === 修改: 构建异构图时不包含模板边 ===
        hetero_graph = self.build_hetero_graph(
            [
                ("question", "has", "skill"),
                ("skill", "related_to", "assignment"),
                # 跳过: ("question", "belongs_to", "template")
            ]
        )

        # ... 其余逻辑不变

        return {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "skill_hypergraph": skill_hypergraph,
            "hetero_graph": hetero_graph,  # 修改后的图
            "question_skill_matrix": question_skill_matrix,
        }
```

然后在训练器中使用自定义数据类：

```python
# 在训练器 __init__ 中
from variants.hgikt_no_template_edges_data import HGIKTNoTemplateEdgesData

model_data = HGIKTNoTemplateEdgesData(data_src)  # 使用自定义数据类
data_dict = model_data.prepare_data(args)
```

### 步骤 4: 创建批量实验配置

**文件**: `configs/ablation/hgikt_study.json`

```json
{
  "study_name": "hgikt_ablation_study",
  "base_model": "HGIKT",
  "dataset": "assistments09",
  "shared_params": {
    "epochs": 120,
    "learning_rate": 0.0003,
    "batch_size": 64,
    "fold": 0,
    "seed": 42,
    "hidden_dim": 250,
    "n_hop": 4,
    "heads": 1,
    "lstm_layers": 1,
    "history_neighbour": 5,
    "att_bound": 0.1,
    "num_difficulty_clusters": 5,
    "dropout": 0.25,
    "weight_decay": 0.00001,
    "device": null,
    "checkpoint_path": null,
    "lr_decay": null,
    "es_patience": null,
    "base_dir": "runs",
    "data_base_path": "./data"
  },
  "ablations": [
    {
      "name": "baseline",
      "variant": "HGIKT",
      "description": "完整模型（无消融）"
    },
    {
      "name": "no_hypergraph",
      "variant": "HGIKT_NoHypergraph",
      "description": "禁用超图分支 - 仅使用异构图"
    },
    {
      "name": "no_template_edges",
      "variant": "HGIKT_NoTemplateEdges",
      "description": "移除问题-模板边"
    }
  ]
}
```

**配置说明:**
- `study_name`: 消融研究名称
- `base_model`: 基模型名称
- `dataset`: 数据集名称
- `shared_params`: 所有消融实验共享的参数
- `ablations`: 消融实验列表
  - `name`: 实验名称
  - `variant`: 变体训练器名称（必须在 TRAINERS 中注册）
  - `description`: 可选描述
  - `params`: 可选，覆盖共享参数的特定参数

## 运行消融实验

### 列出所有可用变体

```bash
pixi run python ablation_study.py --list-variants
```

输出:
```
Available model variants (registered trainers):
  - ABKT
  - GIKT
  - GIKTEdmine
  - HGIKT
  - HGIKT_NoHypergraph
  - HGIKT_NoTemplateEdges
  - SGKT
  - SQGKT
```

### 运行批量消融实验

```bash
pixi run python ablation_study.py --config configs/ablation/hgikt_study.json
```

这将依次运行配置文件中定义的所有消融实验，每个实验独立训练并记录结果。

### 训练单个变体

也可以直接使用 `train.py` 训练单个变体：

```bash
pixi run python train.py -m HGIKT_NoHypergraph -d assistments09
```

## 消融类型总结

| 消融目标          | 实现方式                                                   | 示例               |
| ----------------- | ---------------------------------------------------------- | ------------------ |
| 移除特定边类型    | 创建 `*Data` 类，重写 `prepare_data()`                     | 移除模板边         |
| 禁用超图分支      | 创建模型子类，重写 `forward()`                             | HGIKT_NoHypergraph |
| 禁用异构图分支    | 创建模型子类，重写 `forward()`                             | 使用零张量替代     |
| 替换 MoE 为拼接   | 创建模型子类，修改 `__init__()`                            | 替换融合模块       |
| 禁用 LSTM         | 创建模型子类，替换 `__init__()` 中的模块                   | 用占位符替代       |
| 禁用 HistoryRecap | 创建模型子类，替换 `__init__()` 中的模块                   | 用占位符替代       |
| 冻结参数          | 创建模型子类，在 `__init__()` 中设置 `requires_grad=False` | 冻结嵌入层         |

## 常见问题

### Q: 如何确保接口兼容？

**A**: 保持 `forward` 方法签名与原模型相同：

```python
def forward(
    self,
    user_sequence: torch.Tensor,
    user_response: torch.Tensor,
    user_mask: torch.Tensor,
    hetero_graph: Any,
    hypergraph: Any,  # 即使不使用也保留
    question_skill_matrix: torch.Tensor,
) -> torch.Tensor:
    ...
```

### Q: 变体可以继承原模型类吗？

**A**: 可以，但不是必需的。继承可以减少代码重复：

```python
from model.HGIKT.HGIKT_model import HGIKT

@register_model("HGIKT_NoHypergraph")
class HGIKT_NoHypergraph(HGIKT):
    def __init__(self, ...):
        super().__init__(...)  # 继承所有初始化

    def forward(self, ...):
        # 仅重写需要修改的部分
        ...
```

但需要注意：如果父类初始化了不需要的模块（如 `self.hgnn_conv`），它们仍会占用内存。在这种情况下，直接继承 `nn.Module` 并重写所有内容可能更合适。
