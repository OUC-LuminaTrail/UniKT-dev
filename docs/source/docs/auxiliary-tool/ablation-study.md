# 消融实验

通过系统消融分析模型各组件的贡献。

## 概述

```{mermaid}
flowchart TB
 A[基础模型] --> B[变体 A<br/>移除组件 X]
 A --> C[变体 B<br/>移除组件 Y]
 A --> D[变体 C<br/>移除组件 Z]
 B --> E[训练]
 C --> F[训练]
 D --> G[训练]
 E --> H[比较]
 F --> H
 G --> H
```

## 快速上手

```bash
# 运行消融实验
python ablation_study.py \
 --config configs/ablation/hgikt_study.json \
 -d assistments09 \
 --fold 0
```

## 参数说明

| 参数 | 默认值 | 描述 |
| --- | --- | --- |
| ``--config`` | 必填 | 消融实验配置 JSON 路径 |
| ``-d, --dataset`` | 必填 | 数据集名称 |
| ``-f, --fold`` | 0 | K 折索引 |


## 配置

在 JSON 中定义消融变体：

```json
{
 "study_name": "hgikt_ablation_study",
 "base_model": "HDHKT",
 "shared_params": {
 "epochs": 120,
 "learning_rate": 0.0003,
 "batch_size": 64,
 "hidden_dim": 250,
 "dropout": 0.25,
 "weight_decay": 0.00001,
 "es_patience": 10
 },
 "ablations": [
 {
 "name": "baseline",
 "variant": "HDHKT",
 "description": "完整模型（无消融）"
 },
 {
 "name": "hetero_only",
 "variant": "HDHKT_HeteroOnly",
 "description": "仅保留异构图分支"
 },
 {
 "name": "hyper_only",
 "variant": "HDHKT_HyperOnly",
 "description": "仅保留超图分支（难度加权）"
 }
 ]
}
```

**配置字段说明：**

| 字段 | 描述 |
| --- | --- |
| ``study_name`` | 消融实验名称 |
| ``base_model`` | 作为基准的模型 |
| ``shared_params`` | 所有变体共享的训练参数 |
| ``ablations`` | 消融变体列表 |
| ``ablations[].name`` | 变体简称 |
| ``ablations[].variant`` | 已注册的模型变体名称 |
| ``ablations[].description`` | 变更说明 |


## 输出

结果保存到 ``runs/ablation/<study_name>_<timestamp>/``：

```
runs/ablation/hgikt_ablation_study_20240403-120000/
├── results.csv # CSV 格式的所有变体结果
└── <variant_name>/ # 各变体的运行目录
 ├── best_model.pth
 └── hyperparameters.json
```

**示例比较：**

| 模型 | AUC | ACC | Δ AUC |
| --- | --- | --- | --- |
| HDHKT（完整） | 0.785 | 0.742 | - |
| HeteroOnly | 0.762 | 0.721 | -2.9% |
| HyperOnly | 0.751 | 0.710 | -4.3% |
| SimpleFusion | 0.768 | 0.725 | -2.2% |


## 创建消融变体

### 步骤一：创建变体模型

```python
# model/HDHKT/variants/hgikt_hetero_only.py
from utils.core import register_trainer
from model.HDHKT.hgikt_model import HDHKT


@register_trainer("HDHKT_HeteroOnly")
class HDHKT_HeteroOnly(HDHKT):
 """消融：仅保留异构图分支"""

 def __init__(self, **kwargs):
 super.__init__(**kwargs)
 # 移除超图组件
 self.hypergraph_conv = None
```

### 步骤二：添加到配置

```json
{
 "ablations": [
 {"name": "hetero_only", "variant": "HDHKT_HeteroOnly", "description": "..."}
 ]
}
```

## 多数据集消融

```bash
for dataset in assistments09 assistments12 assistments17; do
 python ablation_study.py \
 --config configs/ablation/hgikt_study.json \
 -d $dataset \
 --fold 0
done
```

## 最佳实践

1. **单一变量**：每个变体只更改一个组件
2. **相同超参数**：使用相同设置以进行公平比较
3. **多次运行**：用不同折运行 3-5 次，报告均值 ± 标准差
4. **统计检验**：使用 t 检验评估显著性
5. **文档化**：清晰描述每个变体移除或修改的内容
