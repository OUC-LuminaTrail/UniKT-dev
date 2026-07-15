# 快速上手

UniKT 是一个基于 PyTorch 和 PyTorch Geometric 构建的统一知识追踪研究实验框架。

## 什么是知识追踪

知识追踪是对学习者的知识状态随时间进行建模的任务，基于其历史交互数据（回答的题目、练习的技能以及正确性结果）。其目标是预测学习者在未来题目上的表现，从而实现个性化学习。

## UniKT 提供的功能

UniKT 提供从数据到评估的完整流程：

- **统一 API**：使用相同的 CLI 和训练器 API 训练任何模型
- **35 个内置模型**：覆盖图神经网络（GKT、GIKT、SGKT、DyGKT、HDHKT）和序列/注意力架构（DKT、AKT、SimpleKT）等方向
- **11 个支持的数据集**：ASSISTments（2009/2012/2015/2017）、Algebra（2005/2006）、Bridge2006、EdNet-KT1、Junyi、NIPS2020_T34、Slepemapy
- **实验管理**：K 折交叉验证、超参数搜索（Optuna）

## 框架设计

UniKT 通过装饰器自动发现模型、训练器、数据源和配置——只需在代码中添加一行装饰器，无需手动编辑注册文件。

```{mermaid}
graph TB
 subgraph Entry["入口"]
 E1[train.py]
 E2[optuna_search.py]
 E4[case_analysis.py]
 end

 subgraph Framework["框架层"]
 F1[训练基础设施]
 F2[配置管理]
 F3[数据处理流水线]
 end

 subgraph Models["模型层"]
 M1[问题级模型]
 M2[KC级模型]
 end

 Entry --> Framework
 Framework --> Models
```

```{toctree}
:maxdepth: 1

quick-start
setup
```
