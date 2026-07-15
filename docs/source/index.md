# UniKT

**统一知识追踪实验框架** —— 内置 35 个知识追踪模型（GNN、注意力、Mamba 等），覆盖训练、评估、超参数搜索与案例分析全流程。

```{rubric} 快速导航
```

- [快速上手](getting-started/quick-start.md) —— 安装与第一次训练
- [模型基准](model-zoo.md) —— 各模型在标准数据集上的性能对比
- [框架数据流](advanced/data-flow.md) —— 端到端架构详解

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} 多模型支持
- 35 个内置知识追踪模型（GNN、注意力、Mamba）
- 装饰器注册 + 静态发现懒加载
- 统一的训练器 API
:::

:::{grid-item-card} 实验管理
- K 折交叉验证
- 基于 Optuna 的超参数搜索
- SwanLab 实验追踪
:::

:::{grid-item-card} 数据处理
- 11 个支持的数据集
- 自动下载与预处理
- 可配置的采样策略
- 标准化 Parquet 输出
:::

:::{grid-item-card} 案例分析
- 模型推理与预测
- 按策略选择学生
- 知识状态热力图
:::
::::

```{toctree}
:caption: 目录
:maxdepth: 2
:hidden:

getting-started/index
model-zoo
datasets/index
user-guide/index
advanced/index
api/index
about/index
```
