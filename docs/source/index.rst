UniKT
======

**统一知识追踪实验框架** —— 内置 35 个知识追踪模型（GNN、注意力、Mamba 等），覆盖训练、评估、超参数搜索、消融实验与案例分析全流程。

.. rubric:: 快速导航

- :doc:`快速上手 <docs/introduction/quick-start>` —— 安装与第一次训练
- :doc:`模型基准 <model-zoo>` —— 各模型在标准数据集上的性能对比
- :doc:`框架数据流 <docs/advanced-guide/data-flow>` —— 端到端架构详解

.. panels::
   :column: col-lg-6 col-md-12 col-12

   .. panel:: 多模型支持

      - 35 个内置知识追踪模型（GNN、注意力、Mamba）
      - 装饰器注册 + 静态发现懒加载
      - 统一的训练器 API

   .. panel:: 实验管理

      - K 折交叉验证
      - 基于 Optuna 的超参数搜索
      - 消融实验框架
      - SwanLab 实验追踪

   .. panel:: 数据处理

      - 11 个支持的数据集
      - 自动下载与预处理
      - 可配置的采样策略
      - 标准化 Parquet 输出

   .. panel:: 案例分析

      - 模型推理与预测
      - 按策略选择学生
      - 知识状态热力图

.. toctree::
   :caption: 目录
   :maxdepth: 2
   :hidden:

   docs/index
   api/index
   datasets/index
   model-zoo
