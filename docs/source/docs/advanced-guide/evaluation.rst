评估系统
========

UniKT 的评估系统分为三层：指标计算（MetricsAccumulator）、指标记录（MetricLogger）和回调节点（Callback）。本页深入介绍每层的设计原理、调用链和数据流。

架构总览
--------

::

   batch_data → forward_pass → {y_hat, y_label, y_predict, y_score, y_prob}
    │
    ┌───────────┴───────────┐
    │ │
    _compute_loss MetricsAccumulator.update
    │ │
    backward │
    │ │
    opt.step per-epoch: accum.compute(phase)
    │
    ┌───────┴────────┐
    │ │
    metric_logger.log_metrics callbacks.on_phase_end
    (CSV + SwanLab) (Checkpoint + EarlyStopping)

指标计算：MetricsAccumulator
----------------------------

``MetricsAccumulator`` 位于 ``utils/training/metrics.py``\ ，负责从 batch 级别的 ``forward_pass`` 输出累积到 epoch 级别的聚合指标。

输入约定
~~~~~~~~

``forward_pass`` 必须返回一个字典，包含以下五个键：

============= ==================== ===========================
键            含义                 用途
============= ==================== ===========================
``y_label``   真实标签（0/1）      所有指标的真值
``y_predict`` 二元预测（0/1）      计算 ACC
``y_score``   排序分数（任意实数） 计算 AUC
``y_prob``    预测概率（[0, 1]）   计算 RMSE
``group_id``  **可选**\ ，题组 ID  测试阶段启用 group 聚合评估
============= ==================== ===========================

以 GIKT 的 ``forward_pass`` 为例（\ ``model/GIKT/GIKT_trainer.py`` ）：

.. code:: python

   def forward_pass(self, batch_data):
    y_hat_full = self._pad_to_full_sequence(
    self.model(user_sequence=sequence, user_response=response, ...)
    )
    y_hat, y_label, _ = self._extract_valid_predictions(
    y_hat_full, response, mask
    )
    return {
    "y_hat": y_hat,
    "y_label": y_label,
    "y_predict": self._generate_binary_predictions(y_hat, threshold=0.0),
    "y_score": y_hat,
    "y_prob": torch.sigmoid(y_hat),
    }

GIKT 使用 ``y_score = y_hat``\ （logits 直接作为排序分数）和 ``y_prob = sigmoid(y_hat)``\ （概率）。二分类阈值取 0.0，因为损失函数是 ``BCEWithLogitsLoss``\ （内部已含 sigmoid）。

累积与聚合
~~~~~~~~~~

每个 batch 训练/验证后，训练循环调用 ``accumulator.update(phase, output)``\ （\ ``base_trainer.py`` 和 ）。累积器将 batch 级 tensor 移到 CPU 后存入列表，避免 GPU 内存堆积。

每个 epoch 结束时调用 ``accum.compute(phase)``\ （\ ``base_trainer.py`` ），将所有 batch 拼接为 numpy 数组后计算三项核心指标：

**ACC（准确率）** — 使用 ``sklearn.metrics.accuracy_score``\ ：

.. code:: python

   metrics["acc"] = float(accuracy_score(y_label, y_pred))

这里 ``y_pred`` 是 ``(y_score >= 0).float`` 产生的 0/1 二值预测。

**AUC（ROC 曲线下面积）** — 使用 ``sklearn.metrics.roc_auc_score``\ ：

.. code:: python

   try:
    metrics["auc"] = float(roc_auc_score(y_label, y_score))
   except ValueError:
    metrics["auc"] = 0.0

``roc_auc_score`` 对排序分数做全排序后计算 TPR/FPR 曲线下面积。当 ``y_label`` 全为 0 或全为 1 时 ``ValueError``\ ，此时 AUC 设为 0.0。

**RMSE（均方根误差）** — 使用 ``sklearn.metrics.root_mean_squared_error``\ ：

.. code:: python

   metrics["rmse"] = float(root_mean_squared_error(y_label, y_prob))

``y_prob`` 必须落在 [0, 1] 区间，由模型通过 sigmoid/softmax 保证。

测试阶段的 Group 聚合
~~~~~~~~~~~~~~~~~~~~~

当 ``phase="test"`` 且提供了 ``group_id`` 时，\ ``compute("test")`` 进入分组评估模式（\ ``metrics.py`` ）。每个 group 内的多条题目记录被聚合为一条后计算指标，支持三种聚合策略：

======== ===============================================================
策略     行为
======== ===============================================================
``mean`` 组内预测分数取均值
``vote`` 按组内多数方向（对/错）取对应子集的均值；子集为空时回退整组均值
``all``  全对/全错组取整组均值；其余组按多数方向取子集均值
======== ===============================================================

三种策略的输出指标名分别前缀 ``mean_``\ 、\ ``vote_``\ 、\ ``all_``——例如 ``mean_acc``\ 、\ ``vote_auc``\ 、\ ``all_rmse``\ 。Group 内标签必须一致，否则抛 ``ValueError``\ （）。

MetricLogger：记录后端
----------------------

MetricLogger 位于 ``utils/training/metric_logger.py``\ ，提供统一的指标记录抽象，管理后端实例。

注册机制
~~~~~~~~

使用 ``@register_metric_logger("name")`` 装饰器注册，\ ``METRIC_LOGGERS.get("name")`` 获取：

.. code:: python

   # metric_logger.py 
   @register_metric_logger("local")
   class LocalMetricLogger(MetricLogger): ...


   # metric_logger.py 
   @register_metric_logger("swanlab")
   class SwanLabMetricLogger(MetricLogger): ...


   # metric_logger.py 
   def get_metric_logger(name: str, **kwargs) -> MetricLogger:
    cls = METRIC_LOGGERS.get(name)
    return cls(**kwargs)

MetricLogger 与其他组件（训练器、参数配置、数据源、分析器）使用相同的注册机制。

工厂函数：build_default_metric_loggers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

训练器在 ``build`` 中通过 ``build_default_metric_loggers`` 创建后端组合（\ ``base_trainer.py`` ）：

.. code:: python

   self.metric_logger = build_default_metric_loggers(
    log_dir=self.log_dir,
    log_batch_metrics=self.log_batch_metrics,
    no_swanlab=self.no_swanlab,
   )

该工厂函数（\ ``metric_logger.py`` ）保证：

1. ``LocalMetricLogger`` — **始终启用**\ ，写入本地 CSV
2. ``SwanLabMetricLogger`` — 除非 ``--no_swanlab`` 为 True，否则自动附加
3. 两者通过 ``MetricLoggerComposite`` 组合，任一后端异常不影响另一个（\ ``metric_logger.py`` ）

LocalMetricLogger：本地 CSV 记录
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``LocalMetricLogger``\ （\ ``metric_logger.py`` ）在 ``log_dir`` 下按 phase 写入 CSV：

::

   runs/normal/GIKT_assistments09_20240403-120000_fold0_bs128/
   ├── metrics_train.csv # 每 epoch 的训练聚合指标
   ├── metrics_val.csv # 每 epoch 的验证聚合指标
   ├── metrics_test.csv # 测试集聚合指标
   ├── early_stopping.csv # 早停轨迹：best_score / num_bad_epochs
   ├── batch_metrics_train.csv # 每 batch 的 loss（仅 --log_batch_metrics）
   └── metrics_final.csv # 最终摘要指标

多阶段训练场景下，文件名以阶段名前缀区分：\ ``metrics_Stage1_train.csv``\ 、\ ``metrics_Stage2_val.csv``\ 。CSV 表头惰性写入，句柄复用整个 run 期间不关闭，按需 ``flush`` 保证崩溃安全（）。

SwanLabMetricLogger：SwanLab 集成
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SwanLabMetricLogger``\ （\ ``metric_logger.py`` ）将指标写入 SwanLab 云端/本地面板。

**初始化流程（\ ``init_run``\ ，）：**

.. code:: python

   def init_run(self, *, log_dir, experiment_name, group, tags, config):
    import swanlab
    from dotenv import load_dotenv
    from swanlab.plugin.notification import LarkCallback

    load_dotenv
    callbacks = []
    webhook = os.getenv("LARK_WEBHOOK_URL")
    secret = os.getenv("LARK_SECRET")
    if webhook:
    callbacks.append(LarkCallback(webhook_url=webhook, secret=secret))

    swanlab.init(
    workspace=os.getenv("SWANLAB_WORKSPACE", None),
    project_name="kt-exp-graph",
    experiment_name=f"Run_{experiment_name}",
    config=config,
    callbacks=callbacks,
    group=group,
    tags=tags,
    )
    self._initialized = True

要点：

- **惰性导入**\ ：\ ``swanlab`` 在方法内 ``import``\ ，保持为可选依赖——标准环境下缺少 ``swanlab`` 不影响 ``import utils.training``
- **环境变量**\ ：通过 ``dotenv.load_dotenv`` 加载 ``.env`` 文件，支持 ``SWANLAB_WORKSPACE``\ 、\ ``LARK_WEBHOOK_URL``\ 、\ ``LARK_SECRET``
- **飞书通知**\ ：若配置了 ``LARK_WEBHOOK_URL``\ ，训练异常时通过 ``LarkCallback`` 自动推送飞书消息
- **project_name** 固定为 ``"kt-exp-graph"``\ ，\ ``experiment_name`` 格式为 ``Run_<log_dir_basename>``
- **group** 按模型类名分组（\ ``model.__class__.__name__``\ ），tags 标注 ``cuda`` 或 ``cpu``

**指标记录（\ ``log_metrics``\ ，）：**

.. code:: python

   def log_metrics(self, *, phase, metrics, step, epoch, stage=None):
    prefix = self._prefix(phase, stage) # e.g. "Train/" or "Val/"
    payload = {
    f"{prefix}{name.upper}-epoch": v
    for name, v in metrics.items
    if v is not None
    }
    swanlab.log(payload, step=step)

指标名按 ``{Stage/}{Phase/}{NAME}-epoch`` 格式上送，例如单阶段训练的 ``Train/AUC-epoch``\ 、\ ``Val/ACC-epoch``\ ，多阶段训练的 ``STAGE1/Train/AUC-epoch``\ 。

**生命周期：** ``init_run`` 在 ``_init_metric_logger`` 中调用（\ ``base_trainer.py`` ），\ ``finish`` 在 ``_finish_metric_logger`` 中调用（），对应训练的开始和结束。

SwanLab 环境变量
~~~~~~~~~~~~~~~~

相关变量在 ``init_run`` 中通过 ``os.getenv`` 读取：

+-----------------------+-------------------------------------------------------+---------------------------------+
| 变量                  | 说明                                                  | 缺失影响                        |
+=======================+=======================================================+=================================+
| ``SWANLAB_WORKSPACE`` | SwanLab 工作空间名                                    | 使用默认空间                    |
+-----------------------+-------------------------------------------------------+---------------------------------+
| ``SWANLAB_MODE``      | ``cloud``\ （上传云端）或 ``local``\ （仅本地，默认） | 仅本地记录                      |
+-----------------------+-------------------------------------------------------+---------------------------------+
| ``LARK_WEBHOOK_URL``  | 飞书机器人 Webhook 地址                               | 不推送飞书通知                  |
+-----------------------+-------------------------------------------------------+---------------------------------+
| ``LARK_SECRET``       | 飞书机器人签名密钥                                    | ``LARK_WEBHOOK_URL`` 存在时必填 |
+-----------------------+-------------------------------------------------------+---------------------------------+

建议在项目根目录创建 ``.env`` 文件配置这些变量（该文件已加入 ``.gitignore``\ ）。

关闭 SwanLab
~~~~~~~~~~~~

通过 ``--no_swanlab`` 或 ``--nsl`` 标志关闭 SwanLab：

.. code:: bash

   python train.py -m GIKT -d assistments09 --no_swanlab

关闭后 ``build_default_metric_loggers`` 只创建 ``LocalMetricLogger``\ ，SwanLab 端不初始化。

回调节点中的指标流转
--------------------

EarlyStoppingCallback
~~~~~~~~~~~~~~~~~~~~~

``EarlyStoppingCallback``\ （\ ``callbacks.py`` ）在验证阶段结束时（\ ``on_phase_end``\ ，phase="val"）执行两步操作：

1. 从 metrics 中提取监控指标值（\ ``_select_monitor_value``\ ，），按 ``es_monitor`` 指定的名称查找，找不到时按 auc → acc → rmse 顺序回退
2. 调用 ``early_stopping.step(current, epoch, metrics)`` 判断是否触发早停
3. 将早停状态写入 ``metric_logger.log_early_stopping``\ ，同时记录到 SwanLab 的 ``ES/Best``\ 、\ ``ES/Num_Bad_Epochs``\ 、\ ``ES/Best_{METRIC}`` 面板

CheckpointCallback
~~~~~~~~~~~~~~~~~~

``CheckpointCallback``\ （\ ``callbacks.py`` ）在两个时机操作：

- ``on_epoch_end``\ ：保存 ``last_checkpoint.pth``\ （完整状态：model + optimizer + scheduler + early_stopping）
- ``on_phase_end``\ （phase="val"）：根据 ``monitor`` 指标判断当前是否是最佳 epoch，若是则保存 ``best_model.pth``\ （仅模型权重）

``CheckpointCallback`` 的 ``monitor`` 可与早停的 ``monitor`` 解耦——通过构造函数的 ``monitor`` 参数显式传入（），默认跟随 ``early_stopping``\ 。

TestEvaluationCallback
~~~~~~~~~~~~~~~~~~~~~~

``TestEvaluationCallback``\ （\ ``callbacks.py`` ）在 ``on_train_end`` 时触发，调用 ``_evaluate_on_test_set(use_best_model=True)``\ （\ ``base_trainer.py`` ）：

1. 恢复到最佳模型权重（\ ``best_model_state``\ ）
2. 遍历测试集 DataLoader，每条样本调 ``test_forward_pass``\ （默认同 ``forward_pass``\ ）
3. 累积到 ``MetricsAccumulator`` 的 "test" phase
4. 调用 ``metric_logger.log_metrics(phase="test", ...)`` 写入 CSV + SwanLab
5. 恢复训练结束时的模型权重

指标数据流全貌
--------------

::

   epoch 循环
    ├─ _process_epoch(epoch, is_train=True)
    │ ├─ accum.reset("train")
    │ ├─ for batch in train_data:
    │ │ ├─ forward_pass(batch) → output
    │ │ ├─ loss = _compute_loss(output)
    │ │ ├─ loss.backward → opt.step
    │ │ ├─ accum.update("train", output)
    │ │ └─ [可选] metric_logger.log_batch(phase="train", ...)
    │ ├─ metrics = accum.compute("train")
    │ ├─ metric_logger.log_metrics(phase="train", metrics, ...)
    │ └─ callbacks.on_phase_end(trainer=self)
    │
    ├─ _process_epoch(epoch, is_train=False)
    │ └─ ...（同上，accum 用 "val"，loss 不反向传播）
    │ └─ callbacks.on_phase_end →
    │ ├─ CheckpointCallback: 可能保存 best_model
    │ └─ EarlyStoppingCallback: 可能触发早停
    │
    └─ 训练结束
    └─ TestEvaluationCallback.on_train_end →
    ├─ 恢复 best_model
    ├─ 遍历 test_data → accum.compute("test")
    └─ metric_logger.log_metrics(phase="test", ...)

扩展到新指标
------------

若需添加新指标（如 F1、MCC），修改两个位置：

1. ``MetricsAccumulator.compute`` 中添加计算逻辑——从 ``y_label`` / ``y_predict`` / ``y_score`` 计算并加入 ``metrics`` 字典
2. 若指标需提前感知（如 EarlyStopping 监控），在 ``EarlyStoppingCallback._select_monitor_value`` 的 fallback 链中添加
