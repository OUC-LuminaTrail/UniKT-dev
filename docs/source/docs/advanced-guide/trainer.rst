训练器架构
==========

深入理解 UniKT 训练器系统的内部架构：BaseTrainer 的链式构建器、前向传播规范、数据对齐机制，以及 MultiTrainer 的多阶段训练编排。

架构概览
--------

.. mermaid::

   flowchart TD
    Builder[链式构建器] --> Base[BaseTrainer]
    Builder --> Multi[MultiTrainer]
    Base --> Loop[单阶段训练循环]
    Multi --> Stages[多阶段编排]
    Stages --> Loop
    Loop --> Forward[forward_pass 抽象]
    Loop --> Align[_extract_valid_predictions 对齐]

UniKT 提供两条训练路径：

- **BaseTrainer**\ ：单阶段训练器，通过链式构建器（Builder Pattern）装配模型、数据、优化器和实验管理，然后执行标准的 train/val/test 循环。
- **MultiTrainer**\ ：多阶段训练器，继承 BaseTrainer，将训练拆分为多个顺序阶段，每个阶段可拥有独立的模型组件、数据和早停策略。

BaseTrainer 链式构建器
----------------------

BaseTrainer 的设计遵循 **链式构建器（Fluent Builder）** 模式：通过一组 ``with_*`` 方法配置训练器的各个维度，最后调用 ``build`` 完成初始化，\ ``run`` 执行训练。

完整调用链
~~~~~~~~~~

.. code:: python

   trainer = (
    MyTrainer(model)
    .with_training(epochs=150, seed=42)
    .with_data(train_dataset, val_dataset, batch_size=128)
    .with_optimization(optimizer, loss_fn, lr_scheduler)
    .with_experiment(exp_manager, hyperparams=args)
    .build
   )

   trainer.run

.. _with_training--训练参数:

with_training —— 训练参数
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def with_training(
    self,
    epochs: int = 200,
    seed: int = 42,
    device: torch.device | None = None,
    checkpoint_path: str | None = None,
   ) -> "BaseTrainer":

创建 ``TrainingConfig`` dataclass 实例，指定训练轮数、随机种子、计算设备和断点续训路径。

- ``device=None`` 时，\ ``build`` 阶段自动调用 ``_try_gpu`` 检测可用 GPU，无 GPU 则回退到 CPU。
- ``checkpoint_path`` 不为空时，\ ``build`` 会加载检查点并恢复 epoch、optimizer、lr_scheduler 和 early_stopping 状态，从 ``start_epoch`` 继续训练。

**调用时机**\ ：必须第一个调用，\ ``build`` 会校验 ``_training_config is not None``\ ，否则抛出 ``ValueError``\ 。

.. _with_data--数据配置:

with_data —— 数据配置
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def with_data(
    self,
    train_data,
    batch_size,
    val_data,
    test_data=None,
    collate_fn=None,
    val_collate_fn=None,
    test_collate_fn=None,
   ) -> "BaseTrainer":

创建 ``DataConfig``\ 。内部 ``_setup_data_loaders`` 会自动将传入的 ``Dataset`` 包装为优化过的 ``DataLoader``\ （使用 ``create_optimized_dataloader``\ ，启用 ``pin_memory``\ 、\ ``prefetch_factor=2``\ 、\ ``persistent_workers``\ ）；如果传入的已经是 ``DataLoader`` 实例，则原样使用不做二次包装。

.. _with_optimization--优化配置:

with_optimization —— 优化配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def with_optimization(
    self,
    optimizer,
    loss_fn,
    max_clip_grad_norm: float | None = None,
    lr_scheduler=None,
    early_stopping: EarlyStoppingConfig | None = None,
   ) -> "BaseTrainer":

创建 ``OptimizationConfig``\ 。其中 ``early_stopping`` 接收 ``EarlyStoppingConfig`` dataclass（定义 ``monitor``\ 、\ ``mode``\ 、\ ``patience``\ 、\ ``min_delta``\ ），\ ``build`` 阶段内部会据此构造 ``EarlyStopping`` 工具实例。

``max_clip_grad_norm`` 用于梯度裁剪，在 ``_run_train_batch`` 中 ``loss.backward`` 之后、\ ``opt.step`` 之前执行：

.. code:: python

   if self.max_clip_grad_norm is not None:
    torch.nn.utils.clip_grad_norm_(
    self.model.parameters, max_norm=self.max_clip_grad_norm
    )

.. _with_experiment--实验管理:

with_experiment —— 实验管理
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def with_experiment(
    self,
    exp_manager,
    hyperparams=None,
    no_swanlab: bool | None = None,
    log_batch_metrics: bool | None = None,
    model_name: str = "",
    dataset_name: str = "",
    skip_test: bool = False,
   ) -> "BaseTrainer":

创建 ``ExperimentConfig``\ ，负责日志目录、超参数保存和指标记录后端。关键行为：

- ``no_swanlab`` 和 ``log_batch_metrics`` 支持双通道解析：显式传入优先，否则回退到 ``hyperparams``\ （CLI 参数）中的对应字段。这意味着 ``--no_swanlab`` / ``--log_batch_metrics`` 命令行标志对所有模型生效。
- 本地 CSV 记录始终启用，SwanLab 仅在未设置 ``--no_swanlab`` 时启动。
- ``skip_test`` 控制训练完成后的测试集评估。

.. _build--装配与校验:

build —— 装配与校验
~~~~~~~~~~~~~~~~~~~

``build`` 按 12 个步骤顺序执行初始化：

1.  **校验**\ ：四个配置对象（training / data / optimization / experiment）必须都已调用，否则抛 ``ValueError``
2.  **设备**\ ：自动检测或使用显式指定的设备
3.  **日志开关**\ ：调用 ``resolve_metric_logging_flags`` 解析 ``no_swanlab`` / ``log_batch_metrics``\ （显式 > CLI）
4.  **随机种子**\ ：调用 ``seed_everything(seed, deterministic)``\ ，同时设置 PyTorch / NumPy / Python random 种子
5.  **DataLoader**\ ：\ ``_setup_data_loaders`` 包装 Dataset → DataLoader
6.  **优化器/损失/调度器/梯度裁剪**\ ：从 ``OptimizationConfig`` 解包挂到实例属性
7.  **早停**\ ：\ ``EarlyStoppingConfig → EarlyStopping`` 实例化
8.  **日志目录**\ ：\ ``exp_manager.get_log_dir`` 创建 ``runs/<type>/<name>/``
9.  **组件初始化**\ ：\ ``MetricsAccumulator``\ 、\ ``CheckpointManager``\ 、\ ``MetricLogger``\ 、\ ``CallbackManager``
10. **超参数保存**\ ：调用 ``HyperparameterManager`` 保存模型参数量、优化器、损失函数、设备信息等元数据
11. **检查点恢复**\ ：若指定了 ``checkpoint_path``\ ，加载模型权重和训练状态
12. **torch.compile**\ ：若通过 ``with_compile`` 或 CLI 参数启用了编译优化

..

   **重要**\ ：\ ``build`` 必须显式调用。\ ``run`` 检查 ``self._built`` 标志，若未构建直接抛出 ``RuntimeError``\ 。重复调用 ``build`` 会被跳过并记录警告。

回调系统
~~~~~~~~

``build`` 自动注入以下回调（按优先级从高到低）：

+----------------------------+---------------+-------------------------------------------------------------------------+
| 回调                       | 触发时机      | 职责                                                                    |
+============================+===============+=========================================================================+
| ``FunctionCallback``       | 各事件        | 将用户通过 ``with_callbacks(functions={...})`` 注册的普通函数包装为回调 |
+----------------------------+---------------+-------------------------------------------------------------------------+
| ``MemoryCleanupCallback``  | 每 5 个 epoch | 调用 ``torch.cuda.empty_cache`` 释放显存                                |
+----------------------------+---------------+-------------------------------------------------------------------------+
| ``EarlyStoppingCallback``  | epoch 结束    | 检查早停条件，触发 ``should_stop``                                      |
+----------------------------+---------------+-------------------------------------------------------------------------+
| ``CheckpointCallback``     | epoch 结束    | 保存 ``last_checkpoint.pth`` 和 ``best_model.pth``                      |
+----------------------------+---------------+-------------------------------------------------------------------------+
| ``TestEvaluationCallback`` | 训练结束      | 加载最佳模型权重，在测试集上评估                                        |
+----------------------------+---------------+-------------------------------------------------------------------------+

用户可通过 ``with_callbacks(callbacks=[...], functions={"on_epoch_end": my_fn})`` 注入自定义回调。

forward_pass 返回规范
---------------------

``forward_pass`` 是 BaseTrainer 唯一的抽象方法，所有模型子类必须实现：

.. code:: python

   @abstractmethod
   def forward_pass(self, batch_data: tuple[Any, ...]) -> dict:
    """模型前向传播。

    Returns:
    包含 "y_hat", "y_label", "y_predict" 的字典
    """

三个必需的返回键：

+---------------+--------------------------+------------------------------------------------------------------------------------------------+
| 键            | 形状                     | 含义                                                                                           |
+===============+==========================+================================================================================================+
| ``y_hat``     | ``(N,)`` 或 ``(B, S)``   | 模型原始输出（logits），未经过阈值处理或 sigmoid                                               |
+---------------+--------------------------+------------------------------------------------------------------------------------------------+
| ``y_label``   | ``(N,)`` 或 ``(B, S)``   | 真实标签，float 类型                                                                           |
+---------------+--------------------------+------------------------------------------------------------------------------------------------+
| ``y_predict`` | ``(N,)`` 或 ``(B, S)``   | 二值预测结果（0/1），int 类型。由 ``_generate_binary_predictions`` 默认使用 threshold=0.0 生成 |
+---------------+--------------------------+------------------------------------------------------------------------------------------------+

训练和验证阶段均调用 ``forward_pass``\ 。测试评估默认调用 ``test_forward_pass``\ ，它默认转发到 ``forward_pass``\ ，子类可重写以在测试时使用不同逻辑（如加载集成模型）。

``_compute_loss`` 从 outputs 字典取出 ``y_hat`` 和 ``y_label`` 计算损失：

.. code:: python

   def _compute_loss(self, outputs: dict) -> torch.Tensor:
    y_hat = outputs["y_hat"]
    y_label = outputs["y_label"]
    return self.loss(y_hat, y_label)

子类如有自定义损失逻辑（如多任务损失、正则化项），可重写 ``_compute_loss``\ 。

训练循环中的调用流
~~~~~~~~~~~~~~~~~~

::

   _run_train_batch(batch_data)
    → opt.zero_grad
    → output = forward_pass(batch_data) # 模型前向
    → loss = _compute_loss(output) # 损失计算
    → loss.backward # 反向传播
    → clip_grad_norm (if configured) # 梯度裁剪
    → opt.step # 参数更新
    → metrics_accumulator.update("train", output) # 累积预测（用于 epoch 级指标计算）

   _run_eval_batch(batch_data) # @torch.inference_mode
    → output = forward_pass(batch_data) # 无梯度前向
    → loss = _compute_loss(output)
    → metrics_accumulator.update("val", output)

数据对齐与预测位提取
--------------------

知识追踪任务的核心约定是 **next-item 预测**\ ：用前 t 个交互预测第 t+1 个交互的答题正确性。如果对齐出错，会导致评估指标虚高。

.. _`_extract_valid_predictions`:

\_extract_valid_predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

BaseTrainer 提供了静态方法 ``_extract_valid_predictions`` 来正确处理对齐：

.. code:: python

   def _extract_valid_predictions(
    self,
    y_hat_full: torch.Tensor, # [B, S]
    response: torch.Tensor, # [B, S]
    mask: torch.Tensor, # [B, S]
    same_position: bool = False,
   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

**核心逻辑**\ ：

1. **next-item 对齐**\ ：取 ``y_hat_full[:, :-1]`` 作为预测，配 ``response[:, 1:]`` 作为标签。即 t 时刻的模型输出用于预测 t+1 时刻的答题结果。
2. **掩码交叉校验**\ ：只保留 ``mask[:, :-1] & mask[:, 1:]`` 的位置——即当前位置和下一位置都有效的交互对。这确保不会在序列边界或填充位置提取无意义的预测。
3. **same_position 归一化**\ ：部分模型（如输出长度为 S-1 的架构）输出的是同位置约定 ``out[t] → response[t]``\ 。设置 ``same_position=True`` 会将 ``y_hat_full[:, 1:]`` 左移一位并在末尾补零占位，统一转换为 next-item 视图后再对齐。

**为什么需要预测位提取？**

假设序列 ``[q1, q2, q3, q4]`` 对应答案 ``[r1, r2, r3, r4]``\ 。如果直接计算 ``y_hat[t]`` 与 ``response[t]`` 的匹配（同位置），模型在预测 ``q3`` 的结果时可能"看到"了 ``q3`` 本身的信息。正确的做法是用 ``y_hat[:3]`` 与 ``response[1:4]`` 配对——模型看到前 t 个题，预测第 t+1 个题的答案。

.. _`_pad_to_full_sequence`:

\_pad_to_full_sequence
~~~~~~~~~~~~~~~~~~~~~~

部分模型（如 GKT、SAKT、SGKT、MIKT、KQN）的 next-item 输出长度为 ``S-1``\ （因为最后一个时间步没有下一项可预测）。\ ``_pad_to_full_sequence`` 在时间维末尾补一列零占位，将 ``[B, S-1]`` 扩展为 ``[B, S]``\ ，使得 ``_extract_valid_predictions`` 的 ``[:, :-1]`` 切片能正确丢弃这一占位列：

.. code:: python

   def _pad_to_full_sequence(self, y_hat: torch.Tensor) -> torch.Tensor:
    dummy = torch.zeros(y_hat.size(0), 1, device=y_hat.device)
    return torch.cat([y_hat, dummy], dim=1)

.. _`_handle_empty_batch`:

\_handle_empty_batch
~~~~~~~~~~~~~~~~~~~~

当批次内所有位置都被掩码过滤掉时（例如序列长度不足或采样策略排除），\ ``y_label.numel == 0`` 会触发 ``ValueError`` 并给出详细的排查提示：

.. code:: python

   if y_label.numel == 0:
    raise ValueError(
    "Empty valid targets in current batch: no positions satisfy "
    "the training mask alignment. Please check data preprocessing/sampling "
    "settings (e.g., min_seq_len, sample_users, batch_size)."
    )

MultiTrainer 多阶段训练
-----------------------

MultiTrainer 用于需要\ **分阶段训练**\ 的模型（如 ABKT、LPKT 等），每个阶段拥有独立的模型组件和数据配置。

与 BaseTrainer 的关键差异
~~~~~~~~~~~~~~~~~~~~~~~~~

+-------+--------------------------------------------+----------------------------------------------------------------------------+
| 维度  | BaseTrainer                                | MultiTrainer                                                               |
+=======+============================================+============================================================================+
| 构造  | ``__init__(model)``                        | ``__init__(*, device, seed, deterministic)``\ ，\ ``model=None``           |
+-------+--------------------------------------------+----------------------------------------------------------------------------+
| 配置  | 必须在 ``build`` 前调用全部四个 ``with_*`` | 只需 ``with_experiment``\ 。模型/数据/优化器通过 ``build_stages`` 延迟构建 |
+-------+--------------------------------------------+----------------------------------------------------------------------------+
| build | 一次性装配所有组件                         | 只初始化跨阶段共享设施（设备、种子、日志、超参数）                         |
+-------+--------------------------------------------+----------------------------------------------------------------------------+
| run   | 单次 ``_run_training_loop``                | 遍历 ``build_stages`` 返回的阶段列表，每个阶段独立训练                     |
+-------+--------------------------------------------+----------------------------------------------------------------------------+

核心抽象
~~~~~~~~

**StageConfig** —— 阶段的声明式描述：

.. code:: python

   @dataclass
   class StageConfig:
    name: str # 阶段名称
    build: Callable[[], StageComponents] # 延迟构建器，无参可调用

``build`` 是一个无参可调用对象，\ **仅在阶段即将开始时才被调用**\ 。这使得后续阶段的构建可以依赖前序阶段的输出（如 boosting 残差、上一阶段学习到的表示）。

**StageComponents** —— 阶段内的完整组件集合：

.. code:: python

   @dataclass
   class StageComponents:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: torch.nn.Module
    train_data: Any
    val_data: Any | None = None
    test_data: Any | None = None
    epochs: int = 100
    lr_scheduler: Any | None = None
    early_stopping: EarlyStoppingConfig | None = None
    max_clip_grad_norm: float | None = None
    checkpoint_monitor: str | None = None # 保存最佳模型的监控指标（可与早停不同）
    checkpoint_mode: str | None = None # 'max' / 'min'

``checkpoint_monitor`` 和 ``checkpoint_mode`` 允许将"保存最佳模型"的监控指标与"早停"的监控指标解耦。例如，可以用 AUC 做早停但用 ACC 保存最佳检查点。

完整示例
~~~~~~~~

.. code:: python

   from utils.training import MultiTrainer, StageConfig, StageComponents
   from utils.core import register_trainer
   from utils.config import EarlyStoppingConfig


   @register_trainer("ABKT")
   class ABKTTrainer(MultiTrainer):
    def __init__(self, args, data_src, exp_manager):
    super.__init__(device=args.device, seed=args.seed)
    self.args = args
    self.data_src = data_src
    self.with_experiment(exp_manager, hyperparams=args, model_name="ABKT").build

    def build_stages(self) -> list[StageConfig]:
    return [
    StageConfig("km", self._build_km_stage),
    StageConfig("am", self._build_am_stage),
    ]

    def _build_km_stage(self) -> StageComponents:
    # 构建知识掌握（Knowledge Mastery）阶段的组件
    km_model = KMModel(...)
    return StageComponents(
    model=km_model,
    optimizer=torch.optim.Adam(km_model.parameters, lr=1e-3),
    loss_fn=torch.nn.BCEWithLogitsLoss,
    train_data=self.train_loader,
    val_data=self.val_loader,
    epochs=100,
    early_stopping=EarlyStoppingConfig(monitor="auc", patience=10),
    )

    def _build_am_stage(self) -> StageComponents:
    # 构建能力建模（Ability Modeling）阶段的组件
    am_model = AMModel(...)
    return StageComponents(
    model=am_model,
    optimizer=torch.optim.Adam(am_model.parameters, lr=1e-4),
    loss_fn=torch.nn.BCEWithLogitsLoss,
    train_data=self.train_loader,
    val_data=self.val_loader,
    epochs=50,
    early_stopping=EarlyStoppingConfig(monitor="auc", patience=5),
    )

    def forward_pass(self, batch_data):
    if self._current_stage == "km":
    return self._km_forward(batch_data)
    return self._am_forward(batch_data)

阶段执行流程
~~~~~~~~~~~~

::

   run
    → build_stages # 获取阶段列表
    → for each stage:
    → self._current_stage = stage.name # 设置阶段标识
    → on_stage_begin(name) # 阶段前钩子
    → setup = stage.build # 延迟构建组件
    → _apply_stage(name, setup) # 挂载 model/opt/loss/data/epochs
    → _run_training_loop # 复用 BaseTrainer 的训练循环
    → 加载最佳模型权重回 self.model
    → on_stage_complete(name, result) # 阶段后钩子（可传递数据到下一阶段）

阶段钩子
~~~~~~~~

MultiTrainer 提供两个可选重写的钩子：

.. code:: python

   def on_stage_begin(self, name: str) -> None:
    """阶段开始前调用。"""


   def on_stage_complete(self, name: str, result: StageResult) -> None:
    """阶段结束后调用，此时最佳模型已加载回 self.model。
    常用于向下一阶段传递数据（如计算 boosting 残差）。"""

指标记录与检查点
~~~~~~~~~~~~~~~~

- **指标步骤累加**\ ：\ ``_metric_step_offset`` 在阶段间累积，保证 SwanLab 等后端的 x 轴（训练步数）单调递增。
- **阶段独立检查点**\ ：每个阶段的检查点命名为 ``{stage_name}_last_checkpoint.pth`` 和 ``best_{stage_name}_model.pth``\ ，互不覆盖。
- **测试集评估**\ ：多阶段训练通常由 ``on_stage_complete`` 控制测试时机（而非在 ``build`` 中自动注入 ``TestEvaluationCallback``\ ）。

训练循环细节
------------

进度显示
~~~~~~~~

训练过程使用 Rich 库渲染双进度条：

- **总进度条**\ （红色）：Total Epochs，每完成一个 train+val epoch 推进一格
- **工作进度条**\ （绿色/青色）：Training / Validation，实时显示当前阶段的 batch 进度
- **最佳指标**\ （黄色）：实时更新当前最佳监控指标值、取得最佳值的 epoch 和早停剩余耐心

多阶段训练时，所有显示自动添加 ``[STAGE_NAME]`` 前缀。

早停机制
~~~~~~~~

``EarlyStopping`` 工具类通过 ``_cmp_sign`` 统一 min/max 模式的比较方向：

.. code:: python

   _cmp_sign = -1.0 if mode == "min" else 1.0


   def _is_improved(self, current, best):
    return (current - best) * self._cmp_sign > self.cfg.min_delta

``step`` 方法追踪 ``best_score``\ 、\ ``best_epoch``\ 、\ ``num_bad_epochs``\ ，当 ``num_bad_epochs >= patience`` 时返回 ``True``\ ，触发 ``CallbackManager.should_stop`` 中断循环。

测试集评估
~~~~~~~~~~

``_evaluate_on_test_set`` 在训练结束后被 ``TestEvaluationCallback`` 调用：

1. 获取 ``CheckpointCallback`` 中缓存的最佳模型权重（\ ``best_model_state``\ ）
2. 暂存当前模型权重，加载最佳权重
3. 在 ``@torch.inference_mode`` 下遍历测试集，调用 ``test_forward_pass``
4. 记录测试指标，恢复当前权重

训练器注册
----------

训练器通过 ``@register_trainer("NAME")`` 注册，支持命令行 ``-m`` 参数按名检索：

.. code:: python

   from utils.core import register_trainer


   @register_trainer("GIKT")
   class GIKTTrainer(BaseTrainer): ...

``@register_trainer`` 装饰器会自动完成注册。
