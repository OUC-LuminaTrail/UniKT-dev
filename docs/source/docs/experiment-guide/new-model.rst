添加新模型
==========

本指南演示如何为 UniKT 添加一个新模型，使之能被 ``python train.py -m MyModel -d assistments09`` 直接调用。

步骤一：创建模型目录结构
------------------------

每个模型在 ``model/`` 下拥有独立目录，含三个必需文件和一个可选文件。以 GIKT 为例：

::

   model/GIKT/
   ├── GIKT_trainer.py # @register_trainer("GIKT") → GIKTTrainer(BaseTrainer)
   ├── GIKT_data.py # @register_model_params("GIKT") → GIKTModelParams
   ├── GIKT_model.py # GIKT(nn.Module)
   └── GIKT_analyzer.py # 可选：@register_analyzer("GIKT")

注册由 ``model/__init__.py`` 统一的 ``discover_registrations`` 完成，子包 ``__init__.py`` 留空即可。

步骤二：实现 Trainer
--------------------

Trainer 需完成三件事：准备数据、初始化模型、定义 ``forward_pass``\ 。

.. _21-准备数据:

2.1 准备数据
~~~~~~~~~~~~

根据数据粒度选择基类：\ **``SkillModelData``**\ （KC 级，如 DKT、AKT、SimpleKT）使用 ``build_sequence_data``\ ；\ **``QuestionModelData``**\ （题目级，如 GIKT、HDHKT、SGKT）使用 ``load_sequence_data``\ 。两者均通过 ``split_kfold_data`` 划分训练/验证/测试集。

.. code:: python

   from model.MyModel.MyModel_data import MyModelData

   model_data = MyModelData(data_src)
   train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

若需自定义 collate_fn（如 GIKT 在 batch 内构建邻居索引），通过 ``with_data(collate_fn=...)`` 传入。

.. _22-初始化模型与链式构建:

2.2 初始化模型与链式构建
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from model.MyModel.MyModel_model import MyModel

   model = MyModel(args=args, data_metadata=data_src.get_metadata)
   super.__init__(model)

   optimizer = torch.optim.Adam(
    model.parameters, lr=args.learning_rate, weight_decay=args.weight_decay
   )
   lr_scheduler = (
    torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    if args.lr_decay
    else None
   )

   self.with_training(
    epochs=args.epochs,
    seed=args.seed,
    device=args.device,
   ).with_data(
    train_data=train_dataset,
    val_data=val_dataset,
    test_data=test_dataset,
    batch_size=args.batch_size,
    collate_fn=train_collate_fn,
   ).with_optimization(
    optimizer=optimizer,
    loss_fn=torch.nn.BCEWithLogitsLoss,
    lr_scheduler=lr_scheduler,
    early_stopping=EarlyStoppingConfig(monitor="auc", mode="max", patience=10),
   ).with_experiment(
    exp_manager=exp_manager,
    hyperparams=args,
    model_name="MyModel",
    dataset_name=args.dataset,
   ).build

.. important::

   四个 ``with_*`` 必须全部调用，否则 ``build`` 抛 ``ValueError``\ 。


.. _23-实现-forward_pass:

2.3 实现 forward_pass
~~~~~~~~~~~~~~~~~~~~~

``forward_pass`` 返回包含 ``y_hat``\ 、\ ``y_label``\ 、\ ``y_predict`` 的字典。以 GIKT 为例：

.. code:: python

   def forward_pass(self, batch_data):
    sequence = self._move_tensor_to_device(batch_data["sequence"])
    response = self._move_tensor_to_device(batch_data["response"])
    mask = self._move_tensor_to_device(batch_data["mask"])

    # 模型输出 [B, S-1]（next-item：y[t] 预测 response[t+1]）
    y_hat_full = self._pad_to_full_sequence(self.model(sequence, response, mask))
    y_hat, y_label, _ = self._extract_valid_predictions(y_hat_full, response, mask)
    y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

    return {
    "y_hat": y_hat,
    "y_label": y_label,
    "y_predict": self._generate_binary_predictions(y_hat, threshold=0.0),
    "y_score": y_hat,
    "y_prob": torch.sigmoid(y_hat),
    }

BaseTrainer 提供的关键辅助方法：

+-----------------------------------------+------------------------------------------------------+
| 方法                                    | 作用                                                 |
+=========================================+======================================================+
| ``_move_tensor_to_device(t)``           | 张量移到训练设备                                     |
+-----------------------------------------+------------------------------------------------------+
| ``_extract_valid_predictions(y, r, m)`` | next-item 对齐：\ ``y[:,:-1]`` 配 ``r[:,1:]``        |
+-----------------------------------------+------------------------------------------------------+
| ``_pad_to_full_sequence(y)``            | ``[B,S-1]`` 补零到 ``[B,S]``\ ，用于输出缺末位的模型 |
+-----------------------------------------+------------------------------------------------------+
| ``_handle_empty_batch(yh, yl)``         | 空 batch 安全检查（抛 ValueError 防止静默失败）      |
+-----------------------------------------+------------------------------------------------------+
| ``_generate_binary_predictions(y, t)``  | 二分类预测                                           |
+-----------------------------------------+------------------------------------------------------+

步骤三：用装饰器注册
--------------------

在 Trainer 和参数配置类上方添加装饰器：

.. code:: python

   from utils.core import register_trainer
   from utils.config import register_model_params, BaseParamConfig


   @register_model_params("MyModel")
   class MyModelParams(BaseParamConfig):
    def define_params(self) -> tuple[str, dict]:
    return "MyModel Parameters", {
    "embedding_dim": {
    "type": int,
    "default": 128,
    "help": "Embedding dimension",
    },
    "epochs": {"type": int, "default": 100, "short": "ep"},
    "learning_rate": {"type": float, "default": 1e-3, "short": "lr"},
    "batch_size": {"type": int, "default": 128, "short": "bs"},
    }


   @register_trainer("MyModel")
   class MyModelTrainer(BaseTrainer): ...

注册后直接运行 ``python train.py -m MyModel -d assistments09``\ 。

.. important::

   装饰器参数必须是\ **字符串字面量**\ ，变量或表达式不会被识别。


步骤四：预测位提取
------------------

UniKT 通过 ``_extract_valid_predictions`` 自动从模型输出中提取有效的预测位置，确保 ``y_hat[t]`` 对齐到正确的 ``response`` 标签。支持两种输出约定：

**约定 A：next-item（默认）** — ``y_hat_full[t]`` 预测 ``response[t+1]``\ 。GIKT、AKT 采用此约定：

.. code:: python

   y_hat, y_label, _ = self._extract_valid_predictions(y_hat_full, response, mask)

若模型只输出 ``[B, S-1]``\ ，先用 ``_pad_to_full_sequence`` 补零。

**约定 B：same-position** — ``y_hat_full[t]`` 预测 ``response[t]``\ 。SimpleKT 采用此约定，传入 ``same_position=True``\ ，内部自动左移一位（丢弃无历史的第 0 位）以遵守 next-item 契约。

检查清单
~~~~~~~~

- ☐ ``forward_pass`` 中模型输入\ **不含当前题目的 response**\ （\ ``r[t]`` 不在 ``y_hat[t]`` 的计算路径）
- ☐ ``response[:, 0]`` 的含义与模型约定一致
- ☐ 若模型内部做移位，确认方向正确（\ ``r[t-1]`` 而非 ``r[t+1]``\ ）
- ☐ 打一次 ``y_hat.shape`` vs ``response.shape``\ ，确认时序对齐
- ☐ Windowlate 评估时确认 ``response`` 占位值不影响预测

步骤五：运行验证
----------------

.. code:: bash

   # 1. 确认模型已注册
   python -c "from utils.core import TRAINERS; print(TRAINERS.keys)"

   # 2. 查看模型参数
   python train.py -m MyModel -h

   # 3. 小规模验证训练
   python train.py -m MyModel -d assistments09 --epochs 1 --es_patience 0

   # 4. K 折交叉验证
   python train.py -m MyModel -d assistments09 --fold 0

输出保存在 ``runs/normal/MyModel_assistments09_<timestamp>/``\ ：

::

   runs/normal/MyModel_assistments09_20240703-120000_fold0_bs128/
   ├── best_model.pth # 最佳检查点
   ├── last_checkpoint.pth # 断点续训用
   ├── hyperparameters.json # 超参数配置
   └── training.log # 训练日志

更多训练参数请参阅\ :doc:`训练与评估 <training-evaluation>`\ 。
