模型评估
========

UniKT 在训练过程中自动执行验证和测试评估，支持 K 折交叉验证、早停控制和多后端指标追踪。本页介绍评估机制的工作原理、配置方式和输出解读。

K 折交叉验证
------------

K 折交叉验证是知识追踪领域评估模型稳定性的标准做法。UniKT 在数据预处理阶段（\ ``add_kfold_labels``\ ）为用户级别的交互序列打上折标签，确保同一用户的所有交互始终落入同一折，从根本上杜绝数据泄露。

折标签含义
~~~~~~~~~~

折标签存储在序列数据的 ``fold`` 列中：

- **-1**\ ：测试集用户。这些用户完全不参与训练和验证，仅在最终测试阶段使用
- **0 ~ K-1**\ ：训练/验证集用户。K 折交叉验证时轮流将其中一份作为验证集，其余作为训练集

折的划分在以用户为粒度随机打乱后进行。例如，在 ASSISTments 2009 上运行 5 折交叉验证，\ ``test_ratio=0.2`` 时：

.. code:: bash

   # 运行单个折（第 1 折）
   python train.py -m GIKT -d assistments09 --fold 0

   # 运行所有折
   for i in {0..4}; do
    python train.py -m GIKT -d assistments09 --fold $i
   done

训练时，框架根据 ``--fold`` 参数选择当前折作为验证集，其余折作为训练集。验证集用户的数据不参与训练，包括用于构建关系矩阵和题目难度统计时也被排除。

.. note::

   折的划分在数据预处理时固定（通过 ``data_process.py process --kfold 5 --test_ratio 0.2 --seed 42``\ ）。所有模型使用相同的折分配，确保结果可比。改变 ``--seed`` 会得到不同的折划分。


验证集评估流程
~~~~~~~~~~~~~~

每个 epoch 分为训练阶段和验证阶段，两个阶段各自累积指标后计算 epoch 级汇总：

1. **训练阶段**\ （phase=train）：遍历训练 DataLoader，每个 batch 执行前向传播、损失计算和反向传播，累积预测结果到 ``MetricsAccumulator``
2. **验证阶段**\ （phase=val）：以 ``torch.inference_mode`` 遍历验证 DataLoader，只做前向传播和损失计算，不更新参数
3. **指标计算**\ ：每个 phase 结束时调用 ``MetricsAccumulator.compute`` 计算 ``acc``\ （准确率）、\ ``auc``\ （AUC 分数）和 ``rmse``\ （均方根误差）

这三个指标的计算基于 ``forward_pass`` 输出的四个字段：

============= ======== =====================
字段          含义     用途
============= ======== =====================
``y_label``   真实标签 acc/auc/rmse 的真实值
``y_predict`` 二元预测 acc 的预测值
``y_score``   排序分数 auc 的输入分数
``y_prob``    预测概率 rmse 的预测值
============= ======== =====================

这些字段在 ``forward_pass`` 的返回值字典中定义，所有模型训练器子类必须实现。

早停
----

早停机制在验证阶段结束时触发，根据监控指标的变化趋势决定是否提前终止训练。UniKT 的早停由两部分协作完成：配置对象 ``EarlyStoppingConfig`` 负责存储参数，工具类 ``EarlyStopping`` 负责状态跟踪和判断逻辑。

命令行参数
~~~~~~~~~~

所有早停参数在命令行中通过 ``--es_*`` 前缀指定：

.. code:: bash

   python train.py -m GIKT -d assistments09 \
    --es_patience 10 \
    --es_monitor auc \
    --es_mode max \
    --es_min_delta 0.001 \
    --es_restore_best

+-----------------------+--------+--------------------------------------------------------------+
| 参数                  | 默认值 | 描述                                                         |
+=======================+========+==============================================================+
| ``--es_patience``     | 10     | 容忍的 epoch 数。设为 0 禁用早停                             |
+-----------------------+--------+--------------------------------------------------------------+
| ``--es_monitor``      | auc    | 监控的指标名（auc / acc / rmse / loss）                      |
+-----------------------+--------+--------------------------------------------------------------+
| ``--es_mode``         | max    | 优化方向：\ ``max`` 用于 auc/acc，\ ``min`` 用于 rmse/loss   |
+-----------------------+--------+--------------------------------------------------------------+
| ``--es_min_delta``    | 0.0    | 最小改善阈值，当前指标与历史最优的差值超过此值才视为有效改进 |
+-----------------------+--------+--------------------------------------------------------------+
| ``--es_restore_best`` | False  | 早停触发后是否恢复最佳模型权重                               |
+-----------------------+--------+--------------------------------------------------------------+

工作原理
~~~~~~~~

每个 epoch 验证阶段结束后，\ ``EarlyStoppingCallback.on_phase_end`` 被调用：

1. 从 ``metrics`` 字典中按 ``--es_monitor`` 指定的名称取当前指标值。若名称对应的值不存在，按 ``auc → acc → rmse`` 的优先级自动回退
2. 调用 ``EarlyStopping.step(current, epoch, metrics)`` 进行判断：

- 首次调用时记录当前值为 ``best_score``\ ，返回不停止
- 之后每次比较当前值与 ``best_score``\ ，若改善超过 ``min_delta`` 则更新 ``best_score`` 和 ``best_epoch``\ ，重置 ``num_bad_epochs``
- 若未改善则 ``num_bad_epochs`` 累加，达到 ``patience`` 时返回停止信号

3. 若启用了 ``--es_restore_best``\ ，训练结束后自动恢复 ``best_epoch`` 对应的模型权重

在验证集 AUC 波动较大的场景（如数据量小或模型不稳定时），适当增大 ``--es_min_delta``\ （如 0.005）可以避免因噪声提前停止。在 ASSISTments 2009 上训练 GIKT，AUC 通常在前 20 个 epoch 上升显著，设置 ``--es_patience 10`` 足以捕捉最佳点。

.. important::

   早停只基于验证集表现，不接触测试集数据。测试集在训练完全结束后才参与最终评估。


测试集评估
----------

训练循环结束后，\ ``TestEvaluationCallback`` 自动触发测试集评估。测试集用户（fold=-1）在整个训练过程中从未被使用，确保评估结果反映模型在未见过的学生上的泛化能力。

评估流程
~~~~~~~~

1. 加载最佳模型：从 ``CheckpointCallback`` 中获取验证集最佳 epoch 对应的模型权重（\ ``best_model_state``\ ），加载到模型上
2. 遍历测试集：以 ``torch.inference_mode`` 遍历测试 DataLoader，调用 ``test_forward_pass`` 执行前向传播
3. 计算指标：调用 ``MetricsAccumulator.compute("test")`` 计算测试指标

测试集评估会额外输出三种 group 聚合指标。当 ``forward_pass`` 的返回值包含 ``group_id``\ （题目级分组 ID）时，\ ``MetricsAccumulator`` 对每组内的预测分数聚合后再计算指标。数据集 ``assistments09`` 按题目分组后，输出示例：

::

   test/acc: 0.7523
   test/auc: 0.7831
   test/rmse: 0.4432
   test/mean_acc: 0.7612
   test/mean_auc: 0.7956
   test/mean_rmse: 0.4378
   test/vote_acc: 0.7598
   test/vote_auc: 0.7934
   test/vote_rmse: 0.4391
   test/all_acc: 0.7615
   test/all_auc: 0.7961
   test/all_rmse: 0.4375

三种聚合方式的含义：

- **mean**\ ：组内所有时间步的预测分数取均值后计算指标
- **vote**\ ：组内按多数正确的方向取模型输出，组内多数答对的取正确子集、多数答错的取错误子集
- **all**\ ：组内全对或全错时用整组预测，其余按多数方向取子集

训练时可以通过 ``--skip_test`` 跳过测试集评估（仅执行训练和验证）：

.. code:: bash

   python train.py -m GIKT -d assistments09 --skip_test

SwanLab 指标追踪
----------------

UniKT 内置两层指标记录：本地 CSV 始终开启，SwanLab 云端追踪可按需启用。两部分通过 ``MetricLoggerComposite`` 统一 fan-out，单个后端异常不会影响其他后端。

初始设置
~~~~~~~~

首次使用 SwanLab 需要登录认证：

.. code:: bash

   swanlab login

SwanLab 的配置通过环境变量控制，而非命令行参数。在项目根目录创建 ``.env`` 文件（已加入 ``.gitignore``\ ）：

.. code:: bash

   # .env
   SWANLAB_WORKSPACE=my-lab
   SWANLAB_MODE=cloud

+-----------------------+-------------------------------------------------------------+
| 变量                  | 说明                                                        |
+=======================+=============================================================+
| ``SWANLAB_WORKSPACE`` | SwanLab 工作空间名称，不设置则使用默认空间                  |
+-----------------------+-------------------------------------------------------------+
| ``SWANLAB_MODE``      | ``cloud``\ （上传到云端）或 ``local``\ （仅本地记录，默认） |
+-----------------------+-------------------------------------------------------------+

代码中通过 ``dotenv.load_dotenv`` 自动加载 ``.env`` 文件。

关闭 SwanLab
~~~~~~~~~~~~

使用 ``--no_swanlab``\ （或短选项 ``--nsl``\ ）关闭 SwanLab，只保留本地 CSV 记录：

.. code:: bash

   python train.py -m GIKT -d assistments09 --no_swanlab

本地记录始终生效，数据保存在 ``runs/`` 目录下，不受此开关影响。

追踪的指标
~~~~~~~~~~

SwanLab 中指标按 ``阶段/指标名`` 的形式组织：

===================== ==============================
SwanLab 指标路径      含义
===================== ==============================
``Train/LOSS-epoch``  训练集每 epoch 的平均损失
``Val/LOSS-epoch``    验证集每 epoch 的平均损失
``Val/ACC-epoch``     验证集准确率
``Val/AUC-epoch``     验证集 AUC
``Val/RMSE-epoch``    验证集 RMSE
``ES/Best``           早停当前最佳指标值
``ES/Num_Bad_Epochs`` 连续未改善的 epoch 数
``Test/ACC-epoch``    测试集准确率（训练结束后写入）
``Test/AUC-epoch``    测试集 AUC（训练结束后写入）
``Test/RMSE-epoch``   测试集 RMSE（训练结束后写入）
===================== ==============================

多阶段训练（\ ``MultiTrainer``\ ）时，指标名会附加阶段前缀，如 ``STAGE1/Train/LOSS-epoch``\ 。

飞书告警
~~~~~~~~

SwanLab 支持通过飞书机器人推送训练异常通知。在 ``.env`` 中添加：

.. code:: bash

   LARK_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxx
   LARK_SECRET=your-secret

设置后，训练异常（如梯度爆炸）会通过飞书机器人实时通知。\ ``LARK_SECRET`` 为签名密钥，与 ``LARK_WEBHOOK_URL`` 配对使用。

本地指标文件
------------

本地 CSV 记录始终开启，保存在实验目录 ``runs/normal/<run_id>/`` 下：

+------------------------+----------------------------------------------------+
| 文件                   | 内容                                               |
+========================+====================================================+
| ``metrics_train.csv``  | 每 epoch 的训练指标（epoch, loss, acc, auc, rmse） |
+------------------------+----------------------------------------------------+
| ``metrics_val.csv``    | 每 epoch 的验证指标（epoch, loss, acc, auc, rmse） |
+------------------------+----------------------------------------------------+
| ``metrics_test.csv``   | 测试集指标                                         |
+------------------------+----------------------------------------------------+
| ``early_stopping.csv`` | 早停轨迹（best_score, num_bad_epochs, best\_*）    |
+------------------------+----------------------------------------------------+
| ``metrics_final.csv``  | 最终摘要指标                                       |
+------------------------+----------------------------------------------------+

如果启用了 ``--log_batch_metrics``\ ，还会额外生成 ``batch_metrics_train.csv`` 和 ``batch_metrics_val.csv``\ ，记录每个 batch 的损失值，用于更细粒度的训练诊断。

指标计算细则
------------

准确率（ACC）
~~~~~~~~~~~~~

二元预测 ``y_predict`` 与真实标签 ``y_label`` 比较，使用 sklearn 的 ``accuracy_score``\ ：

.. code:: python

   from sklearn.metrics import accuracy_score

   acc = accuracy_score(y_label, y_predict)

``y_predict`` 通常由 ``(y_score >= 0.5).float`` 得到，阈值为 0.5。

AUC
~~~

排序分数 ``y_score`` 与真实标签 ``y_label`` 比较，使用 sklearn 的 ``roc_auc_score``\ ：

.. code:: python

   from sklearn.metrics import roc_auc_score

   auc = roc_auc_score(y_label, y_score)

当数据中只有单一类别（全 0 或全 1）时，\ ``roc_auc_score`` 会抛出异常，此时 AUC 记录为 0.0。

RMSE
~~~~

预测概率 ``y_prob`` 与真实标签 ``y_label`` 比较，使用 sklearn 的 ``root_mean_squared_error``\ ：

.. code:: python

   from sklearn.metrics import root_mean_squared_error

   rmse = root_mean_squared_error(y_label, y_prob)

``y_prob`` 通常由 ``torch.sigmoid(y_score)`` 得到，将排序分数映射到 [0,1] 区间。

.. note::

   这三个指标的计算都在 ``MetricsAccumulator.compute`` 中完成（\ ``utils/training/metrics.py``\ ）。指标值在验证集上按 batch 累积全部预测后一次性计算，而非取各 batch 指标的平均值，避免了小 batch 引起的指标偏差。

