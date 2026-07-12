# 配置系统

UniKT 的命令行参数由多个预定义配置类统一管理，同时支持每个模型通过装饰器注册自己的专属参数。本页介绍如何查看、使用和扩展这些参数。

## 查看可用参数

训练脚本支持分层参数解析：先加载通用参数，再按 ``-m`` 指定的模型注入模型专属参数。在模型名后加 ``-h`` 即可看到完整的参数列表：

```bash
python train.py -m GIKT -h
```

输出会按参数组分类，依次显示 General Parameters、Data Parameters、Early Stopping Parameters、Compile Parameters 和 GIKT Parameters。直接运行 ``python train.py -h`` 则只显示通用参数组。

## 通用参数组

这些参数在 ``utils/config/param_config.py`` 中定义为 ``BaseParamConfig`` 的子类，由 ``train.py`` 无条件加载。

(generalparams--运行控制)=

### GeneralParams — 运行控制

控制设备、日志、种子等基础行为：

```bash
python train.py -m GIKT -d assistments09 \
 --seed 123 \
 --device cuda \
 --log_dir ./my_experiments \
 --no_swanlab
```

``--seed`` 用于复现实验，设置为不同值可以测试模型对随机性的敏感度。``--device`` 默认自动检测 GPU 可用性，但在多 GPU 机器上建议显式指定（如 ``cuda:0``）。``--no_swanlab`` 关闭 SwanLab 记录，适合本地快速迭代时减少网络开销。

**注意：** 确定性算法默认开启，会调用 ``torch.use_deterministic_algorithms`` 确保可复现性。若某个算子不支持 deterministic 模式导致训练失败，可以用 ``--no_deterministic`` 关闭。

(dataparams--数据加载)=

### DataParams — 数据加载

```bash
python train.py -m GIKT -d assistments09 \
 --fold 2 \
 --kfold 5 \
 --max_seq_len 100 \
 --data_base_path /data/datasets
```

``--dataset`` / ``-d`` 是必选参数，可选值包括 ``algebra2005``、``algebra2006``、``assistments09``、``assistments12``、``assistments15``、``assistments17``、``bridge2006``、``ednet_kt1``、``junyi2015``、``nips2020_t34``、``slepemapy``，共 11 个数据集。

``--fold`` 指定当前使用第几折（0-indexed），与 ``--kfold`` 配合完成 K 折交叉验证。例如 ``--kfold 5 --fold 0`` 表示 5 折中的第 1 折。``--test_ratio`` 控制测试集用户占比。测试集用户不参与训练和验证，与 ``--kfold`` 同时生效——``test_ratio`` 决定测试集比例，``kfold`` 对剩余数据划分训练/验证折。

``--max_seq_len`` 会截断过长的交互序列，减小显存占用。数据集 ``assistments09`` 序列通常较短，设 200 足够；``ednet_kt1`` 序列较长，可适当增大至 500。

(earlystoppingparams--早停控制)=

### EarlyStoppingParams — 早停控制

```bash
python train.py -m GIKT -d assistments09 \
 --es_patience 10 \
 --es_monitor auc \
 --es_mode max \
 --es_min_delta 0.001
```

``--es_monitor`` 指定监控指标，在验证集上计算。分类任务通常用 ``auc``（AUC 越高越好，``--es_mode max``），回归任务用 ``rmse``（越低越好，``--es_mode min``）。``--es_patience`` 设为 0 可以完全禁用早停。

``--es_min_delta`` 决定"有改进"的阈值：只有当前指标与历史最优的差值超过此值，才视为有效改进。面对波动较大的训练曲线时，适当增大此值（如 0.005）可以避免因噪声提前停止。

## 模型特定参数

每个模型通过 ``@register_model_params`` 装饰器将自己独有的超参数暴露给 CLI。这套机制由三步组成：定义参数类 → 注册到全局表 → CLI 自动注入。

### 如何工作

以 GIKT 为例，在 ``model/GIKT/GIKT_trainer.py`` 中：

```python
from utils.config import BaseParamConfig, register_model_params


@register_model_params("GIKT")
class GIKTModelParams(BaseParamConfig):
 def define_params(self) -> tuple[str, dict]:
 return "GIKT Parameters", {
 "embedding_dim": {
 "type": int,
 "default": 100,
 "short": "ed",
 "help": "Embedding dimension (default: 100)",
 },
 "n_hop": {
 "type": int,
 "default": 3,
 "short": "nh",
 "help": "Number of GNN aggregation hops (default: 3)",
 },
 "batch_size": {
 "type": int,
 "default": 32,
 "short": "bs",
 "help": "Batch size (default: 32)",
 },
 "learning_rate": {
 "type": float,
 "default": 0.001,
 "short": "lr",
 "help": "Learning rate (default: 0.001)",
 },
 }
```

``define_params`` 返回一个 ``(组名, 参数字典)`` 的元组。参数字典的键是 CLI 参数名，值为配置字典，支持以下字段：

- ``type``：参数类型，``int`` / ``float`` / ``str`` / ``bool``。``bool`` 类型会自动转换为 ``store_true`` 或 ``store_false`` action
- ``default``：默认值
- ``short``：短选项名（如 ``"ed"`` → ``-ed``）
- ``help``：帮助文本
- ``choices``：可选值列表（用于 ``str`` 类型）

注册到全局表 ``PARAM_CONFIGS`` 后，``train.py`` 在解析参数时会调用 ``get_model_params("GIKT").add_args(parser)``，将这些参数注入到 ArgumentParser 中。

### 使用模型参数

注册后即可像通用参数一样在命令行中使用：

```bash
python train.py -m GIKT -d assistments09 \
 --embedding_dim 128 \
 --n_hop 4 \
 --batch_size 64 \
 --learning_rate 0.0005
```

GIKT 特有的参数如 ``--skill_neighbor_num``（每跳技能邻居采样数）、``--hist_neighbor_num``（同技能历史邻居数 M）只在使用 GIKT 模型时出现，切换到其他模型会显示不同的专属参数集。

### 为新模型添加参数

如果你在添加一个新模型，按三步完成参数集成：

**1. 定义参数类**（在 ``model/YourModel/YourModel_trainer.py``）：

```python
from utils.config import BaseParamConfig, register_model_params


@register_model_params("YourModel")
class YourModelParams(BaseParamConfig):
 def define_params(self) -> tuple[str, dict]:
 return "YourModel Parameters", {
 "hidden_dim": {"type": int, "default": 256, "help": "Hidden dimension"},
 "n_layers": {"type": int, "default": 2, "help": "Number of layers"},
 "dropout": {"type": float, "default": 0.1, "help": "Dropout rate"},
 }
```

**2. 注册**：装饰器 ``@register_model_params("YourModel")`` 已完成注册。**注意**：装饰器参数必须是字符串字面量，不能使用变量或表达式，否则 AST 静态发现（``utils/core/discovery.py``）无法识别。

**3. 配置使用**：完成定义后无需任何额外步骤，``train.py`` 会自动发现并加载。最终用法与其他模型一致：

```bash
python train.py -m YourModel -d assistments09 --hidden_dim 512
```

**注意**：若无法通过 ``-h`` 看到新参数，检查装饰器参数是否为字符串字面量，且文件位于 ``model/`` 目录下（静态发现只扫描该目录）。

## SwanLab 环境变量

SwanLab 作为实验追踪后端通过环境变量配置，而非命令行参数。相关变量在 ``utils/training/metric_logger.py`` 的 ``SwanLabMetricLogger.init_run`` 中读取：

| 变量 | 说明 |
| --- | --- |
| ``SWANLAB_WORKSPACE`` | SwanLab 工作空间名称，不设置则使用默认空间 |
| ``SWANLAB_MODE`` | ``cloud``（上传到云端）或 ``local``（仅本地记录，默认） |
| ``LARK_WEBHOOK_URL`` | 飞书机器人 Webhook 地址，设置后训练异常自动推送通知 |
| ``LARK_SECRET`` | 飞书机器人签名密钥，与 ``LARK_WEBHOOK_URL`` 配对使用 |


建议在项目根目录创建 ``.env`` 文件（已加入 ``.gitignore``）：

```bash
# .env
SWANLAB_WORKSPACE=my-lab
SWANLAB_MODE=cloud
LARK_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxx
LARK_SECRET=your-secret
```

代码中通过 ``dotenv.load_dotenv`` 自动加载。**注意**：SwanLab 需要在首次使用前完成登录认证，在终端执行 ``swanlab login`` 获取 API 密钥。``--no_swanlab`` 或 ``--nsl`` 标志可以完全关闭 SwanLab 后端，只保留本地 CSV 记录。
