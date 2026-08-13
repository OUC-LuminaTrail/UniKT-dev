# 贡献指南

欢迎向 UniKT 贡献代码。本文档包含代码规范、数据安全要求、PR 流程和模型接入清单。

## 代码格式化

提交前运行：

```bash
ruff format . # 自动格式化
ruff check . # lint 检查
ruff check --fix . # 自动修复 lint
```

## 命名约定

| 类型 | 规则 | 示例 |
| --- | --- | --- |
| 模块 | 小写 + 下划线 | ``base_trainer.py`` |
| 类 | PascalCase | ``BaseTrainer`` |
| 方法/函数 | 小写 + 下划线 | ``forward_pass`` |
| 常量 | 大写 + 下划线 | ``TRAINERS`` |
| 私有属性 | 前缀 ``_`` | ``_registry`` |


## 注释规范

### 文件级

```python
"""训练回调系统

提供训练过程中的回调机制，包括早停、检查点、内存管理等。
"""
```

仅描述模块职责，不含作者、日期等元信息。

### 类级

```python
class MetricsAccumulator:
    """指标累积器。

    职责：
    1. 收集 batch 级别的预测和标签
    2. 计算 epoch 级别的聚合指标

    指标的持久化记录由 MetricLogger 负责，本类只负责计算。
    """
```

### 方法级

```python
def split_kfold_data(self, *arrays, fold_idx: int):
    """根据 K 折交叉验证的 fold 索引划分数据。

    参数:
    *arrays: 任意个数、首维为用户数的数组或张量
    fold_idx: 当前 fold 索引（关键字参数，必填）

    返回:
    train_data: 训练集切片
    val_data: 验证集切片
    test_data: 测试集切片
    """
```

使用中文描述，``r"""..."""`` 原始字符串避免反斜杠转义。

### 行内注释

注释解释「为什么」而非「做什么」：

```python
# 好：解释设计意图
# GIKT 在 max_step = full_seq_len - 1 上运行（next-item 约定）
model_seq_len = full_seq_len - 1

# 避免：复述代码行为
result = a + b  # 将 a 和 b 相加 ← 无意义
```

## 类型注解

- 公开函数和方法必须标注参数类型和返回值类型
- 抽象方法必须在 docstring 中说明子类实现要求

## 预测位提取

知识追踪的核心约束是**时序因果性**：``y_hat[t]`` 只能依赖 ``q[0:t+1]`` 和 ``r[0:t]``，绝不能包含 ``r[t]``。

UniKT 通过 ``BaseTrainer._extract_valid_predictions`` 统一处理对齐：

```python
# 默认 next-item：y_hat[:,:-1] 预测 response[:,1:]
y_hat, y_label, _ = self._extract_valid_predictions(y_hat_full, response, mask)

# same-position：y_hat[:,1:] 预测 response[:,1:]
y_hat, y_label, _ = self._extract_valid_predictions(
    y_hat_full, response, mask, same_position=True
)
```

### 检查清单

实现 ``forward_pass`` 前逐项确认：

- ☐ ``y_hat[t]`` 的计算路径中不包含 ``response[t]``
- ☐ 输出维度与对齐方式一致（next-item 输出 ``[B, S-1]``，same-position 输出 ``[B, S]`` 对齐后左移）
- ☐ 若模型内部自建移位，确认方向正确（``r[t-1]`` 而非 ``r[t+1]``）
- ☐ ``response[:, 0]`` 为占位符，不参与损失计算（由 mask 控制）
- ☐ Windowlate 评估时确认占位值不影响预测

### 设备管理

不要手动调用 ``.cuda``——使用 ``_move_tensor_to_device`` 以兼容 CPU 模式：

```python
sequence = self._move_tensor_to_device(batch_data["sequence"])
response = self._move_tensor_to_device(batch_data["response"])
```

## 注册约定

### 装饰器参数必须使用字符串字面量

```python
# 正确
@register_trainer("GIKT")

# 错误：变量不会被识别
NAME = "GIKT"
@register_trainer(NAME)
```

### 注册名必须唯一

同一注册名下不能有两个不同的类使用相同名字，否则抛出 ``KeyError``。

### 注册名与目录名保持一致

```
model/GIKT/
├── GIKT_trainer.py → @register_trainer("GIKT")
├── GIKT_data.py → @register_model_params("GIKT")
├── GIKT_model.py → GIKT(torch.nn.Module)
└── GIKT_analyzer.py → @register_analyzer("GIKT")（可选）
```

### 添加新注册组件

以 MetricLogger 为例：

1. 继承抽象基类，实现所有抽象方法
2. 用 ``@register_metric_logger("name")`` 装饰
3. 第三方库在方法内惰性 ``import``（不在模块顶层），缺少该库时 ``import utils.training`` 仍正常

```python
from utils.core import register_metric_logger
from utils.training import MetricLogger

@register_metric_logger("wandb")
class WandbMetricLogger(MetricLogger):
    def init_run(self, *, log_dir, experiment_name, ...):
        import wandb
        wandb.init(...)
        self._initialized = True

    def log_metrics(self, *, phase, metrics, step, ...):
        if not self._initialized:
            return
        ...

    def finish(self):
        if self._initialized:
            import wandb
            wandb.finish()
```

## PR 流程

### 创建分支

```bash
git checkout -b feat/my-feature
```

分支命名：``feat/``（新功能）、``fix/``（修复）、``refactor/``（重构）、``perf/``（性能）、``docs/``（文档）。

### 提交消息

遵循 Conventional Commits：

```text
feat(grkt): add GRKT model with sparse CSR matmul
fix(data): correct clean_raw_data return type
refactor(training): rewrite MultiTrainer on BaseTrainer
```

### 提交 PR

- 标题与提交消息格式一致
- 描述中说明做了什么、为什么、如何测试
- 关联相关 Issue（如有）

## 模型提交清单

添加新模型需要以下文件：

```
model/<NAME>/
├── <NAME>_trainer.py # 训练器
├── <NAME>_data.py # 数据处理
├── <NAME>_model.py # 模型定义
└── <NAME>_analyzer.py # 案例分析器（可选）
```

### 必须实现

**训练器**：继承 ``BaseTrainer`` 或 ``MultiTrainer``，用 ``@register_trainer("NAME")`` 注册，实现 ``forward_pass(batch_data) → dict``。

**参数配置**：用 ``@register_model_params("NAME")`` 注册，继承 ``BaseParamConfig``，实现 ``define_params``。

**模型定义**：继承 ``torch.nn.Module``，``forward`` 适配训练器的调用约定。

### 验证步骤

1. ``TRAINERS.keys`` 包含新模型名
2. ``python train.py -m NAME -h`` 显示模型专属参数
3. 至少在一个数据集上完成完整训练流程
