# 快速上手

在几分钟内运行你的第一个 UniKT 实验。

## 环境配置

```bash
# 安装 pixi
curl -fsSL https://pixi.sh/install.sh | bash

# 进入 GPU 环境
pixi shell
```

:::{tip}
CPU、Mamba、DHG 等环境或手动 Conda 安装方式请参考[环境配置](setup.md)。
:::

(5-分钟示例)=

## 5 分钟示例

```bash
# 1. 下载数据集
python data_process.py download -d assistments09

# 2. 处理数据
python data_process.py process -d assistments09

# 3. 训练模型
python train.py -m GIKT -d assistments09

# 4. （可选）登录 SwanLab 以跟踪实验
swanlab login
```

## 工作流程概览

```{mermaid}
flowchart LR
 A[下载数据] --> B[处理数据]
 B --> C[训练模型]
 C --> D[评估]
```

典型工作流程：

1. [下载](../user-guide/data-processing.md) 原始数据集
2. [处理](../user-guide/data-processing.md) 数据为标准化格式，生成 K 折划分
3. [训练](../user-guide/training-evaluation.md) 知识追踪模型
4. [评估](../user-guide/evaluation.md) 模型性能

## 输出

实验输出保存在 ``runs/normal/<model>_<dataset>_<timestamp>/``：

```
runs/normal/GIKT_assistments09_20240403-120000_fold0_bs128/
├── best_model.pth # 最佳模型检查点
├── last_checkpoint.pth # 最后检查点
├── hyperparameters.json # 超参数配置
└── metrics_train.csv # 训练指标
```

## 下一步

- [配置系统](../advanced/config.md) — 了解命令行参数
- [模型评估](../user-guide/evaluation.md) — 解读评估指标
- [添加新模型](../user-guide/new-model.md) — 接入自己的模型
