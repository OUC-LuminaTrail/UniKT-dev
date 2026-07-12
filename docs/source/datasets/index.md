# 支持的数据集

UniKT 支持 11 个知识追踪数据集，提供自动下载和预处理功能。

## 支持的数据集

| 数据集 | 来源 | 学生数 | 交互数 | 技能数 |
| --- | --- | --- | --- | --- |
| ASSISTments 2009 | [ASSISTmentsData](https://www.etrialstestbed.org/data-sets) | 4,151 | 324,975 | 110 |
| ASSISTments 2012 | [ASSISTmentsData](https://www.etrialstestbed.org/data-sets) | 26,688 | 2,535,057 | 265 |
| ASSISTments 2015 | [ASSISTmentsData](https://www.etrialstestbed.org/data-sets) | 19,840 | 683,601 | 100 |
| ASSISTments 2017 | [ASSISTmentsData](https://www.etrialstestbed.org/data-sets) | 1,709 | 942,816 | 102 |
| EdNet-KT1 | [GitHub](https://github.com/riiid/ednet) | 784,309 | 131,441,538 | — |
| Junyi 2015 | [Kaggle](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset) | 247,606 | 25,925,922 | 179 |
| Slepemapy | [SLEP](https://www.fi.muni.cz/adaptivelearning/) | 91,001 | 9,164,780 | 1,457 |


## 数据格式

所有数据集都会被处理为标准化的 Parquet 文件：

```
data/{dataset}/
├── {dataset}_question.parquet       # 题目特征
├── {dataset}_sequence.parquet       # 用户交互序列
├── {dataset}_split_question_sequence.parquet  # 按折划分的序列
├── {dataset}_split_skill_sequence.parquet     # 按折划分的技能序列
└── metadata.json                    # 处理元数据
```

### 序列字段

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| ``user`` | ``Int32`` | 用户标识符（重映射） |
| ``question`` | ``Int32`` | 题目标识符 |
| ``skill`` | ``Int32`` | 技能/概念标识符 |
| ``label`` | ``Int8`` | 回答正确性（0 或 1） |
| ``timestamp`` | ``Int64`` | Unix 时间戳（毫秒） |
| ``fold`` | ``Int32`` | K 折标签（-1 = 测试集） |
| ``seq_pos`` | ``Int32`` | 序列中的位置 |


## 使用方法

```bash
# 下载
python data_process.py download -d assistments09

# 使用默认设置处理
python data_process.py process -d assistments09

# 使用自定义采样处理
python data_process.py process -d assistments09 \
    --min_seq_len 3 \
    --max_seq_len 200 \
    --kfold 5 \
    --seed 42
```

详细流程文档请参阅[数据预处理](../user-guide/data-processing.md)。

```{toctree}
:maxdepth: 1

analysis/index
```
