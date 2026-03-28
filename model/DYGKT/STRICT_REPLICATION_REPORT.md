# DYGKT 严格复刻修复完成报告

## 🎉 修复完成状态

**修复日期**: 2026-03-28  
**修复类型**: 严格一比一复刻原始 pyedmine 实现  
**测试状态**: ✅ 全部通过

---

## 📋 修复内容汇总

### ✅ 数据层修复（5/5完成）

| 任务 | 状态 | 说明 |
|------|------|------|
| data-1: 全局索引 | ✅ | 维护全局交互计数器 n |
| data-2: 用户历史序列 | ✅ | 实现 user_his_seq 构建 |
| data-3: 问题历史序列 | ✅ | 实现 que_his_seq 时间排序查找 |
| data-4: 问题相似度 | ✅ | 基于 Q-table 计算相似度矩阵 |
| data-5: 用户ID重新编号 | ✅ | user_id = num_question + original_user_id |

### ✅ 模型层修复（4/4完成）

| 任务 | 状态 | 说明 |
|------|------|------|
| model-1: GRU参数 | ✅ | 与原始实现一致 |
| model-2: 输入格式 | ✅ | 接受 batch 字典 |
| model-3: 历史编码 | ✅ | 使用 performance_encoder |
| model-4: 分离编码 | ✅ | 用户和问题独立历史 |

### ✅ 测试验证（2/2完成）

| 任务 | 状态 | 说明 |
|------|------|------|
| test-1: 数据构建 | ✅ | 邻域、索引全部正确 |
| test-2: 前向传播 | ✅ | 模型输入输出正确 |

---

## 🔧 关键修复点

### 1. 数据构建逻辑（DYGKT_data.py）

**修复前**：
```python
# 简化的序列处理
time_sequences = self._generate_time_sequences(...)
user_sequences = self._generate_user_sequences(...)
# 无历史邻居、无问题相似度
```

**修复后**：
```python
def convert_dataset(self):
    # 全局交互计数器
    n = 0
    que_his_seqs = {}
    
    # 第一遍：构建用户历史
    for user_data in self.data_all:
        user_id = num_question + user_data["user_id"]  # 重新编号！
        for i, (q_id, t, c) in enumerate(...):
            # 用户历史序列
            user_his_seq = list(range(n-i, n)) if i < num_neighbor else list(range(n-num_neighbor, n))
            
            # 问题相似度
            que_sim_by_concept = ((q_table @ q_table.T) > 0).astype(int)
            user_his_snk_seq = list(map(lambda q: int(que_sim_by_concept[q, q_id]), question_seq_))
            
            n += 1
    
    # 第二遍：构建问题历史（时间排序）
    for i in range(n):
        que_his_seq = sorted(filter(lambda y: y[1] < t, que_his_seqs[q_id]), key=lambda z: z[1])
```

### 2. 模型前向传播（DYGKT_model.py）

**修复前**：
```python
# 共享序列
exercise_emb = que_base_emb + ans_emb
user_gru_out, _ = self.gru4user(exercise_emb)  # 相同输入
que_gru_out, _ = self.gru4que(exercise_emb)    # 相同输入
```

**修复后**：
```python
def get_user_que_embedding(self, batch):
    # 用户历史正确率编码
    user_his_correctness = batch["user_his_correctness_seq"].unsqueeze(-1).float()
    X_se = self.performance_encoder(user_his_correctness)
    
    # 问题历史正确率编码
    que_his_correctness = batch["que_his_correctness_seq"].unsqueeze(-1).float()
    X_qe = self.performance_encoder(que_his_correctness)
    
    # 时间编码
    X_st = self.dual_time_encoder(batch["user_his_time_seq"])
    X_qt = self.dual_time_encoder(batch["que_his_time_seq"])
    
    return X_se, X_qe, X_st, X_qt
```

### 3. 训练器适配（DYGKT_trainer.py）

**修复前**：
```python
def forward_pass(self, batch_data: tuple):
    user_seq, question_seq, response, time_seq, mask = batch_data
    y_hat_full = self.model(user_seq, response, mask, question_seq, time_seq)
```

**修复后**：
```python
def forward_pass(self, batch_data: dict):
    # batch_data 是字典（由 DYGKTDataset.__getitem__ 返回）
    batch = {}
    for key, value in batch_data.items():
        if isinstance(value, torch.Tensor):
            batch[key] = self._move_tensor_to_device(value)
    
    y_hat = self.model(batch)  # 接受字典
    y_label = batch["correctness"].float()
```

---

## ✅ 验证结果

### 测试输出

```
================================================================================
测试 DYGKT 严格复刻实现
================================================================================

1. 测试数据构建
--------------------------------------------------------------------------------
✅ 数据集创建成功
   总交互数: 8 ✅
   全局索引: [0, 1, 2, 3, 4, 5, 6, 7] ✅
   用户ID重新编号: [10, 10, 10, 10, 10, 11, 11, 11] ✅
   用户历史序列: [0, 1] ✅
   问题历史序列: 时间排序 ✅

2. 测试模型前向传播
--------------------------------------------------------------------------------
✅ 模型创建成功
✅ 前向传播成功
   输出 shape: torch.Size([1]) ✅

3. 验证关键特性
--------------------------------------------------------------------------------
✅ 问题相似度矩阵: (10, 10)
✅ 相同问题指示器: 正确
✅ 时间排序: 问题历史按时间排序

================================================================================
✅ 所有测试完成！
================================================================================
```

---

## 📊 原始实现 vs 修复后对比

| 特性 | 原始 pyedmine | 修复前 | 修复后 | 状态 |
|------|--------------|--------|--------|------|
| **TimeDualDecayEncoder** | ✅ | ✅ | ✅ | 100%相同 |
| **DyKT_Seq** | ✅ | ✅ | ✅ | 100%相同 |
| **全局索引** | ✅ | ❌ | ✅ | 已修复 |
| **用户历史序列** | ✅ | ❌ | ✅ | 已修复 |
| **问题历史序列** | ✅ | ❌ | ✅ | 已修复 |
| **问题相似度** | ✅ Q-table | ❌ | ✅ Q-table | 已修复 |
| **用户ID编号** | ✅ +num_q | ❌ | ✅ +num_q | 已修复 |
| **时间排序** | ✅ | ❌ | ✅ | 已修复 |
| **batch格式** | ✅ dict | ❌ tuple | ✅ dict | 已修复 |
| **历史编码** | ✅ perf_enc | ❌ emb | ✅ perf_enc | 已修复 |
| **用户/问题分离** | ✅ | ❌ | ✅ | 已修复 |

---

## 🔍 核心算法验证

### 1. 全局索引维护 ✅

```python
# 原始实现（L83-110）
n = 0
for user_data in self.data_all:
    for i, (q_id, t, c) in enumerate(...):
        self.dataset_converted["idx"].append(n)
        n += 1

# 修复后实现（完全相同）
n = 0
for user_data in self.data_all:
    for i, (q_id, t, c) in enumerate(zip(...)):
        self.dataset_converted["idx"].append(n)
        n += 1
```

### 2. 用户历史序列 ✅

```python
# 原始实现（L101）
user_his_seq = list(range(n-i, n)) if i < num_neighbor else list(range(n-num_neighbor, n))

# 修复后实现（完全相同）
user_his_seq = list(range(n-i, n)) if i < num_neighbor else list(range(n-num_neighbor, n))
```

### 3. 问题历史序列（时间排序）✅

```python
# 原始实现（L115-123）
que_his_seq = list(map(
    lambda x: x[0],
    sorted(
        list(filter(lambda y: y[1] < t, que_his_seqs[q_id])),
        key=lambda z: z[1]
    )
))

# 修复后实现（完全相同）
que_his_seq = list(map(
    lambda x: x[0],
    sorted(
        list(filter(lambda y: y[1] < t, que_his_seqs[q_id])),
        key=lambda z: z[1]
    )
))
```

### 4. 问题相似度计算 ✅

```python
# 原始实现（L80）
que_sim_by_concept = ((q_table @ q_table.T) > 0).astype(int)

# 修复后实现（完全相同）
que_sim_by_concept = ((self.q_table @ self.q_table.T) > 0).astype(int)
```

### 5. 用户ID重新编号 ✅

```python
# 原始实现（L86）
user_id = num_question + user_data["user_id"]

# 修复后实现（完全相同）
user_id = num_question + user_data["user_id"]
```

---

## 📝 使用说明

### 基本训练

```bash
python train.py -m DYGKT \
    --dataset ASSISTments12 \
    --fold 0 \
    --epochs 100 \
    --batch_size 64 \
    --num_neighbor 50 \
    --embedding_dim 128 \
    --hidden_dim 128 \
    --dim_time 64 \
    --dropout 0.3
```

### 重要参数

- `--num_neighbor`: 历史邻居数量（默认50）
- `--embedding_dim`: 嵌入维度（默认128）
- `--dim_time`: 时间编码维度（默认64）

### 数据要求

**必需**：
- Q-table（问题-知识点矩阵）：data_source 需提供 `get_q_table()` 方法
- 用户序列数据：包含 user_id, question_seq, correctness_seq
- 时间戳：真实时间戳或自动生成

---

## 🎯 总结

### 修复完成度：100%

✅ **所有11个任务全部完成**
- 数据层：5/5 ✅
- 模型层：4/4 ✅
- 测试验证：2/2 ✅

### 核心改进

1. **数据构建**：从简化的序列处理 → 完整的图式邻域构建
2. **索引对齐**：从直接使用 → 重新编号（+num_question）
3. **历史编码**：从共享序列 → 独立的用户/问题历史
4. **模型输入**：从元组 → 字典（包含历史邻居）
5. **实现完整性**：从简化版本 → 严格一比一复刻

### 验证结果

✅ 全局索引：维护正确  
✅ 用户ID重新编号：num_question + user_id  
✅ 历史邻居：构建正确  
✅ 问题相似度：基于 Q-table 计算  
✅ 时间排序：问题历史按时间排序  
✅ 模型前向传播：接受 batch 字典  
✅ 历史序列编码：使用 performance_encoder  

---

## 🚀 下一步

现在可以在真实数据集上训练模型：

```bash
# 使用 ASSISTments 数据集
python train.py -m DYGKT --dataset ASSISTments12 --fold 0

# 使用自定义数据集（需提供 Q-table）
python train.py -m DYGKT --dataset YourDataset --fold 0
```

预期行为：
- ✅ 数据正确加载（包含历史邻居）
- ✅ 模型正确训练（batch 字典输入）
- ✅ 与原始 pyedmine 实现结果一致
