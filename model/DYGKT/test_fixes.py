#!/usr/bin/env python
"""快速测试 DYGKT 修复后的功能"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from model.DYGKT.DYGKT_data import DYGKTDataset

print("=" * 60)
print("测试 DYGKT 修复")
print("=" * 60)

# 测试1: DYGKTDataset 性能优化
print("\n1. 测试 DYGKTDataset 数据转换...")
user_seqs = [[0] * 10 for _ in range(5)]
question_seqs = [[i for i in range(10)] for _ in range(5)]
responses = [[0, 1] * 5 for _ in range(5)]
time_seqs = [[i * 3600.0 for i in range(10)] for _ in range(5)]
masks = [[1] * 10 for _ in range(5)]

dataset = DYGKTDataset(user_seqs, question_seqs, responses, time_seqs, masks)
batch = dataset[0]

print(f"✅ Batch 类型: {[type(x) for x in batch]}")
print(f"✅ Batch 形状: {[x.shape for x in batch]}")
print(f"✅ 所有元素都是 Tensor: {all(isinstance(x, torch.Tensor) for x in batch)}")

# 测试2: 模型维度处理
print("\n2. 测试模型维度处理...")
from model.DYGKT.DYGKT_model import DYGKT

class Args:
    embedding_dim = 64
    hidden_dim = 64
    dim_time = 32
    dropout = 0.3

data_metadata = {
    "num_questions": 100,
    "num_users": 50,
}

model = DYGKT(Args(), data_metadata)
model.eval()

# 测试不同维度的输入
test_cases = [
    ("2D input", torch.randint(0, 50, (4, 10))),
    ("3D input (batched)", torch.randint(0, 50, (1, 4, 10))),
]

for name, user_seq in test_cases:
    try:
        question_seq = torch.randint(0, 100, user_seq.shape)
        response = torch.randint(0, 2, user_seq.shape)
        time_seq = torch.arange(user_seq.numel(), dtype=torch.float).view(user_seq.shape)
        mask = torch.ones(user_seq.shape, dtype=torch.bool)
        
        with torch.no_grad():
            logits = model(user_seq, response, mask, question_seq, time_seq)
        
        print(f"✅ {name}: 输入形状 {user_seq.shape} -> 输出形状 {logits.shape}")
    except Exception as e:
        print(f"❌ {name}: {e}")

# 测试3: GPU 自动检测
print("\n3. 测试 GPU 自动检测...")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"推荐设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")

print("\n" + "=" * 60)
print("✅ 所有测试完成!")
print("=" * 60)
