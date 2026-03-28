#!/usr/bin/env python
"""DYGKT 模型单元测试。

测试模型的基本功能是否正常。
"""

import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from model.DYGKT import DYGKT


def test_model_forward():
    """测试模型前向传播"""
    print("Testing DYGKT forward pass...")
    
    # 创建模拟参数
    class Args:
        embedding_dim = 64
        hidden_dim = 64
        dim_time = 32
        dropout = 0.3
    
    # 创建模拟数据元数据
    data_metadata = {
        "num_questions": 100,
        "num_users": 50,
    }
    
    # 初始化模型
    model = DYGKT(Args(), data_metadata)
    model.eval()
    
    # 创建模拟输入
    batch_size = 4
    seq_len = 10
    
    user_seq = torch.randint(0, 50, (batch_size, seq_len))
    question_seq = torch.randint(0, 100, (batch_size, seq_len))
    response = torch.randint(0, 2, (batch_size, seq_len))
    time_seq = torch.arange(batch_size * seq_len, dtype=torch.float).view(batch_size, seq_len) * 3600
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # 前向传播
    with torch.no_grad():
        logits = model(user_seq, response, mask, question_seq, time_seq)
    
    # 验证输出形状
    assert logits.shape == (batch_size, seq_len), f"Expected shape {(batch_size, seq_len)}, got {logits.shape}"
    
    print(f"✅ Forward pass successful! Output shape: {logits.shape}")
    print(f"   Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    
    # 测试 return_states
    logits, user_emb, que_emb = model(user_seq, response, mask, question_seq, time_seq, return_states=True)
    print(f"✅ Forward pass with return_states successful!")
    print(f"   User embedding shape: {user_emb.shape}")
    print(f"   Question embedding shape: {que_emb.shape}")


def test_time_encoder():
    """测试时间编码器"""
    print("\nTesting TimeDualDecayEncoder...")
    
    from model.DYGKT.DYGKT_model import TimeDualDecayEncoder
    
    encoder = TimeDualDecayEncoder(dim_time=32)
    
    # 创建模拟时间戳
    batch_size = 4
    seq_len = 10
    timestamps = torch.arange(batch_size * seq_len, dtype=torch.float).view(batch_size, seq_len) * 3600
    
    # 编码
    time_emb = encoder(timestamps)
    
    # 验证输出形状
    assert time_emb.shape == (batch_size, seq_len, 32), f"Expected shape {(batch_size, seq_len, 32)}, got {time_emb.shape}"
    
    print(f"✅ Time encoder successful! Output shape: {time_emb.shape}")
    print(f"   Time embedding range: [{time_emb.min():.4f}, {time_emb.max():.4f}]")


def test_parameter_count():
    """测试模型参数数量"""
    print("\nTesting parameter count...")
    
    class Args:
        embedding_dim = 128
        hidden_dim = 128
        dim_time = 64
        dropout = 0.3
    
    data_metadata = {
        "num_questions": 1000,
        "num_users": 500,
    }
    
    model = DYGKT(Args(), data_metadata)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Parameter count:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("DYGKT Model Unit Tests")
    print("=" * 60)
    
    try:
        test_model_forward()
        test_time_encoder()
        test_parameter_count()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
