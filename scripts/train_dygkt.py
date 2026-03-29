#!/usr/bin/env python
"""DYGKT 模型训练脚本示例。

使用方法:
    python scripts/train_dygkt.py --dataset ASSISTments12 --fold 0
    python scripts/train_dygkt.py --dataset ASSISTments12 --fold 0 --epochs 150 --batch_size 64
"""

import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse

from utils.config import DataParams, EarlyStoppingParams, GeneralParams, get_model_params
from utils.experiment_manager import ExperimentManager, ExperimentType
from utils.data_process import get_data_source

from model.DYGKT import DYGKTTrainer

def get_args_parser():
    parser = argparse.ArgumentParser(description="DYGKT Training Script")
    GeneralParams.add_args(parser)
    DataParams.add_args(parser)
    EarlyStoppingParams.add_args(parser)
    
    model_params_cls = get_model_params("DYGKT")
    if model_params_cls:
        model_params_cls.add_args(parser)
        
    return parser

def main():
    """主函数：训练 DYGKT 模型"""
    
    # 1. 解析命令行参数
    parser = get_args_parser()
    args = parser.parse_args()
    
    # 设置默认模型为 DYGKT
    args.model = "DYGKT"
    
    # 2. 创建实验管理器
    exp_manager = ExperimentManager.from_args(args, ExperimentType.NORMAL)
    
    # 3. 加载数据源
    print(f"Loading dataset: {args.dataset}")
    data_src = get_data_source(dataset_name=args.dataset, args=args)
    
    # 4. 创建训练器
    print("Initializing DYGKT trainer...")
    trainer = DYGKTTrainer(
        args=args,
        data_src=data_src,
        exp_manager=exp_manager,
    )
    
    # 5. 开始训练
    print(f"Starting training for {args.epochs} epochs...")
    trainer.run()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
