"""
GIKT 模型 Optuna 超参数搜索脚本

使用 Optuna 进行 GIKT 模型的自动超参数搜索
"""

import argparse
import os
import logging

from model.GIKT.GIKT_trainer import GIKTTrainer
from utils.data_process import get_data_source
from utils.optuna_utils import (
    load_config_from_json,
    load_param_space_from_json,
    TrainerObjectiveWrapper,
    OptunaTunerBuilder,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="GIKT Optuna Hyperparameter Search")

    # Optuna配置
    optuna_group = parser.add_argument_group("Optuna Configuration")
    optuna_group.add_argument(
        "--optuna_config",
        type=str,
        default="./configs/optuna_config.json",
        help="Path to Optuna config JSON file",
    )
    optuna_group.add_argument(
        "--param_space",
        type=str,
        default="./configs/param_space_gikt.json",
        help="Path to parameter space JSON file",
    )
    optuna_group.add_argument(
        "--metric",
        type=str,
        choices=["auc", "acc", "rmse", "loss"],
        default="auc",
        help="Metric to optimize",
    )

    # 数据参数
    data_params = parser.add_argument_group("Data Parameters")
    data_params.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["assistments09", "assistments12", "assistments17", "ednet_kt1"],
        required=True,
        help="Select dataset to use",
    )
    data_params.add_argument(
        "--data_base_path",
        type=str,
        default="./data",
        help="Path to the data files",
    )
    data_params.add_argument(
        "--fold", type=int, default=0, help="Index of folds for K-Fold cross-validation"
    )

    # 基础训练参数
    train_params = parser.add_argument_group("Base Training Parameters")
    train_params.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs"
    )
    train_params.add_argument(
        "--checkpoint_path", type=str, default=None, help="Path to model checkpoints"
    )
    train_params.add_argument(
        "--es_patience",
        type=int,
        default=10,
        help="Patience for early stopping",
    )
    train_params.add_argument(
        "--es_monitor",
        type=str,
        choices=["auc", "acc", "rmse", "loss"],
        default="auc",
        help="Metric to monitor for early stopping",
    )
    train_params.add_argument(
        "--lr_decay",
        type=float,
        default=None,
        help="Learning rate decay factor applied via StepLR",
    )

    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./runs/optuna_search",
        help="Directory to save logs",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument(
        "--use_swanlab",
        action="store_true",
        default=True,
        help="Use SwanLab for tracking",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("GIKT Optuna Hyperparameter Search")
    logger.info("=" * 60)

    # 加载Optuna配置
    logger.info(f"Loading Optuna config from: {args.optuna_config}")
    optuna_config = load_config_from_json(args.optuna_config)
    optuna_config.save_dir = args.log_dir

    # 加载参数空间
    logger.info(f"Loading parameter space from: {args.param_space}")
    param_spaces = load_param_space_from_json(args.param_space)

    # 创建数据源工厂函数
    def data_src_factory():
        return get_data_source(dataset_name=args.dataset, args=args)

    # 创建目标函数包装器
    objective_wrapper = TrainerObjectiveWrapper(
        trainer_class=GIKTTrainer,
        data_src_fn=data_src_factory,
        base_args=args,
        metric_name=args.metric,
        max_epochs=args.epochs,
    )

    # 使用构建器创建OptunaTuner
    tuner = (
        OptunaTunerBuilder()
        .with_config(optuna_config)
        .with_param_spaces(param_spaces)
        .with_objective(objective_wrapper)
        .build()
    )

    # 执行超参数搜索
    logger.info(
        f"Starting hyperparameter search with {optuna_config.n_trials} trials..."
    )
    best_params = tuner.search()

    # 打印结果
    tuner.print_summary()

    # 获取和保存数据框
    df = tuner.get_dataframe()
    if df is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        df_path = os.path.join(args.log_dir, "trials_history.csv")
        df.to_csv(df_path, index=False)
        logger.info(f"Trials history saved to: {df_path}")

    logger.info("=" * 60)
    logger.info("Search completed successfully!")
    logger.info("=" * 60)

    return best_params


if __name__ == "__main__":
    best_params = main()
