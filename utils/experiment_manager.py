"""统一实验管理模块

提供统一的实验日志目录管理，支持普通训练和超参数搜索。
"""

from datetime import datetime
from enum import Enum
from pathlib import Path

from utils.core import get_logger

logger = get_logger(__name__)


class ExperimentType(Enum):
    """实验类型枚举"""

    NORMAL = "normal"
    HYPERPARAM_SEARCH = "hyperparam_search"
    ABLATION = "ablation"


class ExperimentManager:
    """统一的实验管理器

    职责：
    1. 创建符合规范的实验目录结构
    2. 生成统一的命名格式
    3. 管理实验子目录
    4. 提供工厂方法从命令行参数创建

    Example:
        >>> # 方式1：直接创建
        >>> manager = ExperimentManager(
        ...     exp_type=ExperimentType.NORMAL,
        ...     model_name="GIKT",
        ...     dataset_name="assist09",
        ...     tags=["fold0"]
        ... )
        >>> log_dir = manager.get_log_dir()
        >>> # runs/normal/GIKT_assist09_20241201-120000_fold0/

        >>> # 方式2：从命令行参数创建
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument("--model", type=str, default="GIKT")
        >>> parser.add_argument("--dataset", type=str, default="assist09")
        >>> parser.add_argument("--fold", type=int, default=0)
        >>> args = parser.parse_args()
        >>> manager = ExperimentManager.from_args(args, ExperimentType.NORMAL)
    """

    def __init__(
        self,
        exp_type: ExperimentType,
        model_name: str,
        dataset_name: str,
        base_dir: str = "runs",
        tags: list[str] | None = None,
    ):
        """初始化实验管理器

        Args:
            exp_type: 实验类型（NORMAL/HYPERPARAM_SEARCH）
            model_name: 模型名称（GIKT/HDHKT/SQGKT）
            dataset_name: 数据集名称（assist09/assist12/assist17/ednet）
            base_dir: 基础目录（默认: runs）
            tags: 可选标签列表（fold0, bs64等）
        """
        self.exp_type = exp_type
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.base_dir = Path(base_dir)
        self.tags = tags or []

        # 创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = f"{model_name}_{dataset_name}_{timestamp}"
        if self.tags:
            exp_name += "_" + "_".join(self.tags)

        self.exp_dir = self.base_dir / exp_type.value / exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Experiment directory created: {self.exp_dir}")

    def get_log_dir(self) -> str:
        """获取日志目录路径

        Returns:
            日志目录的绝对路径字符串
        """
        return str(self.exp_dir)

    def create_sub_experiment(self, sub_name: str) -> "ExperimentManager":
        """创建子实验管理器，共享时间戳但使用子目录

        用于超参数搜索等需要多个子实验的场景，所有子实验共享同一个时间戳。

        Args:
            sub_name: 子实验名称（如 "trial_0", "full_model", "no_gnn"）

        Returns:
            新的 ExperimentManager 实例，指向子目录

        Example:
            >>> parent_manager = ExperimentManager(...)
            >>> child_manager = parent_manager.create_sub_experiment("trial_0")
            >>> # parent: runs/hyperparam_search/GIKT_assist09_20241201-120000/
            >>> # child:  runs/hyperparam_search/GIKT_assist09_20241201-120000/trial_0/
        """
        # 创建子实验目录
        sub_dir = self.exp_dir / sub_name
        sub_dir.mkdir(parents=True, exist_ok=True)

        # 创建一个新的 ExperimentManager，复用父管理器的时间戳和配置
        sub_manager = ExperimentManager.__new__(ExperimentManager)
        sub_manager.exp_type = self.exp_type
        sub_manager.model_name = self.model_name
        sub_manager.dataset_name = self.dataset_name
        sub_manager.base_dir = self.base_dir
        sub_manager.tags = self.tags + [sub_name]
        sub_manager.exp_dir = sub_dir

        logger.debug(f"Sub-experiment created: {sub_dir}")
        return sub_manager

    def create_subdir(self, name: str) -> Path:
        """创建子目录

        用于需要多个子目录的场景。

        Args:
            name: 子目录名称

        Returns:
            子目录的 Path 对象

        Example:
            >>> manager = ExperimentManager(...)
            >>> full_model_dir = manager.create_subdir("full_model")
            >>> # runs/normal/GIKT_assist09_20241201-120000/full_model/
        """
        subdir = self.exp_dir / name
        subdir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Subdirectory created: {subdir}")
        return subdir

    @staticmethod
    def from_args(args, exp_type: ExperimentType) -> "ExperimentManager":
        """从命令行参数创建管理器

        自动从 args 中提取模型名称、数据集名称和常用标签。

        Args:
            args: 命令行参数对象（argparse.Namespace）
            exp_type: 实验类型

        Returns:
            ExperimentManager 实例
        """
        model = getattr(args, "model", "unknown")
        dataset = getattr(args, "dataset", "unknown")

        tags = []
        # 提取fold标签
        if hasattr(args, "fold") and args.fold is not None:
            tags.append(f"fold{args.fold}")
        # 提取batch_size标签
        if hasattr(args, "batch_size"):
            tags.append(f"bs{args.batch_size}")

        return ExperimentManager(
            exp_type=exp_type,
            model_name=model,
            dataset_name=dataset,
            base_dir=getattr(args, "base_dir", "runs"),
            tags=tags,
        )

    @staticmethod
    def from_run_dir(run_dir: str | Path) -> "ExperimentManager":
        """Create an ExperimentManager wrapping an existing run directory.

        Unlike the normal constructor, this does NOT create a new timestamped
        directory. Used for evaluation/inference on already-trained models.

        Args:
            run_dir: Path to an existing run directory.

        Returns:
            ExperimentManager pointing to the existing directory.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        run_path = Path(run_dir).resolve()
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        manager = ExperimentManager.__new__(ExperimentManager)
        manager.exp_type = ExperimentType.NORMAL
        manager.model_name = ""
        manager.dataset_name = ""
        manager.base_dir = run_path.parent.parent
        manager.tags = []
        manager.exp_dir = run_path

        logger.debug(f"ExperimentManager bound to existing dir: {run_path}")
        return manager

    def get_experiment_info(self) -> dict:
        """获取实验信息字典

        Returns:
            包含实验类型、模型、数据集等信息的字典
        """
        return {
            "experiment_type": self.exp_type.value,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "base_dir": str(self.base_dir),
            "experiment_dir": str(self.exp_dir),
            "tags": self.tags,
        }


__all__ = ["ExperimentManager", "ExperimentType"]
