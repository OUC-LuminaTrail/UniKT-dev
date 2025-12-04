"""
Optuna configuration and parameter space helpers
"""
import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from optuna.pruners import BasePruner, MedianPruner, PercentilePruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, TPESampler, RandomSampler, GridSampler, CmaEsSampler
from optuna.trial import Trial

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """超参数搜索空间定义"""
    name: str
    type: str  # 'int', 'float', 'categorical'
    low: Optional[float] = None
    high: Optional[float] = None
    log: Optional[bool] = None  # 用于数值参数的对数采样
    step: Optional[float] = None  # 用于整数参数的步长
    choices: Optional[List[Any]] = None  # 用于分类参数
    default: Optional[Any] = None

    def validate(self):
        """验证参数空间配置的完整性"""
        if self.type == 'int':
            if self.low is None or self.high is None:
                raise ValueError(f"Integer parameter '{self.name}' requires 'low' and 'high'")
            if self.low >= self.high:
                raise ValueError(f"Parameter '{self.name}': low must be less than high")
        elif self.type == 'float':
            if self.low is None or self.high is None:
                raise ValueError(f"Float parameter '{self.name}' requires 'low' and 'high'")
            if self.low >= self.high:
                raise ValueError(f"Parameter '{self.name}': low must be less than high")
        elif self.type == 'categorical':
            if not self.choices:
                raise ValueError(f"Categorical parameter '{self.name}' requires 'choices'")
        else:
            raise ValueError(f"Unsupported parameter type: {self.type}")

    def suggest(self, trial: Trial) -> Any:
        """从Optuna trial中采样参数值"""
        self.validate()
        
        if self.type == 'int':
            return trial.suggest_int(
                self.name,
                low=int(self.low),
                high=int(self.high),
                step=int(self.step) if self.step else 1,
                log=self.log or False
            )
        elif self.type == 'float':
            return trial.suggest_float(
                self.name,
                low=float(self.low),
                high=float(self.high),
                log=self.log or False
            )
        elif self.type == 'categorical':
            return trial.suggest_categorical(self.name, self.choices)


@dataclass
class OptunaConfig:
    """Optuna搜索配置"""
    # 采样器配置
    sampler: str = "tpe"  # 'tpe', 'random', 'grid', 'cmaes'
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # 修剪器配置
    pruner: str = "median"  # 'median', 'percentile', 'successive_halving', None
    pruner_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # 搜索配置
    n_trials: int = 100
    n_jobs: int = 1  # 并行任务数
    timeout: Optional[int] = None  # 单位：秒
    directions: List[str] = field(default_factory=lambda: ["maximize"])  # 'maximize' 或 'minimize'
    
    # 存储和日志
    study_name: Optional[str] = None
    db_url: Optional[str] = None  # 持久化数据库URL，如 "sqlite:///study.db"
    save_dir: Optional[str] = None
    verbose: int = 1  # 0=quiet, 1=normal, 2=verbose

    def get_sampler(self) -> BaseSampler:
        """根据配置创建采样器"""
        sampler_name = self.sampler.lower()
        kwargs = self.sampler_kwargs.copy()
        
        if sampler_name == "tpe":
            return TPESampler(**kwargs)
        elif sampler_name == "random":
            return RandomSampler(**kwargs)
        elif sampler_name == "grid":
            return GridSampler(**kwargs)
        elif sampler_name == "cmaes":
            return CmaEsSampler(**kwargs)
        else:
            raise ValueError(f"Unsupported sampler: {sampler_name}")

    def get_pruner(self) -> Optional[BasePruner]:
        """根据配置创建修剪器"""
        if self.pruner is None:
            return None
            
        pruner_name = self.pruner.lower()
        kwargs = self.pruner_kwargs.copy()
        
        if pruner_name == "median":
            return MedianPruner(**kwargs)
        elif pruner_name == "percentile":
            return PercentilePruner(**kwargs)
        elif pruner_name == "successive_halving":
            return SuccessiveHalvingPruner(**kwargs)
        else:
            raise ValueError(f"Unsupported pruner: {pruner_name}")



def load_config_from_json(config_path: str) -> OptunaConfig:
    """从JSON文件加载Optuna配置"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return OptunaConfig(**config_dict)


def load_param_space_from_json(space_path: str) -> List[HyperparameterSpace]:
    """从JSON文件加载参数空间定义"""
    with open(space_path, 'r') as f:
        spaces_dict = json.load(f)
    
    param_spaces = []
    for space_config in spaces_dict:
        if isinstance(space_config, dict):
            param_spaces.append(HyperparameterSpace(**space_config))
    
    return param_spaces


__all__ = [
    "HyperparameterSpace",
    "OptunaConfig",
    "load_config_from_json",
    "load_param_space_from_json",
]
