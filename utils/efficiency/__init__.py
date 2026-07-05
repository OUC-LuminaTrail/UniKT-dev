"""模型效率评估模块。

独立效率基准：构造模型+数据后，在受控条件下测量模型规模、计算量、推理效率、
训练效率（显存峰值 + 吞吐）以及系统资源/环境元数据，用于全面、公平、可复现的效率评估。

入口：``python efficiency.py -m MODEL -d DATASET``

设计要点：
- 通过 ``trainer.forward_pass`` 驱动前向，屏蔽不同模型的 forward 签名差异（GIKT dict batch、
  AKT 4 元组、SAKT 3 元组等自动处理）。
- FLOPs 用 ``torch.utils.flop_counter.FlopCounterMode``（aten 级，覆盖 mm/bmm/SDPA/LSTM 分解），
  测量时临时关 cuDNN 以让 LSTM 类算子可分解计数。
- 推理延迟用 CUDA Event + 每次迭代 ``end.synchronize()``；训练吞吐/显存用镜像
  ``_run_train_batch`` 的伪训练循环。
- 后台线程采样 CPU/RAM/GPU 利用率与功耗（psutil + pynvml）。
"""

from .args import EfficiencyParams
from .environment import (
    EnvironmentInfo,
    ResourceSampler,
    ResourceStats,
    ResourceSummary,
    collect_environment,
)
from .inference import InferenceMetrics, benchmark_inference
from .model_profile import ModelProfile, profile_model
from .report import EfficiencyReport
from .session import EfficiencyConfig, EfficiencySession
from .training import TrainingMetrics, benchmark_training

__all__ = [
    "EfficiencyParams",
    "EfficiencyConfig",
    "EfficiencySession",
    "EfficiencyReport",
    "ModelProfile",
    "InferenceMetrics",
    "TrainingMetrics",
    "EnvironmentInfo",
    "ResourceStats",
    "ResourceSummary",
    "ResourceSampler",
    "profile_model",
    "benchmark_inference",
    "benchmark_training",
    "collect_environment",
]
