"""Efficiency benchmark CLI parameters."""

from utils.config import BaseParamConfig


class EfficiencyParams(BaseParamConfig):
    """效率基准测试专属参数。

    与 ``GeneralParams``/``DataParams``/模型参数一同注册到 ``efficiency.py`` 的 argparse。
    不加 ``@register_model_params`` —— 否则会把 "Efficiency" 当作模型污染 ``PARAM_CONFIGS``。
    """

    def define_params(self) -> tuple[str, dict]:
        return "Efficiency Parameters", {
            "modes": {
                "type": str,
                "default": "profile,inference,train",
                "help": "Comma-separated benchmark stages: profile,inference,train (default: all)",
            },
            "benchmark_iters": {
                "type": int,
                "default": 200,
                "help": "Forward passes for inference latency measurement (default: 200)",
            },
            "warmup_iters": {
                "type": int,
                "default": 50,
                "help": "Discarded iterations before timing; covers cuDNN autotune / JIT / GPU clock ramp (default: 50)",
            },
            "repeats": {
                "type": int,
                "default": 3,
                "help": "Full benchmark repeats; median latency reported (default: 3)",
            },
            "train_iters": {
                "type": int,
                "default": 50,
                "help": "Forward+backward+step iterations for training memory/throughput (default: 50)",
            },
            "sample_interval": {
                "type": float,
                "default": 0.05,
                "help": "Background resource sampling interval in seconds (default: 0.05)",
            },
            "profile_flops": {
                "type": bool,
                "default": True,
                "help": "Estimate forward FLOPs/MACs via torch.utils.flop_counter (default: True)",
            },
            "run_dir": {
                "type": str,
                "default": None,
                "help": "Trained run dir; reconstruct args from its hyperparameters.json and load best_model.pth",
            },
            "checkpoint": {
                "type": str,
                "default": "best_model.pth",
                "help": "Checkpoint filename inside --run_dir (default: best_model.pth)",
            },
            "weights": {
                "type": str,
                "default": None,
                "help": "Standalone checkpoint file to load after building the model (random init if omitted)",
            },
            "output_dir": {
                "type": str,
                "default": None,
                "help": "Where to write efficiency_report.json/.csv; default = <exp dir>/efficiency",
            },
        }
