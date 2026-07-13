"""Efficiency benchmark config node (entry-point-specific, not part of RunConfig).

Exposed via ``ConfigParser(extra_nodes={"efficiency": EfficiencyConfig})`` in
``efficiency.py``; read at runtime as ``rc.efficiency.*``. Lives outside the
shared :class:`RunConfig` tree so it never appears in ``train.py``'s flags.
"""

from dataclasses import dataclass, field


@dataclass
class EfficiencyConfig:
    """Efficiency benchmark knobs.

    Attributes:
        modes: Stages to run (empty = all discovered stages). Names match
            ``@register_efficiency_stage("name")`` registrations.
        benchmark_iters: Forward passes per repeat for inference latency.
        warmup_iters: Discarded iters before timing (cuDNN autotune / clock ramp).
        repeats: Full benchmark repeats; median latency reported.
        train_iters: Forward+backward+step iters for training memory/throughput.
        resource_sample_interval: Background resource sampling interval (s).
        profile_flops: Estimate forward FLOPs via torch flop_counter.
        run_dir: Trained run dir; reconstruct rc from its run_config.yaml and
            load the checkpoint for benchmarking a trained model.
        checkpoint: Checkpoint filename inside ``run_dir``.
        weights: Standalone checkpoint to load after building the model.
        output_dir: Where to write efficiency_report.json; default = exp dir.
    """

    modes: list[str] = field(
        default_factory=list,
        metadata={
            "help": "Stages to run, empty = all (e.g. --efficiency.modes profile inference)",
            "nargs": "+",
        },
    )
    benchmark_iters: int = field(
        default=200,
        metadata={"help": "Forward passes for inference latency (default: 200)"},
    )
    warmup_iters: int = field(
        default=50,
        metadata={
            "help": "Discarded iters before timing; cuDNN autotune / clock ramp (default: 50)"
        },
    )
    repeats: int = field(
        default=3,
        metadata={"help": "Full benchmark repeats; median latency reported (default: 3)"},
    )
    train_iters: int = field(
        default=50,
        metadata={
            "help": "Forward+backward+step iters for training memory/throughput (default: 50)"
        },
    )
    resource_sample_interval: float = field(
        default=0.05,
        metadata={"help": "Background resource sampling interval in seconds (default: 0.05)"},
    )
    profile_flops: bool = field(
        default=True,
        metadata={"help": "Estimate forward FLOPs via torch flop_counter (default: True)"},
    )
    run_dir: str | None = field(
        default=None,
        metadata={
            "help": "Trained run dir; reconstruct rc from run_config.yaml and load checkpoint"
        },
    )
    checkpoint: str = field(
        default="best_model.pth",
        metadata={"help": "Checkpoint filename inside --efficiency.run_dir (default: best_model.pth)"},
    )
    weights: str | None = field(
        default=None,
        metadata={"help": "Standalone checkpoint to load after building the model"},
    )
    output_dir: str | None = field(
        default=None,
        metadata={"help": "Where to write efficiency_report.json; default = exp dir"},
    )
