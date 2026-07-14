"""Efficiency benchmark config node (entry-point-specific, not part of RunConfig).

Exposed via ``ConfigParser(extra_nodes={"efficiency": EfficiencyConfig})`` in
``efficiency.py``. Lives outside the shared :class:`RunConfig` tree so it never
appears in ``train.py``'s flags.
"""

from dataclasses import dataclass


@dataclass
class EfficiencyConfig:
    """Efficiency benchmark knobs.

    Args:
        modes: Comma-separated stages to run (empty = all discovered stages);
            e.g. ``--efficiency.modes profile,inference``.
        benchmark_iters: Forward passes per repeat for inference latency.
        warmup_iters: Discarded iters before timing (cuDNN autotune / clock ramp).
        repeats: Independent repeats; per-repeat mean spread (repeat_cv) gauges run-to-run stability.
        train_iters: Forward+backward+step iters for training memory/throughput.
        resource_sample_interval: Background resource sampling interval (s).
        profile_flops: Estimate forward FLOPs via torch flop_counter.
        run_dir: Trained run dir; seed rc from its run_config.yaml and benchmark
            its checkpoint.
        checkpoint: Checkpoint filename inside ``run_dir``.
        weights: Standalone checkpoint to load after building the model.
        output_dir: Where to write efficiency_report.json; default = exp dir.
        trace_iters: Forward (and train-step) iterations captured by torch.profiler
            per segment; kept below benchmark_iters since profiling adds overhead.
        trace_top_ops: Operators retained per segment in the report, sorted by
            self device time.
        trace_export: Export a chrome/perfetto trace JSON per segment.
        batch_sizes: Comma-separated batch sizes to sweep (empty = single run). When
            set, a fresh trainer is rebuilt per size, giving each measurement a clean
            CUDA allocator and isolating peak memory across sizes;
            e.g. ``--efficiency.batch_sizes 32,64,128``.
    """

    modes: str = ""
    benchmark_iters: int = 200
    warmup_iters: int = 50
    repeats: int = 3
    train_iters: int = 50
    resource_sample_interval: float = 0.05
    profile_flops: bool = True
    run_dir: str | None = None
    checkpoint: str = "best_model.pth"
    weights: str | None = None
    output_dir: str | None = None
    trace_iters: int = 20
    trace_top_ops: int = 15
    trace_export: bool = True
    batch_sizes: str = ""
