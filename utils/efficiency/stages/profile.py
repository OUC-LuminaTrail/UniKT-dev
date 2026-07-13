"""Profile stage: parameter counts, disk size, forward FLOPs."""

from rich.table import Table

from utils.core import register_efficiency_stage

from ..model_profile import ModelProfile, profile_model
from .base import EfficiencyStage, StageContext, format_flops


@register_efficiency_stage("profile")
class ProfileStage(EfficiencyStage):
    """Static model profile: parameter counts, disk size, forward FLOPs."""

    name = "profile"
    priority = 10

    def run(self, ctx: StageContext) -> ModelProfile:
        """Measure parameter counts, disk size, and optional forward FLOPs."""
        return profile_model(
            ctx.trainer.model,
            forward_fn=lambda: ctx.trainer.forward_pass(ctx.sample_batch),
            device=ctx.device,
            count_flops=ctx.cfg.profile_flops,
        )

    def format_table(self, result: ModelProfile) -> Table | None:
        """Render the model profile as a Rich table."""
        table = Table(
            title="Model Profile", title_style="bold cyan", show_header=False, box=None
        )
        table.add_column("Key", style="yellow", no_wrap=True)
        table.add_column("Value", style="white")
        table.add_row("Parameters", f"{result.params:,}")
        table.add_row("Trainable params", f"{result.trainable_params:,}")
        table.add_row("Model size", f"{result.model_size_mb:.2f} MB")
        if result.flops_forward is not None:
            table.add_row("FLOPs / forward", format_flops(result.flops_forward))
        if result.op_breakdown:
            top = list(result.op_breakdown.items())[:3]
            breakdown = ", ".join(f"{k}={format_flops(v)}" for k, v in top)
            table.add_row("Top ops", breakdown)
        return table
