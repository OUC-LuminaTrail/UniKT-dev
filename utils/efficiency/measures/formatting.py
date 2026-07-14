"""Human-readable formatters for efficiency report values."""


def format_flops(flops: int) -> str:
    """Human-readable FLOPs (K/M/G)."""
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} G"
    if flops >= 1e6:
        return f"{flops / 1e6:.2f} M"
    if flops >= 1e3:
        return f"{flops / 1e3:.2f} K"
    return f"{flops}"


def format_duration(seconds: float) -> str:
    """Human-readable wall-clock duration."""
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {sec}s"
    if minutes > 0:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def format_mib(mib: float | None) -> str:
    """Human-readable GPU memory in MiB; ``—`` when unset."""
    if mib is None:
        return "—"
    return f"{mib:,.0f} MiB"


def format_throughput(per_sec: float) -> str:
    """Human-readable throughput (interactions/s)."""
    return f"{per_sec:,.0f} interactions/s"
