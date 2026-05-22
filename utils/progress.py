"""Shared Rich progress bar factory."""

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Column


def create_progress() -> Progress:
    """Create a Rich Progress bar with the project-standard style.

    Returns:
        A configured ``Progress`` instance ready for ``add_task()``.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(table_column=Column(justify="right")),
        TimeRemainingColumn(),
        expand=True,
    )
