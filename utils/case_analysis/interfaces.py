"""Minimal plugin interfaces for the case analysis framework.

The framework defines only these contracts; concrete behavior lives in
registered default plugins (sinks, selectors, visualizers) and can be
replaced wholesale. Registries reject re-binding an existing name, so
"replacing" a plugin means registering a new name and pointing the CLI
at it (``--sink`` / ``--selector`` / ``--visualizer``).

Model-specific analyzers (``ANALYZERS`` registry) produce the case data
that sinks consume; see ``base_analyzer.BaseCaseAnalyzer`` for the
analyzer contract.
"""

from abc import ABC, abstractmethod
from typing import Any


class CaseDataSink(ABC):
    """Receives extracted case data batches during inference."""

    @abstractmethod
    def add_batch(self, case_data: dict[str, Any]) -> None:
        """Accumulate one batch of extracted case data.

        Args:
            case_data: Per-batch dict of parallel lists produced by the
                analyzer's ``extract_case_data``. The key contract is
                defined by the concrete sink, not the framework.
        """
        raise NotImplementedError

    @abstractmethod
    def result(self) -> Any:
        """Return the accumulated result after all batches."""
        raise NotImplementedError


class UserSelector(ABC):
    """Selects user IDs from collected case analysis results."""

    @abstractmethod
    def select(self, results: Any, **options: Any) -> list:
        """Select users from results.

        Args:
            results: Sink output (``CaseDataSink.result()``).
            **options: Selector-specific options; concrete selectors
                declare named keyword parameters and ignore the rest of
                the framework contract.

        Returns:
            List of selected user identifiers.
        """
        raise NotImplementedError


class CaseVisualizer(ABC):
    """Visualizes a single user's case data."""

    @abstractmethod
    def plot_user(
        self, user_data: Any, user_id: Any, *, output_path: str | None = None
    ) -> Any:
        """Produce a visualization for one user.

        Args:
            user_data: Per-user data (e.g. a DataFrame row slice).
            user_id: User identifier.
            output_path: Optional path to save the figure.

        Returns:
            The produced figure object.
        """
        raise NotImplementedError


__all__ = ["CaseDataSink", "CaseVisualizer", "UserSelector"]
