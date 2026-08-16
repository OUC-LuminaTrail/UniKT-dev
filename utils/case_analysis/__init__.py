"""Case analysis framework for KT models.

Minimal plugin interfaces plus default implementations:

- Analyzers (``ANALYZERS`` registry): model-specific inference wrappers
  built from an archived run (see ``base_analyzer.BaseCaseAnalyzer``).
- Sinks (``CASE_SINKS``): consume extracted case data during inference.
- Selectors (``CASE_SELECTORS``): pick representative users.
- Visualizers (``CASE_VISUALIZERS``): render per-user views.

Importing this package registers the default plugins below.
"""

from .base_analyzer import BaseCaseAnalyzer
from .interfaces import CaseDataSink, CaseVisualizer, UserSelector
from .selectors import DiverseSelector, ExtremeSelector, RandomSelector
from .sinks.dataframe_sink import DataFrameSink, get_user_sequence, load_case_results
from .user_metrics import compute_user_metrics
from .visualizers import HeatmapVisualizer

__all__ = [
    "BaseCaseAnalyzer",
    "CaseDataSink",
    "CaseVisualizer",
    "DataFrameSink",
    "DiverseSelector",
    "ExtremeSelector",
    "HeatmapVisualizer",
    "RandomSelector",
    "UserSelector",
    "compute_user_metrics",
    "get_user_sequence",
    "load_case_results",
]
