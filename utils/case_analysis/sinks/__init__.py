"""Case data sinks: consume analyzer output during inference."""

from .dataframe_sink import DataFrameSink, get_user_sequence, load_case_results

__all__ = ["DataFrameSink", "get_user_sequence", "load_case_results"]
