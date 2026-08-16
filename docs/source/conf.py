"""Sphinx configuration for UniKT docs."""

import os
import sys

from sphinx.ext.autodoc.mock import _MockObject


# Mock the __or__ and __ror__ methods of _MockObject to
# avoid errors when using the | operator in type hints.
def _mock_or(self, _other):
    return self


_MockObject.__or__ = _mock_or
_MockObject.__ror__ = _mock_or

project = "UniKT"
author = "UniKT Team"
language = os.environ.get("READTHEDOCS_LANGUAGE", "zh_CN")
project_copyright = "2025-%Y, UniKT Team"

locale_dirs = ["locale/"]
gettext_compact = False
gettext_uuid = True

# Let sphinx find the modules to document
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

extensions = [
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

source_suffix = ".md"

myst_enable_extensions = ["colon_fence"]

# autodoc
autodoc_mock_imports = [
    "torch",
    "numpy",
    "polars",
    "scipy",
    "sklearn",
    "torch_geometric",
    "dhg",
    "mamba_ssm",
    "causal_conv1d",
    "xlstm",
    "swanlab",
    "wandb",
    "optuna",
    "pandas",
    "matplotlib",
    "rich",
    "tqdm",
    "pyarrow",
]
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_typehints = "description"

# napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable", None),
    "pyg": ("https://pytorch-geometric.readthedocs.io/en/latest", None),
}

# linkcheck
linkcheck_ignore = [
    r"https://github\.com/.*#L\d+",
    r"https://www\.sciencedirect\.com/.*",
    r"https://dl\.acm\.org/.*",
]

html_theme = "furo"

html_static_path = ["_static"]
html_css_files = ["arena.css"]

mermaid_version = "11"

exclude_patterns = ["_build", "Thumbs.db", "**/.DS_Store"]
