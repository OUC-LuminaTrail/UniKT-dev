"""Sphinx configuration for UniKT docs."""

project = "UniKT"
author = "UniKT Team"
language = "zh_CN"
release = ""

extensions = [
    "sphinxcontrib.mermaid",
    "sphinx_design",
    "sphinx_copybutton",
]

source_suffix = ".rst"

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "titles_only": False,
}

html_static_path = ["_static"]
html_css_files = ["arena.css"]

mermaid_version = "11"

exclude_patterns = ["_build", "Thumbs.db", "**/.DS_Store"]
