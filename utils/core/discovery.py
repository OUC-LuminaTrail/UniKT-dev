"""Static registration discovery: scans source code without importing modules to build a name->module lazy index.

Scans for ``@register_<role>("name")`` decorators and populates the corresponding
registry's ``index``. This allows ``keys()`` to list all components at startup while
component code is only imported on ``get()`` (lazy loading, multi-environment safe).

Only recognizes the uniform ``@register_<role>("literal")`` form; non-literal arguments
(variables, expressions) are ignored.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .logger import get_logger
from .registry import (
    ANALYZERS,
    DATA_SOURCES,
    EFFICIENCY_STAGES,
    MODEL_CONFIGS,
    TRAINERS,
    UniversalRegistry,
)

_logger = get_logger(__name__)

# @register_<role>("name") -> target registry
_DECORATORS: dict[str, UniversalRegistry] = {
    "register_trainer": TRAINERS,
    "register_model_config": MODEL_CONFIGS,
    "register_data_source": DATA_SOURCES,
    "register_analyzer": ANALYZERS,
    "register_efficiency_stage": EFFICIENCY_STAGES,
}


def discover_registrations(root: str | Path, package: str) -> None:
    """Scan all ``.py`` files under ``root`` (excluding ``__init__``) and write decorator registrations into the lazy index.

    Args:
        root: Directory to scan (pass ``__file__``'s parent directory, independent of the current working directory).
        package: Dotted module name for the directory (e.g. ``"model"``, ``"utils.data_process"``).
    """
    root_path = Path(root)
    found = 0
    for py in root_path.rglob("*.py"):
        if py.name == "__init__.py":
            continue
        module_path = _to_module_path(py, root_path, package)
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            _logger.warning("Skip %s: %s", py, exc)
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for dec in node.decorator_list:
                    registry = _target_registry(dec)
                    if registry is not None:
                        registry.index(dec.args[0].value, module_path)
                        found += 1
    _logger.debug("discovery: in %s, found %d registrations", root, found)


def _to_module_path(py: Path, root: Path, package: str) -> str:
    rel = py.relative_to(root).with_suffix("")
    return ".".join((package, *rel.parts))


def _target_registry(dec: ast.expr) -> UniversalRegistry | None:
    """Identify ``@register_<role>("literal")`` and return the target registry, or ``None``."""
    if not isinstance(dec, ast.Call) or not dec.args:
        return None
    if not (
        isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str)
    ):
        return None
    return _DECORATORS.get(dec.func.id) if isinstance(dec.func, ast.Name) else None
