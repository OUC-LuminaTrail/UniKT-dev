"""静态注册发现:在不导入模块的前提下扫描源码,建立名字->模块的懒索引。

扫描 ``@register_<role>("name")`` 装饰器,把发现的 ``(name, module_path)`` 写入对应注册表的
``index``。这样 ``keys()`` 在启动期即可列出全部组件,而组件代码只在 ``get()`` 时才被导入
(懒加载,多环境安全)。

只识别 ``@register_<role>("字面量")`` 这一统一形式;非字面量参数(变量、表达式)会被忽略。
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from .registry import (
    ANALYZERS,
    DATA_SOURCES,
    PARAM_CONFIGS,
    TRAINERS,
    UniversalRegistry,
)

_logger = logging.getLogger(__name__)

# @register_<role>("name") -> 目标注册表
_DECORATORS: dict[str, UniversalRegistry] = {
    "register_trainer": TRAINERS,
    "register_model_params": PARAM_CONFIGS,
    "register_data_source": DATA_SOURCES,
    "register_analyzer": ANALYZERS,
}


def discover_registrations(root: str | Path, package: str) -> None:
    """扫描 ``root`` 下所有 ``.py``(不含 ``__init__``),把装饰器注册写入懒索引。

    Args:
        root: 要扫描的目录(传 ``__file__`` 所在目录即可,与当前工作目录无关)。
        package: 该目录对应的点分模块名(如 ``"model"``、``"utils.data_process"``)。
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
    """识别 ``@register_<role>("字面量")``,返回目标注册表;否则 ``None``。"""
    if not isinstance(dec, ast.Call) or not dec.args:
        return None
    if not (
        isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str)
    ):
        return None
    return _DECORATORS.get(dec.func.id) if isinstance(dec.func, ast.Name) else None
