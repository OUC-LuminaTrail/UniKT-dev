import json
import sys
import typing
from dataclasses import MISSING, fields

from docstring_parser import parse

sys.path.insert(0, ".")

import model  # noqa: F401
from utils.config import (
    CompileConfig,
    EarlyStoppingConfig,
    GeneralConfig,
    RunDataConfig,
)
from utils.core import MODEL_CONFIGS, get_supported_models

# Fixed framework groups, in display order: (group_name, RunConfig node, dataclass).
# The node key lets the backend route flat params into RunConfig nodes without
# re-importing the model config under torch.
_FRAMEWORK_GROUPS = [
    ("General", "general", GeneralConfig),
    ("Compile", "compile", CompileConfig),
    ("Early Stopping", "early_stopping", EarlyStoppingConfig),
    ("Data", "data", RunDataConfig),
]


def _base_type(tp):
    args = typing.get_args(tp)
    if not args:
        return tp
    if typing.get_origin(tp) is list:
        return args[0] if args else tp
    non_none = [a for a in args if a is not type(None)]
    if type(None) in args and len(non_none) == 1:
        return non_none[0]
    return tp


def _field_spec(f, help_map: dict[str, str] | None = None):
    ftype = f.type
    meta = f.metadata

    # Extract Literal choices from the type annotation when metadata is missing
    # them (e.g. compile_mode: Literal["default", "reduce-overhead", ...]).
    choices = meta.get("choices")
    if choices is None:
        origin = typing.get_origin(ftype)
        if origin is typing.Literal:
            choices = list(typing.get_args(ftype))

    if typing.get_origin(ftype) is list:
        type_str = "list"
    elif ftype is bool:
        type_str = "bool"
    else:
        scalar = _base_type(ftype)
        type_str = {int: "int", float: "float", str: "str", bool: "bool"}.get(
            scalar, "str"
        )
    default = f.default_factory() if f.default_factory is not MISSING else f.default

    # Prefer metadata-driven help; fall back to the class docstring Args: section.
    help_text = meta.get("help", "")
    if not help_text and help_map:
        help_text = help_map.get(f.name, "")

    return {
        "type": type_str,
        "default": default,
        "help": help_text,
        "required": False,
        "choices": choices,
        "short": meta.get("short"),
        "nargs": meta.get("nargs"),
    }


def _parse_docstring_helps(cls: type) -> dict[str, str]:
    """Extract ``{field_name: help_text}`` from the class docstring Args: section.

    Delegates to ``docstring_parser`` which handles Google, NumPy, and Sphinx
    styles uniformly, including multi-line descriptions.
    """
    doc = parse(cls.__doc__ or "")
    return {p.arg_name: p.description for p in doc.params if p.description}


def _group(group_name, node, cls):
    help_map = _parse_docstring_helps(cls)
    params = {f.name: _field_spec(f, help_map) for f in fields(cls)}
    return {"group_name": group_name, "node": node, "params": params}


models = get_supported_models()
print(json.dumps({"type": "models", "data": models}))

for model_name in models:
    try:
        model_cls = MODEL_CONFIGS.get(model_name)
        if model_cls is None:
            continue
        groups = [_group(name, node, cls) for name, node, cls in _FRAMEWORK_GROUPS]
        groups.append(_group(model_name, "model", model_cls))
        print(json.dumps({"type": "schema", "model": model_name, "data": groups}))
    except Exception as e:
        print(json.dumps({"type": "error", "model": model_name, "error": str(e)}))
