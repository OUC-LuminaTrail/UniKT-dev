import json
import sys
import typing
from dataclasses import MISSING, fields

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


def _field_spec(f):
    ftype = f.type
    meta = f.metadata
    if typing.get_origin(ftype) is list:
        type_str = "list"
    elif ftype is bool:
        type_str = "bool"
    else:
        scalar = _base_type(ftype)
        type_str = {int: "int", float: "float", str: "str"}.get(scalar, "str")
    default = f.default_factory() if f.default_factory is not MISSING else f.default
    return {
        "type": type_str,
        "default": default,
        "help": meta.get("help", ""),
        "required": False,
        "choices": meta.get("choices"),
        "short": meta.get("short"),
        "nargs": meta.get("nargs"),
    }


def _group(group_name, node, cls):
    params = {f.name: _field_spec(f) for f in fields(cls)}
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
