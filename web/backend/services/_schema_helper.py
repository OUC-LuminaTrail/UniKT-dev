"""Schema reflection helper — run as a subprocess via SchemaExtractor.

Two modes (argv[1]):
  - ``models`` (default): reflect every registered model's config + the fixed
    framework groups; emits ``models`` / ``schema`` / ``error`` envelopes.
  - ``preprocess <action>``: reflect the data_process.py download/process
    parameter dataclasses (with a UI-level field whitelist); emits one
    ``preprocess_schema`` envelope.

The reflection unit is a dataclass: ``reflect_group`` walks
``dataclasses.fields(cls)`` and maps each field to ``{type, default, help,
choices, ...}``, so model configs and preprocess configs share one path.
"""

import json
import sys
import typing
from dataclasses import MISSING, fields

from docstring_parser import parse

sys.path.insert(0, ".")


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


def _field_spec(f, help_map=None):
    ftype = f.type
    meta = f.metadata

    # Extract Literal choices from the annotation when metadata lacks them
    # (e.g. compile_mode: Literal["default", "reduce-overhead", ...]).
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

    # Prefer metadata help; fall back to the class docstring Args: section.
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

    Delegates to ``docstring_parser`` (Google/NumPy/Sphinx, multi-line).
    """
    doc = parse(cls.__doc__ or "")
    return {p.arg_name: p.description for p in doc.params if p.description}


def reflect_group(group_name, node, cls, only=None, skip=None):
    """Reflect a dataclass into a ``{group_name, node, params}`` schema group.

    ``only``/``skip`` restrict which fields are exposed — a UI-level selection
    for the preprocess schema (which need not surface every RunDataConfig /
    GeneralConfig knob); the model schema leaves them unset to expose all.
    """
    help_map = _parse_docstring_helps(cls)
    params = {}
    for f in fields(cls):
        if only is not None and f.name not in only:
            continue
        if skip and f.name in skip:
            continue
        params[f.name] = _field_spec(f, help_map)
    return {"group_name": group_name, "node": node, "params": params}


def _emit_models():
    import model  # noqa: F401  triggers @register_model_config discovery
    from utils.config import (
        CompileConfig,
        EarlyStoppingConfig,
        GeneralConfig,
        RunDataConfig,
    )
    from utils.core import MODEL_CONFIGS, get_supported_models

    # Fixed framework groups, in display order: (group_name, RunConfig node, dataclass).
    framework_groups = [
        ("General", "general", GeneralConfig),
        ("Compile", "compile", CompileConfig),
        ("Early Stopping", "early_stopping", EarlyStoppingConfig),
        ("Data", "data", RunDataConfig),
    ]

    models = get_supported_models()
    print(json.dumps({"type": "models", "data": models}))
    for model_name in models:
        try:
            model_cls = MODEL_CONFIGS.get(model_name)
            if model_cls is None:
                continue
            groups = [
                reflect_group(name, node, cls) for name, node, cls in framework_groups
            ]
            groups.append(reflect_group(model_name, "model", model_cls))
            print(json.dumps({"type": "schema", "model": model_name, "data": groups}))
        except Exception as e:
            print(json.dumps({"type": "error", "model": model_name, "error": str(e)}))


def _emit_preprocess(action: str):
    from dataclasses import fields as dc_fields

    from utils.config import (
        DownloadConfig,
        GeneralConfig,
        ProcessConfig,
        RunDataConfig,
    )

    # Visible preprocess fields are marked on the dataclass itself via
    # field(metadata={"preprocess_ui": True}), so adding a param only needs
    # that marker — no parallel whitelist here.
    def visible(cls):
        return {f.name for f in dc_fields(cls) if f.metadata.get("preprocess_ui")}

    if action == "download":
        # data_url is resolved from dataset metadata; hide it from the UI.
        groups = [reflect_group("下载选项", "", DownloadConfig, skip={"data_url"})]
    elif action == "process":
        groups = [
            reflect_group("数据", "data", RunDataConfig, only=visible(RunDataConfig)),
            reflect_group(
                "通用", "general", GeneralConfig, only=visible(GeneralConfig)
            ),
            reflect_group("额外", "", ProcessConfig),
        ]
    else:
        print(
            json.dumps(
                {
                    "type": "error",
                    "action": action,
                    "error": f"unknown action '{action}'",
                }
            )
        )
        return
    print(json.dumps({"type": "preprocess_schema", "action": action, "data": groups}))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "models"
    if mode == "preprocess":
        _emit_preprocess(sys.argv[2] if len(sys.argv) > 2 else "")
    else:
        _emit_models()
