"""Reflective CLI parser: derives argparse flags from RunConfig dataclass fields.

Two-pass parse: first resolve the model name (needed to pick the concrete
:class:`~utils.config.run_config.ModelConfig` subclass), then register dot-path
flags (``--general.seed``, ``--model.hidden_dim``) for every field and parse
fully. Returns an OmegaConf ``DictConfig``: the structured defaults merged with
user-supplied overrides (and an optional ``--config`` yaml base).

Defaults are ``SUPPRESS``ed so only user-provided flags land in the parsed
namespace; the structured schema supplies every default. This yields correct
"override-on-top-of-defaults" semantics and lets a yaml base config combine
cleanly with CLI flags.
"""

from __future__ import annotations

import argparse
import sys
import typing
from dataclasses import fields
from typing import Any

from omegaconf import OmegaConf

from .run_config import build_run_config_schema


def _base_type(tp: Any) -> Any:
    """Reduce ``Optional[X]`` and ``list[X]`` to their scalar element type.

    ``int``/``float``/``str``/``bool`` pass through unchanged.
    """
    args = typing.get_args(tp)
    if not args:
        return tp
    origin = typing.get_origin(tp)
    if origin is list:
        return args[0] if args else tp
    # Optional[X] / X | None: drop NoneType
    non_none = [a for a in args if a is not type(None)]
    if type(None) in args and len(non_none) == 1:
        return non_none[0]
    return tp


def _is_list(tp: Any) -> bool:
    return typing.get_origin(tp) is list


def register_config_group(
    parser: argparse.ArgumentParser, node: str, cls: type
) -> None:
    """Register a config dataclass's fields as dot-path argparse flags.

    Shared by :class:`ConfigParser` (full RunConfig) and standalone tools that
    only need a subset of nodes (e.g. ``data_process.py``). Defaults are
    ``SUPPRESS``ed so the structured schema supplies them on merge.
    """
    hints = typing.get_type_hints(cls)
    for f in fields(cls):
        ftype = hints.get(f.name, f.type)
        flag = f"--{node}.{f.name}"
        meta = f.metadata
        kwargs: dict[str, Any] = {
            "dest": f"{node}.{f.name}",
            "default": argparse.SUPPRESS,
            "help": meta.get("help", ""),
        }
        if meta.get("choices") is not None:
            kwargs["choices"] = meta["choices"]

        if _is_list(ftype):
            kwargs["nargs"] = meta.get("nargs", "+")
            elem = _base_type(ftype)
            if elem in (int, float, str):
                kwargs["type"] = elem
        else:
            scalar = _base_type(ftype)
            if scalar is bool:
                # default True -> store_false, else store_true (covers bool and bool | None)
                kwargs["action"] = "store_false" if f.default is True else "store_true"
            elif scalar in (int, float, str):
                kwargs["type"] = scalar

        arg_names = [flag]
        if meta.get("short"):
            arg_names.insert(0, f"-{meta['short']}")
        parser.add_argument(*arg_names, **kwargs)


def _register_nodes(parser: argparse.ArgumentParser, nodes: dict[str, type]) -> None:
    """Register every node's dataclass fields as dot-path flags on ``parser``."""
    for node, cls in nodes.items():
        register_config_group(parser, node, cls)


class ConfigParser:
    """Build argparse flags reflectively from a RunConfig schema and parse to an OmegaConf node."""

    def __init__(self, prog: str | None = None, description: str | None = None):
        """Store the argparse prog/description for the generated parser."""
        self.prog = prog
        self.description = description

    def parse_args(self, argv: list[str] | None = None) -> Any:
        """Parse ``argv`` (default: ``sys.argv[1:]``) into a structured OmegaConf DictConfig."""
        model_name, base_cfg = self._pass_one(argv)
        schema_nodes = build_run_config_schema(model_name)
        schema = OmegaConf.structured(schema_nodes)

        parser = self._build_full_parser(schema_nodes)
        ns = parser.parse_args(argv)
        # Config flags use ``node.field`` dot dests; the only non-dot dest
        # (``config``) is SUPPRESSed, so dot-presence marks a user override.
        overrides = {k: v for k, v in vars(ns).items() if "." in k}
        nested = _dot_to_nested(overrides)

        merge_sources = [schema]
        if base_cfg is not None:
            merge_sources.append(base_cfg)
        if nested:
            merge_sources.append(OmegaConf.create(nested))
        return OmegaConf.merge(*merge_sources)

    def _pass_one(self, argv: list[str] | None) -> tuple[str | None, Any]:
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument(
            "-m", "--experiment.model_name", dest="model_name", default=None
        )
        pre.add_argument("--config", dest="config", default=None)
        pre_args, _ = pre.parse_known_args(argv)

        base_cfg = None
        model_name = pre_args.model_name
        if pre_args.config:
            base_cfg = OmegaConf.load(pre_args.config)
            if not model_name:
                try:
                    model_name = base_cfg.experiment.model_name
                except (AttributeError, KeyError):
                    model_name = None
        if not model_name:
            argv_list = argv if argv is not None else sys.argv[1:]
            if "-h" in argv_list or "--help" in argv_list:
                # No model yet: show framework-level help and exit 0 (model-specific
                # flags require `-m <model> -h`).
                self._print_framework_help()
            pre.error(
                "model name is required: pass -m/--experiment.model_name, "
                "or provide --config pointing at a yaml with experiment.model_name"
            )
        return model_name, base_cfg

    def _print_framework_help(self) -> None:
        """Print help for the framework-level flags (when no model is given)."""
        from utils.core import get_supported_models

        from .run_config import _FRAMEWORK_NODES

        available = get_supported_models()
        parser = argparse.ArgumentParser(
            prog=self.prog,
            description=(
                (self.description or "")
                + f"\nAvailable models: {', '.join(available)}"
                + "\nPass -m <model> -h for model-specific flags."
            ),
            allow_abbrev=False,
        )
        parser.add_argument("--config", help="Path to a RunConfig yaml base.")
        # The experiment node registers -m/--experiment.model_name via its field
        # metadata; do not add -m manually (would conflict).
        _register_nodes(parser, _FRAMEWORK_NODES)
        parser.parse_args(["-h"])  # prints help and exits 0

    def _build_full_parser(
        self, schema_nodes: dict[str, type]
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog=self.prog, description=self.description, allow_abbrev=False
        )
        # --config is consumed in pass one; register here so pass two accepts it.
        parser.add_argument(
            "--config", dest="config", default=argparse.SUPPRESS, help=argparse.SUPPRESS
        )
        _register_nodes(parser, schema_nodes)
        return parser


def _dot_to_nested(dot_dict: dict[str, Any]) -> dict[str, Any]:
    """Expand ``{"a.b": 1}`` into ``{"a": {"b": 1}}``."""
    nested: dict[str, Any] = {}
    for key, value in dot_dict.items():
        parts = key.split(".")
        cursor = nested
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return nested


__all__ = ["ConfigParser"]
