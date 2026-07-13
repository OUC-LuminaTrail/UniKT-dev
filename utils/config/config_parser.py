"""Reflective CLI parser built on ``jsonargparse`` (PyTorch Lightning-aligned).

The RunConfig dataclass tree is the single schema. jsonargparse derives CLI flags
from it (types, ``Literal`` choices, list nargs, bool), merges ``--config`` yaml
bases with CLI overrides, and dumps/loads the same shape — replacing the former
hand-rolled argparse reflection + OmegaConf merge.

Two-pass model resolution is retained: the model name (``-m`` /
``--experiment.model_name`` / a ``--config`` or ``default_config`` yaml) selects
the concrete :class:`ModelConfig` subclass *before* the schema is built, because
model configs are lazily discovered across multiple environments and cannot all
be imported at parse time.

``parse_args`` returns a typed :class:`RunConfig` instance. Entry points that
need their own extra nodes (e.g. ``efficiency.py``) use ``parse_with_extras``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace

from .run_config import RunConfig, build_run_config_schema


class ConfigParser:
    """Build CLI flags reflectively from the RunConfig schema and parse to a RunConfig."""

    def __init__(
        self,
        prog: str | None = None,
        description: str | None = None,
        extra_nodes: dict[str, type] | None = None,
        default_config: str | Path | None = None,
    ):
        """Store parser identity and optional extension points.

        Args:
            prog: argparse ``prog``.
            description: argparse ``description``.
            extra_nodes: Extra ``{node_name: dataclass}`` nodes (e.g.
                ``{"efficiency": EfficiencyConfig}``) exposed as ``--<node>.<field>``
                flags without polluting the shared RunConfig tree.
            default_config: Path to a yaml loaded as the base config (CLI overrides
                on top); serves the same role as ``--config`` but programmatic.
        """
        self.prog = prog
        self.description = description
        self.extra_nodes = dict(extra_nodes) if extra_nodes else {}
        self.default_config = str(default_config) if default_config else None

    def parse_args(self, argv: list[str] | None = None) -> RunConfig:
        """Parse ``argv`` (default ``sys.argv[1:]``) into a :class:`RunConfig` instance."""
        rc, _ = self.parse_with_extras(argv)
        return rc

    def parse_with_extras(self, argv: list[str] | None = None) -> tuple[RunConfig, Namespace]:
        """Parse and return ``(RunConfig, full Namespace)`` — Namespace exposes extra nodes."""
        argv = sys.argv[1:] if argv is None else list(argv)
        argv = _expand_short_flags(argv)

        model_name = self._resolve_model_name(argv)
        schema_nodes = {**build_run_config_schema(model_name), **self.extra_nodes}
        parser = self._build_parser(schema_nodes)

        ns = parser.parse_args(argv)
        rc = _namespace_to_run_config(ns, schema_nodes)
        return rc, ns

    def _resolve_model_name(self, argv: list[str]) -> str:
        """Pass one: find the model name from -m / --config / default_config."""
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("-m", "--experiment.model_name", dest="model_name", default=None)
        pre.add_argument("--config", dest="config", default=None)
        pre_args, _ = pre.parse_known_args(argv)

        for source in (pre_args.config, self.default_config):
            if source:
                name = _read_model_name(source)
                if name:
                    return name
        if pre_args.model_name:
            return pre_args.model_name

        argv_list = argv if argv is not None else sys.argv[1:]
        if "-h" in argv_list or "--help" in argv_list:
            self._print_framework_help()
        raise SystemExit(
            "model name is required: pass -m/--experiment.model_name, or a --config / "
            "default_config yaml carrying experiment.model_name"
        )

    def _build_parser(self, schema_nodes: dict[str, type]) -> ArgumentParser:
        default_files = [self.default_config] if self.default_config else None
        parser = ArgumentParser(
            prog=self.prog,
            description=self.description,
            default_config_files=default_files,
        )
        parser.add_argument("--config", action=ActionConfigFile)
        for node, cls in schema_nodes.items():
            parser.add_class_arguments(cls, node)
        return parser

    def _print_framework_help(self) -> None:
        """Print framework-level help (model-specific flags need ``-m <model> -h``)."""
        from utils.core import get_supported_models

        from .run_config import _FRAMEWORK_NODES

        available = get_supported_models()
        parser = ArgumentParser(
            prog=self.prog,
            description=(
                (self.description or "")
                + f"\nAvailable models: {', '.join(available)}"
                + "\nPass -m <model> -h for model-specific flags."
            ),
        )
        parser.add_argument("--config", action=ActionConfigFile)
        for node, cls in {**_FRAMEWORK_NODES, **self.extra_nodes}.items():
            parser.add_class_arguments(cls, node)
        parser.parse_args(["-h"])  # prints help and exits 0


def _expand_short_flags(argv: list[str]) -> list[str]:
    """Rewrite the two essential short flags to their full jsonargparse forms.

    ``-m X`` -> ``--experiment.model_name X`` and ``-d X`` -> ``--data.dataset X``.
    Kept as the only short flags for ergonomics; everything else uses readable
    dotted flags. Done by argv rewriting so jsonargparse only ever sees full flags.
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        long_form, inline = None, None
        if token == "-m":
            long_form = "--experiment.model_name"
        elif token.startswith("-m="):
            inline = "--experiment.model_name=" + token[3:]
        elif token == "-d":
            long_form = "--data.dataset"
        elif token.startswith("-d="):
            inline = "--data.dataset=" + token[3:]
        if inline is not None:
            out.append(inline)
            i += 1
            continue
        if long_form is not None and i + 1 < len(argv):
            out.append(long_form)
            out.append(argv[i + 1])
            i += 2
            continue
        out.append(token)
        i += 1
    return out


def _read_model_name(config_path: str) -> str | None:
    """Read ``experiment.model_name`` from a yaml config file (no schema needed)."""
    try:
        with Path(config_path).open() as f:
            data = yaml.safe_load(f) or {}
        return data.get("experiment", {}).get("model_name")
    except (OSError, yaml.YAMLError):
        return None


def _namespace_to_run_config(ns: Namespace, schema_nodes: dict[str, type]) -> RunConfig:
    """Construct a :class:`RunConfig` from a jsonargparse Namespace.

    Only the standard RunConfig nodes are built; entry-point ``extra_nodes`` stay
    in the Namespace (returned alongside by ``parse_with_extras``).
    """
    rc_fields = {f.name for f in fields(RunConfig)}
    node_instances: dict[str, Any] = {}
    for node, cls in schema_nodes.items():
        if node in rc_fields:
            node_instances[node] = _build_node(cls, ns[node])
    return RunConfig(**node_instances)


def _build_node(cls: type, node_ns: Namespace) -> Any:
    """Instantiate a config dataclass from its Namespace slice."""
    return cls(**{f.name: node_ns[f.name] for f in fields(cls)})


__all__ = ["ConfigParser"]
