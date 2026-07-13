"""Schema extractor — model discovery and parameter schema retrieval.

Runs a helper subprocess that introspects available training models and their
CLI parameter schemas, returning structured data for the schemas API.
"""

import json
import logging
import subprocess
import sys

from config import PROJECT_ROOT
from schemas import ModelSchemaResponse, ParamField, ParamGroup

from services.python_env import EnvironmentNotConfigured

logger = logging.getLogger(__name__)

HELPER_SCRIPT = str(PROJECT_ROOT / "web" / "backend" / "services" / "_schema_helper.py")


def _parse_group(data: dict) -> ParamGroup:
    """Parse a raw parameter group dict into a ParamGroup model.

    Args:
        data: Raw dict with ``group_name`` and ``params`` keys.

    Returns:
        A ParamGroup instance with parsed ParamField entries.
    """
    fields = {}
    for name, cfg in data["params"].items():
        fields[name] = ParamField(
            type=cfg.get("type", "str"),
            default=cfg.get("default"),
            help=cfg.get("help", ""),
            required=cfg.get("required", False),
            choices=cfg.get("choices"),
            short=cfg.get("short"),
            nargs=cfg.get("nargs"),
        )
    return ParamGroup(group_name=data["group_name"], params=fields)


class SchemaExtractor:
    """Extracts model names and parameter schemas by running a helper script.

    Caches results after the first successful run to avoid repeated subprocess
    invocations.

    Args:
        env_manager: A PythonEnvManager instance (or None) used to resolve
            the Python command for the helper script.
    """

    def __init__(self, env_manager):
        """Initialize the SchemaExtractor.

        Args:
            env_manager: PythonEnvManager (or compatible) for command resolution.
        """
        self._env_manager = env_manager
        self._models: list[str] | None = None
        self._schemas: dict[str, list[dict]] = {}

    def _resolve_base_cmd(self) -> list[str]:
        """Resolve the base Python command for the helper script.

        Uses the wizard-configured default environment.

        Raises:
            EnvironmentNotConfigured: If no default environment is configured.
        """
        if self._env_manager:
            return self._env_manager.resolve_command()
        return [sys.executable]

    def _run_helper(self) -> None:
        """Run the schema helper subprocess and cache the results.

        Parses stdout lines into model lists and parameter schemas.
        Errors are recorded so errored models are excluded from the model list.
        """
        if self._models is not None:
            return
        try:
            base = self._resolve_base_cmd()
            cmd = [*base, HELPER_SCRIPT]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, cwd=str(PROJECT_ROOT)
            )
        except EnvironmentNotConfigured as e:
            # No env configured (user skipped the setup wizard) — fail safe with
            # an empty model list rather than crashing the schemas endpoint.
            logger.error("Schema extraction skipped — %s", e)
            self._models = []
            return
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            self._models = []
            return
        all_models: list[str] = []
        errored: set[str] = set()
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry["type"] == "models":
                all_models = entry["data"]
            elif entry["type"] == "schema":
                self._schemas[entry["model"]] = entry["data"]
            elif entry["type"] == "error":
                errored.add(entry["model"])
        self._models = [m for m in all_models if m not in errored]

    def list_models(self) -> list[str]:
        """Return the list of available model names.

        Returns:
            A list of model name strings.
        """
        self._run_helper()
        return self._models

    def get_model_schema(self, model_name: str) -> ModelSchemaResponse:
        """Return the parameter schema for a specific model.

        Args:
            model_name: The model name to look up.

        Returns:
            A ModelSchemaResponse with parameter groups and fields.

        Raises:
            KeyError: If the model name is not found.
        """
        self._run_helper()
        raw_groups = self._schemas.get(model_name)
        if raw_groups is None:
            raise KeyError(f"Model '{model_name}' not found")
        groups = [_parse_group(g) for g in raw_groups]
        return ModelSchemaResponse(model_name=model_name, param_groups=groups)
