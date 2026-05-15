import json
import subprocess
import sys

from config import PROJECT_ROOT
from schemas import ModelSchemaResponse, ParamField, ParamGroup

HELPER_SCRIPT = str(
    PROJECT_ROOT / "web" / "backend" / "services" / "_schema_helper.py"
)


def _parse_group(data: dict) -> ParamGroup:
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
    def __init__(self, env_manager):
        self._env_manager = env_manager
        self._models: list[str] | None = None
        self._schemas: dict[str, list[dict]] = {}

    def _resolve_base_cmd(self) -> list[str]:
        if self._env_manager:
            return self._env_manager.resolve_default_command()
        return [sys.executable]

    def _run_helper(self) -> None:
        if self._models is not None:
            return
        base = self._resolve_base_cmd()
        cmd = base + [HELPER_SCRIPT]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, cwd=str(PROJECT_ROOT)
        )
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
        self._run_helper()
        return self._models

    def get_model_schema(self, model_name: str) -> ModelSchemaResponse:
        self._run_helper()
        raw_groups = self._schemas.get(model_name)
        if raw_groups is None:
            raise KeyError(f"Model '{model_name}' not found")
        groups = [_parse_group(g) for g in raw_groups]
        return ModelSchemaResponse(model_name=model_name, param_groups=groups)
