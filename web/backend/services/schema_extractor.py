import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from schemas import ModelSchemaResponse, ParamField, ParamGroup


def _extract_group(group_name: str, params: dict) -> ParamGroup:
    fields = {}
    for name, cfg in params.items():
        ptype = cfg.get("type")
        if ptype is bool:
            type_str = "bool"
        elif ptype is int:
            type_str = "int"
        elif ptype is float:
            type_str = "float"
        elif ptype is str:
            type_str = "str"
        else:
            type_str = "str"
        fields[name] = ParamField(
            type=type_str,
            default=cfg.get("default"),
            help=cfg.get("help", ""),
            required=cfg.get("required", False),
            choices=cfg.get("choices"),
            short=cfg.get("short"),
            nargs=cfg.get("nargs"),
        )
    return ParamGroup(group_name=group_name, params=fields)


class SchemaExtractor:
    def __init__(self):
        import model  # noqa: F401
        from utils.config import DataParams, EarlyStoppingParams, GeneralParams
        from utils.core import PARAM_CONFIGS

        self._param_configs = PARAM_CONFIGS
        self._general = GeneralParams
        self._data = DataParams
        self._early_stopping = EarlyStoppingParams

    def list_models(self) -> list[str]:
        return list(self._param_configs.keys())

    def get_model_schema(self, model_name: str) -> ModelSchemaResponse:
        model_cls = self._param_configs.get(model_name)
        if model_cls is None:
            raise KeyError(f"Model '{model_name}' not found")

        groups: list[ParamGroup] = []

        for cls in [self._general, self._data, self._early_stopping, model_cls]:
            inst = cls()
            groups.append(_extract_group(inst.group_name, inst.params))

        return ModelSchemaResponse(model_name=model_name, param_groups=groups)
