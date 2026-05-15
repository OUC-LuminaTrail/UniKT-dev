import json
import sys

sys.path.insert(0, ".")

from utils.config.param_config import DataParams, EarlyStoppingParams, GeneralParams
from utils.core import PARAM_CONFIGS

models = list(PARAM_CONFIGS.keys())
print(json.dumps({"type": "models", "data": models}))

for model_name in models:
    try:
        model_cls = PARAM_CONFIGS.get(model_name)
        if model_cls is None:
            continue
        groups = []
        for cls in [GeneralParams, DataParams, EarlyStoppingParams, model_cls]:
            inst = cls()
            fields = {}
            for name, cfg in inst.params.items():
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
                fields[name] = {
                    "type": type_str,
                    "default": cfg.get("default"),
                    "help": cfg.get("help", ""),
                    "required": cfg.get("required", False),
                    "choices": cfg.get("choices"),
                    "short": cfg.get("short"),
                    "nargs": cfg.get("nargs"),
                }
            groups.append({"group_name": inst.group_name, "params": fields})
        print(json.dumps({"type": "schema", "model": model_name, "data": groups}))
    except Exception as e:
        print(json.dumps({"type": "error", "model": model_name, "error": str(e)}))
