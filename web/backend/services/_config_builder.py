"""Route flat task params into a RunConfig nested dict in a torch-capable subprocess.

The web backend is polars/torch-free, so it cannot import model configs to route
flat frontend params (``{field: value}``) into RunConfig nodes. This helper runs
under the default training env's Python (which has torch), discovers model
configs via ``import model``, and emits exactly one JSON line on stdout:

    {"nested": {node: {field: value, ...}, ...}}

Routing mirrors ``ProcessManager._build_cli_args`` so the produced yaml is
identical to what an in-process build would write. Any failure is reported as:

    {"nested": null, "error": "<message>"}
"""

import json
import sys
from dataclasses import fields as dc_fields

sys.path.insert(0, ".")

import model  # noqa: F401  — triggers @register_model_config discovery
from utils.config.run_config import build_run_config_schema


def build_nested(model_name: str, params: dict) -> dict:
    """Route params into RunConfig nodes by schema field-name lookup."""
    schema = build_run_config_schema(model_name)
    nested: dict = {}
    for node, cls in schema.items():
        for f in dc_fields(cls):
            if f.name in params and params[f.name] is not None:
                nested.setdefault(node, {})[f.name] = params[f.name]
    nested.setdefault("experiment", {})["model_name"] = model_name
    return nested


def main() -> None:
    payload = json.loads(sys.stdin.read())
    try:
        nested = build_nested(payload["model_name"], payload["params"])
        print(json.dumps({"nested": nested}))
    except Exception as e:  # surfaced to the backend as a structured error
        print(json.dumps({"nested": None, "error": str(e)}))


if __name__ == "__main__":
    main()
