"""Shared fixtures for config-area tests: yaml config writer factory."""

import pytest


@pytest.fixture
def write_model_yaml(tmp_path):
    """Factory: write a --config yaml naming a model (plus optional overrides)."""

    def _write(model_name="TinyTestModel", overrides=None, filename="config.yaml"):
        import yaml

        data = {"experiment": {"model_name": model_name}}
        for dotted, value in (overrides or {}).items():
            node, _, field = dotted.partition(".")
            data.setdefault(node, {})[field] = value
        path = tmp_path / filename
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return _write
