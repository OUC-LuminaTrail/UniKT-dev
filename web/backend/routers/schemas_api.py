from dependencies import get_settings_manager
from fastapi import APIRouter, Depends, HTTPException
from schemas import ModelSchemaResponse
from services.schema_extractor import SchemaExtractor
from services.settings_manager import SettingsManager

router = APIRouter(prefix="/api/schemas", tags=["schemas"])

_extractor: SchemaExtractor | None = None


def _get_extractor(sm: SettingsManager = Depends(get_settings_manager)) -> SchemaExtractor:
    global _extractor
    if _extractor is None:
        from services.environment_resolver import EnvironmentResolver
        _extractor = SchemaExtractor(resolver=EnvironmentResolver(), settings_manager=sm)
    return _extractor


@router.get("/models", response_model=list[str])
def list_models():
    return _get_extractor().list_models()


@router.get("/models/{model_name}/params", response_model=ModelSchemaResponse)
def get_model_params(model_name: str):
    try:
        return _get_extractor().get_model_schema(model_name)
    except KeyError:
        raise HTTPException(404, f"Model '{model_name}' not found")
