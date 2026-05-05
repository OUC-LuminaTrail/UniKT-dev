from fastapi import APIRouter, HTTPException
from schemas import ModelSchemaResponse
from services.schema_extractor import SchemaExtractor

router = APIRouter(prefix="/api/schemas", tags=["schemas"])

_extractor: SchemaExtractor | None = None


def _get_extractor() -> SchemaExtractor:
    global _extractor
    if _extractor is None:
        _extractor = SchemaExtractor()
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
