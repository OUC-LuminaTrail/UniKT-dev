from dependencies import get_python_env_manager
from fastapi import APIRouter, Depends, HTTPException
from schemas import ModelSchemaResponse
from services.python_env import PythonEnvManager
from services.schema_extractor import SchemaExtractor

router = APIRouter(prefix="/api/schemas", tags=["schemas"])

_extractor: SchemaExtractor | None = None


def _get_extractor(
    mgr: PythonEnvManager = Depends(get_python_env_manager),
) -> SchemaExtractor:
    global _extractor
    if _extractor is None:
        _extractor = SchemaExtractor(env_manager=mgr)
    return _extractor


@router.get("/models", response_model=list[str])
def list_models(mgr: PythonEnvManager = Depends(get_python_env_manager)):
    return _get_extractor(mgr).list_models()


@router.get("/models/{model_name}/params", response_model=ModelSchemaResponse)
def get_model_params(
    model_name: str, mgr: PythonEnvManager = Depends(get_python_env_manager)
):
    try:
        return _get_extractor(mgr).get_model_schema(model_name)
    except KeyError:
        raise HTTPException(404, f"Model '{model_name}' not found")
