"""Schemas API router — model discovery and parameter schema retrieval.

Provides endpoints to list available model names and retrieve the
parameter schema (groups, fields, defaults) for a specific model.
"""

import threading

from dependencies import get_python_env_manager
from fastapi import APIRouter, Depends, HTTPException
from schemas import ModelSchemaResponse
from services.python_env import PythonEnvManager
from services.schema_extractor import SchemaExtractor

router = APIRouter(prefix="/api/schemas", tags=["schemas"])

_extractor: SchemaExtractor | None = None
_extractor_lock = threading.Lock()


def _get_extractor(
    mgr: PythonEnvManager = Depends(get_python_env_manager),
) -> SchemaExtractor:
    """Return the lazy-initialized SchemaExtractor singleton.

    Args:
        mgr: Injected PythonEnvManager used by the extractor.

    Returns:
        The SchemaExtractor instance.
    """
    global _extractor
    if _extractor is None:
        with _extractor_lock:
            if _extractor is None:
                _extractor = SchemaExtractor(env_manager=mgr)
    return _extractor


@router.get("/models", response_model=list[str])
def list_models(mgr: PythonEnvManager = Depends(get_python_env_manager)):
    """List all available model names.

    Args:
        mgr: Injected PythonEnvManager singleton.

    Returns:
        A list of model name strings.
    """
    return _get_extractor(mgr).list_models()


@router.get("/models/{model_name}/params", response_model=ModelSchemaResponse)
def get_model_params(
    model_name: str, mgr: PythonEnvManager = Depends(get_python_env_manager)
):
    """Return the parameter schema for a specific model.

    Args:
        model_name: The model name to look up.
        mgr: Injected PythonEnvManager singleton.

    Returns:
        A ModelSchemaResponse with parameter groups and fields.

    Raises:
        HTTPException: 404 if the model is not found.
    """
    try:
        return _get_extractor(mgr).get_model_schema(model_name)
    except KeyError:
        raise HTTPException(404, f"Model '{model_name}' not found")
