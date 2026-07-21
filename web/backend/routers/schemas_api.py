"""Schemas API router — model discovery and parameter schema retrieval.

Provides endpoints to list available model names and retrieve the
parameter schema (groups, fields, defaults) for a specific model.
"""

from dependencies import get_schema_extractor
from fastapi import APIRouter, HTTPException
from schemas import ModelSchemaResponse, ParamGroup
from services.schema_extractor import SchemaExtractor

router = APIRouter(prefix="/api/schemas", tags=["schemas"])


def _get_extractor() -> SchemaExtractor:
    """Return the shared SchemaExtractor singleton.

    Returns:
        The application-wide SchemaExtractor instance.
    """
    return get_schema_extractor()


@router.get("/models", response_model=list[str])
def list_models():
    """List all available model names.

    Returns:
        A list of model name strings.
    """
    return _get_extractor().list_models()


@router.get("/models/{model_name}/params", response_model=ModelSchemaResponse)
def get_model_params(model_name: str):
    """Return the parameter schema for a specific model.

    Args:
        model_name: The model name to look up.

    Returns:
        A ModelSchemaResponse with parameter groups and fields.

    Raises:
        HTTPException: 404 if the model is not found.
    """
    try:
        return _get_extractor().get_model_schema(model_name)
    except KeyError:
        raise HTTPException(404, f"Model '{model_name}' not found")


@router.get("/preprocess/{action}", response_model=list[ParamGroup])
def get_preprocess_params(action: str):
    """Return the parameter schema for a preprocess action (download/process).

    Args:
        action: Preprocess action ('download' or 'process').

    Returns:
        A list of ParamGroup describing the action's parameters.

    Raises:
        HTTPException: 404 if the action is unknown.
    """
    try:
        return _get_extractor().get_preprocess_schema(action)
    except KeyError:
        raise HTTPException(404, f"Unknown preprocess action '{action}'")
