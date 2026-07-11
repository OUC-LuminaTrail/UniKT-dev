"""Datasets router — listing and metadata retrieval.

Provides endpoints to list all available datasets and retrieve metadata
for a specific dataset by name.
"""

import json

from config import PROJECT_ROOT
from fastapi import APIRouter, HTTPException

from utils.core import get_supported_datasets

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

DATA_DIR = PROJECT_ROOT / "data"


@router.get("")
def list_datasets():
    """List all available datasets with optional metadata summaries.

    Returns:
        A list of dicts, each containing ``name``, ``num_users``,
        ``num_questions``, and ``num_skills`` from the dataset's
        ``metadata.json`` if available.
    """
    supported = get_supported_datasets()
    if not supported:
        raise HTTPException(500, "No datasets registered in DATA_SOURCES")

    result = []
    for name in supported:
        meta_path = DATA_DIR / name / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
        else:
            meta = {}
        result.append(
            {
                "name": name,
                "num_users": meta.get("num_users"),
                "num_questions": meta.get("num_questions"),
                "num_skills": meta.get("num_skills"),
            }
        )
    return result


@router.get("/{dataset_name}/metadata")
def get_dataset_metadata(dataset_name: str):
    """Return the metadata JSON for a specific dataset.

    Args:
        dataset_name: Name of the dataset.

    Returns:
        The parsed metadata dictionary.

    Raises:
        HTTPException: 404 if the dataset metadata file does not exist,
            500 if it cannot be read.
    """
    meta_path = DATA_DIR / dataset_name / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(404, f"Dataset '{dataset_name}' not found")
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        raise HTTPException(500, "Failed to read metadata")
