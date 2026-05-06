import importlib
import json

from config import PROJECT_ROOT
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

DATA_DIR = PROJECT_ROOT / "data"


def _get_supported_datasets() -> list[str]:
    try:
        mod = importlib.import_module("data_process")
        return list(getattr(mod, "SUPPORTED_DATASETS", []))
    except Exception:
        return []


@router.get("")
def list_datasets():
    supported = _get_supported_datasets()
    if not supported:
        supported = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())

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
    meta_path = DATA_DIR / dataset_name / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(404, f"Dataset '{dataset_name}' not found")
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        raise HTTPException(500, "Failed to read metadata")
