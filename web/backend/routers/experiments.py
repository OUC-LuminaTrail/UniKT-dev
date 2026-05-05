import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from config import RUNS_DIR
from schemas import ExperimentDetail, ExperimentInfo

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

_EXP_PATTERN = re.compile(
    r"^(?P<model>\w+)_(?P<dataset>[\w]+)_(?P<timestamp>\d{8}-\d{6})"
)


@router.get("", response_model=list[ExperimentInfo])
def list_experiments(
    type: str = "normal",
    model: str | None = None,
    dataset: str | None = None,
):
    base = RUNS_DIR / type
    if not base.exists():
        return []
    results = []
    for d in sorted(base.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        m = _EXP_PATTERN.match(d.name)
        info = ExperimentInfo(
            name=d.name,
            path=str(d.relative_to(RUNS_DIR)),
            model_name=m.group("model") if m else None,
            dataset_name=m.group("dataset") if m else None,
            timestamp=m.group("timestamp") if m else None,
            type=type,
        )
        if model and info.model_name != model:
            continue
        if dataset and info.dataset_name != dataset:
            continue
        results.append(info)
    return results


@router.get("/{exp_path:path}", response_model=ExperimentDetail)
def get_experiment(exp_path: str):
    full_path = RUNS_DIR / exp_path
    if not full_path.exists() or not full_path.is_dir():
        raise HTTPException(404, "Experiment not found")

    files = [f.name for f in sorted(full_path.iterdir())]

    hyperparams = None
    hp_file = full_path / "hyperparameters.json"
    if hp_file.exists():
        try:
            hyperparams = json.loads(hp_file.read_text())
        except json.JSONDecodeError:
            pass

    return ExperimentDetail(
        name=full_path.name,
        path=exp_path,
        files=files,
        hyperparams=hyperparams,
    )


@router.get("/{exp_path:path}/files/{file_name}")
def read_experiment_file(exp_path: str, file_name: str):
    full_path = RUNS_DIR / exp_path / file_name
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(404, "File not found")
    if not str(full_path.resolve()).startswith(str(RUNS_DIR.resolve())):
        raise HTTPException(403, "Access denied")
    try:
        content = full_path.read_text(errors="replace")
        return {"name": file_name, "content": content}
    except Exception as e:
        raise HTTPException(500, str(e))
