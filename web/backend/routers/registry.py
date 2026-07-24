"""Registry router — on-demand refresh of model/dataset discovery.

Invalidates the backend's cached discovery (SchemaExtractor's model list and
the DATA_SOURCES index) so components added or removed on disk after startup
become visible without restarting the backend.
"""

from dependencies import get_schema_extractor
from fastapi import APIRouter
from services.registry_sync import registry_lock

from utils.core import DATA_SOURCES, get_supported_datasets

router = APIRouter(prefix="/api/registry", tags=["registry"])


@router.post("/refresh")
def refresh_registry():
    """Reset cached discovery and warm the freshly extracted caches."""
    extractor = get_schema_extractor()
    extractor.reset_cache()
    with registry_lock:
        DATA_SOURCES.clear()
        get_supported_datasets()  # warm: rediscover under the same lock
    extractor.list_models()  # warm: reload schemas now
    return {"ok": True}
