"""Global memory routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from research_agent.api.deps import ServiceBundle, get_service_bundle
from research_agent.api.schemas import MemoryBundlesResponse

router = APIRouter(prefix="/api/memories", tags=["memories"])


@router.get("/global-bundles", response_model=MemoryBundlesResponse)
def get_global_memory_bundles(
    services: ServiceBundle = Depends(get_service_bundle),
) -> MemoryBundlesResponse:
    """Return paper-centric bundles across all persisted memories."""

    bundle = services.memory_bundles.get_global_bundle()
    return MemoryBundlesResponse.from_domain(bundle)
