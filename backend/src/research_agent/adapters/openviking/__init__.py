"""OpenViking adapter boundaries."""

from __future__ import annotations

from importlib import import_module

from research_agent.adapters.openviking.runtime_surfaces import (
    InMemoryOpenVikingMemorySurface,
    InMemoryOpenVikingMessageSurface,
    InMemoryOpenVikingSessionSurface,
    OpenVikingSurfaceConfig,
    build_embedded_openviking_surface_bundle,
    SDKBackedOpenVikingMemorySurface,
    SDKBackedOpenVikingMessageSurface,
    SDKBackedOpenVikingSessionSurface,
    build_inmemory_openviking_surface_bundle,
    build_sdk_openviking_surface_bundle,
)
from research_agent.adapters.openviking.retrieval_adapter import (
    OpenVikingMemorySearchResult,
    OpenVikingRetrievalAdapter,
)
from research_agent.adapters.openviking.surfaces import (
    NoopOpenVikingMemorySurface,
    NoopOpenVikingMessageSurface,
    NoopOpenVikingSessionSurface,
    OpenVikingAdapterSurfaceBundle,
    OpenVikingMemoryRecord,
    OpenVikingMemorySurface,
    OpenVikingMessageRecord,
    OpenVikingMessageSurface,
    OpenVikingSearchHit,
    OpenVikingSessionSnapshot,
    OpenVikingSessionSurface,
)

_LAZY_EXPORTS = {
    "HttpOpenVikingMemoryGateway": "research_agent.adapters.openviking.memory_gateway",
    "NoopOpenVikingMemoryGateway": "research_agent.adapters.openviking.memory_gateway",
    "OpenVikingMemoryGateway": "research_agent.adapters.openviking.memory_gateway",
    "OpenVikingMemoryGatewayConfig": "research_agent.adapters.openviking.memory_gateway",
    "SurfaceBackedOpenVikingMemoryGateway": "research_agent.adapters.openviking.memory_gateway",
}

__all__ = [
    "NoopOpenVikingMemorySurface",
    "NoopOpenVikingMessageSurface",
    "NoopOpenVikingSessionSurface",
    "InMemoryOpenVikingMemorySurface",
    "InMemoryOpenVikingMessageSurface",
    "InMemoryOpenVikingSessionSurface",
    "OpenVikingAdapterSurfaceBundle",
    "OpenVikingMemorySearchResult",
    "OpenVikingMemoryRecord",
    "OpenVikingMemorySurface",
    "OpenVikingMessageRecord",
    "OpenVikingMessageSurface",
    "OpenVikingRetrievalAdapter",
    "OpenVikingSearchHit",
    "OpenVikingSurfaceConfig",
    "OpenVikingSessionSnapshot",
    "OpenVikingSessionSurface",
    "build_embedded_openviking_surface_bundle",
    "SDKBackedOpenVikingMemorySurface",
    "SDKBackedOpenVikingMessageSurface",
    "SDKBackedOpenVikingSessionSurface",
    "build_inmemory_openviking_surface_bundle",
    "build_sdk_openviking_surface_bundle",
]


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(name)
    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
