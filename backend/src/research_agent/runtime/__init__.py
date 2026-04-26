"""Runtime layer for handwritten orchestration."""

from __future__ import annotations

from importlib import import_module

from research_agent.runtime.agent_protocol import (
    AgentActionType,
    AgentObservation,
    AgentStopReason,
    AgentTurnClient,
    AgentTurnDecision,
    AgentTurnRequest,
)
from research_agent.runtime.query_orchestration import QueryOrchestrationRunner, QueryTurnSelection
from research_agent.runtime.query_turn import (
    QueryTurnClient,
    QueryTurnDecision,
    QueryTurnState,
)

_LAZY_EXPORTS = {
    "QueryRuntimeService": "research_agent.runtime.query_runtime_service",
    "RuntimeEventBroker": "research_agent.runtime.streaming",
    "RuntimeStreamEvent": "research_agent.runtime.streaming",
    "TaskRuntimeService": "research_agent.runtime.task_runtime_service",
}

__all__ = [
    "AgentActionType",
    "AgentObservation",
    "AgentStopReason",
    "AgentTurnClient",
    "AgentTurnDecision",
    "AgentTurnRequest",
    "QueryTurnClient",
    "QueryTurnDecision",
    "QueryTurnState",
    "QueryOrchestrationRunner",
    "QueryTurnSelection",
    "QueryRuntimeService",
    "RuntimeEventBroker",
    "RuntimeStreamEvent",
    "TaskRuntimeService",
]


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(name)
    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
