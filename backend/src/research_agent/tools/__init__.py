"""Internal tool registry and frozen query tool protocol."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    # registry
    "InternalToolRegistry": ("research_agent.tools.registry", "InternalToolRegistry"),
    "PaperRegistrationResult": ("research_agent.tools.registry", "PaperRegistrationResult"),
    "RegistryToolEntry": ("research_agent.tools.registry", "RegistryToolEntry"),
    # protocol enums
    "QueryToolName": ("research_agent.tools.protocol", "QueryToolName"),
    "ToolErrorCode": ("research_agent.tools.protocol", "ToolErrorCode"),
    # protocol descriptors
    "ChunkDescriptor": ("research_agent.tools.protocol", "ChunkDescriptor"),
    "ConversationEvidenceRefDescriptor": ("research_agent.tools.protocol", "ConversationEvidenceRefDescriptor"),
    "GetConversationContextInput": ("research_agent.tools.protocol", "GetConversationContextInput"),
    "GetConversationContextOutput": ("research_agent.tools.protocol", "GetConversationContextOutput"),
    "GetPaperMemoryBundleInput": ("research_agent.tools.protocol", "GetPaperMemoryBundleInput"),
    "GetPaperMemoryBundleOutput": ("research_agent.tools.protocol", "GetPaperMemoryBundleOutput"),
    "MemoryDescriptor": ("research_agent.tools.protocol", "MemoryDescriptor"),
    "OpenVikingHitDescriptor": ("research_agent.tools.protocol", "OpenVikingHitDescriptor"),
    "ListRecentMessagesInput": ("research_agent.tools.protocol", "ListRecentMessagesInput"),
    "ListRecentMessagesOutput": ("research_agent.tools.protocol", "ListRecentMessagesOutput"),
    "ListSessionPapersInput": ("research_agent.tools.protocol", "ListSessionPapersInput"),
    "ListSessionPapersOutput": ("research_agent.tools.protocol", "ListSessionPapersOutput"),
    "PaperInfoDescriptor": ("research_agent.tools.protocol", "PaperInfoDescriptor"),
    "PaperMemoryBundleDescriptor": ("research_agent.tools.protocol", "PaperMemoryBundleDescriptor"),
    "RecentConversationContextDescriptor": ("research_agent.tools.protocol", "RecentConversationContextDescriptor"),
    "RecentConversationMessageDescriptor": ("research_agent.tools.protocol", "RecentConversationMessageDescriptor"),
    # protocol input models
    "SearchSessionMemoryInput": ("research_agent.tools.protocol", "SearchSessionMemoryInput"),
    "SearchGlobalMemoryInput": ("research_agent.tools.protocol", "SearchGlobalMemoryInput"),
    "SearchOpenVikingMemoryInput": ("research_agent.tools.protocol", "SearchOpenVikingMemoryInput"),
    "SearchSourceChunksInput": ("research_agent.tools.protocol", "SearchSourceChunksInput"),
    "RerankCandidatesInput": ("research_agent.tools.protocol", "RerankCandidatesInput"),
    "ReadSourcePassagesInput": ("research_agent.tools.protocol", "ReadSourcePassagesInput"),
    "ComposeAnswerInput": ("research_agent.tools.protocol", "ComposeAnswerInput"),
    # protocol output models
    "SearchSessionMemoryOutput": ("research_agent.tools.protocol", "SearchSessionMemoryOutput"),
    "SearchGlobalMemoryOutput": ("research_agent.tools.protocol", "SearchGlobalMemoryOutput"),
    "SearchOpenVikingMemoryOutput": ("research_agent.tools.protocol", "SearchOpenVikingMemoryOutput"),
    "SearchSourceChunksOutput": ("research_agent.tools.protocol", "SearchSourceChunksOutput"),
    "RerankCandidatesOutput": ("research_agent.tools.protocol", "RerankCandidatesOutput"),
    "ReadSourcePassagesOutput": ("research_agent.tools.protocol", "ReadSourcePassagesOutput"),
    "ComposeAnswerOutput": ("research_agent.tools.protocol", "ComposeAnswerOutput"),
    # protocol envelopes
    "ToolDefinition": ("research_agent.tools.protocol", "ToolDefinition"),
    "ToolError": ("research_agent.tools.protocol", "ToolError"),
    "ToolRequest": ("research_agent.tools.protocol", "ToolRequest"),
    "ToolResponse": ("research_agent.tools.protocol", "ToolResponse"),
    "ToolOutcome": ("research_agent.tools.protocol", "ToolOutcome"),
    # protocol constants
    "QUERY_TOOL_DEFINITIONS": ("research_agent.tools.protocol", "QUERY_TOOL_DEFINITIONS"),
    "QUERY_TOOL_BY_NAME": ("research_agent.tools.protocol", "QUERY_TOOL_BY_NAME"),
    # protocol helpers
    "get_query_tool_definition": ("research_agent.tools.protocol", "get_query_tool_definition"),
    "is_query_tool": ("research_agent.tools.protocol", "is_query_tool"),
    "validate_tool_request": ("research_agent.tools.protocol", "validate_tool_request"),
    # planner
    "HOST_CONTROLLED_QUERY_TOOLS": ("research_agent.tools.query_planner", "HOST_CONTROLLED_QUERY_TOOLS"),
    "HeuristicQueryToolPlannerClient": ("research_agent.tools.query_planner", "HeuristicQueryToolPlannerClient"),
    "QueryToolPlannerClient": ("research_agent.tools.query_planner", "QueryToolPlannerClient"),
    "QueryToolPlannerDecision": ("research_agent.tools.query_planner", "QueryToolPlannerDecision"),
    "QueryToolPlannerState": ("research_agent.tools.query_planner", "QueryToolPlannerState"),
    # executor
    "QueryToolExecutor": ("research_agent.tools.query_executor", "QueryToolExecutor"),
    "ToolExecutionEnvelope": ("research_agent.tools.query_executor", "ToolExecutionEnvelope"),
    # query agent compatibility aliases
    "PlannerBackedQueryAgentClient": ("research_agent.tools.query_agent", "PlannerBackedQueryAgentClient"),
    "QueryAgentClient": ("research_agent.tools.query_agent", "QueryAgentClient"),
    "QueryAgentDecision": ("research_agent.tools.query_agent", "QueryAgentDecision"),
    "QueryAgentState": ("research_agent.tools.query_agent", "QueryAgentState"),
    "QueryAgentTurnClient": ("research_agent.tools.query_agent", "QueryAgentTurnClient"),
    "StaticFinalAnswerQueryAgentClient": ("research_agent.tools.query_agent", "StaticFinalAnswerQueryAgentClient"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
