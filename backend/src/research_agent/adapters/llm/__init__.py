"""LLM adapter boundaries for future model-backed planning and synthesis."""

from research_agent.adapters.llm.query_agent import (
    DeepSeekStructuredQueryAgentTransport,
    ModelBackedQueryAgentClient,
    StaticStructuredQueryAgentTransport,
    StructuredQueryAgentChoice,
    StructuredQueryAgentPrompt,
    StructuredQueryAgentTransport,
    UnavailableStructuredQueryAgentTransport,
)
from research_agent.adapters.llm.ingest_extraction import (
    DeepSeekStructuredIngestExtractionTransport,
    ModelBackedIngestExtractionClient,
    StaticStructuredIngestExtractionTransport,
    StructuredIngestExtractionChoice,
    StructuredIngestExtractionPrompt,
    StructuredIngestExtractionTransport,
    UnavailableStructuredIngestExtractionTransport,
)
from research_agent.adapters.llm.query_tool_planner import (
    DeepSeekStructuredPlannerTransport,
    ModelBackedQueryToolPlannerClient,
    StaticStructuredPlannerTransport,
    StructuredPlannerChoice,
    StructuredPlannerPrompt,
    StructuredPlannerTransport,
    UnavailableStructuredPlannerTransport,
)

__all__ = [
    "DeepSeekStructuredQueryAgentTransport",
    "DeepSeekStructuredIngestExtractionTransport",
    "DeepSeekStructuredPlannerTransport",
    "ModelBackedIngestExtractionClient",
    "ModelBackedQueryAgentClient",
    "ModelBackedQueryToolPlannerClient",
    "StaticStructuredIngestExtractionTransport",
    "StaticStructuredQueryAgentTransport",
    "StaticStructuredPlannerTransport",
    "StructuredIngestExtractionChoice",
    "StructuredIngestExtractionPrompt",
    "StructuredIngestExtractionTransport",
    "StructuredQueryAgentChoice",
    "StructuredQueryAgentPrompt",
    "StructuredQueryAgentTransport",
    "StructuredPlannerChoice",
    "StructuredPlannerPrompt",
    "StructuredPlannerTransport",
    "UnavailableStructuredIngestExtractionTransport",
    "UnavailableStructuredQueryAgentTransport",
    "UnavailableStructuredPlannerTransport",
]
