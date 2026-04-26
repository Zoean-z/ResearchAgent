"""Framework-agnostic policy helpers for identity, memory, retrieval, and termination."""

from research_agent.domain.policies.canonical_key import build_canonical_key, normalize_arxiv_id
from research_agent.domain.policies.memory import (
    MemoryUpsertDecision,
    merge_open_question_memory,
    merge_paper_memory,
    merge_relation_memory,
)
from research_agent.domain.policies.retrieval import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    CONTEXT_CANDIDATE_TOP_K,
    CONTEXT_RERANK_TOP_K,
    GLOBAL_MEMORY_TOP_K,
    SESSION_MEMORY_TOP_K,
    RetrievalStep,
    build_reread_reason,
    get_followup_retrieval_plan,
    should_reread_source,
)
from research_agent.domain.policies.openviking import (
    OpenVikingDeletionPlan,
    OpenVikingOwnershipPlan,
    build_dialogue_deletion_plan,
    build_memory_deletion_plan,
    build_openviking_ownership_plan,
    should_mirror_memory_to_openviking,
    should_mirror_message_to_openviking,
)
from research_agent.domain.policies.termination import should_terminate_run

__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "CONTEXT_CANDIDATE_TOP_K",
    "CONTEXT_RERANK_TOP_K",
    "GLOBAL_MEMORY_TOP_K",
    "MemoryUpsertDecision",
    "OpenVikingDeletionPlan",
    "OpenVikingOwnershipPlan",
    "RetrievalStep",
    "SESSION_MEMORY_TOP_K",
    "build_dialogue_deletion_plan",
    "build_canonical_key",
    "build_reread_reason",
    "build_memory_deletion_plan",
    "build_openviking_ownership_plan",
    "get_followup_retrieval_plan",
    "merge_open_question_memory",
    "merge_paper_memory",
    "merge_relation_memory",
    "normalize_arxiv_id",
    "should_mirror_memory_to_openviking",
    "should_mirror_message_to_openviking",
    "should_reread_source",
    "should_terminate_run",
]
