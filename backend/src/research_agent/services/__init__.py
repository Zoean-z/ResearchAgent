"""Lazy application service exports for the thin orchestration layer."""

from __future__ import annotations

from importlib import import_module


_EXPORT_MAP = {
    "ArxivHttpResponse": ("research_agent.services.arxiv_search_service", "ArxivHttpResponse"),
    "ArxivImportToolResult": ("research_agent.services.arxiv_import_tool_service", "ArxivImportToolResult"),
    "ArxivImportToolService": ("research_agent.services.arxiv_import_tool_service", "ArxivImportToolService"),
    "ArxivSearchError": ("research_agent.services.arxiv_search_service", "ArxivSearchError"),
    "ArxivSearchPaper": ("research_agent.services.arxiv_search_service", "ArxivSearchPaper"),
    "ArxivSearchResult": ("research_agent.services.arxiv_search_service", "ArxivSearchResult"),
    "ArxivSearchService": ("research_agent.services.arxiv_search_service", "ArxivSearchService"),
    "AcceptedTaskRun": ("research_agent.services.task_run_service", "AcceptedTaskRun"),
    "ChunkRerankResult": ("research_agent.services.context_rerank_service", "ChunkRerankResult"),
    "ContextRerankerClient": ("research_agent.services.context_rerank_service", "ContextRerankerClient"),
    "ContextRerankService": ("research_agent.services.context_rerank_service", "ContextRerankService"),
    "DeletionService": ("research_agent.services.deletion_service", "DeletionService"),
    "DeleteMemoryResult": ("research_agent.services.deletion_service", "DeleteMemoryResult"),
    "DeleteSessionResult": ("research_agent.services.deletion_service", "DeleteSessionResult"),
    "EntityNotFoundError": ("research_agent.services.errors", "EntityNotFoundError"),
    "HeuristicContextRerankerClient": ("research_agent.services.context_rerank_service", "HeuristicContextRerankerClient"),
    "IngestExecutionResult": ("research_agent.services.ingest_execution_service", "IngestExecutionResult"),
    "IngestExecutionService": ("research_agent.services.ingest_execution_service", "IngestExecutionService"),
    "IngestAnalysisService": ("research_agent.services.ingest_analysis_service", "IngestAnalysisService"),
    "MemoryAnalysisResult": ("research_agent.services.ingest_analysis_service", "MemoryAnalysisResult"),
    "IngestMaterializationResult": ("research_agent.services.ingest_materialization_service", "IngestMaterializationResult"),
    "IngestMaterializationService": ("research_agent.services.ingest_materialization_service", "IngestMaterializationService"),
    "InvalidIngestSourceError": ("research_agent.services.errors", "InvalidIngestSourceError"),
    "InvalidTaskRunStateError": ("research_agent.services.errors", "InvalidTaskRunStateError"),
    "MemoryExtractionResult": ("research_agent.services.memory_extraction_service", "MemoryExtractionResult"),
    "MemoryExtractionService": ("research_agent.services.memory_extraction_service", "MemoryExtractionService"),
    "MemoryBundleCatalog": ("research_agent.services.memory_bundle_service", "MemoryBundleCatalog"),
    "MemoryBundleGroup": ("research_agent.services.memory_bundle_service", "MemoryBundleGroup"),
    "MemoryBundleItem": ("research_agent.services.memory_bundle_service", "MemoryBundleItem"),
    "MemoryBundlePaperInfo": ("research_agent.services.memory_bundle_service", "MemoryBundlePaperInfo"),
    "MemoryBundleService": ("research_agent.services.memory_bundle_service", "MemoryBundleService"),
    "MemoryBundleSourceChunk": ("research_agent.services.memory_bundle_service", "MemoryBundleSourceChunk"),
    "MemoryRetrievalResult": ("research_agent.services.retrieval_service", "MemoryRetrievalResult"),
    "MemoryRerankResult": ("research_agent.services.context_rerank_service", "MemoryRerankResult"),
    "MemorySnapshot": ("research_agent.services.memory_snapshot_service", "MemorySnapshot"),
    "MemorySnapshotService": ("research_agent.services.memory_snapshot_service", "MemorySnapshotService"),
    "MessageIntakeRequest": ("research_agent.services.message_intake_service", "MessageIntakeRequest"),
    "MessageIntakeService": ("research_agent.services.message_intake_service", "MessageIntakeService"),
    "MessageQueryService": ("research_agent.services.message_query_service", "MessageQueryService"),
    "QueryExecutionResult": ("research_agent.services.query_execution_service", "QueryExecutionResult"),
    "QueryExecutionService": ("research_agent.services.query_execution_service", "QueryExecutionService"),
    "RetrievalPlan": ("research_agent.services.retrieval_service", "RetrievalPlan"),
    "RetrievalService": ("research_agent.services.retrieval_service", "RetrievalService"),
    "SessionService": ("research_agent.services.session_service", "SessionService"),
    "SourceRereadResult": ("research_agent.services.retrieval_service", "SourceRereadResult"),
    "SubmittedMessage": ("research_agent.services.message_intake_service", "SubmittedMessage"),
    "TaskRunService": ("research_agent.services.task_run_service", "TaskRunService"),
    "TimelineQueryService": ("research_agent.services.timeline_query_service", "TimelineQueryService"),
    "TraceQueryResult": ("research_agent.services.trace_query_service", "TraceQueryResult"),
    "TraceQueryService": ("research_agent.services.trace_query_service", "TraceQueryService"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(name)
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    return getattr(module, attr_name)


__all__ = list(_EXPORT_MAP)
