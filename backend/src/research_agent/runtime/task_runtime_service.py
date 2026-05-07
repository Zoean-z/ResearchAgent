"""Handwritten runtime coordination for mock task execution."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from research_agent.domain.enums import MessageType, TaskRunStatus
from research_agent.domain.models import OpenQuestionMemory, PaperMemory, RelationMemory, TraceNarrative
from research_agent.domain.ports import TraceRepositoryPort
from research_agent.runtime.streaming import RuntimeEventBroker, RuntimeStreamEvent
from research_agent.services.errors import EntityNotFoundError, InvalidTaskRunStateError
from research_agent.services.ingest_execution_service import IngestExecutionResult, IngestExecutionService
from research_agent.services.query_execution_models import QueryExecutionError
from research_agent.services.query_execution_service import QueryExecutionResult, QueryExecutionService
from research_agent.services.task_run_service import TaskRunService


class TaskRuntimeService:
    """Host-controlled runtime wrapper for mock task execution."""

    def __init__(
        self,
        task_run_service: TaskRunService,
        query_execution_service: QueryExecutionService,
        ingest_execution_service: IngestExecutionService,
        trace_repository: TraceRepositoryPort,
        max_steps: int = 8,
        runtime_event_broker: RuntimeEventBroker | None = None,
    ) -> None:
        self._task_run_service = task_run_service
        self._query_execution_service = query_execution_service
        self._ingest_execution_service = ingest_execution_service
        self._trace_repository = trace_repository
        self._max_steps = max_steps
        self._runtime_event_broker = runtime_event_broker

    def execute_query_run(self, session_id: str, run_id: str) -> QueryExecutionResult:
        """Execute a pending query run through the handwritten runtime shell."""

        message = self._task_run_service.get_message_for_run(session_id, run_id)
        if message.type is not MessageType.FOLLOWUP_QUERY:
            raise InvalidTaskRunStateError(run_id, MessageType.FOLLOWUP_QUERY.value, message.type.value)
        return cast(QueryExecutionResult, self.execute_task_run(session_id=session_id, run_id=run_id))

    def execute_ingest_run(self, session_id: str, run_id: str) -> IngestExecutionResult:
        """Execute a pending ingest run through the handwritten runtime shell."""

        message = self._task_run_service.get_message_for_run(session_id, run_id)
        if message.type not in {MessageType.INGEST_ARXIV, MessageType.INGEST_PDF}:
            raise InvalidTaskRunStateError(run_id, "ingest_message", message.type.value)
        return cast(IngestExecutionResult, self.execute_task_run(session_id=session_id, run_id=run_id))

    def execute_task_run(self, session_id: str, run_id: str) -> QueryExecutionResult | IngestExecutionResult:
        """Execute the accepted task run using the runtime-dispatched worker."""

        task_run = self._task_run_service.get_run(session_id, run_id)
        if task_run.status is not TaskRunStatus.PENDING:
            raise InvalidTaskRunStateError(run_id, TaskRunStatus.PENDING.value, task_run.status.value)

        message = self._task_run_service.get_message_for_run(session_id, run_id)
        running_run = self._task_run_service.mark_running(session_id, run_id)
        if self._runtime_event_broker is not None:
            self._runtime_event_broker.publish_run_started(running_run, message)

        return self.execute_running_task_run(session_id=session_id, run_id=run_id)

    def execute_running_task_run(self, session_id: str, run_id: str) -> QueryExecutionResult | IngestExecutionResult:
        """Execute a task run that has already transitioned to running."""

        task_run = self._task_run_service.get_run(session_id, run_id)
        if task_run.status is not TaskRunStatus.RUNNING:
            raise InvalidTaskRunStateError(run_id, TaskRunStatus.RUNNING.value, task_run.status.value)

        message = self._task_run_service.get_message_for_run(session_id, run_id)
        try:
            if message.type is MessageType.FOLLOWUP_QUERY:
                execution_result = self._query_execution_service.execute_query_run(session_id=session_id, run_id=run_id)
            elif message.type in {MessageType.INGEST_ARXIV, MessageType.INGEST_PDF}:
                execution_result = self._ingest_execution_service.execute_ingest_run(session_id=session_id, run_id=run_id)
            else:
                raise InvalidTaskRunStateError(run_id, "supported task message", message.type.value)

            self._write_trace_narratives(run_id, execution_result)
            step_count = len(self._trace_repository.list_steps(run_id))

            if step_count > self._max_steps:
                step_limited_run = self._task_run_service.mark_step_limit_reached(session_id, run_id, step_count)
                if self._runtime_event_broker is not None:
                    self._runtime_event_broker.publish(
                        RuntimeStreamEvent(
                            event_type="run_step_limit_reached",
                            session_id=step_limited_run.session_id,
                            run_id=step_limited_run.id,
                            payload={
                                "task_run": {
                                    "id": step_limited_run.id,
                                    "status": step_limited_run.status.value,
                                    "step_count": step_limited_run.step_count,
                                    "finished_at": step_limited_run.finished_at.isoformat() if step_limited_run.finished_at is not None else None,
                                    "finish_reason": step_limited_run.finish_reason,
                                }
                            },
                        )
                    )
                return replace(execution_result, task_run=step_limited_run)

            self._task_run_service.update_step_count(session_id, run_id, step_count)
            finished_run = self._task_run_service.finish_run(session_id, run_id, finish_reason=self._finish_reason(execution_result))
            if self._runtime_event_broker is not None:
                self._runtime_event_broker.publish_run_finished(finished_run)
            return replace(execution_result, task_run=finished_run)
        except Exception as error:
            self._mark_run_failed_if_still_running(session_id=session_id, run_id=run_id, error=error)
            raise

    def fail_running_task_run(self, session_id: str, run_id: str, reason: str, error: dict[str, object] | None = None):
        """Mark a running task as failed and publish a terminal event."""

        step_count = len(self._trace_repository.list_steps(run_id))
        self._task_run_service.update_step_count(session_id, run_id, step_count)
        failed_run = self._task_run_service.fail_run(session_id, run_id, reason)
        if self._runtime_event_broker is not None:
            self._runtime_event_broker.publish_run_failed(failed_run, reason, error)
        return failed_run

    def _mark_run_failed_if_still_running(self, *, session_id: str, run_id: str, error: Exception) -> None:
        current = self._task_run_service.get_run(session_id, run_id)
        if current.status is not TaskRunStatus.RUNNING:
            return
        if isinstance(error, QueryExecutionError):
            detail = error.to_dict()
            self.fail_running_task_run(session_id, run_id, error.detail.error_code, detail)
            return
        self.fail_running_task_run(
            session_id,
            run_id,
            type(error).__name__,
            {"error_message": str(error), "error_type": type(error).__name__},
        )

    def _write_trace_narratives(self, run_id: str, execution_result) -> None:
        existing = {narrative.trace_step_id for narrative in self._trace_repository.list_narratives(run_id)}
        for trace_step in self._trace_repository.list_steps(run_id):
            if trace_step.id in existing:
                continue
            self._trace_repository.save_narrative(
                TraceNarrative(
                    trace_step_id=trace_step.id,
                    reason_text=self._build_reason_text(trace_step.action, execution_result),
                    impact_text=self._build_impact_text(trace_step.action, execution_result),
                )
            )

    def _build_reason_text(self, action: str, execution_result) -> str:
        if action == "retrieve_session_memories":
            return "Session memory is checked first to preserve memory-first retrieval."
        if action == "retrieve_global_memories":
            return "Global memory is checked next to widen recall when session memory is not enough."
        if action == "rerank_context_candidates":
            return "The runtime reranked context candidates before deciding whether a reread is needed."
        if action == "decide_reread_source":
            reread_reason = getattr(execution_result, "reread_reason", None)
            if reread_reason is not None:
                return f"The runtime evaluates the post-rerank choice between rereading and answering: {reread_reason}."
            return "The runtime evaluates whether to reread source passages or answer directly."
        if action == "compose_mock_answer":
            return "A mock answer is composed from the reranked context output."
        if action == "direct_final_answer":
            return "The host answered a simple conversational turn without retrieval."
        if action == "reread_source_passages":
            return "The runtime reranks source passages when memory-first retrieval is insufficient."
        if action == "inspect_ingest_request":
            return "The runtime validates the accepted ingest request before continuing."
        if action == "stage_mock_artifact":
            return "A mock artifact staging step preserves the future adapter boundary."
        if action == "skip_real_extraction":
            return "Real ingestion is deferred; the scaffold only records that extraction would happen later."
        if action == "extract_local_pdf_text":
            return "The runtime extracts text from the local PDF before chunking it for retrieval."
        if action == "extract_arxiv_pdf_text":
            return "The runtime extracts text from the downloaded arXiv PDF before chunking it for retrieval."
        if action == "persist_pdf_chunks":
            return "Extracted PDF chunks are persisted so later retrieval can read source passages."
        if action == "persist_arxiv_chunks":
            return "Extracted arXiv PDF chunks are persisted so later retrieval can read source passages."
        if action == "compose_ingest_summary":
            source_type = getattr(execution_result, "source_type", None)
            if source_type is not None:
                return f"A source-backed ingest summary is composed from the extracted {source_type.value} text."
            return "A source-backed ingest summary is composed from the extracted source text."
        if action == "compose_mock_ingest_summary":
            summary = getattr(execution_result, "ingest_summary", None)
            if summary:
                return summary
            return "A mock ingest summary is composed from the accepted source."
        if action == "extract_paper_memory":
            return "The runtime distills the parsed source into paper-level memory."
        if action == "derive_relation_memory":
            return "The runtime looks for a relationship to an existing paper before finishing ingest."
        if action == "capture_open_questions":
            return "The runtime records any unresolved questions surfaced by the source."
        return "A runtime step was recorded for the task run."

    def _build_impact_text(self, action: str, execution_result) -> str:
        if action == "retrieve_session_memories":
            retrieval_plan = getattr(execution_result, "retrieval_plan", None)
            if retrieval_plan is not None:
                descriptors = self._memory_descriptors(retrieval_plan.session_memories.memories)
                reasons = getattr(retrieval_plan.session_memories, "selection_reasons", ())
                if descriptors:
                    if reasons:
                        return (
                            f"{len(retrieval_plan.session_memories.memories)} session memories were considered: {descriptors}. "
                            f"Selection reasons: {' | '.join(reasons)}."
                        )
                    return f"{len(retrieval_plan.session_memories.memories)} session memories were considered: {descriptors}."
                return f"{len(retrieval_plan.session_memories.memories)} session memories were considered."
            return "Session memory was considered before other sources."
        if action == "retrieve_global_memories":
            retrieval_plan = getattr(execution_result, "retrieval_plan", None)
            if retrieval_plan is not None:
                descriptors = self._memory_descriptors(retrieval_plan.global_memories.memories)
                reasons = getattr(retrieval_plan.global_memories, "selection_reasons", ())
                if descriptors:
                    if reasons:
                        return (
                            f"{len(retrieval_plan.global_memories.memories)} global memories were considered: {descriptors}. "
                            f"Selection reasons: {' | '.join(reasons)}."
                        )
                    return f"{len(retrieval_plan.global_memories.memories)} global memories were considered: {descriptors}."
                return f"{len(retrieval_plan.global_memories.memories)} global memories were considered."
            return "Global memory was considered after session memory."
        if action == "rerank_context_candidates":
            selection_source = getattr(execution_result, "memory_selection_source", None)
            fallback_used = getattr(execution_result, "memory_selection_fallback_used", False)
            citations = getattr(execution_result, "used_memory_citations", ())
            if citations:
                memory_text = ", ".join(f"{citation.memory_type}:{citation.memory_id}" for citation in citations)
                if selection_source is not None:
                    if fallback_used:
                        return f"Bounded memory candidates were reranked via fallback and selected as {memory_text}."
                    return f"Bounded memory candidates were reranked by {selection_source} and selected as {memory_text}."
                return f"Bounded memory candidates were reranked and selected as {memory_text}."
            return "Bounded memory candidates were reranked before reread gating."
        if action == "decide_reread_source":
            reread_reason = getattr(execution_result, "reread_reason", None)
            if reread_reason is not None:
                citations = getattr(execution_result, "used_memory_citations", ())
                if citations:
                    memory_text = ", ".join(f"{citation.memory_type}:{citation.memory_id}" for citation in citations)
                    return f"Reread decision: {getattr(execution_result, 'should_reread_source', False)} based on {memory_text}. Reason: {reread_reason}."
                return f"Reread decision: {getattr(execution_result, 'should_reread_source', False)}. Reason: {reread_reason}."
            return "The runtime captured the reread decision point."
        if action == "compose_mock_answer":
            citations = getattr(execution_result, "used_memory_citations", ())
            source_reread_chunks = getattr(execution_result, "source_reread_chunks", ())
            if citations:
                memory_text = ", ".join(f"{citation.memory_type}:{citation.memory_id}" for citation in citations)
                if source_reread_chunks:
                    chunk_text = ", ".join(citation.chunk_id for citation in source_reread_chunks)
                    memory_strategy = getattr(execution_result, "memory_selection_source", "model")
                    source_strategy = getattr(execution_result, "source_selection_source", None) or "none"
                    return f"The mock answer was grounded in {memory_text} (memory rerank={memory_strategy}) and reread chunks {chunk_text} (source rerank={source_strategy})."
                memory_strategy = getattr(execution_result, "memory_selection_source", "model")
                return f"The mock answer was grounded in {memory_text} (memory rerank={memory_strategy})."
            return "The run can finish without a real source reread."
        if action == "direct_final_answer":
            return "The memory-first tool loop was skipped because retrieval would not improve this turn."
        if action == "reread_source_passages":
            source_reread_chunks = getattr(execution_result, "source_reread_chunks", ())
            if source_reread_chunks:
                chunk_text = ", ".join(citation.chunk_id for citation in source_reread_chunks)
                reason_text = " | ".join(citation.selection_reason for citation in source_reread_chunks)
                source_strategy = getattr(execution_result, "source_selection_source", None)
                if source_strategy is not None:
                    return (
                        f"Source reread reranked {len(source_reread_chunks)} chunks: {chunk_text}. "
                        f"Selection reasons: {reason_text}. Strategy: {source_strategy}."
                    )
                return f"Source reread selected {len(source_reread_chunks)} chunks: {chunk_text}. Selection reasons: {reason_text}."
            return "Source reread found no matching chunks."
        if action == "inspect_ingest_request":
            source_type = getattr(execution_result, "source_type", None)
            if source_type is not None:
                return f"Ingest request type: {source_type.value}."
            return "The ingest request is validated before mock execution."
        if action == "stage_mock_artifact":
            return "A placeholder artifact record is reserved for later adapter work."
        if action == "skip_real_extraction":
            return "No real parser or model is invoked in this scaffold step."
        if action == "extract_local_pdf_text":
            chunk_count = getattr(execution_result, "chunk_count", None)
            if chunk_count is not None:
                return f"{chunk_count} PDF chunks were extracted for later retrieval."
            return "Local PDF text was extracted for later retrieval."
        if action == "extract_arxiv_pdf_text":
            chunk_count = getattr(execution_result, "chunk_count", None)
            if chunk_count is not None:
                return f"{chunk_count} arXiv PDF chunks were extracted for later retrieval."
            return "ArXiv PDF text was extracted for later retrieval."
        if action == "persist_pdf_chunks":
            chunk_count = getattr(execution_result, "chunk_count", None)
            if chunk_count is not None:
                return f"{chunk_count} extracted chunks were persisted."
            return "The extracted PDF chunks were persisted."
        if action == "persist_arxiv_chunks":
            chunk_count = getattr(execution_result, "chunk_count", None)
            if chunk_count is not None:
                return f"{chunk_count} extracted arXiv chunks were persisted."
            return "The extracted arXiv PDF chunks were persisted."
        if action == "compose_ingest_summary":
            return "The ingest run can finish from extracted source text without a model call."
        if action == "compose_mock_ingest_summary":
            return "The ingest run can finish without a real parser or model call."
        if action == "extract_paper_memory":
            memory_extraction = getattr(execution_result, "memory_extraction", None)
            if memory_extraction is not None:
                return f"Paper memory {memory_extraction.paper_operation}."
            return "Paper memory was distilled from the parsed source."
        if action == "derive_relation_memory":
            memory_extraction = getattr(execution_result, "memory_extraction", None)
            if memory_extraction is not None:
                if memory_extraction.relation_memory is not None:
                    return f"Relation memory {memory_extraction.relation_operation}."
                return "No related paper was available, so relation memory was skipped."
            return "The runtime checked for a paper-to-paper relation."
        if action == "capture_open_questions":
            memory_extraction = getattr(execution_result, "memory_extraction", None)
            if memory_extraction is not None:
                return f"Open question memory {memory_extraction.open_question_operation}."
            return "Open questions were recorded from the parsed source."
        return "The runtime state advanced."

    def _memory_descriptors(self, memories) -> str:
        descriptors: list[str] = []
        for memory in memories:
            if isinstance(memory, PaperMemory):
                memory_type = "paper_memory"
            elif isinstance(memory, RelationMemory):
                memory_type = "relation_memory"
            elif isinstance(memory, OpenQuestionMemory):
                memory_type = "open_question_memory"
            else:
                memory_type = memory.__class__.__name__.lower()
            descriptors.append(f"{memory_type}:{memory.id}")
        return ", ".join(descriptors)

    def _finish_reason(self, execution_result) -> str:
        if hasattr(execution_result, "retrieval_plan"):
            return "mock_query_completed"
        return "mock_ingest_completed"
