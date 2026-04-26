"""Thin execution service for mock ingest runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from research_agent.adapters.openviking import OpenVikingAdapterSurfaceBundle, OpenVikingMessageRecord
from research_agent.domain.enums import MessageType, SourceType, TaskRunStatus
from research_agent.domain.models import Message, TaskRun, TimelineEvent, TraceStep
from research_agent.domain.ports import MessageRepositoryPort, TimelineRepositoryPort, TraceRepositoryPort
from research_agent.runtime.streaming import RuntimeEventBroker
from research_agent.runtime.ingest_extraction import IngestPaperSummaryDraft
from research_agent.tools.registry import InternalToolRegistry
from research_agent.services.errors import EntityNotFoundError, InvalidTaskRunStateError
from research_agent.services.memory_extraction_service import MemoryExtractionResult, MemoryExtractionService
from research_agent.services.ingest_materialization_service import (
    IngestMaterializationResult,
    IngestMaterializationService,
)


@dataclass(frozen=True, slots=True)
class IngestExecutionResult:
    """Structured ingest execution output."""

    task_run: TaskRun
    source_type: SourceType
    materialization: IngestMaterializationResult
    memory_extraction: MemoryExtractionResult
    paper_summary: IngestPaperSummaryDraft
    ingest_summary: str
    chunk_count: int


class IngestExecutionService:
    """Execute a pending ingest run using the thin handwritten runtime."""

    def __init__(
        self,
        message_repository: MessageRepositoryPort,
        materialization_service: IngestMaterializationService,
        memory_extraction_service: MemoryExtractionService,
        trace_repository: TraceRepositoryPort,
        timeline_repository: TimelineRepositoryPort,
        tool_registry: InternalToolRegistry | None = None,
        openviking_bundle: OpenVikingAdapterSurfaceBundle | None = None,
        runtime_event_broker: RuntimeEventBroker | None = None,
    ) -> None:
        self._message_repository = message_repository
        self._materialization_service = materialization_service
        self._memory_extraction_service = memory_extraction_service
        self._trace_repository = trace_repository
        self._timeline_repository = timeline_repository
        self._tool_registry = tool_registry
        self._openviking_bundle = openviking_bundle or OpenVikingAdapterSurfaceBundle()
        self._runtime_event_broker = runtime_event_broker

    def execute_ingest_run(self, session_id: str, run_id: str) -> IngestExecutionResult:
        """Run the mock ingest chain and persist trace/timeline placeholders."""

        task_run = self._trace_repository.get_run(run_id)
        if task_run is None:
            raise EntityNotFoundError("TaskRun", run_id)
        if task_run.session_id != session_id:
            raise EntityNotFoundError("TaskRun", run_id)
        if task_run.status is not TaskRunStatus.RUNNING:
            raise InvalidTaskRunStateError(run_id, TaskRunStatus.RUNNING.value, task_run.status.value)

        message = self._message_repository.get_by_id(task_run.message_id)
        if message is None:
            raise EntityNotFoundError("Message", task_run.message_id)
        if message.type not in {MessageType.INGEST_ARXIV, MessageType.INGEST_PDF}:
            raise InvalidTaskRunStateError(run_id, "ingest_message", message.type.value)

        source_type = SourceType.ARXIV if message.type is MessageType.INGEST_ARXIV else SourceType.PDF
        materialization = self._materialize_source(session_id=session_id, source_type=source_type, source_value=message.content)
        memory_extraction = self._extract_memories(session_id=session_id, paper_id=materialization.paper.id)
        self._write_trace_and_timeline(
            session_id=session_id,
            run_id=run_id,
            source_type=source_type,
            source_value=message.content,
            materialization=materialization,
            memory_extraction=memory_extraction,
        )

        summary = self._compose_real_summary(
            source_type,
            message.content,
            materialization.chunk_count,
            memory_extraction.paper_summary,
        )
        self._persist_assistant_summary(
            session_id=session_id,
            run_id=run_id,
            source_type=source_type,
            summary=summary,
            user_source_value=message.content,
            paper_id=materialization.paper.id,
        )
        return IngestExecutionResult(
            task_run=task_run,
            source_type=source_type,
            materialization=materialization,
            memory_extraction=memory_extraction,
            paper_summary=memory_extraction.paper_summary,
            ingest_summary=summary,
            chunk_count=materialization.chunk_count,
        )

    def _materialize_source(self, *, session_id: str, source_type: SourceType, source_value: str) -> IngestMaterializationResult:
        if source_type is SourceType.ARXIV:
            return self._materialization_service.materialize_arxiv_source(session_id=session_id, arxiv_url=source_value)
        return self._materialization_service.materialize_pdf_source(session_id=session_id, file_path=source_value)

    def _extract_memories(self, *, session_id: str, paper_id: str) -> MemoryExtractionResult:
        if self._tool_registry is not None:
            return self._tool_registry.extract_memories(session_id=session_id, paper_id=paper_id)
        return self._memory_extraction_service.extract_and_store_memories(session_id=session_id, paper_id=paper_id)

    def _write_trace_and_timeline(
        self,
        *,
        session_id: str,
        run_id: str,
        source_type: SourceType,
        source_value: str,
        materialization: IngestMaterializationResult,
        memory_extraction: MemoryExtractionResult,
    ) -> None:
        self._save_trace_step(
            session_id=session_id,
            run_id=run_id,
            trace_step=TraceStep(
                run_id=run_id,
                action="inspect_ingest_request",
                input_payload={"session_id": session_id, "source_type": source_type.value, "source_value": source_value},
                result_payload={
                    "accepted": True,
                    "paper_id": materialization.paper.id,
                    "artifact_id": materialization.artifact.id,
                    "session_document_id": materialization.session_document.id,
                },
            ),
        )
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="step_completed",
                summary="已检查导入请求",
                related_paper_ids=[materialization.paper.id],
            )
        )

        source_label = "本地 PDF" if source_type is SourceType.PDF else "arXiv PDF"
        extract_action = "extract_local_pdf_text" if source_type is SourceType.PDF else "extract_arxiv_pdf_text"
        persist_action = "persist_pdf_chunks" if source_type is SourceType.PDF else "persist_arxiv_chunks"
        extract_summary = "已抽取 PDF 文本" if source_type is SourceType.PDF else "已抽取 arXiv PDF 文本"
        persist_summary = "已保存 PDF 分块" if source_type is SourceType.PDF else "已保存 arXiv 分块"

        self._save_trace_step(
            session_id=session_id,
            run_id=run_id,
            trace_step=TraceStep(
                run_id=run_id,
                action=extract_action,
                input_payload={"source_type": source_type.value},
                result_payload={
                    "artifact_id": materialization.artifact.id,
                    "paper_id": materialization.paper.id,
                    "chunk_count": materialization.chunk_count,
                },
            ),
        )
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="step_completed",
                summary=extract_summary,
                related_paper_ids=[materialization.paper.id],
            )
        )

        self._save_trace_step(
            session_id=session_id,
            run_id=run_id,
            trace_step=TraceStep(
                run_id=run_id,
                action=persist_action,
                input_payload={"source_type": source_type.value},
                result_payload={
                    "chunk_count": materialization.chunk_count,
                    "session_document_id": materialization.session_document.id,
                },
            ),
        )
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="step_completed",
                summary=persist_summary,
                related_paper_ids=[materialization.paper.id],
            )
        )

        self._save_trace_step(
            session_id=session_id,
            run_id=run_id,
            trace_step=TraceStep(
                run_id=run_id,
                action="compose_ingest_summary",
                input_payload={"source_type": source_type.value},
                result_payload={
                    "summary": self._compose_real_summary(
                        source_type,
                        source_value,
                        materialization.chunk_count,
                        memory_extraction.paper_summary,
                    ),
                    "paper_summary": asdict(memory_extraction.paper_summary),
                    "paper_id": materialization.paper.id,
                },
            ),
        )
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="step_completed",
                summary=f"已生成{source_label}摘要",
                related_paper_ids=[materialization.paper.id],
            )
        )

        self._save_trace_step(
            session_id=session_id,
            run_id=run_id,
            trace_step=TraceStep(
                run_id=run_id,
                action="extract_paper_memory",
                input_payload={"paper_id": materialization.paper.id},
                result_payload={
                    "paper_memory_id": memory_extraction.paper_memory.id,
                    "paper_operation": memory_extraction.paper_operation,
                },
            ),
        )
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="memory_completed",
                summary="已抽取论文记忆",
                related_paper_ids=[materialization.paper.id],
                related_memory_ids=[memory_extraction.paper_memory.id],
            )
        )

        self._save_trace_step(
            session_id=session_id,
            run_id=run_id,
            trace_step=TraceStep(
                run_id=run_id,
                action="derive_relation_memory",
                input_payload={"paper_id": materialization.paper.id},
                result_payload={
                    "relation_memory_id": memory_extraction.relation_memory.id if memory_extraction.relation_memory else None,
                    "relation_operation": memory_extraction.relation_operation,
                },
            ),
        )
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="memory_completed",
                summary="已生成关系记忆",
                related_paper_ids=[materialization.paper.id],
                related_memory_ids=[memory_extraction.relation_memory.id] if memory_extraction.relation_memory else [],
            )
        )

        self._save_trace_step(
            session_id=session_id,
            run_id=run_id,
            trace_step=TraceStep(
                run_id=run_id,
                action="capture_open_questions",
                input_payload={"paper_id": materialization.paper.id},
                result_payload={
                    "open_question_memory_id": memory_extraction.open_question_memory.id,
                    "open_question_operation": memory_extraction.open_question_operation,
                },
            ),
        )
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="memory_completed",
                summary="已记录开放问题",
                related_paper_ids=[materialization.paper.id],
                related_memory_ids=[memory_extraction.open_question_memory.id],
            )
        )
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="run_finished",
                summary="导入运行已完成",
                related_paper_ids=[materialization.paper.id],
            )
        )

    def _compose_mock_summary(self, source_type: SourceType, source_value: str) -> str:
        source_label = "本地 PDF" if source_type is SourceType.PDF else "arXiv PDF"
        return f"模拟导入 {source_label}：{source_value}"

    def _compose_real_summary(
        self,
        source_type: SourceType,
        source_value: str,
        chunk_count: int,
        paper_summary: IngestPaperSummaryDraft,
    ) -> str:
        source_label = "本地 PDF" if source_type is SourceType.PDF else "arXiv PDF"
        lines = [
            f"已解析 {source_label}：{source_value}（{chunk_count} 个文本分块）",
            f"- 论文主题：{paper_summary.what_it_is_about}",
            f"- 解决的问题：{paper_summary.problem_solved}",
        ]
        if paper_summary.new_ideas:
            lines.append(f"- 新想法：{'；'.join(paper_summary.new_ideas)}")
        if paper_summary.limitations:
            lines.append(f"- 局限性：{'；'.join(paper_summary.limitations)}")
        if paper_summary.suggestions_or_questions:
            lines.append(f"- 后续建议或问题：{'；'.join(paper_summary.suggestions_or_questions)}")
        return "\n".join(lines)

    def _persist_assistant_summary(
        self,
        *,
        session_id: str,
        run_id: str,
        source_type: SourceType,
        summary: str,
        user_source_value: str,
        paper_id: str,
    ) -> Message:
        message_type = MessageType.INGEST_ARXIV if source_type is SourceType.ARXIV else MessageType.INGEST_PDF
        assistant_message = self._message_repository.save(
            Message(
                session_id=session_id,
                role="assistant",
                type=message_type,
                content=summary,
                status="completed",
            )
        )
        self._openviking_bundle.sessions.ensure_session(session_id)
        self._openviking_bundle.messages.mirror_message(
            OpenVikingMessageRecord(
                session_id=session_id,
                message_id=assistant_message.id,
                role="assistant",
                content=assistant_message.content,
                metadata={
                    "message_type": assistant_message.type.value,
                    "status": assistant_message.status,
                    "run_id": run_id,
                    "source_type": source_type.value,
                    "source_value": user_source_value,
                    "paper_id": paper_id,
                },
            )
        )
        self._openviking_bundle.sessions.commit_session(session_id)
        self._publish_assistant_message(run_id=run_id, message=assistant_message)
        return assistant_message

    def _save_trace_step(self, *, session_id: str, run_id: str, trace_step: TraceStep) -> None:
        self._trace_repository.save_step(trace_step)
        if self._runtime_event_broker is None:
            return
        self._runtime_event_broker.publish_step_completed(session_id, run_id, trace_step)

    def _publish_assistant_message(self, *, run_id: str, message: Message) -> None:
        if self._runtime_event_broker is None:
            return
        self._runtime_event_broker.publish_assistant_message(run_id, message)
