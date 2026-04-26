"""Unified message intake for query and ingest entry points."""

from __future__ import annotations

from dataclasses import dataclass

from research_agent.adapters.openviking import OpenVikingAdapterSurfaceBundle, OpenVikingMessageRecord
from research_agent.domain.enums import MessageType
from research_agent.domain.policies import should_mirror_message_to_openviking
from research_agent.domain.models import Message, TaskRun
from research_agent.services.errors import EntityNotFoundError
from research_agent.services.task_run_service import TaskRunService


@dataclass(frozen=True, slots=True)
class SubmittedMessage:
    """Unified message intake result."""

    message: Message
    task_run: TaskRun
    message_type: MessageType


@dataclass(frozen=True, slots=True)
class MessageIntakeRequest:
    """Unified message intake payload."""

    text: str | None = None
    arxiv_url: str | None = None
    file_path: str | None = None


class MessageIntakeService:
    """Classify message input and create the matching accepted task run."""

    def __init__(
        self,
        task_run_service: TaskRunService,
        openviking_bundle: OpenVikingAdapterSurfaceBundle | None = None,
    ) -> None:
        self._task_run_service = task_run_service
        self._openviking_bundle = openviking_bundle or OpenVikingAdapterSurfaceBundle()

    def submit(self, session_id: str, request: MessageIntakeRequest) -> SubmittedMessage:
        """Accept a unified message submission."""

        kind, value = self._classify(request)
        if kind is MessageType.FOLLOWUP_QUERY:
            accepted = self._task_run_service.accept_followup_query(session_id=session_id, query=value)
        elif kind is MessageType.INGEST_ARXIV:
            accepted = self._task_run_service.accept_arxiv_ingest(session_id=session_id, arxiv_url=value)
        elif kind is MessageType.INGEST_PDF:
            accepted = self._task_run_service.accept_pdf_ingest(session_id=session_id, file_path=value)
        else:  # pragma: no cover - defensive branch
            raise EntityNotFoundError("MessageType", kind.value)
        self._mirror_to_openviking(accepted.message)
        return SubmittedMessage(message=accepted.message, task_run=accepted.task_run, message_type=kind)

    def submit_followup_query(self, session_id: str, query: str) -> SubmittedMessage:
        """Compatibility wrapper for follow-up query submissions."""

        return self.submit(session_id, MessageIntakeRequest(text=query))

    def submit_arxiv_ingest(self, session_id: str, arxiv_url: str) -> SubmittedMessage:
        """Compatibility wrapper for arXiv ingest submissions."""

        return self.submit(session_id, MessageIntakeRequest(arxiv_url=arxiv_url))

    def submit_pdf_ingest(self, session_id: str, file_path: str) -> SubmittedMessage:
        """Compatibility wrapper for local PDF ingest submissions."""

        return self.submit(session_id, MessageIntakeRequest(file_path=file_path))

    def _classify(self, request: MessageIntakeRequest) -> tuple[MessageType, str]:
        provided = [item for item in (request.text, request.arxiv_url, request.file_path) if item is not None and item.strip()]
        if len(provided) != 1:
            raise ValueError("Provide exactly one of text, arxiv_url, or file_path.")
        if request.arxiv_url is not None and request.arxiv_url.strip():
            return MessageType.INGEST_ARXIV, request.arxiv_url.strip()
        if request.file_path is not None and request.file_path.strip():
            return MessageType.INGEST_PDF, request.file_path.strip()
        return MessageType.FOLLOWUP_QUERY, request.text.strip() if request.text is not None else ""

    def _mirror_to_openviking(self, message: Message) -> None:
        if not should_mirror_message_to_openviking(message.type):
            return
        self._openviking_bundle.sessions.ensure_session(message.session_id)
        self._openviking_bundle.messages.mirror_message(
            OpenVikingMessageRecord(
                session_id=message.session_id,
                message_id=message.id,
                role=message.role,
                content=message.content,
                metadata={"message_type": message.type.value, "status": message.status},
            )
        )
        self._openviking_bundle.sessions.commit_session(message.session_id)


__all__ = [
    "MessageIntakeRequest",
    "MessageIntakeService",
    "SubmittedMessage",
]
