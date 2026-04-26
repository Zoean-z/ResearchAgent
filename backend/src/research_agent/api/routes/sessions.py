"""Mock session routes backed by in-memory repositories."""

from __future__ import annotations

from pathlib import Path
import re
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from research_agent.api.deps import ServiceBundle, get_service_bundle
from research_agent.api.schemas import (
    CreateSessionRequest,
    DeleteMemoryResponse,
    DeleteSessionResponse,
    MemorySnapshotResponse,
    MessageIntakeRequest,
    MessageListResponse,
    MessageResponse,
    MessageSubmissionResponse,
    OpenQuestionMemoryResponse,
    PaperMemoryResponse,
    RelationMemoryResponse,
    SessionListResponse,
    SessionResponse,
    TimelineEventResponse,
    TimelineResponse,
)
from research_agent.services.message_intake_service import MessageIntakeRequest as MessageIntakePayload
from research_agent.services import EntityNotFoundError

router = APIRouter(prefix="/api/sessions", tags=["sessions"])
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


@router.post("/{session_id}/messages", response_model=MessageSubmissionResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_message(
    session_id: str,
    request: MessageIntakeRequest,
    services: ServiceBundle = Depends(get_service_bundle),
) -> MessageSubmissionResponse:
    """Submit a unified message that the system classifies into a query or ingest run."""

    submitted = services.message_intake.submit(
        session_id=session_id,
        request=MessageIntakePayload(text=request.text, arxiv_url=request.arxiv_url, file_path=request.file_path),
    )
    return _submission_response(session_id, submitted.message.id, submitted.task_run.id, submitted.message_type.value)


@router.post("/{session_id}/uploads/pdf", response_model=MessageSubmissionResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_pdf(
    session_id: str,
    file: UploadFile = File(...),
    services: ServiceBundle = Depends(get_service_bundle),
) -> MessageSubmissionResponse:
    """Upload a PDF from the browser and route it into the existing ingest path."""

    _get_session_or_404(session_id, services)
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF uploads are supported.")

    upload_path = _build_upload_path(session_id, filename)
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded PDF is empty.")
    upload_path.write_bytes(payload)

    submitted = services.message_intake.submit_pdf_ingest(session_id=session_id, file_path=str(upload_path))
    return _submission_response(session_id, submitted.message.id, submitted.task_run.id, submitted.message_type.value)


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    request: CreateSessionRequest,
    services: ServiceBundle = Depends(get_service_bundle),
) -> SessionResponse:
    """Create a new session using the thin application service layer."""

    session = services.sessions.create_session(request.title)
    return SessionResponse.from_domain(session)


@router.get("", response_model=SessionListResponse)
def list_sessions(
    services: ServiceBundle = Depends(get_service_bundle),
) -> SessionListResponse:
    """List all known sessions."""

    sessions = [SessionResponse.from_domain(session) for session in services.sessions.list_sessions()]
    return SessionListResponse(items=sessions)


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(
    session_id: str,
    services: ServiceBundle = Depends(get_service_bundle),
) -> SessionResponse:
    """Fetch a single session."""

    session = _get_session_or_404(session_id, services)
    return SessionResponse.from_domain(session)


@router.get("/{session_id}/messages", response_model=MessageListResponse)
def list_messages(
    session_id: str,
    services: ServiceBundle = Depends(get_service_bundle),
) -> MessageListResponse:
    """List session messages from the in-memory mock repository."""

    items = [MessageResponse.from_domain(message) for message in services.messages.list_messages(session_id)]
    return MessageListResponse(items=items)


@router.get("/{session_id}/timeline", response_model=TimelineResponse)
def list_timeline(
    session_id: str,
    services: ServiceBundle = Depends(get_service_bundle),
) -> TimelineResponse:
    """List timeline events for a session."""

    items = [TimelineEventResponse.from_domain(event) for event in services.timeline.list_timeline(session_id)]
    return TimelineResponse(items=items)


@router.get("/{session_id}/memory-snapshot", response_model=MemorySnapshotResponse)
def get_memory_snapshot(
    session_id: str,
    services: ServiceBundle = Depends(get_service_bundle),
) -> MemorySnapshotResponse:
    """Return the current memory snapshot for a session."""

    snapshot = _get_memory_snapshot_or_404(session_id, services)
    return MemorySnapshotResponse(
        paper_memories=[
            PaperMemoryResponse.from_domain(memory)
            for memory in snapshot.paper_memories
        ],
        relation_memories=[
            RelationMemoryResponse.from_domain(memory)
            for memory in snapshot.relation_memories
        ],
        open_question_memories=[
            OpenQuestionMemoryResponse.from_domain(memory)
            for memory in snapshot.open_question_memories
        ],
    )


@router.delete("/{session_id}", response_model=DeleteSessionResponse)
def delete_session(
    session_id: str,
    services: ServiceBundle = Depends(get_service_bundle),
) -> DeleteSessionResponse:
    """Delete a session, its messages, runs, timeline, and mirrored memories."""

    try:
        deleted = services.deletions.delete_session(session_id)
    except EntityNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.") from error
    return DeleteSessionResponse(
        session=SessionResponse.from_domain(deleted.session),
        deleted_documents=deleted.deleted_documents,
        deleted_messages=deleted.deleted_messages,
        deleted_runs=deleted.deleted_runs,
        deleted_timeline_events=deleted.deleted_timeline_events,
        deleted_memories=deleted.deleted_memories,
        mirrored_to_openviking=deleted.mirrored_to_openviking,
    )


@router.delete("/{session_id}/memories/{memory_kind}/{memory_id}", response_model=DeleteMemoryResponse)
def delete_memory(
    session_id: str,
    memory_kind: str,
    memory_id: str,
    services: ServiceBundle = Depends(get_service_bundle),
) -> DeleteMemoryResponse:
    """Delete a single session-scoped memory item."""

    try:
        deleted = services.deletions.delete_memory(session_id, memory_kind, memory_id)
    except EntityNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found.") from error
    return DeleteMemoryResponse(
        session_id=deleted.session_id,
        memory_kind=deleted.memory_kind,
        memory_id=deleted.memory_id,
        deleted=deleted.deleted,
        mirrored_to_openviking=deleted.mirrored_to_openviking,
    )


def _get_session_or_404(session_id: str, services: ServiceBundle):
    try:
        return services.sessions.get_session(session_id)
    except EntityNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.") from error


def _get_memory_snapshot_or_404(session_id: str, services: ServiceBundle):
    try:
        return services.memory_snapshot.get_snapshot(session_id)
    except EntityNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.") from error


def _submission_response(session_id: str, message_id: str, run_id: str, message_type: str) -> MessageSubmissionResponse:
    return MessageSubmissionResponse(
        session_id=session_id,
        message_id=message_id,
        run_id=run_id,
        message_type=message_type,
    )


def _build_upload_path(session_id: str, filename: str) -> Path:
    safe_name = _UNSAFE_FILENAME_CHARS.sub("-", Path(filename).name).strip("-") or "upload.pdf"
    repo_root = Path(__file__).resolve().parents[5]
    upload_dir = repo_root / "data" / "artifacts" / "uploads" / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir / f"{uuid4()}-{safe_name}"
