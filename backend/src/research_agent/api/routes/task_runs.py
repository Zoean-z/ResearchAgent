"""Session-scoped acceptance and task-run routes."""

from __future__ import annotations

from collections.abc import Callable
import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from research_agent.api.deps import ServiceBundle, get_service_bundle
from research_agent.api.schemas import (
    FollowupQueryRequest,
    IngestAcceptedResponse,
    IngestArxivRequest,
    IngestPdfRequest,
    IngestExecutionResponse,
    QueryExecutionResponse,
    QueryAcceptedResponse,
    PaperSummaryResponse,
    TaskRunListResponse,
    TaskRunResponse,
    TimelineResponse,
    TraceResponse,
    TimelineEventResponse,
    TraceNarrativeResponse,
    TraceStepResponse,
)
from research_agent.services import AcceptedTaskRun, EntityNotFoundError, InvalidIngestSourceError, InvalidTaskRunStateError

router = APIRouter(prefix="/api/sessions/{session_id}", tags=["task-runs"])


@router.post("/ingest/arxiv", response_model=IngestAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
def accept_arxiv_ingest(
    session_id: str,
    request: IngestArxivRequest,
    services: ServiceBundle = Depends(get_service_bundle),
) -> IngestAcceptedResponse:
    """Accept an arXiv ingest request without executing real ingestion."""

    accepted = _accept_or_404(lambda: services.message_intake.submit_arxiv_ingest(session_id=session_id, arxiv_url=request.arxiv_url))
    return IngestAcceptedResponse(
        session_id=session_id,
        message_id=accepted.message.id,
        run_id=accepted.task_run.id,
    )


@router.post("/ingest/pdf", response_model=IngestAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
def accept_pdf_ingest(
    session_id: str,
    request: IngestPdfRequest,
    services: ServiceBundle = Depends(get_service_bundle),
) -> IngestAcceptedResponse:
    """Accept a local PDF ingest request without executing real ingestion."""

    accepted = _accept_or_404(lambda: services.message_intake.submit_pdf_ingest(session_id=session_id, file_path=request.file_path))
    return IngestAcceptedResponse(
        session_id=session_id,
        message_id=accepted.message.id,
        run_id=accepted.task_run.id,
    )


@router.post("/queries", response_model=QueryAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
def accept_followup_query(
    session_id: str,
    request: FollowupQueryRequest,
    services: ServiceBundle = Depends(get_service_bundle),
) -> QueryAcceptedResponse:
    """Accept a follow-up query request without executing retrieval or runtime logic."""

    accepted = _accept_or_404(lambda: services.message_intake.submit_followup_query(session_id=session_id, query=request.query))
    return QueryAcceptedResponse(
        session_id=session_id,
        message_id=accepted.message.id,
        run_id=accepted.task_run.id,
    )


@router.get("/runs/{run_id}", response_model=TaskRunResponse)
def get_task_run(
    session_id: str,
    run_id: str,
    services: ServiceBundle = Depends(get_service_bundle),
) -> TaskRunResponse:
    """Return the minimal task-run status view for a session-scoped run."""

    try:
        return TaskRunResponse.from_domain(services.task_runs.get_run(session_id=session_id, run_id=run_id))
    except EntityNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="TaskRun not found.") from error


@router.get("/runs", response_model=TaskRunListResponse)
def list_task_runs(
    session_id: str,
    services: ServiceBundle = Depends(get_service_bundle),
) -> TaskRunListResponse:
    """Return task runs for a session, newest first."""

    try:
        runs = [TaskRunResponse.from_domain(run) for run in services.task_runs.list_runs(session_id)]
    except EntityNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.") from error
    return TaskRunListResponse(items=runs)


@router.post("/queries/{run_id}/execute", response_model=QueryExecutionResponse)
def execute_query_run(
    session_id: str,
    run_id: str,
    services: ServiceBundle = Depends(get_service_bundle),
) -> QueryExecutionResponse:
    """Execute a pending follow-up query run through the mock retrieval chain."""

    try:
        result = services.task_runtime.execute_query_run(session_id=session_id, run_id=run_id)
        return QueryExecutionResponse.from_result(result)
    except EntityNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except InvalidTaskRunStateError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.post("/queries/{run_id}/start", response_model=TaskRunResponse, status_code=status.HTTP_202_ACCEPTED)
def start_query_run(
    session_id: str,
    run_id: str,
    services: ServiceBundle = Depends(get_service_bundle),
) -> TaskRunResponse:
    """Start a query run in the background for live streaming."""

    try:
        task_run = services.task_stream.start_query_run(session_id=session_id, run_id=run_id)
        return TaskRunResponse.from_domain(task_run)
    except EntityNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except InvalidTaskRunStateError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.post("/ingest/{run_id}/start", response_model=TaskRunResponse, status_code=status.HTTP_202_ACCEPTED)
def start_ingest_run(
    session_id: str,
    run_id: str,
    services: ServiceBundle = Depends(get_service_bundle),
) -> TaskRunResponse:
    """Start an ingest run in the background for live streaming."""

    try:
        task_run = services.task_stream.start_ingest_run(session_id=session_id, run_id=run_id)
        return TaskRunResponse.from_domain(task_run)
    except EntityNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except InvalidTaskRunStateError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.post("/ingest/{run_id}/execute", response_model=IngestExecutionResponse)
def execute_ingest_run(
    session_id: str,
    run_id: str,
    services: ServiceBundle = Depends(get_service_bundle),
) -> IngestExecutionResponse:
    """Execute a pending ingest run through the same handwritten runtime shell."""

    try:
        result = services.task_runtime.execute_ingest_run(session_id=session_id, run_id=run_id)
        return IngestExecutionResponse(
            task_run=TaskRunResponse.from_domain(result.task_run),
            source_type=result.source_type.value,
            paper_id=result.materialization.paper.id,
            artifact_id=result.materialization.artifact.id,
            session_document_id=result.materialization.session_document.id,
            chunk_count=result.chunk_count,
            operation=result.materialization.operation,
            summary=result.ingest_summary,
            paper_summary=PaperSummaryResponse.from_domain(result.paper_summary),
        )
    except EntityNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except InvalidIngestSourceError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except InvalidTaskRunStateError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.get("/runs/{run_id}/trace", response_model=TraceResponse)
def get_task_run_trace(
    session_id: str,
    run_id: str,
    services: ServiceBundle = Depends(get_service_bundle),
) -> TraceResponse:
    """Return the raw trace and narrative placeholders for a run."""

    try:
        trace = services.trace.get_trace(session_id=session_id, run_id=run_id)
        return TraceResponse(
            steps=[TraceStepResponse.from_domain(step) for step in trace.steps],
            narratives=[TraceNarrativeResponse.from_domain(narrative) for narrative in trace.narratives],
        )
    except EntityNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@router.get("/runs/{run_id}/events", response_model=TimelineResponse)
def get_task_run_events(
    session_id: str,
    run_id: str,
    services: ServiceBundle = Depends(get_service_bundle),
) -> TimelineResponse:
    """Return the timeline events associated with a run."""

    try:
        events = services.timeline.list_events_for_run(session_id=session_id, run_id=run_id)
        return TimelineResponse(items=[TimelineEventResponse.from_domain(event) for event in events])
    except EntityNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@router.get("/runs/{run_id}/stream")
def stream_task_run(
    session_id: str,
    run_id: str,
    services: ServiceBundle = Depends(get_service_bundle),
) -> StreamingResponse:
    """Stream live or replayed run events via SSE."""

    try:
        subscription = services.task_stream.subscribe(session_id=session_id, run_id=run_id)
    except EntityNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    def event_iterator():
        try:
            for event in subscription.iter_events():
                if event is None:
                    yield ": keep-alive\n\n"
                    continue
                payload = json.dumps(event.to_dict(), ensure_ascii=False)
                yield f"event: {event.event_type}\n"
                yield f"data: {payload}\n\n"
        finally:
            subscription.close()

    return StreamingResponse(
        event_iterator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _accept_or_404(factory: Callable[[], AcceptedTaskRun]) -> AcceptedTaskRun:
    try:
        return factory()
    except EntityNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.") from error
