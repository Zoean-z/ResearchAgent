# API

## Current Scope

This document describes the current runnable API surface. It is intentionally limited to session-oriented reads, a unified message acceptance entry, compatibility acceptance endpoints, a handwritten-runtime-backed execution path for query and ingest runs, and minimal task-run lookup backed by repository ports. The default integration storage is SQLite for the runnable paths, while InMemory remains available for explicit test/control use. The ingest path now creates durable artifact, paper, session-document, chunk, and memory bindings for arXiv PDFs and local PDFs, and the ingest analysis layer now has a candidate-first model-backed extraction boundary with heuristic fallback when the model transport is unavailable or asks for more context. It can also mirror accepted intake messages, assistant query answers, assistant ingest summaries, and ingest-created structured memories into an OpenViking surface bundle when configured. Query runs now generate a bounded memory candidate pool, rerank those candidates through a thin model-facing layer with rule fallback, and only reread source chunks when the reranked memory signal is insufficient. The fixed query and ingest chains have also started calling a first internal tool registry so the runtime can later migrate toward model-driven tool use without rewriting service boundaries. OpenViking now acts as a retrieval-facing bridge for session/global memory lookup, while SQLite remains the repo-owned runtime/display snapshot store.

## Environment Loading

The API now loads a local `.env` file before building repositories and services.

Resolution order:

1. `RESEARCH_AGENT_ENV_FILE` if set
2. repository-root `.env`

This keeps local storage and planner configuration out of committed code. See `.env.example` for the current variables.

Planner-related variables currently include:

- `RESEARCH_AGENT_QUERY_PLANNER_BACKEND`
- `RESEARCH_AGENT_QUERY_PLANNER_PROVIDER`
- `RESEARCH_AGENT_QUERY_PLANNER_MODEL`
- `RESEARCH_AGENT_QUERY_PLANNER_BASE_URL`
- `RESEARCH_AGENT_QUERY_PLANNER_TIMEOUT_SECONDS`
- `RESEARCH_AGENT_QUERY_AGENT_BACKEND`
- `RESEARCH_AGENT_QUERY_AGENT_PROVIDER`
- `RESEARCH_AGENT_QUERY_AGENT_MODEL`
- `RESEARCH_AGENT_QUERY_AGENT_BASE_URL`
- `RESEARCH_AGENT_QUERY_AGENT_TIMEOUT_SECONDS`
- `RESEARCH_AGENT_INGEST_EXTRACTION_BACKEND`
- `RESEARCH_AGENT_INGEST_EXTRACTION_PROVIDER`
- `RESEARCH_AGENT_INGEST_EXTRACTION_MODEL`
- `RESEARCH_AGENT_INGEST_EXTRACTION_BASE_URL`
- `RESEARCH_AGENT_INGEST_EXTRACTION_TIMEOUT_SECONDS`
- `DEEPSEEK_API_KEY`
- `RESEARCH_AGENT_OPENVIKING_BACKEND`
- `RESEARCH_AGENT_OPENVIKING_DATA_PATH`
- `RESEARCH_AGENT_OPENVIKING_URL`
- `RESEARCH_AGENT_OPENVIKING_API_KEY`

Current model-backed query behavior:

- `heuristic` keeps the deterministic host-side planner
- `model_adapter` now attempts a provider-specific structured query-agent transport
- `pydantic_ai` activates the PydanticAI-backed query-turn adapter
- `RESEARCH_AGENT_INGEST_EXTRACTION_BACKEND=model_adapter` activates the candidate-first ingest extraction adapter, with heuristic fallback if the model transport is unavailable or returns invalid structure
- when the provider transport fails, returns invalid JSON, selects a tool outside the host-allowed set, or emits an invalid final-answer decision, the runtime falls back to heuristic host-controlled execution
- after memory rerank, the query agent can choose whether to reread source passages or finish directly, while the host still owns validation, trace, timeline, and task-run lifecycle

Current OpenViking bridge behavior:

- `noop` keeps OpenViking disabled
- `inmemory` enables a shared in-process OpenViking test double
- `embedded` enables the local-library OpenViking backend through `ov.OpenViking(path=...)`
- `mirror` and `sdk` enable the SDK-backed best-effort surface bundle
- when `embedded` is active, `RESEARCH_AGENT_OPENVIKING_DATA_PATH` points to the local OpenViking data directory used by the repo
- to isolate data from `C:\Users\Lenovo\.openviking`, set `OPENVIKING_CONFIG_FILE=data/openviking/ov.conf` and copy your existing `~/.openviking/ov.conf` into that repo-local path
- accepted intake messages are mirrored into an OpenViking session
- assistant query answers are also mirrored into the same OpenViking session
- assistant ingest summaries are also mirrored into the same OpenViking session
- ingest-created paper, relation, and open-question memories are mirrored as individually addressable records
- query session/global memory lookup can be reordered by OpenViking search hits before the local SQLite-backed memory payloads are exposed to the runtime
- delete-session and delete-memory flows propagate through the same OpenViking surface bundle

## Implemented Endpoints

### `GET /health`

Response:

```json
{
  "status": "ok"
}
```

### `POST /api/sessions`

Request:

```json
{
  "title": "Research session"
}
```

Response:

```json
{
  "id": "session-id",
  "title": "Research session",
  "created_at": "2026-04-23T12:00:00Z",
  "updated_at": "2026-04-23T12:00:00Z",
  "status": "active"
}
```

### `GET /api/sessions`

Response:

```json
{
  "items": [
    {
      "id": "session-id",
      "title": "Research session",
      "created_at": "2026-04-23T12:00:00Z",
      "updated_at": "2026-04-23T12:00:00Z",
      "status": "active"
    }
  ]
}
```

### `GET /api/sessions/{session_id}`

Returns a single `SessionResponse`. Missing sessions return `404`.

### `POST /api/sessions/{session_id}/messages`

Request for a text query:

```json
{
  "text": "What changed after reading this paper?"
}
```

Request for an arXiv ingest:

```json
{
  "arxiv_url": "https://arxiv.org/abs/2401.12345"
}
```

Request for a local PDF ingest:

```json
{
  "file_path": "C:/papers/example.pdf"
}
```

Response:

```json
{
  "accepted": true,
  "session_id": "session-id",
  "message_id": "message-id",
  "run_id": "run-id",
  "message_type": "followup_query",
  "status": "accepted"
}
```

Current behavior:

- validates that exactly one supported input field is present
- classifies the payload into `followup_query`, `ingest_arxiv`, or `ingest_pdf`
- writes a user `Message` and creates a `TaskRun` in `pending`
- mirrors the accepted intake message into the configured OpenViking session surface when enabled
- preserves the existing session/run/trace/timeline flow after classification

### `POST /api/sessions/{session_id}/ingest/arxiv`

Request:

```json
{
  "arxiv_url": "https://arxiv.org/abs/2401.12345"
}
```

Response:

```json
{
  "accepted": true,
  "session_id": "session-id",
  "message_id": "message-id",
  "run_id": "run-id",
  "status": "accepted"
}
```

Current behavior:

- delegates to the unified message intake service for backward compatibility
- validates that the session exists
- writes a user `Message` with type `ingest_arxiv`
- creates a `TaskRun` in `pending`
- does not execute real ingest work

### `POST /api/sessions/{session_id}/ingest/pdf`

Request:

```json
{
  "file_path": "C:/papers/example.pdf"
}
```

Response shape matches `IngestAcceptedResponse`.

Current behavior:

- delegates to the unified message intake service for backward compatibility
- validates that the session exists
- writes a user `Message` with type `ingest_pdf`
- creates a `TaskRun` in `pending`
- reads the local PDF
- extracts page text into chunk records
- persists artifact, paper, session-document, and chunk bindings
- writes paper, relation, and open-question memories when the current session has enough context

### `POST /api/sessions/{session_id}/queries`

Request:

```json
{
  "query": "What changed after reading this paper?"
}
```

Response:

```json
{
  "accepted": true,
  "session_id": "session-id",
  "message_id": "message-id",
  "run_id": "run-id",
  "status": "accepted"
}
```

Current behavior:

- delegates to the unified message intake service for backward compatibility
- validates that the session exists
- writes a user `Message` with type `followup_query`
- creates a `TaskRun` in `pending`
- does not execute retrieval, answering, or runtime steps

### `POST /api/sessions/{session_id}/queries/{run_id}/execute`

Current behavior:

- validates that the run belongs to the session
- routes the run through the handwritten runtime service
- transitions the run from `pending` to `running` to `finished`
- performs memory repository queries for session-scoped and global memories
- reranks the bounded memory candidate pool through a thin model-facing layer, with rule fallback when the model selection is invalid or insufficient
- runs the retrieval/rerank/answer composition path through the first internal tool registry rather than direct route-specific glue
- records the specific memory ids, summaries, and selection reasons that were used
- decides whether reread would be needed in a real runtime using the reranked memory selection
- rereads stored source chunks when the reranked memory signal is insufficient, then reranks those source candidates too
- writes trace, narrative, and timeline entries that reference the retrieved memories and reread chunks
- returns a mock answer payload that includes the retrieved memory citations, the rerank source, and any reread chunk citations

### `POST /api/sessions/{session_id}/ingest/{run_id}/execute`

Response:

```json
{
  "task_run": {
    "id": "run-id",
    "session_id": "session-id",
    "message_id": "message-id",
    "status": "finished",
    "step_count": 7,
    "started_at": "2026-04-23T12:00:00Z",
    "finished_at": "2026-04-23T12:00:01Z",
    "finish_reason": "mock_ingest_completed"
  },
  "source_type": "pdf",
  "paper_id": "paper-id",
  "artifact_id": "artifact-id",
  "session_document_id": "session-document-id",
  "chunk_count": 1,
  "operation": "created",
  "summary": "Parsed local PDF: C:/papers/example.pdf (1 chunks)\n- What it is about: ...\n- Problem solved: ...",
  "paper_summary": {
    "what_it_is_about": "...",
    "problem_solved": "...",
    "new_ideas": ["..."],
    "limitations": ["..."],
    "suggestions_or_questions": ["..."],
    "evidence_candidate_ids": ["title", "abstract"],
    "confidence": 0.5
  }
}
```

Current behavior:

- validates that the run belongs to the session
- routes the run through the same handwritten runtime service
- transitions the run from `pending` to `running` to `finished`
- routes paper registration and memory extraction through the first internal tool registry
- creates durable artifact, paper, session-document, and chunk bindings for arXiv PDFs and local PDFs
- writes the first paper, relation, and open-question memory records from parsed content when available
- mirrors those structured memory records into the configured OpenViking surface bundle when enabled
- writes ingest trace, narrative, and timeline entries
- returns a source-aware ingest summary payload plus a structured paper summary

### `GET /api/sessions/{session_id}/runs/{run_id}/trace`

Response:

```json
{
  "steps": [
    {
      "id": "step-id",
      "run_id": "run-id",
      "action": "retrieve_session_memories",
      "input_payload": {},
      "result_payload": {},
      "status": "completed",
      "started_at": "2026-04-23T12:00:00Z",
      "finished_at": null
    }
  ],
  "narratives": [
    {
      "id": "narrative-id",
      "run_id": "run-id",
      "step_id": "step-id",
      "text": "checked session memory",
      "created_at": "2026-04-23T12:00:00Z"
    }
  ]
}
```

Current behavior:

- validates that the session owns the run
- returns the raw trace step list and any stored narratives
- returns the raw trace steps and narratives written by runtime execution

### `GET /api/sessions/{session_id}/runs/{run_id}/events`

Response:

```json
{
  "items": [
    {
      "id": "event-id",
      "session_id": "session-id",
      "run_id": "run-id",
      "event_type": "step_completed",
      "summary": "checked session memory",
      "related_memory_ids": [],
      "related_paper_ids": [],
      "created_at": "2026-04-23T12:00:00Z"
    }
  ]
}
```

Current behavior:

- validates that the session owns the run
- returns only timeline events associated with that run
- returns the timeline events written by runtime execution

### `GET /api/sessions/{session_id}/messages`

Response:

```json
{
  "items": []
}
```

Current behavior:

- validates that the session exists
- returns the stable message list shape even when empty
- now includes accepted ingest/query user messages
- now also includes persisted assistant query answers and assistant ingest summaries, with `role` set to `assistant`

### `GET /api/sessions/{session_id}/timeline`

Response:

```json
{
  "items": []
}
```

Current behavior:

- validates that the session exists
- reads from the in-memory timeline repository
- now shows the timeline events written by runtime execution

### `GET /api/sessions/{session_id}/memory-snapshot`

Response:

```json
{
  "paper_memories": [],
  "relation_memories": [],
  "open_question_memories": []
}
```

Current behavior:

- validates that the session exists
- resolves related paper ids from `session_documents`
- maps domain memory models to API-specific response models
- returns the paper, relation, and open-question memories associated with the session's documents

### `GET /api/sessions/{session_id}/runs/{run_id}`

Response:

```json
{
  "id": "run-id",
  "session_id": "session-id",
  "message_id": "message-id",
  "status": "pending",
  "step_count": 0,
  "started_at": "2026-04-23T12:00:00Z",
  "finished_at": null,
  "finish_reason": null
}
```

Current behavior:

- validates that the session exists
- returns only runs owned by that session
- exposes minimal task-run lifecycle state
- does not expose trace or timeline execution details yet

## Defined But Not Yet Routed

The following request/response schemas already exist in code but are not connected to live endpoints yet:

- `IngestPdfResponse`

Those remain placeholders for the next vertical slice and preserve stable API boundaries while runtime logic is still missing.

## Internal Tools

The runtime now has a first internal tool registry with these stable names:

- `register_paper`
- `extract_memories`
- `search_session_memory`
- `search_global_memory`
- `search_source_chunks`
- `rerank_candidates`
- `read_source_passages`
- `compose_answer`

These tools do not handle HTTP directly. They wrap existing services and repository-backed logic so the handwritten runtime can move toward tool-oriented orchestration without changing API boundaries.

## Source Of Truth

Behavioral intent still comes from [project_spec.md](./project_spec.md). This file now tracks the concrete mock API surface that actually runs in the repository.
