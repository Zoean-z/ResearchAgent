# OpenViking Memory-Routed Paper Agent

OpenViking Memory-Routed Paper Agent is a research assistant project that reads arXiv papers or local PDFs, stores structured research memory, and uses that memory to shape later answers.

## Current Status

This repository now includes the first runnable foundation slice on top of the original scaffold.

Completed so far:

- project documentation and planning memory files are in place
- domain models, enums, value objects, repository ports, and policy modules exist
- in-memory mock repositories exist for sessions, messages, papers, artifacts, memories, trace, timeline, and chunks
- thin services exist for session reads, acceptance, memory snapshots, and task-run lookup
- a minimal SQLite schema placeholder exists in `backend/migrations/0001_initial_schema.sql`
- the FastAPI app now defaults to SQLite-backed storage for the runnable paths
- the FastAPI app exposes a handwritten runtime-backed execution path for both query and ingest runs that drives `pending -> running -> finished`
- query responses now include the specific memory citations and selection reasons used during retrieval, not just counts
- query execution also rereads stored source chunks when memory is insufficient and surfaces the reread chunk citations and reasons
- the ingest path parses arXiv PDFs and local PDFs into chunks and creates durable artifact, paper, session-document, chunk, and first-pass memory bindings
- the ingest analysis stage now has a candidate-first model-backed boundary with heuristic fallback, while memory persistence remains separate, and the completed ingest run now emits a structured paper summary alongside the three memory types
- OpenViking can be enabled as a retrieval-facing bridge: accepted intake messages, assistant query answers, assistant ingest summaries, and ingest-created memories are mirrored into an OpenViking surface bundle while SQLite remains the canonical repo-owned runtime/display store
- the API can be configured to use SQLite for sessions, messages, task runs, papers, artifacts, session documents, trace, and timeline
- the API can now also use SQLite for structured memories and chunks
- the FastAPI app also exposes trace and timeline read endpoints for executed runs, including generated narratives
- query execution now uses a bounded candidate pool, reranks it through a thin model-facing layer with rule fallback, and only rereads source chunks when the reranked memory signal is insufficient
- the API now exposes a unified session message entry that classifies plain text, arXiv links, and PDF paths into the existing query/ingest task-run flow
- the API now exposes a browser-facing PDF upload entry and a runtime-status endpoint for frontend settings
- the backend now includes a first internal tool registry so query and ingest execution can start calling structured tool capabilities instead of expanding business routes
- the repository now includes a real frontend workbench with session navigation, a unified composer, PDF upload, recent-run history, a trace/timeline inspector, and memory/settings drawers
- the frontend workbench now also shows live run progress, recent-run state, and query/ingest evidence summaries on top of the shared SSE event stream
- baseline unit and API tests pass

## Local Startup

For the repo-local embedded OpenViking path, run:

```powershell
.\scripts\start-dev.ps1
```

The script prepares `data/openviking/ov.conf`, isolates OpenViking data under this repo, starts the backend, and waits for `/health` to report `ok`.

For the frontend:

```powershell
cd frontend
npm install
npm run build
cd ..
.\scripts\start-dev.ps1 -BindPort 8011
```

After the frontend is built, the backend serves it from `/`, so the workbench is available at `http://127.0.0.1:8011/`.

To quickly inspect the current end-to-end effect:

1. open the workbench at `http://127.0.0.1:8011/`
2. paste a pure arXiv link or upload a PDF to start an ingest run
3. watch the right-side inspector:
   - `Reasoning` now shows live progress, evidence summaries, and per-step trace output
   - `Timeline` now shows live run events before the final persisted timeline refresh
4. ask a follow-up question in the same session and compare the query run:
   - recent-run history on the left updates immediately
   - the status strip shows the selected run step count
   - evidence blocks show which memory or source chunks shaped the answer

Still intentionally not implemented:

- OCR-based ingest variations and broader remote source handling beyond direct arXiv PDF downloads
- real runtime loop execution for all task types
- broader frontend workflows such as multi-file upload orchestration, richer multi-run comparison, and live progress streaming

## Repository Structure

- `backend/`: Python service, domain layers, runtime, adapters, migrations, API, tests
- `frontend/`: UI shell and feature folders for sessions, chat, ingest, timeline, trace, and memory
- `data/`: local artifact and SQLite storage directories
- `docs/`: project specification and API documentation

## Local Environment

Copy `.env.example` to `.env` when you want to run the repo with local configuration instead of shell-exported environment variables.

Current variables that matter:

- `RESEARCH_AGENT_STORAGE_BACKEND`
- `RESEARCH_AGENT_SQLITE_PATH`
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

Current note:

- `RESEARCH_AGENT_QUERY_PLANNER_BACKEND=model_adapter` activates the bounded DeepSeek planner transport.
- `RESEARCH_AGENT_QUERY_AGENT_BACKEND=pydantic_ai` activates the PydanticAI query-turn adapter. The host still owns lifecycle, fallback, validation, trace, and timeline persistence.
- `RESEARCH_AGENT_INGEST_EXTRACTION_BACKEND=model_adapter` activates the candidate-first ingest extraction adapter. The host still owns parsing, persistence, merge/upsert, OpenViking mirroring, and fallback to heuristic extraction.
- `RESEARCH_AGENT_OPENVIKING_BACKEND=inmemory` enables a repo-local OpenViking test double.
- `RESEARCH_AGENT_OPENVIKING_BACKEND=embedded` enables the local-library OpenViking backend using `ov.OpenViking(path=...)`.
- `RESEARCH_AGENT_OPENVIKING_BACKEND=mirror` or `sdk` enables the SDK-backed best-effort OpenViking surface bundle.
- In embedded mode, set `RESEARCH_AGENT_OPENVIKING_DATA_PATH` to the local OpenViking data directory you want this repo to use.
- To isolate OpenViking data from `C:\Users\Lenovo\.openviking`, copy your existing `~/.openviking/ov.conf` to `data/openviking/ov.conf` in this repo and keep `OPENVIKING_CONFIG_FILE=data/openviking/ov.conf`.
- `scripts/start-dev.ps1` will prepare the repo-local OpenViking config and launch the backend against the embedded OpenViking path.
- In the current bridge, OpenViking search hits can reorder query memory lookup, while SQLite still serves as the local display/runtime snapshot store.
- If you actually want message/history mirroring, OpenViking does need local configuration. At minimum you need:
  - `RESEARCH_AGENT_OPENVIKING_BACKEND`
  - `RESEARCH_AGENT_OPENVIKING_DATA_PATH` when using `embedded`
  - `RESEARCH_AGENT_OPENVIKING_URL`
  - `RESEARCH_AGENT_OPENVIKING_API_KEY` when your server requires auth

## Primary Specification

The main project specification lives at [docs/project_spec.md](docs/project_spec.md).

## Planned Milestones

1. Expose trace and timeline read endpoints for executed runs.
2. Implement session-aware memory lookup and snapshot composition services.
3. Extend SQLite coverage to any remaining in-memory repositories once the core storage boundary is stable.
4. Tighten model-backed rerank selection and answer synthesis over extracted memories and reread chunks.
5. Let the handwritten runtime dispatch more of the fixed query/ingest chain through the internal tool registry before introducing real model tool calling.
6. Build the three-column frontend around timeline, trace, and memory snapshot.

## Current API Surface

The current mock API includes:

- `GET /health`
- `GET /api/system/runtime`
- `POST /api/sessions`
- `GET /api/sessions`
- `GET /api/sessions/{session_id}`
- `POST /api/sessions/{session_id}/messages`
- `POST /api/sessions/{session_id}/uploads/pdf`
- `POST /api/sessions/{session_id}/ingest/arxiv`
- `POST /api/sessions/{session_id}/ingest/pdf`
- `GET /api/sessions/{session_id}/messages`
- `GET /api/sessions/{session_id}/runs`
- `GET /api/sessions/{session_id}/timeline`
- `GET /api/sessions/{session_id}/memory-snapshot`
- `POST /api/sessions/{session_id}/queries`
- `GET /api/sessions/{session_id}/runs/{run_id}`
- `POST /api/sessions/{session_id}/queries/{run_id}/execute`
- `POST /api/sessions/{session_id}/queries/{run_id}/start`
- `POST /api/sessions/{session_id}/ingest/{run_id}/execute`
- `POST /api/sessions/{session_id}/ingest/{run_id}/start`
- `GET /api/sessions/{session_id}/runs/{run_id}/trace`
- `GET /api/sessions/{session_id}/runs/{run_id}/events`
- `GET /api/sessions/{session_id}/runs/{run_id}/stream`

## Notes

The first runtime will be handwritten and intentionally decoupled from agent frameworks so the system can later migrate to PydanticAI without rewriting core services or adapters.
