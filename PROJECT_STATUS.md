# Project Status

## Phase

The backend now exposes browser-usable intake and runtime-status endpoints, and the repository also has a first real frontend workbench that can be built and served by the same FastAPI app; query orchestration still runs through the host-guarded tool loop, OpenViking remains the mirrored retrieval/context layer, SQLite remains the repo-owned runtime/display store, and both query and ingest now share the same background-start plus SSE event-stream observation path while final assistant text is still committed message-by-message rather than token-streamed

## Completed

- Created the baseline repository structure
- Wrote the root project guidance in `AGENTS.md`
- Wrote the consolidated specification in `docs/project_spec.md`
- Implemented concrete domain models, enums, value objects, repository ports, and policy helpers
- Added in-memory mock repositories for session, message, paper, artifact, memory, trace, timeline, and chunk storage
- Added a thin service layer for session reads, acceptance, timeline reads, memory snapshot composition, and task-run lookup
- Added thin retrieval and query execution services that drive mock `pending -> running -> finished` query runs
- Added a handwritten runtime service that wraps both query and ingest execution, manages task lifecycle, and writes trace narratives
- Added a unified message intake service that classifies plain text, arXiv links, and PDF paths into the existing query/ingest task-run flow
- Added a thin ingest materialization service that parses local PDFs into chunks and creates durable artifact, paper, session-document, and chunk bindings
- Added arXiv ingest materialization that downloads arXiv PDFs, parses them into chunks, and creates durable artifact, paper, session-document, and chunk bindings
- Added a thin memory extraction service that derives and upserts paper, relation, and open-question memories from parsed source content
- Added a candidate-first ingest extraction protocol and model-backed adapter boundary, with broad candidate windows, reread-on-insufficient-evidence, and heuristic fallback
- Added a structured paper summary draft to ingest extraction so each completed ingest can produce a readable synthesis alongside the structured memories
- Added PDF page-text normalization before chunking so line-number noise and hyphenated line breaks are cleaned at the source
- Added ingest candidate re-ranking so main-text evidence is preferred and appendix/table-heavy content is downweighted instead of removed
- Added per-candidate content roles so the ingest extractor can distinguish main-text evidence from appendix/table/reference material
- Split ingest analysis from memory persistence and added an optional OpenViking memory mirror boundary
- Added a first internal tool registry and started routing fixed query/ingest execution steps through tool-style capabilities instead of route-specific glue
- Added a frozen query-only tool protocol and a dedicated query tool executor so runtime tool calls now pass through structured request/response/error envelopes instead of calling the registry directly
- Added a host-controlled query tool-calling loop with a small planner boundary so query execution now advances by planner-selected next-tool decisions while the handwritten runtime still owns lifecycle and trace/timeline persistence
- Added a model-backed query planner adapter boundary, heuristic fallback wrapper, and config-driven planner wiring so the bounded query tool loop can switch from pure heuristic planning to a real planner transport later without changing runtime semantics
- Added repo-local `.env` loading plus a checked-in `.env.example` so storage and planner configuration can be prepared for a future DeepSeek-backed planner transport without hardcoding secrets
- Added a DeepSeek-backed structured planner transport so `model_adapter` can now make real bounded next-tool decisions and still fall back to heuristic planning on transport, JSON, or validation failures
- Added a PydanticAI-backed query-turn adapter boundary so the framework layer can now own query orchestration decisions while the host keeps lifecycle and validation control
- Initialized the project as a local git repository
- Added a first `QueryAgentClient` boundary so the host-controlled query loop can now accept either a bounded `tool_call` or a `final_answer`
- Added a DeepSeek-backed structured query-agent transport so `model_adapter` can now return either `tool_call` or `final_answer` directly, with host-side fallback when the model output is invalid
- Phase 4 now lets the query agent choose the next step after memory rerank, so rerank can flow either into source reread or direct answer completion without the host hardcoding that branch
- Query execution now first generates a bounded memory candidate pool, reranks it with a thin model-facing layer, and then falls back to rules when needed before producing an answer
- Query execution now surfaces the specific memory citations and selection reasons it used, and trace/timeline records reference those retrieved memories
- Query execution now also performs a thin source reread from stored chunks when memory is insufficient, reranks the reread candidates, and surfaces the reread chunk citations and reasons
- Query execution now exposes structured observations for session memory, global memory, OpenViking retrieval, rerank, reread gating, source reread, and answer composition so turn adapters can see how each step changes later decisions
- Query execution now leaves the reread-or-answer branch open after rerank so the model can choose between source reread and direct answer completion instead of following a fixed host-selected ladder
- Started extracting the query turn-selection logic into a runtime orchestrator helper so the framework layer can sit behind a runtime-owned orchestration boundary instead of inside the execution service
- The query turn protocol now lives in `runtime/query_turn.py`, and `tools/query_agent.py` has been reduced to a compatibility implementation layer
- The query turn-selection helpers now live in a runtime orchestrator helper, and `QueryExecutionService` delegates to it instead of owning the allowed-action and decision translation logic itself
- Added a thin regression test that exercises both the heuristic and PydanticAI query-turn paths through the runtime boundary
- Removed the runtime-side `QueryAgent*` aliases so the final replaceable boundary is now the `QueryTurn*` runtime protocol
- Added OpenViking ownership and deletion policy helpers for chat history, memory, and mirrored snapshots
- Added OpenViking message, memory, and session adapter surfaces with no-op implementations
- Added concrete OpenViking surface implementations for in-memory tests and SDK-backed best-effort wiring, while keeping no-op fallbacks
- Added delete-session and delete-memory service and API boundaries with SQLite/InMemory propagation
- Added dual-write of accepted intake messages into the OpenViking surface bundle
- Added assistant query-answer persistence and assistant ingest-summary persistence, with both mirrored into the OpenViking session so chat history now includes both user and assistant messages
- Added OpenViking-backed memory mirroring through a surface-backed gateway so ingest-created paper, relation, and open-question memories are mirrored as individually addressable records
- Added retrieval bridging that lets OpenViking search hits reorder session/global memory lookup while SQLite remains the local display/runtime snapshot store
- Added an explicit `OpenVikingRetrievalAdapter` so OpenViking search and local-memory mapping are no longer only an internal detail of `RetrievalService`
- Added an explicit internal `search_openviking_memory` tool and exposed it to the first two retrieval turns
- Added unit coverage for OpenViking dual-write, retrieval preference, gateway mirroring, and deletion propagation
- Added an embedded OpenViking backend path so the repo can run OpenViking as a local library backend without a separate HTTP server
- Added a Windows startup script that prepares repo-local OpenViking config, launches the backend, and health-checks the app
- Added trace and timeline read services for executed runs
- Added a handwritten ingest execution path that shares the same handwritten runtime shell as query runs
- Ingest execution now leaves durable source attachments visible through the API, and both arXiv and local PDF ingestion persist extracted chunks and first-pass memories
- Added SQLite-backed repositories for sessions, messages, task runs, papers, artifacts, memories, chunks, trace, and timeline
- Added a minimal SQLite schema placeholder in `backend/migrations/0001_initial_schema.sql`
- SQLite initialization now prefers the shared migration stub so startup and schema layout stay aligned
- SQLite startup now backfills the `messages.role` column for older local databases so existing dev data can still run the current ingest/query paths
- Wired a working FastAPI API for sessions, acceptance endpoints, runtime-backed query and ingest execution, task-run lookup, trace/timeline reads, memory snapshot, and `/health`
- Added a browser-facing PDF upload endpoint that stores uploaded files under the repo and routes them into the existing ingest execute path
- Added a safe runtime-status endpoint for frontend settings and environment visibility
- Added a first real React/Vite frontend workbench with session navigation, unified text composer, arXiv-aware input detection, PDF upload, trace/timeline inspector, memory/settings drawers, recent-run history, and session deletion UX
- Upgraded the frontend trace view into a reasoning view so query runs now expose planner decisions, selected tools, fallback status, and per-step input/result summaries instead of only final answers
- Added a live query streaming path with background run start, replayable SSE events, per-step tool/trace emission, and frontend live reasoning updates during execution
- Extended the same replayable SSE start/stream path to ingest so ingest requests now run in the background, emit step-level trace events live, and surface the assistant ingest summary without waiting for the full run to finish
- Tightened the frontend workbench around the shared SSE path so selected runs now show live progress rails, recent-run state badges, and query/ingest evidence summaries while streaming
- Replaced the old phase-shaped query allowed-tools ladder with one unified query tool pool and enabled first-turn `final_answer`, while keeping host-side lifecycle, validation, and tool-boundary control
- Added inline chat-stream run cards so the current message flow can show progress, tool calls, reasoning summaries, and evidence directly in the conversation area instead of relying only on the side inspector
- Removed the redundant right-side `Reasoning` inspector tab so tool calls and thinking now live primarily in the inline chat-stream run cards while the side inspector stays focused on memory and timeline
- Added a first host-side anti-stall guardrail for the wider unified query tool pool: after repeated low-yield turns, the host now forces `compose_answer` instead of letting the loop drift through more empty tool calls
- Added host-side duplicate-signature suppression for the wider unified query tool pool: semantically repeated memory-search/rerank/reread requests are normalized into one signature and now collapse into `compose_answer` instead of executing another near-identical tool step
- Tightened both query-agent prompt layers so direct `final_answer` is now the explicit default when retrieval is unnecessary, especially for greetings, acknowledgements, capability questions, and other low-context turns
- Replaced the remaining mock query answer text with a Chinese answer composer, so follow-up answers now read as prose instead of debug strings
- Cleaned up the ingest fallback text so memory summaries and paper summaries stay Chinese by default
- Imported local PDFs now use a full-document ingest window for the model-backed extractor so the whole paper is visible to the paper/relation/open-question summary pass
- Removed the left-side `Recent Runs` panel from the frontend so the sidebar stays focused on session navigation
- Restored scrolling in the right-side memory drawer
- Added same-origin frontend serving from the FastAPI app when `frontend/dist` is present
- Added a session-scoped task-run list endpoint so the frontend can inspect historical runs instead of only the latest one
- The API now defaults to SQLite storage for the current runnable paths, while InMemory remains available for tests and control flows
- Added baseline tests for domain rules, repository adapters, and API endpoints

## Not Yet Implemented

- OCR-based ingestion and broader remote-source coverage beyond direct arXiv PDF downloads
- richer session commit timing beyond the current accepted-intake and assistant-answer checkpoints
- live-server validation of the SDK-backed OpenViking surface bundle in this repository
- live-model ingest smoke validation still needs to be run against representative papers with the new summary output checked explicitly
- token-level text streaming for final assistant answers and ingest summaries
- further host-side guardrail tuning if model autonomy is widened again beyond the current no-repeat, low-yield, and duplicate-signature protections
- real runtime loop
- deeper frontend workflows such as upload progress streaming, richer multi-run comparison views, and broader session management polish

## Current Recommendation

Keep using OpenViking as the conversation-history and memory backend while SQLite remains the repo-owned runtime/display store. The current bridge is now visible end to end in the frontend: accepted intake messages, assistant-visible outputs, and ingest-created memories are mirrored to OpenViking; retrieval still has a dedicated adapter and internal tool boundary; the handwritten runtime still owns lifecycle, step limits, validation, finish/fail, and trace/timeline persistence. Query orchestration is now broader than the earlier phase ladder because the model sees one unified query tool pool and may answer directly from the first turn, and the prompt layer now explicitly prefers that direct-answer path whenever retrieval would not materially improve the response. The current guardrails on that wider pool are: completed tools are not re-offered, repeated low-yield turns force `compose_answer`, and repeated effective query signatures are suppressed before they execute again. The current UI is lighter: the left sidebar is sessions only, the inline chat stream carries run evidence, and the memory drawer scrolls normally again. The next most useful slice is token streaming plus live-model ingest tuning.

## Current Work

The repo now has a usable frontend shell, same-origin serving path, session deletion, live step-by-step observation for both query and ingest runs, inline chat-stream reasoning/evidence cards, and a first host-side guardrail set on top of the wider query tool pool. The left-side recent-run list is gone, the memory drawer scrolls, and query answers now render in Chinese prose instead of the old mock debug string. Imported local PDFs now take the aggressive full-document ingest path so the model-backed extractor can see all chunks at once, and the next check is whether that improves the paper summary before broadening it further. Next work should focus on token streaming vs richer upload/progress polish and multi-run evidence comparison, plus continued live-model ingest smoke and summary tuning in parallel.
