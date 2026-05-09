# Project Status

## Phase

The backend now exposes browser-usable intake and runtime-status endpoints, and the repository also has a first real frontend workbench that can be built and served by the same FastAPI app; query orchestration still runs through the host-guarded tool loop, OpenViking remains the mirrored retrieval/context layer, SQLite remains the repo-owned runtime/display store, both query and ingest now share the same background-start plus SSE event-stream observation path, query turns now receive a compact recent conversation context by default, ingest is being refactored so paper understanding comes from model extraction over cleaned evidence chunks instead of keyword scoring or template fallback, the frontend now shows real-time step progress and failure reasons during streaming execution, the query execution service has been split into six focused modules, the DeepSeek API adapter now handles rate-limit and empty-response conditions with exponential backoff retry, and the repo now also has a static showcase demo plus a Docker one-command deployment path for portfolio-style presentation

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
- Added model-first ingest extraction trace evidence so the ingest run trace now records input chunk ids, raw extraction payloads, and field-level rejection reviews
- Added OpenViking-backed memory mirroring through a surface-backed gateway so ingest-created paper, relation, and open-question memories are mirrored as individually addressable records
- Added retrieval bridging that lets OpenViking search hits reorder session/global memory lookup while SQLite remains the local display/runtime snapshot store
- Added an explicit `OpenVikingRetrievalAdapter` so OpenViking search and local-memory mapping are no longer only an internal detail of `RetrievalService`
- Added an explicit internal `search_openviking_memory` tool and exposed it to the first two retrieval turns
- Added a model-callable `import_arxiv_paper` query tool that reuses the existing arXiv intake + ingest runtime chain, so query-time imports do not create a separate PDF URL pipeline
- Added a model-callable `search_arxiv` query tool that calls the official arXiv API for lightweight paper discovery metadata and keeps import as a separate explicit tool step
- Added a query-time no-result fallback for arXiv search/import so network or download failures surface as structured observations instead of failing the whole query run
- Added a GitHub Pages-ready static demo that uses the real frontend with mocked `/api` responses and fixed showcase session data
- Added Docker deployment files so the real frontend/backend stack can be started with `docker compose up --build`
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
- Injected a compact recent conversation context into query turns and exposed `list_recent_messages` / `get_conversation_context` as optional internal tools for follow-up disambiguation
- Tightened the frontend workbench around the shared SSE path so selected runs now show live progress rails, recent-run state badges, and query/ingest evidence summaries while streaming
- Replaced the old phase-shaped query allowed-tools ladder with one unified query tool pool and enabled first-turn `final_answer`, while keeping host-side lifecycle, validation, and tool-boundary control
- Added inline chat-stream run cards so the current message flow can show progress, tool calls, reasoning summaries, and evidence directly in the conversation area instead of relying only on the side inspector
- Removed the redundant right-side `Reasoning` inspector tab so tool calls and thinking now live primarily in the inline chat-stream run cards while the side inspector stays focused on memory and timeline
- Removed the normal-query mock answer fallback so final user-visible answers now come from the query agent's `final_answer` output rather than host-composed template text
- Recast `compose_answer` into an evidence-packaging tool for the model instead of a direct user-answer builder
- Removed host-side forced-`compose_answer` and duplicate-signature forced-`compose_answer` exits from the normal query loop
- Changed the default query-agent wiring to the model-driven path and kept `test_stub` only as a deterministic test backend
- Tightened both query-agent prompt layers so direct `final_answer` is now the explicit default when retrieval is unnecessary, especially for greetings, acknowledgements, capability questions, and other low-context turns
- Restored the legacy `turn_adapter` query-agent path so existing DeepSeek-backed query setups keep using the working model-adapter transport instead of being remapped into an incompatible `pydantic_ai` request shape
- Added query-agent schema repair so model outputs like `{"final_answer": "..."}` or `{"tool_name": "..."}` now normalize to a valid turn decision instead of failing with `model_agent_returned_no_decision`
- Replaced the remaining mock query answer text with a Chinese answer composer, so follow-up answers now read as prose instead of debug strings
- Cleaned up the ingest fallback text so memory summaries and paper summaries stay Chinese by default
- Imported local PDFs now use a full-document ingest window for the model-backed extractor so the whole paper is visible to the paper/relation/open-question summary pass
- Added an ingest stopgap repair that removed production fallback/template overwrite paths, disabled cross-paper open-question merge, and disabled weak relation generation so repeated imports no longer collapse into the same memory bundle
- Query execution now jumps straight from `get_paper_memory_bundle` into finalization, so the model no longer has to survive an extra structured decision after the bundle is already loaded
- DeepSeek's structured query-decision call now uses JSON mode with explicit tool-call and final-answer JSON examples, while the finalization text pass still stays plain-text
- Removed the left-side `Recent Runs` panel from the frontend so the sidebar stays focused on session navigation
- Restored scrolling in the right-side memory drawer
- Added same-origin frontend serving from the FastAPI app when `frontend/dist` is present
- Added a session-scoped task-run list endpoint so the frontend can inspect historical runs instead of only the latest one
- Updated the Windows startup script so it imports repo-local `.env` values into the backend process and no longer gets stuck on stale shell-level API keys
- The API now defaults to SQLite storage for the current runnable paths, while InMemory remains available for tests and control flows
- Added baseline tests for domain rules, repository adapters, and API endpoints
- Refactored the 1988-line `query_execution_service.py` monolith into six focused modules: data models, citation builder, observation builder, answer composer, trace writer, and the slimmed-down main service
- Added real-time live-step timeline in the frontend so streaming query/ingest runs show per-step progress with planner details instead of piling all steps at the end
- Added failure-reason display in the frontend inline run card so failed runs now show the specific error message instead of only a generic "失败" badge
- Added a pulsing "thinking" indicator with bouncing dots and current-step label in the message stream while queries are running
- Added DeepSeek API rate-limit and empty-response detection with exponential backoff retry (5s/10s/15s) so 429, 503, and empty 200 responses are retried instead of immediately failing
- Promoted DeepSeek API error logging from DEBUG to WARNING so HTTP errors and empty responses are visible in the backend log without enabling debug mode

## Not Yet Implemented

- OCR-based ingestion and broader remote-source coverage beyond direct arXiv PDF downloads
- richer session commit timing beyond the current accepted-intake and assistant-answer checkpoints
- live-server validation of the SDK-backed OpenViking surface bundle in this repository
- live-model ingest has now been refactored toward a model-first full-text / hierarchical extraction flow, and the old production keyword-fallback path is no longer responsible for paper understanding
- token-level text streaming for final assistant answers and ingest summaries
- further host-side guardrail tuning if model autonomy is widened again beyond the current no-repeat, low-yield, and duplicate-signature protections
- real runtime loop
- deeper frontend workflows such as upload progress streaming, richer multi-run comparison views, and broader session management polish

## Current Recommendation

Keep using OpenViking as the conversation-history and memory backend while SQLite remains the repo-owned runtime/display store. The current bridge is now visible end to end in the frontend: accepted intake messages, assistant-visible outputs, and ingest-created memories are mirrored to OpenViking; retrieval still has a dedicated adapter and internal tool boundary; the handwritten runtime still owns lifecycle, step limits, validation, finish/fail, and trace/timeline persistence. Query orchestration is now broader than the earlier phase ladder because the model sees one unified query tool pool and may answer directly from the first turn, and the prompt layer now explicitly prefers that direct-answer path whenever retrieval would not materially improve the response. The current query answer contract is narrower: `compose_answer` now packages evidence for the model, and the final user-visible answer must come from the model's last `final_answer`. Finalization now uses a compact evidence payload, a higher token budget, and a single retry when the provider truncates the answer. The current UI is lighter: the left sidebar is sessions only, the inline chat stream carries run evidence, and the memory drawer scrolls normally again. Streaming runs now show a dynamic live-step timeline and a pulsing thinking indicator, and failed runs display the specific failure reason in the run card. Deleted sessions are now hidden both from the session list and from direct session reads after refresh. The DeepSeek adapter is now more resilient: rate-limit responses (429/503) and empty 200 bodies trigger exponential backoff retries instead of immediate failure, and API errors are logged at WARNING level. The next most useful slice is token streaming plus another round of live-model ingest tuning.

## Current Work

The repo now has a usable frontend shell, same-origin serving path, session deletion, live step-by-step observation for both query and ingest runs, inline chat-stream reasoning/evidence cards, and a model-driven query answer contract on top of the wider query tool pool. The left-side recent-run list is gone, the memory drawer scrolls, and normal query answers no longer fall back to `Mock answer for ...` or `compose_answer.result["answer"]`; `compose_answer` only prepares evidence for the model and the final visible answer comes from the query agent's last `final_answer`. Query-time arXiv import is now available through a model-callable `import_arxiv_paper` tool that reuses the existing arXiv intake + ingest runtime chain, creates a real ingest run in the current session, and returns the imported `paper_id` plus ingest summary instead of inventing a second PDF URL path. Query-time discovery now also exposes a model-callable `search_arxiv` tool that queries the official arXiv API for lightweight metadata only and returns `arxiv_id`, `abs_url`, and `pdf_url` without downloading or importing the paper. When arXiv search or import cannot produce a usable result, the query runtime now surfaces a structured `no_results` observation instead of failing the whole run, so the model can answer cleanly that nothing was found or imported. Imported local PDFs now take the aggressive full-document ingest path so the model-backed extractor can see all chunks at once, and the ingest extraction pipeline now uses cleaned evidence chunks plus model-only full-text or hierarchical extraction instead of keyword fallback or template overwrite. The current ingest flow keeps `paper_memory` and `paper_summary` evidence-bound, keeps `open_question_memory` paper-local, and disables weak automatic relation generation. The ingest trace now records extraction debug evidence, including the chunk ids passed to the model and the per-field rejection reviews. Finalization now uses a compact evidence payload, a higher token budget, and a single retry when the provider truncates the answer. Session deletion semantics are now narrower: deleting a conversation removes the session-scoped dialogue state and bindings, but leaves shared memory records intact so memory must be deleted explicitly from the memory drawer. Query turns now receive a compact recent conversation context by default, and the model can optionally fetch a bounded recent-message/context view when follow-up references need more history than the injected summary. The frontend now shows a dynamic live-step timeline during streaming execution with per-step planner details, a pulsing "thinking" indicator with bouncing dots while queries run, and a failure-reason panel when runs fail. The `query_execution_service.py` monolith (~1988 lines) has been refactored into six focused modules: `query_execution_models`, `query_citation_builder`, `query_observation_builder`, `query_answer_composer`, `query_trace_writer`, and the slimmed-down main service. The DeepSeek API adapter now detects rate-limit and empty-response conditions, retries with exponential backoff (5s/10s/15s), and logs HTTP errors at WARNING level so failures are visible in the backend log without enabling DEBUG. Next work should focus on final-answer token streaming plus any later ingest wording tuning if the model output drifts again.
