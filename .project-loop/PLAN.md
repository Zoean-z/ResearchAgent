# Current Goal

Refactor ingest extraction so paper understanding comes from the model over cleaned evidence chunks, while code only prepares text, manages evidence, and rejects weak or unavailable extractions without template fallback.

# Active Checklist

- [x] Decide canonical ownership for chat history vs memory between SQLite and OpenViking
- [x] Specify deletion semantics for dialogue, memory, and mirrored snapshots
- [x] Define OpenViking mirror and retrieval adapter surfaces for message, memory, and session operations
- [x] Add API and service boundaries for delete-session and delete-memory flows
- [x] Add tests for dual-write, retrieval, and deletion propagation
- [x] Split ingest analysis from memory persistence so model-backed extraction can be swapped in cleanly
- [x] Add a model-backed ingest extraction adapter with heuristic fallback for paper, relation, and open-question memories
- [x] Add a model-backed paper summary draft for ingest output and assistant-visible synthesis
- [x] Define ingest extraction protocol and candidate schema for paper, relation, and open-question memory drafts
- [x] Keep ingest candidate input windows broad enough to preserve recall while still filtering to a manageable evidence set
- [x] Re-rank ingest candidates so main-text evidence is preferred and appendix/table content is downweighted instead of removed
- [x] Add per-candidate content roles so the extractor can distinguish main-text evidence from appendix/table/reference material
- [x] Add coarse screening plus evidence reread for ingest when the candidate set is insufficient
- [x] Wire MemoryExtractionService to the ingest extraction boundary without changing persistence behavior
- [x] Add tests for model-backed ingest extraction, heuristic fallback, and reread-triggered candidate expansion
- [x] Add an OpenViking memory gateway adapter behind the memory boundary
- [x] Mirror ingested paper/memory artifacts into OpenViking without changing query behavior
- [x] Add an explicit OpenViking retrieval adapter instead of only implicit retrieval-service bridging
- [x] Expose `search_openviking_memory` as an internal tool without changing the frozen query tool protocol
- [x] Normalize PDF page text before chunking so line-number and hyphenation noise is removed at the source
- [ ] Refine OpenViking session commit timing so all effective user/assistant messages and ingest summaries land in the same committed session flow
- [x] Add an embedded OpenViking backend that uses the local library path instead of an external HTTP server
- [x] Add a one-command Windows startup script that prepares repo-local OpenViking config, starts the backend, and writes startup logs
- [x] Let query runtime choose between local memory search and explicit OpenViking search on the first two retrieval turns
- [x] Add observations for explicit OpenViking retrieval so the model sees it as a turn-level decision
- [x] Broaden the tool pool further only after the explicit OpenViking retrieval turn is stable
- [x] Add a replayable query SSE path with background run start and per-step trace emission
- [x] Extend the same SSE start/stream path to ingest so summary and memory-write steps are also observable live
- [x] Replace phase-shaped query allowed-tools with a unified query tool pool while keeping host step limits and boundary checks
- [x] Surface live run progress, tool calls, and evidence inline inside the chat stream instead of only in the side inspector
- [x] Remove the redundant right-side `Reasoning` inspector so run thinking stays inline in the chat stream
- [x] Tighten query-agent prompts so direct `final_answer` is preferred whenever retrieval is unnecessary
- [x] Replace the remaining query answer mock text with a Chinese answer composer that uses memory and reread evidence
- [x] Remove the left-side `Recent Runs` panel so the sidebar is focused on sessions only
- [x] Make the right-side memory drawer scrollable again
- [ ] Upgrade final assistant answers from event-level commit to token-level text streaming
- [ ] Bring upload/progress ergonomics and richer run inspection on top of the shared SSE path
- [x] Remove `_compose_mock_answer` from the normal query answer path and require `final_answer` for the user-visible response
- [x] Reduce `compose_answer` to an evidence package tool instead of a user-answer builder
- [x] Remove host-forced `compose_answer` and duplicate-signature-forced `compose_answer` exits from the normal query loop
- [x] Switch the default query-agent wiring to the model-driven path instead of heuristic planner as the main route
- [x] Update query tests so they assert model-generated final answers rather than template/mocked answer text
- [x] Compact finalization inputs, raise finalization token budget, and retry truncated plain-text answers once
- [ ] Keep MCP as a later tool-surface option, not the primary persistence path
- [x] Try an aggressive ingest-extraction trial for imported local PDFs by sending the full document to the model-backed extractor instead of the narrower recall window
- [x] Run a live-model ingest smoke on `docs/CRE_v2.pdf` and verify that imported-local-PDF summaries stay Chinese and stop collapsing `what_it_is_about` / `problem_solved`
- [x] Tighten deleted-session visibility so a refreshed frontend does not resurrect tombstoned sessions
- [x] Restore legacy `turn_adapter` query-agent wiring so DeepSeek-backed query answers do not get misrouted into the incompatible `pydantic_ai` transport
- [x] Make the Windows startup script import `.env` into the child process so stale shell API keys do not override repo-local credentials at runtime
- [x] Add schema repair normalization for query-agent model outputs that omit `action_type` or `rationale`
- [x] After `get_paper_memory_bundle`, jump straight into model finalization instead of asking for another structured decision
- [x] Add JSON mode and explicit JSON examples to the DeepSeek structured query-decision prompt
- [x] Inject compact recent conversation context into query turns and expose `list_recent_messages` / `get_conversation_context` as optional query tools
- [x] Add a model-callable `import_arxiv_paper` query tool that reuses the existing arXiv ingest run chain instead of creating a new PDF URL path
- [x] Add a model-callable `search_arxiv` query tool that returns lightweight arXiv metadata and keeps import as a separate explicit step
- [x] Add a query-time no-result fallback so arXiv search/import failures surface as observations instead of failing the whole run
- [x] Ingest stage-1 stopgap repair removed production fallback/template overwrite paths, disabled cross-paper open-question merge, and disabled weak relation generation
- [x] Refactor ingest extraction to model-first full-text or hierarchical extraction with evidence-bound field outputs
- [x] Remove production dependence on keyword scoring and candidate-ranking for paper understanding
- [x] Keep relation disabled unless there is explicit evidence and stop cross-paper merge entirely

# Decisions

- Start with a handwritten runtime loop and keep orchestration framework-agnostic.
- Use a separate frontend and backend structure from the start.
- Treat `docs/project_spec.md` as the single consolidated specification file.
- Keep API schemas separate from domain models even in the mock slice.
- Use in-memory repositories to stabilize ports before adding SQLite adapters.
- OpenViking is the long-term conversation-history and memory store.
- SQLite remains the repo-owned runtime/display store for sessions, task runs, trace, timeline, and UI snapshots.
- Use direct OpenViking adapter wiring as the primary integration path for message sync, memory sync, retrieval, and deletion propagation.
- The repo should support a local embedded OpenViking backend before depending on a separate HTTP server.
- Keep MCP as a later tool facade on top of the same boundary, not as the source-of-truth write path.
- Deletions must propagate to both stores; SQLite may keep tombstones where needed for UI consistency and referential integrity.
- Host runtime remains the owner of run lifecycle, step limit, finish/fail, and trace/timeline persistence even after model tool calling starts.
- Query runtime now exposes one unified query tool pool and allows direct final answers from the first turn, while the host still owns step limits, finish/fail, trace/timeline, and tool-boundary enforcement.
- Query-agent prompts now explicitly bias toward direct `final_answer` for greetings, acknowledgements, capability questions, and other low-context turns where retrieval would not materially improve the answer.
- The first no-repeat guardrail for the wider query pool is that completed tools are not re-exposed; richer anti-stall behavior still needs a dedicated follow-up slice.
- Chat-stream inline run cards are now the primary place for tool calls, progress, and evidence; the side inspector should stay focused on memory and timeline only.
- Normal query answers must now come from the model-facing `final_answer` path; host-written template answers are no longer part of the accepted query contract.
- `compose_answer` is retained only as an evidence/context packaging tool so the model can draft the final answer from structured retrieval output.

# Blockers

- OpenViking runtime/API wiring is still best-effort and not yet validated against a live server in this repository.
- The repo still defaults to the HTTP-backed OpenViking surface bundle unless `embedded` is selected.
- It is not yet confirmed whether deletion should be hard-delete or tombstone for each data class.
- Assistant-side chat mirroring is now in place for query answers and ingest summaries, but commit timing is still partial and does not yet cover every future assistant/runtime message class.
- Query runtime now has an explicit `search_openviking_memory` tool and the first two retrieval turns can choose it instead of only the local memory search path.
- Query turns now receive a compact recent conversation context by default, and the model can optionally fetch a bounded recent-message/context view when follow-up references need more history than the injected summary.
- Query and ingest now share the same host-owned background start + replayable SSE stream path; the frontend can observe step-level execution live, but assistant text is still committed as a full message rather than token-streamed.
- The old host-side anti-stall and duplicate-signature `compose_answer` exits have been removed from the normal query loop, so the main remaining risk is model unavailability rather than host-composed answer leakage.
- Live-model ingest still needs broader quality tuning: the DeepSeek-backed smoke on `docs/CRE_v2.pdf` now stays Chinese and keeps `what_it_is_about` / `problem_solved` distinct, but `problem_solved` can still drift toward a method-style sentence when fallback logic takes over.

# Next Step

The ingest extraction debug evidence is now recorded in the run trace, including the input chunk ids and the field-level reviews. The next useful slice is final-answer token streaming; after that, only later ingest wording tuning should remain if the model output drifts again.

# Deployment Slice

- [x] Add a GitHub Pages-ready static demo that uses the real frontend with mocked `/api` data and fixed showcase conversations
- [x] Add Docker deployment files so the real frontend/backend stack can be started with `docker compose up --build`
- [x] Keep the existing direct arXiv paste behavior unchanged while exposing the same frontend shell in the static demo
- [ ] Validate Docker build/runtime on a machine with Docker available and confirm the published Pages URL after push
