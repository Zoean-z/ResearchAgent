# Current Goal

Complete the first end-to-end live runtime observation path so both query and ingest runs can start in the background and stream step-level SSE events into the frontend workbench.

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
- [ ] Add a one-command Windows startup script that prepares repo-local OpenViking config, starts the backend, and writes startup logs
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
- [x] Add a first host-side anti-stall guardrail that forces `compose_answer` after repeated low-yield query turns
- [x] Add host-side duplicate-signature suppression so semantically repeated query calls collapse into `compose_answer`
- [ ] Keep MCP as a later tool-surface option, not the primary persistence path

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
- The first anti-stall rule is generic rather than intent-specific: after three consecutive low-yield query turns, the host forces `compose_answer` instead of letting the loop drift further.
- Duplicate suppression is also host-owned: repeated calls against the same effective query surface are normalized into one signature, and a repeat signature forces `compose_answer` instead of executing another near-identical tool step.

# Blockers

- OpenViking runtime/API wiring is still best-effort and not yet validated against a live server in this repository.
- The repo still defaults to the HTTP-backed OpenViking surface bundle unless `embedded` is selected.
- It is not yet confirmed whether deletion should be hard-delete or tombstone for each data class.
- Assistant-side chat mirroring is now in place for query answers and ingest summaries, but commit timing is still partial and does not yet cover every future assistant/runtime message class.
- Query runtime now has an explicit `search_openviking_memory` tool and the first two retrieval turns can choose it instead of only the local memory search path.
- Query and ingest now share the same host-owned background start + replayable SSE stream path; the frontend can observe step-level execution live, but assistant text is still committed as a full message rather than token-streamed.
- Live-model ingest smoke validation still needs to be run against a real DeepSeek-backed extraction transport on representative documents.

# Next Step

The next query-runtime slice is to decide whether the final assistant text should become token-streamed before continuing broader frontend polish, now that the first host-side guardrails cover completed-tool no-repeat, repeated low-yield turns, and duplicate effective query signatures. Keep the Windows startup script and live-model ingest validation in the near-term queue.
