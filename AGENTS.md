# OpenViking Memory-Routed Paper Agent

## Project Goal

Build an agent that can read arXiv papers or local PDFs, write structured research memory, and answer follow-up questions by prioritizing memory before re-reading source papers.

The core product value is not generic paper Q&A. The system must make it visible that memory changes later decisions.

## MVP Scope

- Accept new source material from arXiv links or local PDFs only.
- After a paper is ingested into a session, allow natural-language follow-up questions within that session.
- Support three memory types:
  - `paper_memory`
  - `relation_memory`
  - `open_question_memory`
- Follow-up queries must search memory in this order:
  1. session-related memory
  2. global memory
  3. original paper passages only when memory is insufficient
- Exclude `web_search` from MVP.
- Exclude multi-agent orchestration from MVP.
- Avoid heavy framework coupling at the start.

## Architecture Requirements

The codebase should follow these layers:

- `domain`
- `services`
- `tools`
- `runtime`
- `adapters`
- `api/frontend`

Additional architecture rules:

- Version one uses a handwritten runtime loop.
- Keep runtime semantics decoupled from any agent framework.
- Preserve a clear migration path to PydanticAI later.
- Use SQLite or mock storage first, while preserving adapter boundaries for a future OpenViking integration.

## Coding Constraints

- Do not introduce LangChain, LangGraph, PydanticAI, or OpenViking in the first implementation.
- Do not implement real arXiv downloading in the scaffold phase.
- Do not implement real LLM calls in the scaffold phase.
- Do not implement the full runtime loop in the scaffold phase.
- Keep modules small, explicit, and easy to replace.
- Prefer structured Pydantic-style input/output models at boundaries, even before real implementation.
- Keep trace raw execution data separate from generated narrative text.

## Definition Of Done For The Scaffold Phase

The scaffold phase is complete when:

- The repository contains the planned backend, frontend, docs, and data structure.
- `docs/project_spec.md` contains the consolidated project specification and constraints.
- `README.md` explains the project and points to the spec.
- `backend` contains a minimal FastAPI app with `/health`.
- Placeholder modules exist for the major architectural layers.
- No real business logic, real model integration, or real database integration has been added yet.

## Runtime Direction

The first version must use a handwritten runtime loop with:

- explicit `TaskRun` lifecycle control
- explicit `finish_task` semantics
- explicit termination logic
- explicit step limits enforced by the host runtime

Keep these concerns outside any future agent framework so the system can later migrate to PydanticAI by replacing orchestration, not rewriting core business logic.

## Repository Planning Memory

Use `.project-loop/PLAN.md` as the concise active plan and `PROJECT_STATUS.md` as the current status snapshot for future work in this repository.
