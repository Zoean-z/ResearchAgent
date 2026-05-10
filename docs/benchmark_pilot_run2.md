# Benchmark Pilot Run 2

## Scope

This rerun was designed to remove two confounders from Run 1:

1. `search_arxiv` was removed from the normal query tool pool.
2. Fresh sessions were used so single-paper / cross-paper questions and follow-up questions would not inherit unrelated earlier turns.

Backend instance:

- `http://127.0.0.1:8013`
- restarted after the `search_arxiv` rollback so the loaded runtime matched the current code

## Clean Sessions

- Single / cross-paper session:
  - `5f58908a-9684-42a7-84d2-eb0a5a256c76`
  - imported 4 papers:
    - `2605.05191` LongSeeker
    - `2605.05138` Executable World Models for ARC-AGI-3
    - `2605.06664` BAMI
    - `2605.05017` SPINE

- Follow-up session:
  - `be7d8d83-5843-4335-aa42-83f82241d054`
  - imported 2 papers:
    - `2605.05191` LongSeeker
    - `2605.05138` Executable World Models for ARC-AGI-3

## Most Important Result

`search_arxiv` no longer appeared in any of the 5 rerun query traces.

Observed first actions:

- `Q1`: `retrieve_session_memories`
- `Q3`: `list_session_papers`
- `Q7`: `list_session_papers`
- `Q12`: `search_source_chunks`
- `Q13`: `retrieve_session_memories`

This confirms that the rollback worked for the normal query path: follow-up paper questions are now staying inside session-local retrieval / source-reread tools instead of being hijacked by arXiv discovery.

## Results

| Question | Result | Time | First action | Notes |
| --- | --- | --- | --- | --- |
| `Q1` | wrong | `28.3s` | `retrieve_session_memories` | No arXiv detour, but the answer hallucinated a sparse-attention style mechanism instead of LongSeeker's real Context-ReAct formulation. |
| `Q3` | correct | `62.2s` | `list_session_papers` | Correctly identified the three bias types and why BAMI is training-free. |
| `Q7` | correct | `94.0s` | `list_session_papers` | Correctly contrasted Context-ReAct style context orchestration vs executable world models. |
| `Q12` | partial | `48.9s` | `search_source_chunks` | Captured a real difference, but oversimplified it into external-search vs internal-model execution. |
| `Q13` | correct | `9.7s` | `retrieve_session_memories` | Correct follow-up grounding: LongSeeker is more relevant for long-horizon context management. |

## Read

Compared with Run 1, the rerun is clearly better:

- The query path no longer detours into `search_arxiv`.
- Session-local tools are now the first move.
- `Q3`, `Q7`, and `Q13` are usable.
- `Q12` is at least partially grounded instead of collapsing into unrelated context.

However, one core weakness remains:

- `Q1` is still wrong even though the tool path is now reasonable.
- That means the next bottleneck is no longer tool routing; it is answer grounding / final synthesis quality.

## Current Conclusion

The rollback was the right move.

- It fixed the obvious query-path failure mode.
- It made the benchmark path much cleaner.
- It did not fully solve factual accuracy for single-paper explanation questions.

So the repo is now in a better state for benchmarking, but not yet at the point where a larger 16-question or 30-50-question benchmark would be fully persuasive.

## Recommended Next Step

Do one narrow diagnosis pass on why `Q1` still drifts during final answer synthesis even when:

- the paper is already imported,
- session memory exists,
- and the runtime no longer calls `search_arxiv`.

The likely next target is the final answer composition / source-grounding behavior rather than ingest or import.
