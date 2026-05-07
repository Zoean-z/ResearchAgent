"""Pure functions for building memory and source citations from query results."""

from __future__ import annotations

import re
from collections.abc import Sequence

from research_agent.domain.models import Chunk, OpenQuestionMemory, PaperMemory, RelationMemory
from research_agent.services.query_execution_models import RetrievedMemoryCitation, SourceRereadCitation


# ---------------------------------------------------------------------------
# Citation building
# ---------------------------------------------------------------------------


def build_memory_citations(
    memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
    query: str,
    selection_source: str,
) -> tuple[RetrievedMemoryCitation, ...]:
    citations: list[RetrievedMemoryCitation] = []
    seen: set[str] = set()
    for memory in memories:
        if memory.id in seen:
            continue
        seen.add(memory.id)
        citations.append(citation_for(memory, query, selection_source))
    return tuple(citations)


def build_source_reread_citations(
    source_reread_chunks: Sequence[Chunk],
    query: str,
    selection_source: str | None,
) -> tuple[SourceRereadCitation, ...]:
    if not source_reread_chunks:
        return ()
    return tuple(source_reread_citation(chunk, query, selection_source) for chunk in source_reread_chunks)


def unique_memories(
    memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
) -> tuple[PaperMemory | RelationMemory | OpenQuestionMemory, ...]:
    unique: list[PaperMemory | RelationMemory | OpenQuestionMemory] = []
    seen: set[str] = set()
    for memory in memories:
        if memory.id in seen:
            continue
        seen.add(memory.id)
        unique.append(memory)
    return tuple(unique)


def citation_for(
    memory: PaperMemory | RelationMemory | OpenQuestionMemory,
    query: str,
    selection_source: str,
) -> RetrievedMemoryCitation:
    selection_reason = memory_selection_reason(memory, query)
    if selection_source != "retrieval":
        selection_reason = f"{selection_reason}; rerank_strategy={selection_source}"
    if isinstance(memory, PaperMemory):
        summary = paper_memory_summary(memory)
        return RetrievedMemoryCitation(
            memory_id=memory.id,
            memory_type="paper_memory",
            summary=summary,
            selection_reason=selection_reason,
        )
    if isinstance(memory, RelationMemory):
        summary = relation_memory_summary(memory)
        return RetrievedMemoryCitation(
            memory_id=memory.id,
            memory_type="relation_memory",
            summary=summary,
            selection_reason=selection_reason,
        )
    summary = open_question_memory_summary(memory)
    return RetrievedMemoryCitation(
        memory_id=memory.id,
        memory_type="open_question_memory",
        summary=summary,
        selection_reason=selection_reason,
    )


def citations_for(
    memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
    query: str,
    selection_source: str = "retrieval",
) -> tuple[RetrievedMemoryCitation, ...]:
    return tuple(citation_for(memory, query, selection_source) for memory in memories)


# ---------------------------------------------------------------------------
# Memory summaries
# ---------------------------------------------------------------------------


def paper_memory_summary(memory: PaperMemory) -> str:
    parts = [memory.problem or memory.method or memory.novelty_claim or "paper memory"]
    if memory.key_results:
        parts.append(memory.key_results[0])
    return " | ".join(parts)


def relation_memory_summary(memory: RelationMemory) -> str:
    return f"{memory.relation_type.value}: {memory.summary}"


def open_question_memory_summary(memory: OpenQuestionMemory) -> str:
    return memory.unresolved_question


# ---------------------------------------------------------------------------
# Selection reasons
# ---------------------------------------------------------------------------


def memory_selection_reason(memory: PaperMemory | RelationMemory | OpenQuestionMemory, query: str) -> str:
    matched_terms = matched_terms_for_memories(query, [memory])
    if isinstance(memory, PaperMemory):
        memory_type = "paper_memory"
        evidence_score = 2 if any(ref.quote for ref in memory.source_refs) else 1 if memory.source_refs else 0
    elif isinstance(memory, RelationMemory):
        memory_type = "relation_memory"
        evidence_score = 2 if memory.evidence else 0
    else:
        memory_type = "open_question_memory"
        evidence_score = 1 if memory.why_open or memory.possible_followup else 0
    matched_text = ",".join(matched_terms) if matched_terms else "none"
    return f"type={memory_type}; matched_terms={matched_text}; evidence_score={evidence_score}; confidence={memory.confidence.value:.2f}"


def chunk_selection_reason(chunk: Chunk, query: str, selection_source: str | None) -> str:
    matched_terms = matched_terms_for_text(query, [chunk], lambda item: item.text)
    matched_text = ",".join(matched_terms) if matched_terms else "none"
    section = chunk.section or "unknown-section"
    page = str(chunk.page) if chunk.page is not None else "unknown-page"
    suffix = f"; rerank_strategy={selection_source}" if selection_source is not None else ""
    return f"matched_terms={matched_text}; section={section}; page={page}{suffix}"


def source_reread_citation(
    chunk: Chunk,
    query: str,
    selection_source: str | None,
) -> SourceRereadCitation:
    excerpt = chunk.text.strip().replace("\n", " ")
    if len(excerpt) > 220:
        excerpt = excerpt[:217].rstrip() + "..."
    return SourceRereadCitation(
        chunk_id=chunk.id,
        paper_id=chunk.paper_id,
        page=chunk.page,
        section=chunk.section,
        excerpt=excerpt,
        selection_reason=chunk_selection_reason(chunk, query, selection_source),
    )


# ---------------------------------------------------------------------------
# Text matching utilities
# ---------------------------------------------------------------------------


def query_terms(query: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", query.lower())


def normalize_query_text(query: str) -> str:
    return " ".join(query.lower().split())


def memory_text(memory: PaperMemory | RelationMemory | OpenQuestionMemory) -> str:
    if isinstance(memory, PaperMemory):
        source_quotes = " ".join(ref.quote or "" for ref in memory.source_refs)
        return " ".join(
            [
                memory.problem or "",
                memory.method or "",
                " ".join(memory.key_results),
                " ".join(memory.limitations),
                memory.novelty_claim or "",
                source_quotes,
            ]
        ).lower()
    if isinstance(memory, RelationMemory):
        return " ".join([memory.summary, " ".join(memory.evidence), memory.source_paper, memory.target_paper]).lower()
    return " ".join([memory.unresolved_question, " ".join(memory.why_open), " ".join(memory.possible_followup)]).lower()


def matched_terms_for_memories(
    query: str,
    memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
) -> list[str]:
    terms = query_terms(query)
    matched: list[str] = []
    for term in terms:
        if any(term in memory_text(memory) for memory in memories):
            matched.append(term)
    return matched


def matched_terms_for_text(
    query: str,
    items: Sequence[Chunk],
    text_getter,
) -> list[str]:
    terms = query_terms(query)
    matched: list[str] = []
    for term in terms:
        if any(term in text_getter(item).lower() for item in items):
            matched.append(term)
    return matched


def combined_matched_terms(
    query: str,
    memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
    chunks: Sequence[Chunk],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            [
                *matched_terms_for_memories(query, memories),
                *matched_terms_for_text(query, chunks, lambda chunk: chunk.text),
            ]
        )
    )


def memory_has_evidence(memory: PaperMemory | RelationMemory | OpenQuestionMemory) -> bool:
    if isinstance(memory, PaperMemory):
        return any(ref.quote for ref in memory.source_refs)
    if isinstance(memory, RelationMemory):
        return any(memory.evidence)
    return any(memory.why_open) or any(memory.possible_followup)


def max_confidence(
    memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
) -> float:
    if not memories:
        return 0.0
    return max(memory.confidence.value for memory in memories)


def memory_descriptor(memory: PaperMemory | RelationMemory | OpenQuestionMemory) -> str:
    if isinstance(memory, PaperMemory):
        memory_type = "paper_memory"
    elif isinstance(memory, RelationMemory):
        memory_type = "relation_memory"
    else:
        memory_type = "open_question_memory"
    return f"{memory_type}:{memory.id}"
