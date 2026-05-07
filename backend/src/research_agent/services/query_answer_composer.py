"""Pure functions for composing query answer text from citations and plans."""

from __future__ import annotations

from collections.abc import Sequence

from research_agent.services.context_rerank_service import ChunkRerankResult, MemoryRerankResult
from research_agent.services.query_citation_builder import RetrievedMemoryCitation, SourceRereadCitation
from research_agent.services.retrieval_service import RetrievalPlan


def compose_mock_answer(
    query: str,
    plan: RetrievalPlan,
    memory_selection: MemoryRerankResult,
    should_reread_source: bool,
    reread_reason: str,
    used_memory_citations: tuple[RetrievedMemoryCitation, ...],
    source_reread_chunks: tuple[SourceRereadCitation, ...],
    source_selection_source: str | None,
) -> str:
    memory_notes = [trim_answer_text(citation.summary, 120) for citation in used_memory_citations if trim_answer_text(citation.summary, 120)]
    source_notes = [format_source_note(citation.page, citation.section, citation.excerpt) for citation in source_reread_chunks if trim_answer_text(citation.excerpt, 140)]

    if memory_notes:
        lead = f"根据当前记忆，关于「{trim_answer_text(query, 80)}」可以先概括为：{memory_notes[0]}。"
        if len(memory_notes) > 1:
            lead += f"补充记忆还包括：{'；'.join(memory_notes[1:3])}。"
    else:
        lead = f"当前关于「{trim_answer_text(query, 80)}」的记忆还不够完整，暂时只能给出保守回答。"

    if source_notes:
        lead += f"原文回读到的关键片段包括：{'；'.join(source_notes[:2])}。"

    if should_reread_source:
        lead += f"系统仍建议继续回读原文，原因是：{trim_answer_text(reread_reason, 120)}。"
    else:
        lead += "当前记忆已经足够直接回答。"

    if memory_selection.fallback_used:
        lead += " 这次记忆选择包含规则兜底。"
    if source_selection_source:
        lead += f" 原文选择方式：{trim_answer_text(source_selection_source, 40)}。"
    return lead.strip()


def compose_mock_answer_preview(
    plan: RetrievalPlan,
    memory_selection: MemoryRerankResult,
    should_reread_source: bool,
    source_selection: ChunkRerankResult | None,
) -> str:
    source_strategy = source_selection.selection_source if source_selection is not None else "none"
    return (
        f"session={len(plan.session_memories.memories)} "
        f"global={len(plan.global_memories.memories)} "
        f"memory_rerank={memory_selection.selection_source} "
        f"reread={should_reread_source} "
        f"source_rerank={source_strategy}"
    )


def trim_answer_text(text: str | None, limit: int) -> str:
    cleaned = (text or "").strip().replace("\n", " ")
    if not cleaned:
        return ""
    return cleaned[:limit]


def format_source_note(page: int | None, section: str | None, excerpt: str) -> str:
    location = []
    if page is not None:
        location.append(f"第{page}页")
    if section:
        location.append(section)
    location_text = " ".join(location) if location else "原文片段"
    excerpt_text = trim_answer_text(excerpt, 120)
    return f"{location_text}：{excerpt_text}"
