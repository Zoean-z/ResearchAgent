"""Model-facing analysis service for parsed ingest source content."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from research_agent.domain.enums import RelationType
from research_agent.domain.models import Chunk, OpenQuestionMemory, Paper, PaperMemory, RelationMemory, SourceRef
from research_agent.domain.ports import ChunkRepositoryPort, MemoryRepositoryPort, PaperRepositoryPort, SessionRepositoryPort
from research_agent.domain.value_objects import ConfidenceScore
from research_agent.runtime.ingest_extraction import (
    IngestExtractionCandidate,
    IngestExtractionClient,
    IngestExtractionDecision,
    IngestExtractionRequest,
    IngestExtractionWindow,
    IngestOpenQuestionMemoryDraft,
    IngestPaperMemoryDraft,
    IngestPaperSummaryDraft,
    IngestRelationMemoryDraft,
)
from research_agent.services.errors import EntityNotFoundError


_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class MemoryAnalysisResult:
    """Structured memory drafts extracted from parsed ingest content."""

    paper_memory: PaperMemory
    relation_memory: RelationMemory | None
    open_question_memory: OpenQuestionMemory
    paper_summary: IngestPaperSummaryDraft
    context_summary: str


class IngestAnalysisService:
    """Build structured paper, relation, and open-question memory drafts."""

    def __init__(
        self,
        session_repository: SessionRepositoryPort,
        paper_repository: PaperRepositoryPort,
        chunk_repository: ChunkRepositoryPort,
        memory_repository: MemoryRepositoryPort,
        extraction_client: IngestExtractionClient | None = None,
    ) -> None:
        self._session_repository = session_repository
        self._paper_repository = paper_repository
        self._chunk_repository = chunk_repository
        self._memory_repository = memory_repository
        self._extraction_client = extraction_client

    def analyze(self, session_id: str, paper_id: str) -> MemoryAnalysisResult:
        """Analyze parsed source content without persisting memories."""

        session = self._require_session(session_id)
        paper = self._require_paper(paper_id)
        chunks = list(self._chunk_repository.list_by_paper_ids([paper_id]))
        session_documents = list(self._session_repository.list_documents(session.id))
        session_document = next((document for document in session_documents if document.paper_id == paper_id), None)
        if session_document is None:
            raise EntityNotFoundError("SessionDocument", paper_id)

        related_paper = self._select_related_paper(session_id=session_id, current_paper_id=paper_id)
        extraction_result = self._extract_with_client(
            session_id=session_id,
            paper=paper,
            artifact_id=session_document.artifact_id,
            chunks=chunks,
            related_paper=related_paper,
        )
        if extraction_result is None:
            context_text = self._build_context_text(paper, chunks)
            candidate_passages = self._build_candidate_passages(
                paper=paper,
                artifact_id=session_document.artifact_id,
                chunks=chunks,
                window_kind="broad",
            )
            paper_memory = self._build_paper_memory(paper, session_document.artifact_id, chunks, context_text)
            relation_memory = self._build_relation_memory(paper, related_paper, context_text) if related_paper is not None else None
            open_question_memory = self._build_open_question_memory(paper, chunks, context_text, related_paper)
            paper_summary = self._build_paper_summary(
                paper=paper,
                paper_memory=paper_memory,
                open_question_memory=open_question_memory,
                candidate_passages=candidate_passages,
                context_summary=context_text,
            )
            return MemoryAnalysisResult(
                paper_memory=paper_memory,
                relation_memory=relation_memory,
                open_question_memory=open_question_memory,
                paper_summary=paper_summary,
                context_summary=context_text[:240],
            )
        return extraction_result

    def _extract_with_client(
        self,
        *,
        session_id: str,
        paper: Paper,
        artifact_id: str,
        chunks: list[Chunk],
        related_paper: Paper | None,
    ) -> MemoryAnalysisResult | None:
        if self._extraction_client is None:
            return None

        broad_request = self._build_extraction_request(
            session_id=session_id,
            paper=paper,
            chunks=chunks,
            related_paper=related_paper,
            artifact_id=artifact_id,
            window_kind="broad",
        )
        decision = self._invoke_extractor(broad_request)
        if decision is None:
            return None
        if decision.needs_more_context:
            expanded_request = self._build_extraction_request(
                session_id=session_id,
                paper=paper,
                chunks=chunks,
                related_paper=related_paper,
                artifact_id=artifact_id,
                window_kind="expanded",
            )
            expanded_decision = self._invoke_extractor(expanded_request)
            if expanded_decision is not None and not expanded_decision.needs_more_context:
                decision = expanded_decision
                broad_request = expanded_request
            else:
                return None
        return self._result_from_decision(
            decision=decision,
            paper=paper,
            artifact_id=artifact_id,
            chunks=chunks,
            related_paper=related_paper,
            candidate_passages=broad_request.window.candidate_passages,
            context_summary=broad_request.window.context_summary,
        )

    def _invoke_extractor(self, request: IngestExtractionRequest) -> IngestExtractionDecision | None:
        try:
            return self._extraction_client.extract(request)
        except Exception:
            return None

    def _build_extraction_request(
        self,
        *,
        session_id: str,
        paper: Paper,
        chunks: list[Chunk],
        related_paper: Paper | None,
        artifact_id: str,
        window_kind: Literal["broad", "expanded"],
    ) -> IngestExtractionRequest:
        candidate_passages = self._build_candidate_passages(
            paper=paper,
            artifact_id=artifact_id,
            chunks=chunks,
            window_kind=window_kind,
        )
        context_summary = self._build_candidate_context_summary(paper, chunks, candidate_passages)
        return IngestExtractionRequest(
            session_id=session_id,
            paper=paper,
            related_papers=(related_paper,) if related_paper is not None else (),
            window=IngestExtractionWindow(
                kind=window_kind,  # type: ignore[arg-type]
                context_summary=context_summary,
                candidate_passages=candidate_passages,
            ),
        )

    def _build_candidate_passages(
        self,
        *,
        paper: Paper,
        artifact_id: str,
        chunks: list[Chunk],
        window_kind: str,
    ) -> tuple[IngestExtractionCandidate, ...]:
        candidates: list[IngestExtractionCandidate] = [
            IngestExtractionCandidate(
                candidate_id="title",
                chunk_id=None,
                page=None,
                section="title",
                content_role="title",
                excerpt=paper.title,
                relevance_reason="title always included",
            ),
        ]
        if paper.abstract:
            candidates.append(
                IngestExtractionCandidate(
                    candidate_id="abstract",
                    chunk_id=None,
                    page=None,
                    section="abstract",
                    content_role="abstract",
                    excerpt=paper.abstract,
                    relevance_reason="abstract always included",
                )
            )

        if not chunks:
            return tuple(candidates)

        keyword_hit_ids = self._keyword_hit_ids(chunks)
        ranked_chunks = self._rank_candidate_chunks(chunks, keyword_hit_ids)
        if window_kind == "expanded":
            window_chunks = [chunk for chunk, _, _, _ in ranked_chunks]
        else:
            main_limit, appendix_limit, reference_limit = self._broad_candidate_limits(len(chunks))
            role_counts = {"title": 0, "abstract": 0, "main": 0, "appendix": 0, "table": 0, "reference": 0, "unknown": 0}
            window_chunks = []
            for chunk, _, _, content_role in ranked_chunks:
                if content_role in {"title", "abstract", "main", "unknown"}:
                    if role_counts["main"] < main_limit:
                        window_chunks.append(chunk)
                        role_counts["main"] += 1
                    continue
                if content_role in {"appendix", "table"}:
                    if role_counts["appendix"] < appendix_limit:
                        window_chunks.append(chunk)
                        role_counts["appendix"] += 1
                    continue
                if content_role == "reference" and role_counts["reference"] < reference_limit:
                    window_chunks.append(chunk)
                    role_counts["reference"] += 1
        seen_chunk_ids: set[str] = set()
        for index, chunk in enumerate(window_chunks):
            if chunk.id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.id)
            reason_parts = [f"chunk index {index}"]
            if chunk.id in keyword_hit_ids:
                reason_parts.append("keyword hit")
            section = (chunk.section or "").lower()
            content_role = self._classify_chunk_role(chunk)
            if content_role in {"title", "abstract", "main"}:
                reason_parts.append("main text priority")
            elif content_role in {"appendix", "table"}:
                reason_parts.append("appendix downweighted")
            elif content_role == "reference":
                reason_parts.append("reference downweighted")
            candidates.append(
                IngestExtractionCandidate(
                    candidate_id=chunk.id,
                    chunk_id=chunk.id,
                    page=chunk.page,
                    section=chunk.section,
                    content_role=content_role,
                    excerpt=chunk.text[:900],
                    relevance_reason=", ".join(reason_parts),
                )
            )
        return tuple(candidates)

    def _rank_candidate_chunks(
        self,
        chunks: list[Chunk],
        keyword_hit_ids: set[str],
    ) -> list[tuple[Chunk, int, list[str], str]]:
        ranked: list[tuple[Chunk, int, list[str], str]] = []
        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks):
            score = 0
            reasons: list[str] = []
            section = (chunk.section or "").lower()
            text_lower = chunk.text.lower()
            content_role = self._classify_chunk_role(chunk)
            if content_role == "title":
                score += 40
                reasons.append("title section")
            if content_role == "abstract":
                score += 35
                reasons.append("abstract section")
            if content_role == "main" and any(token in section or token in text_lower for token in ("intro", "introduction", "background")):
                score += 28
                reasons.append("main text section")
            if content_role == "main" and any(token in section or token in text_lower for token in ("method", "approach", "experiment", "evaluation")):
                score += 24
                reasons.append("main text section")
            if content_role == "main" and any(token in section or token in text_lower for token in ("result", "discussion", "conclusion", "analysis")):
                score += 22
                reasons.append("main text section")
            if any(token in section or token in text_lower for token in ("limit", "future work")):
                score += 18
                reasons.append("limitations section")
            if content_role in {"appendix", "table"}:
                score -= 15
                reasons.append("appendix section")
            if content_role == "reference":
                score -= 60
                reasons.append("reference section")
            if chunk.id in keyword_hit_ids:
                score += 20
                reasons.append("keyword hit")

            digit_count = sum(character.isdigit() for character in chunk.text)
            letter_count = sum(character.isalpha() for character in chunk.text)
            if digit_count >= 8 and digit_count > max(letter_count // 4, 1):
                score -= 8
                reasons.append("numeric heavy")
            if "table" in text_lower and digit_count >= 4:
                score -= 6
                reasons.append("table heavy")
            if index < 3:
                score += 6
                reasons.append("early section")
            elif index >= max(total_chunks - 3, 0):
                score += 4
                reasons.append("late section")
            ranked.append((chunk, score, reasons, content_role))
        return sorted(ranked, key=lambda item: (-item[1], item[0].page if item[0].page is not None else 10**9, item[0].id))

    def _broad_candidate_limits(self, chunk_count: int) -> tuple[int, int, int]:
        main_limit = min(14, max(6, chunk_count // 3))
        appendix_limit = min(2, max(1, chunk_count // 18))
        reference_limit = 0
        return main_limit, appendix_limit, reference_limit

    def _classify_chunk_role(self, chunk: Chunk) -> Literal["title", "abstract", "main", "appendix", "table", "reference", "unknown"]:
        section = (chunk.section or "").lower()
        text_lower = chunk.text.lower()
        if any(token in section for token in ("title",)):
            return "title"
        if "abstract" in section or text_lower.startswith("abstract"):
            return "abstract"
        if any(token in section or token in text_lower for token in ("appendix", "appendices", "supplement", "supplementary")):
            return "appendix"
        if any(token in section or token in text_lower for token in ("reference", "bibliography")):
            return "reference"
        if "table" in text_lower or any(token in section or token in text_lower for token in ("intro", "introduction", "background", "method", "approach", "experiment", "evaluation", "result", "discussion", "conclusion", "analysis", "limit", "future work")):
            return "main"
        if len(text_lower.split()) < 18 and sum(character.isdigit() for character in chunk.text) >= 4:
            return "table"
        return "unknown"

    def _keyword_hit_ids(self, chunks: list[Chunk]) -> set[str]:
        return {chunk.id for chunk in self._candidate_chunks_by_keywords(chunks)}

    def _build_candidate_context_summary(
        self,
        paper: Paper,
        chunks: list[Chunk],
        candidate_passages: tuple[IngestExtractionCandidate, ...],
    ) -> str:
        parts = [paper.title, paper.abstract or ""]
        parts.extend(candidate.excerpt for candidate in candidate_passages[:12])
        if len(candidate_passages) < 3 and chunks:
            parts.extend(chunk.text[:240] for chunk in chunks[:3])
        summary = " ".join(part for part in parts if part).strip()
        return summary[:4000]

    def _candidate_chunks_by_keywords(self, chunks: list[Chunk]) -> list[Chunk]:
        keyword_chunks: list[Chunk] = []
        keywords = (
            self._problem_keywords()
            + self._method_keywords()
            + self._result_keywords()
            + self._limitation_keywords()
            + self._novelty_keywords()
            + self._relation_keywords()
        )
        for chunk in chunks:
            lowered = chunk.text.lower()
            if any(keyword in lowered for keyword in keywords):
                keyword_chunks.append(chunk)
        return keyword_chunks

    def _result_from_decision(
        self,
        *,
        decision: IngestExtractionDecision,
        paper: Paper,
        artifact_id: str,
        chunks: list[Chunk],
        related_paper: Paper | None,
        candidate_passages: tuple[IngestExtractionCandidate, ...],
        context_summary: str,
    ) -> MemoryAnalysisResult:
        paper_memory = self._paper_memory_from_draft(
            paper=paper,
            artifact_id=artifact_id,
            chunks=chunks,
            context_summary=context_summary,
            candidate_passages=candidate_passages,
            draft=decision.paper,
        )
        relation_memory = None
        if related_paper is not None and decision.relation is not None:
            relation_memory = self._relation_memory_from_draft(
                paper=paper,
                related_paper=related_paper,
                candidate_passages=candidate_passages,
                draft=decision.relation,
            )

        open_question_memory = self._open_question_memory_from_draft(
            paper=paper,
            related_paper=related_paper,
            chunks=chunks,
            context_summary=context_summary,
            draft=decision.open_question,
        )
        paper_summary = self._paper_summary_from_draft(
            paper=paper,
            paper_memory=paper_memory,
            open_question_memory=open_question_memory,
            candidate_passages=candidate_passages,
            draft=decision.paper_summary,
        )
        return MemoryAnalysisResult(
            paper_memory=paper_memory,
            relation_memory=relation_memory,
            open_question_memory=open_question_memory,
            paper_summary=paper_summary,
            context_summary=context_summary[:240],
        )

    def _paper_memory_from_draft(
        self,
        *,
        paper: Paper,
        artifact_id: str,
        chunks: list[Chunk],
        context_summary: str,
        candidate_passages: tuple[IngestExtractionCandidate, ...],
        draft: IngestPaperMemoryDraft,
    ) -> PaperMemory:
        if self._is_placeholder_source_title(paper.title):
            summary_seed = self._summary_seed_text(
                draft.problem,
                draft.method,
                draft.novelty_claim,
                *draft.key_results,
                *draft.limitations,
                context_summary,
            )
            source_refs = self._source_refs_for_candidate_ids(
                paper_id=paper.id,
                artifact_id=artifact_id,
                chunks=chunks,
                candidate_passages=candidate_passages,
                candidate_ids=draft.evidence_candidate_ids,
                fallback_text=context_summary,
            )
            key_results = self._prefer_summary_items(draft.key_results, self._fallback_key_result_texts(summary_seed), prefer_chinese=True)
            limitations = self._prefer_summary_items(draft.limitations, self._fallback_limitation_texts(summary_seed), prefer_chinese=True)
            return PaperMemory(
                paper_id=paper.id,
                problem=self._prefer_summary_text(draft.problem, self._fallback_topic_text(summary_seed, paper.title), prefer_chinese=True),
                method=self._prefer_summary_text(draft.method, self._fallback_method_text(summary_seed), prefer_chinese=True) or None,
                key_results=key_results,
                limitations=limitations,
                novelty_claim=self._prefer_summary_text(draft.novelty_claim, self._fallback_novelty_text(summary_seed), prefer_chinese=True) or None,
                source_refs=source_refs,
                confidence=ConfidenceScore(value=draft.confidence),
            )
        source_refs = self._source_refs_for_candidate_ids(
            paper_id=paper.id,
            artifact_id=artifact_id,
            chunks=chunks,
            candidate_passages=candidate_passages,
            candidate_ids=draft.evidence_candidate_ids,
            fallback_text=context_summary,
        )
        key_results = list(draft.key_results)
        limitations = list(draft.limitations)
        if not key_results and chunks:
            key_results = [chunks[0].text[:240]]
        return PaperMemory(
            paper_id=paper.id,
            problem=draft.problem or paper.title,
            method=draft.method,
            key_results=key_results,
            limitations=limitations,
            novelty_claim=draft.novelty_claim,
            source_refs=source_refs,
            confidence=ConfidenceScore(value=draft.confidence),
        )

    def _relation_memory_from_draft(
        self,
        *,
        paper: Paper,
        related_paper: Paper,
        candidate_passages: tuple[IngestExtractionCandidate, ...],
        draft: IngestRelationMemoryDraft,
    ) -> RelationMemory:
        relation_type = self._coerce_relation_type(draft.relation_type)
        return RelationMemory(
            source_paper=paper.id,
            target_paper=related_paper.id,
            relation_type=relation_type,
            summary=draft.summary,
            evidence=self._evidence_for_candidate_ids(candidate_passages, draft.evidence_candidate_ids, draft.summary),
            confidence=ConfidenceScore(value=draft.confidence),
        )

    def _open_question_memory_from_draft(
        self,
        *,
        paper: Paper,
        related_paper: Paper | None,
        chunks: list[Chunk],
        context_summary: str,
        draft: IngestOpenQuestionMemoryDraft,
    ) -> OpenQuestionMemory:
        if self._is_placeholder_source_title(paper.title):
            related_papers = [paper.id]
            if related_paper is not None:
                related_papers.append(related_paper.id)
            why_open = self._prefer_summary_items(
                draft.why_open,
                self._fallback_why_open_texts(context_summary, len(chunks)),
                prefer_chinese=True,
                max_items=3,
            )
            possible_followup = self._prefer_summary_items(
                draft.possible_followup,
                self._build_followups(why_open, prefer_chinese=True),
                prefer_chinese=True,
                max_items=3,
            )
            return OpenQuestionMemory(
                unresolved_question=self._prefer_summary_text(
                    draft.unresolved_question,
                    self._limitation_to_question(context_summary, prefer_chinese=True),
                    prefer_chinese=True,
                ),
                related_papers=related_papers,
                why_open=why_open,
                possible_followup=possible_followup,
                confidence=ConfidenceScore(value=draft.confidence),
            )
        related_papers = [paper.id]
        if related_paper is not None:
            related_papers.append(related_paper.id)
        why_open = list(draft.why_open) or [f"已解析 {len(chunks)} 个文本分块，但未抽取到明确局限性。"]
        possible_followup = list(draft.possible_followup) or self._build_followups(why_open)
        return OpenQuestionMemory(
            unresolved_question=draft.unresolved_question or self._limitation_to_question(context_summary),
            related_papers=related_papers,
            why_open=why_open,
            possible_followup=possible_followup,
            confidence=ConfidenceScore(value=draft.confidence),
        )

    def _paper_summary_from_draft(
        self,
        *,
        paper: Paper,
        paper_memory: PaperMemory,
        open_question_memory: OpenQuestionMemory,
        candidate_passages: tuple[IngestExtractionCandidate, ...],
        draft: IngestPaperSummaryDraft,
    ) -> IngestPaperSummaryDraft:
        if self._is_placeholder_source_title(paper.title):
            summary_seed = self._summary_seed_text(
                draft.what_it_is_about,
                draft.problem_solved,
                *draft.new_ideas,
                *draft.limitations,
                *draft.suggestions_or_questions,
                paper_memory.problem,
                paper_memory.method,
                paper_memory.novelty_claim,
                *paper_memory.key_results,
                *paper_memory.limitations,
                *open_question_memory.why_open,
                *open_question_memory.possible_followup,
            )
            evidence_candidate_ids = self._preferred_summary_evidence_ids(candidate_passages, draft.evidence_candidate_ids)
            return IngestPaperSummaryDraft(
                what_it_is_about=self._prefer_summary_text(
                    draft.what_it_is_about,
                    self._fallback_topic_text(summary_seed, paper.title),
                    prefer_chinese=True,
                ),
                problem_solved=self._prefer_summary_text(
                    draft.problem_solved,
                    self._fallback_problem_text(summary_seed, paper.title),
                    prefer_chinese=True,
                ),
                new_ideas=self._prefer_summary_items(
                    draft.new_ideas,
                    self._fallback_idea_texts(summary_seed),
                    prefer_chinese=True,
                ),
                limitations=self._prefer_summary_items(
                    draft.limitations,
                    self._fallback_limitation_texts(summary_seed, open_question_memory.why_open),
                    prefer_chinese=True,
                ),
                suggestions_or_questions=self._prefer_summary_items(
                    draft.suggestions_or_questions,
                    self._fallback_suggestion_texts(summary_seed, paper.title, open_question_memory.possible_followup),
                    prefer_chinese=True,
                ),
                evidence_candidate_ids=evidence_candidate_ids,
                confidence=draft.confidence if draft.confidence else paper_memory.confidence.value,
            )
        evidence_candidate_ids = self._preferred_summary_evidence_ids(candidate_passages, draft.evidence_candidate_ids)
        fallback_ideas = tuple(
            item
            for item in (
                paper_memory.method,
                paper_memory.novelty_claim,
                paper_memory.key_results[0] if paper_memory.key_results else None,
            )
            if item
        )[:2]
        fallback_limitations = tuple(open_question_memory.why_open[:2]) or tuple(paper_memory.limitations[:2])
        fallback_suggestions = tuple(open_question_memory.possible_followup[:2]) or (
            f"\u540e\u7eed\u53ef\u4ee5\u56f4\u7ed5\u300a{paper.title}\u300b\u4e2d\u4ecd\u672a\u9a8c\u8bc1\u7684\u90e8\u5206\u7ee7\u7eed\u8ffd\u95ee\u3002",
        )
        return IngestPaperSummaryDraft(
            what_it_is_about=self._sanitize_summary_text(
                draft.what_it_is_about,
                paper.title,
                paper_memory.problem,
            ),
            problem_solved=self._sanitize_summary_text(
                draft.problem_solved,
                paper_memory.method,
                paper_memory.problem,
                paper.title,
            ),
            new_ideas=self._sanitize_summary_items(draft.new_ideas, fallback_ideas),
            limitations=self._sanitize_summary_items(draft.limitations, fallback_limitations),
            suggestions_or_questions=self._sanitize_summary_items(draft.suggestions_or_questions, fallback_suggestions),
            evidence_candidate_ids=evidence_candidate_ids,
            confidence=draft.confidence if draft.confidence else paper_memory.confidence.value,
        )

    def _build_paper_summary(
        self,
        *,
        paper: Paper,
        paper_memory: PaperMemory,
        open_question_memory: OpenQuestionMemory,
        candidate_passages: tuple[IngestExtractionCandidate, ...],
        context_summary: str,
    ) -> IngestPaperSummaryDraft:
        evidence_candidate_ids = tuple(
            candidate.candidate_id
            for candidate in candidate_passages
            if candidate.candidate_id not in {"title", "abstract"}
        )[:4]
        what_it_is_about = paper_memory.problem or paper.title
        problem_solved = paper_memory.problem or paper.title
        new_ideas = tuple(
            item
            for item in (
                paper_memory.method,
                paper_memory.novelty_claim,
                paper_memory.key_results[0] if paper_memory.key_results else None,
            )
            if item
        )[:3]
        limitations = tuple(paper_memory.limitations[:3]) or tuple(open_question_memory.why_open[:3])
        suggestions_or_questions = tuple(open_question_memory.possible_followup[:3])
        if not suggestions_or_questions:
            suggestions_or_questions = (
                f"\u540e\u7eed\u53ef\u4ee5\u56f4\u7ed5\u300a{paper.title}\u300b\u4e2d\u4ecd\u672a\u9a8c\u8bc1\u7684\u90e8\u5206\u7ee7\u7eed\u8ffd\u95ee\u3002",
            )
        confidence = paper_memory.confidence.value if paper_memory.confidence.value else 0.5
        return IngestPaperSummaryDraft(
            what_it_is_about=self._sanitize_summary_text(what_it_is_about, paper.title, paper_memory.problem),
            problem_solved=self._sanitize_summary_text(problem_solved, paper_memory.method, paper_memory.problem, paper.title),
            new_ideas=self._sanitize_summary_items(new_ideas, new_ideas),
            limitations=self._sanitize_summary_items(limitations, limitations),
            suggestions_or_questions=self._sanitize_summary_items(suggestions_or_questions, suggestions_or_questions),
            evidence_candidate_ids=evidence_candidate_ids or ("title", "abstract"),
            confidence=confidence,
        )

    def _preferred_summary_evidence_ids(
        self,
        candidate_passages: tuple[IngestExtractionCandidate, ...],
        draft_candidate_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        candidate_by_id = {candidate.candidate_id: candidate for candidate in candidate_passages}
        preferred_roles = {"title", "abstract", "main"}
        selected = [
            candidate_id
            for candidate_id in draft_candidate_ids
            if candidate_id in candidate_by_id and candidate_by_id[candidate_id].content_role in preferred_roles
        ]
        if selected:
            return tuple(selected[:4])

        preferred = [
            candidate.candidate_id
            for candidate in candidate_passages
            if candidate.content_role in preferred_roles and candidate.candidate_id not in {"title", "abstract"}
        ]
        if preferred:
            return tuple(preferred[:4])

        fallback = [
            candidate.candidate_id
            for candidate in candidate_passages
            if candidate.content_role in {"appendix", "table", "unknown"}
        ]
        return tuple(fallback[:4])

    def _sanitize_summary_text(self, text: str | None, *fallback_candidates: str | None) -> str:
        candidate = (text or "").strip()
        if not candidate:
            return self._first_clean_text(*fallback_candidates)
        if self._looks_like_summary_noise(candidate):
            return self._first_clean_text(*fallback_candidates)
        return candidate[:240]

    def _first_clean_text(self, *candidates: str | None) -> str:
        for candidate in candidates:
            cleaned = (candidate or "").strip()
            if cleaned and not self._looks_like_summary_noise(cleaned):
                return cleaned[:240]
        for candidate in candidates:
            cleaned = (candidate or "").strip()
            if cleaned:
                return cleaned[:240]
        return ""

    def _sanitize_summary_items(
        self,
        items: tuple[str, ...],
        fallback_items: tuple[str, ...],
        *,
        max_items: int = 2,
    ) -> tuple[str, ...]:
        cleaned: list[str] = []
        for item in items:
            normalized = self._normalize_summary_item(item)
            if normalized is None or normalized in cleaned:
                continue
            cleaned.append(normalized)
            if len(cleaned) >= max_items:
                break
        if cleaned:
            return tuple(cleaned)
        fallback_cleaned: list[str] = []
        for item in fallback_items:
            normalized = self._normalize_summary_item(item)
            if normalized is None or normalized in fallback_cleaned:
                continue
            fallback_cleaned.append(normalized)
            if len(fallback_cleaned) >= max_items:
                break
        return tuple(fallback_cleaned)

    def _normalize_summary_item(self, item: str) -> str | None:
        candidate = item.strip()
        if not candidate or self._looks_like_summary_noise(candidate):
            return None
        return candidate[:240]

    def _looks_like_summary_noise(self, text: str) -> bool:
        lowered = text.lower()
        noise_markers = (
            "proceedings of",
            "et al.",
            "doi:",
            "http://",
            "https://",
            "page ",
            "vol.",
            "conference",
            "workshop",
            "in proceedings",
            "tacas",
            "etap",
            "bibliography",
            "references",
        )
        if any(marker in lowered for marker in noise_markers):
            return True
        if "..." in text or re.search(r"\.{4,}", text):
            return True
        if re.match(r"^\s*\d+\s+\d+\.\d+\s+", text):
            return True
        if len(text.split()) > 28 and any(char.isdigit() for char in text):
            return True
        return False

    def _is_placeholder_source_title(self, title: str) -> bool:
        lowered = title.strip().lower()
        return lowered.startswith("imported local pdf") or lowered.startswith("imported arxiv pdf")

    def _summary_seed_text(self, *values: str | None) -> str:
        return " ".join(value.strip() for value in values if value and value.strip())

    def _contains_cjk(self, text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    def _prefer_summary_text(self, candidate: str | None, fallback: str, *, prefer_chinese: bool) -> str:
        cleaned = (candidate or "").strip()
        if cleaned and not self._looks_like_summary_noise(cleaned):
            if not prefer_chinese or self._contains_cjk(cleaned):
                return cleaned[:240]
        fallback_cleaned = fallback.strip()
        if fallback_cleaned:
            return fallback_cleaned[:240]
        return cleaned[:240]

    def _prefer_summary_items(
        self,
        items: tuple[str, ...],
        fallback_items: tuple[str, ...],
        *,
        prefer_chinese: bool,
        max_items: int = 2,
    ) -> tuple[str, ...]:
        cleaned: list[str] = []
        for item in items:
            normalized = item.strip()
            if not normalized or self._looks_like_summary_noise(normalized):
                continue
            if prefer_chinese and not self._contains_cjk(normalized):
                continue
            if normalized in cleaned:
                continue
            cleaned.append(normalized[:240])
            if len(cleaned) >= max_items:
                break
        if cleaned:
            return tuple(cleaned)
        fallback_cleaned: list[str] = []
        for item in fallback_items:
            normalized = item.strip()
            if not normalized or self._looks_like_summary_noise(normalized):
                continue
            if prefer_chinese and not self._contains_cjk(normalized):
                continue
            if normalized in fallback_cleaned:
                continue
            fallback_cleaned.append(normalized[:240])
            if len(fallback_cleaned) >= max_items:
                break
        return tuple(fallback_cleaned)

    def _fallback_topic_text(self, seed_text: str, title: str) -> str:
        lowered = seed_text.lower()
        if any(keyword in lowered for keyword in ("long-context", "long context", "position-aware", "positional", "position sensitive")):
            return "本文主要讨论长上下文场景中的位置敏感性，以及它对检索和推理评估的影响。"
        if "retrieval" in lowered and ("reasoning" in lowered or "evaluation" in lowered):
            return "本文主要讨论检索与推理评估中的差异及其位置偏差。"
        if "benchmark" in lowered and "evaluation" in lowered:
            return "本文主要讨论相关基准上的评估方法及其局限。"
        if any(keyword in lowered for keyword in ("robust", "distribution shift", "shift")):
            return "本文主要讨论模型在分布偏移下的鲁棒性表现。"
        if self._is_placeholder_source_title(title):
            return "本文主要围绕论文中的核心研究问题展开。"
        return f"本文主要围绕《{title}》的核心研究问题展开。"

    def _fallback_problem_text(self, seed_text: str, title: str) -> str:
        lowered = seed_text.lower()
        if any(keyword in lowered for keyword in ("long-context", "long context", "position-aware", "positional", "position sensitive")):
            return "本文试图解决长上下文评估中位置变化带来的偏差问题，并分析检索和推理场景中的脆弱性。"
        if "retrieval" in lowered and ("reasoning" in lowered or "evaluation" in lowered):
            return "本文试图解决检索与推理评估中的差异和位置偏差问题。"
        if "benchmark" in lowered and "evaluation" in lowered:
            return "本文试图补足相关基准评估中的系统性分析。"
        if any(keyword in lowered for keyword in ("robust", "distribution shift", "shift")):
            return "本文试图提升模型在分布偏移下的稳定性。"
        if self._is_placeholder_source_title(title):
            return "本文试图解决论文提出的核心问题，并验证相关方法的效果。"
        return f"本文试图解决《{title}》提出的核心问题，并验证相关方法的效果。"

    def _fallback_method_text(self, seed_text: str) -> str:
        lowered = seed_text.lower()
        if "retrieval" in lowered and "evaluation" in lowered:
            return "本文采用检索与评估结合的方式展开分析。"
        if "benchmark" in lowered:
            return "本文采用基准评估与对比实验来验证结论。"
        return "本文采用文中的方法设计和实验流程来验证上述问题。"

    def _fallback_novelty_text(self, seed_text: str) -> str:
        lowered = seed_text.lower()
        if any(keyword in lowered for keyword in ("long-context", "position-aware", "positional")):
            return "本文提出了位置敏感性的分析视角。"
        if "retrieval" in lowered and "reasoning" in lowered:
            return "本文提出了检索与推理评估的对照分析。"
        return "本文提出了新的方法框架或评估视角。"

    def _fallback_idea_texts(self, seed_text: str) -> tuple[str, ...]:
        lowered = seed_text.lower()
        if any(keyword in lowered for keyword in ("long-context", "position-aware", "positional")):
            return ("本文提出了位置敏感性的分析视角。", "本文比较了位置变化对不同评估设置的影响。")
        if "retrieval" in lowered and "reasoning" in lowered:
            return ("本文提出了检索与推理评估的对照分析。", "本文强调了两类场景中的位置偏差差异。")
        if "benchmark" in lowered and "evaluation" in lowered:
            return ("本文补充了相关基准上的系统评估。", "本文揭示了现有评估方法的局限。")
        return ("本文提出了新的方法框架或评估视角。", "本文通过实验验证了核心结论。")

    def _fallback_key_result_texts(self, seed_text: str) -> tuple[str, ...]:
        lowered = seed_text.lower()
        if "long-context" in lowered or "position-aware" in lowered or "positional" in lowered:
            return ("研究显示，位置变化会显著影响长上下文评估结果。",)
        if "retrieval" in lowered and "reasoning" in lowered:
            return ("研究显示，检索与推理场景中的位置偏差表现不同。",)
        if "benchmark" in lowered and "evaluation" in lowered:
            return ("研究结果强调了基准评估中需要额外关注位置偏差。",)
        return ("实验结果表明该方法在相关设置下取得了改进。",)

    def _fallback_limitation_texts(self, seed_text: str, why_open: tuple[str, ...] = ()) -> tuple[str, ...]:
        lowered = seed_text.lower()
        if any(keyword in lowered for keyword in ("scal", "large")):
            return ("当前结论仍需要在更大规模实验中继续验证。",)
        if any(keyword in lowered for keyword in ("robust", "distribution shift", "shift")):
            return ("当前结论仍需要在分布偏移场景下进一步验证。",)
        if why_open and any(self._contains_cjk(item) for item in why_open):
            return tuple(why_open[:2])
        return ("当前结论仍依赖现有实验设置，后续还需要更多证据。",)

    def _fallback_why_open_texts(self, seed_text: str, chunk_count: int) -> tuple[str, ...]:
        lowered = seed_text.lower()
        if any(keyword in lowered for keyword in ("scal", "large")):
            return ("当前结果仍需要在更大规模设置中继续验证。",)
        if any(keyword in lowered for keyword in ("robust", "distribution shift", "shift")):
            return ("当前结果在分布偏移下的稳定性仍需继续检验。",)
        return (f"已解析 {chunk_count} 个文本分块，但尚未抽取到足够清晰的局限性。",)

    def _fallback_suggestion_texts(self, seed_text: str, title: str, possible_followup: tuple[str, ...] = ()) -> tuple[str, ...]:
        lowered = seed_text.lower()
        if possible_followup and any(self._contains_cjk(item) for item in possible_followup):
            return tuple(possible_followup[:2])
        if any(keyword in lowered for keyword in ("scal", "large")):
            return ("后续可以开展更大规模实验。",)
        if any(keyword in lowered for keyword in ("robust", "distribution shift", "shift")):
            return ("后续可以继续评估方法在分布偏移下的表现。",)
        if self._is_placeholder_source_title(title):
            return ("后续可以继续回读原文，并补充更多证据。",)
        return (f"后续可以围绕《{title}》中仍未验证的部分继续追问。",)

    def _source_refs_for_candidate_ids(
        self,
        *,
        paper_id: str,
        artifact_id: str,
        chunks: list[Chunk],
        candidate_passages: tuple[IngestExtractionCandidate, ...],
        candidate_ids: tuple[str, ...],
        fallback_text: str,
    ) -> list[SourceRef]:
        selected_candidates = [candidate for candidate in candidate_passages if candidate.candidate_id in candidate_ids]
        if selected_candidates:
            return [
                SourceRef(
                    paper_id=paper_id,
                    artifact_id=artifact_id,
                    page=candidate.page,
                    section=candidate.section,
                    chunk_id=candidate.chunk_id,
                    quote=candidate.excerpt[:240],
                )
                for candidate in selected_candidates
            ]

        selected_chunks = [chunk for chunk in chunks if any(candidate.chunk_id == chunk.id for candidate in candidate_passages)]
        if not selected_chunks and not chunks:
            return [
                SourceRef(
                    paper_id=paper_id,
                    artifact_id=artifact_id,
                    section="summary",
                    quote=fallback_text[:240],
                )
            ]
        if not selected_chunks:
            selected_chunks = chunks[:2]
        return [
            SourceRef(
                paper_id=paper_id,
                artifact_id=artifact_id,
                page=chunk.page,
                section=chunk.section,
                chunk_id=chunk.id,
                quote=chunk.text[:240],
            )
            for chunk in selected_chunks
        ]

    def _evidence_for_candidate_ids(
        self,
        candidate_passages: tuple[IngestExtractionCandidate, ...],
        candidate_ids: tuple[str, ...],
        fallback_text: str,
    ) -> list[str]:
        if not candidate_ids:
            return [fallback_text[:240]]
        selected = [candidate.excerpt for candidate in candidate_passages if candidate.candidate_id in candidate_ids]
        return selected or [fallback_text[:240]]

    def _coerce_relation_type(self, relation_type: str) -> RelationType:
        try:
            return RelationType(relation_type)
        except ValueError:
            return RelationType.COMPLEMENTS

    def _build_paper_memory(
        self,
        paper: Paper,
        artifact_id: str,
        chunks: list[Chunk],
        context_text: str,
    ) -> PaperMemory:
        sentences = self._sentences(context_text)
        source_refs = self._build_source_refs(paper.id, artifact_id, chunks, context_text)
        if self._is_placeholder_source_title(paper.title):
            seed_text = self._summary_seed_text(context_text, *sentences[:3])
            return PaperMemory(
                paper_id=paper.id,
                problem=self._fallback_topic_text(seed_text, paper.title),
                method=self._fallback_method_text(seed_text),
                key_results=self._fallback_key_result_texts(seed_text),
                limitations=self._fallback_limitation_texts(seed_text),
                novelty_claim=self._fallback_novelty_text(seed_text),
                source_refs=source_refs,
                confidence=self._paper_confidence(chunks, [], []),
            )
        problem = self._first_sentence(sentences, self._problem_keywords()) or paper.title
        method = self._first_sentence(sentences, self._method_keywords())
        key_results = self._collect_sentences(sentences, self._result_keywords(), limit=3)
        limitations = self._collect_sentences(sentences, self._limitation_keywords(), limit=3)
        novelty_claim = self._first_sentence(sentences, self._novelty_keywords())
        confidence = self._paper_confidence(chunks, key_results, limitations)
        if not key_results and chunks:
            key_results = [chunks[0].text[:240]]
        return PaperMemory(
            paper_id=paper.id,
            problem=problem,
            method=method,
            key_results=key_results,
            limitations=limitations,
            novelty_claim=novelty_claim,
            source_refs=source_refs,
            confidence=confidence,
        )

    def _build_relation_memory(self, paper: Paper, related_paper: Paper, context_text: str) -> RelationMemory:
        relation_type = self._infer_relation_type(context_text)
        evidence = self._collect_sentences(self._sentences(context_text), self._relation_keywords(), limit=2)
        if not evidence:
            evidence = [f"\u5bfc\u5165\u65f6\u5df2\u5c06\u6765\u6e90\u5185\u5bb9\u4e0e\u300a{related_paper.title}\u300b\u8fdb\u884c\u5173\u7cfb\u6bd4\u8f83\u3002"]
        summary = self._build_relation_summary(paper.title, related_paper.title, relation_type)
        return RelationMemory(
            source_paper=paper.id,
            target_paper=related_paper.id,
            relation_type=relation_type,
            summary=summary,
            evidence=evidence,
            confidence=ConfidenceScore(value=0.6 if evidence else 0.5),
        )

    def _build_open_question_memory(
        self,
        paper: Paper,
        chunks: list[Chunk],
        context_text: str,
        related_paper: Paper | None,
    ) -> OpenQuestionMemory:
        sentences = self._sentences(context_text)
        limitation = self._first_sentence(sentences, self._limitation_keywords())
        if self._is_placeholder_source_title(paper.title):
            unresolved_question = self._limitation_to_question(limitation or context_text, prefer_chinese=True)
        else:
            unresolved_question = self._limitation_to_question(limitation) if limitation else f"《{paper.title}》中还有哪些结论尚未充分验证？"
        why_open = self._collect_sentences(sentences, self._limitation_keywords(), limit=3)
        if not why_open:
            why_open = [f"\u5df2\u89e3\u6790 {len(chunks)} \u4e2a\u6587\u672c\u5757\uff0c\u4f46\u5c1a\u672a\u63d0\u53d6\u5230\u660e\u786e\u7684\u5c40\u9650\u6027\u3002"]
        possible_followup = self._build_followups(why_open, prefer_chinese=self._is_placeholder_source_title(paper.title))
        related_papers = [paper.id]
        if related_paper is not None:
            related_papers.append(related_paper.id)
        confidence = ConfidenceScore(value=0.5 if why_open else 0.35)
        return OpenQuestionMemory(
            unresolved_question=unresolved_question,
            related_papers=related_papers,
            why_open=why_open,
            possible_followup=possible_followup,
            confidence=confidence,
        )

    def _build_source_refs(self, paper_id: str, artifact_id: str, chunks: list[Chunk], context_text: str) -> list[SourceRef]:
        if not chunks:
            return [
                SourceRef(
                    paper_id=paper_id,
                    artifact_id=artifact_id,
                    section="title",
                    quote=context_text[:240],
                )
            ]
        refs: list[SourceRef] = []
        for chunk in chunks[:2]:
            refs.append(
                SourceRef(
                    paper_id=paper_id,
                    artifact_id=artifact_id,
                    page=chunk.page,
                    section=chunk.section,
                    chunk_id=chunk.id,
                    quote=chunk.text[:240],
                )
            )
        return refs

    def _select_related_paper(self, session_id: str, current_paper_id: str) -> Paper | None:
        session_documents = [
            document
            for document in self._session_repository.list_documents(session_id)
            if document.paper_id != current_paper_id
        ]
        if session_documents:
            related_ids = [document.paper_id for document in reversed(session_documents)]
            papers = self._paper_repository.list_by_ids(related_ids)
            if papers:
                return papers[0]

        global_related_ids = self._global_related_paper_ids(current_paper_id)
        if global_related_ids:
            papers = self._paper_repository.list_by_ids(global_related_ids)
            if papers:
                return papers[0]
        return None

    def _global_related_paper_ids(self, current_paper_id: str) -> list[str]:
        related_ids: list[str] = []
        for memory in self._memory_repository.list_all_paper_memories():
            if memory.paper_id != current_paper_id:
                related_ids.append(memory.paper_id)
        for memory in self._memory_repository.list_all_relation_memories():
            if memory.source_paper != current_paper_id:
                related_ids.append(memory.source_paper)
            if memory.target_paper != current_paper_id:
                related_ids.append(memory.target_paper)
        return list(dict.fromkeys(related_ids))

    def _build_context_text(self, paper: Paper, chunks: list[Chunk]) -> str:
        chunk_text = " ".join(chunk.text for chunk in chunks if chunk.text.strip())
        parts = [paper.title, paper.abstract or "", chunk_text]
        return " ".join(part for part in parts if part).strip()

    def _sentences(self, text: str) -> list[str]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []
        sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_PATTERN.split(normalized) if sentence.strip()]
        return sentences or [normalized]

    def _first_sentence(self, sentences: list[str], keywords: tuple[str, ...]) -> str | None:
        for sentence in sentences:
            lowered = sentence.lower()
            if any(keyword in lowered for keyword in keywords):
                return sentence
        return None

    def _collect_sentences(self, sentences: list[str], keywords: tuple[str, ...], limit: int) -> list[str]:
        collected: list[str] = []
        for sentence in sentences:
            lowered = sentence.lower()
            if any(keyword in lowered for keyword in keywords):
                collected.append(sentence)
            if len(collected) >= limit:
                break
        return collected

    def _infer_relation_type(self, text: str) -> RelationType:
        lowered = text.lower()
        if any(keyword in lowered for keyword in ("contradict", "conflict", "inconsistent", "opposite")):
            return RelationType.CONFLICTS_WITH
        if any(keyword in lowered for keyword in ("compare", "baseline", "bench", "same benchmark")):
            if "same benchmark" in lowered or "benchmark" in lowered:
                return RelationType.USES_SAME_BENCHMARK
            return RelationType.COMPARES_WITH
        if any(keyword in lowered for keyword in ("improve", "better", "outperform", "surpass", "advance")):
            return RelationType.IMPROVES_ON
        if any(keyword in lowered for keyword in ("similar", "variant", "related", "same approach")):
            return RelationType.SIMILAR_TO
        return RelationType.COMPLEMENTS

    def _build_relation_summary(self, source_title: str, target_title: str, relation_type: RelationType) -> str:
        if relation_type is RelationType.IMPROVES_ON:
            return f"{source_title} improves on {target_title}."
        if relation_type is RelationType.SIMILAR_TO:
            return f"{source_title} is similar to {target_title}."
        if relation_type is RelationType.CONFLICTS_WITH:
            return f"{source_title} conflicts with {target_title}."
        if relation_type is RelationType.USES_SAME_BENCHMARK:
            return f"{source_title} uses the same benchmark as {target_title}."
        if relation_type is RelationType.COMPARES_WITH:
            return f"{source_title} compares with {target_title}."
        return f"{source_title} complements {target_title}."

    def _build_followups(self, why_open: list[str], *, prefer_chinese: bool = False) -> list[str]:
        text = " ".join(why_open).lower()
        followups: list[str] = []
        if "robust" in text or "shift" in text:
            followups.append("评估方法在分布偏移下的鲁棒性。" if prefer_chinese else "Evaluate robustness under distribution shift.")
        if "scal" in text or "large" in text:
            followups.append("开展更大规模的实验。" if prefer_chinese else "Run larger-scale experiments.")
        if "ablation" in text:
            followups.append("补充主要组件的消融实验。" if prefer_chinese else "Add ablation studies for the main components.")
        if "future work" in text or "not yet" in text:
            followups.append("后续结果发布后，再回到这篇论文核对结论。" if prefer_chinese else "作者发布后续结果后，再回到这篇论文核对结论。")
        if not followups:
            followups.append("获得更多证据后，重新阅读原文并更新记忆。" if prefer_chinese else "获得更多证据后，重新阅读原文并更新记忆。")
        return followups

    def _limitation_to_question(self, limitation: str, *, prefer_chinese: bool = False) -> str:
        lowered = limitation.lower()
        if "robust" in lowered or "shift" in lowered:
            return "该方法在分布偏移下是否仍然稳定？" if prefer_chinese else "Does the method remain stable under distribution shift?"
        if "scal" in lowered or "large" in lowered:
            return "该方法在更大规模设置下表现如何？" if prefer_chinese else "How does the method behave at larger scale?"
        if "ablation" in lowered:
            return "根据消融结果，哪些组件是必要的？" if prefer_chinese else "Which components are essential according to the ablation story?"
        return f"关于 {limitation.rstrip('.')} 还有哪些结论尚未充分验证？" if prefer_chinese else f"What remains unresolved about {limitation.rstrip('.')}?"

    def _paper_confidence(self, chunks: list[Chunk], key_results: list[str], limitations: list[str]) -> ConfidenceScore:
        value = 0.45
        if chunks:
            value += 0.15
        if key_results:
            value += 0.1
        if limitations:
            value += 0.05
        return ConfidenceScore(value=min(0.9, value))

    def _problem_keywords(self) -> tuple[str, ...]:
        return ("problem", "challenge", "task", "goal", "aim", "we study", "we investigate", "we propose")

    def _method_keywords(self) -> tuple[str, ...]:
        return ("method", "approach", "pipeline", "framework", "model", "algorithm", "we use", "we train", "we fine-tune")

    def _result_keywords(self) -> tuple[str, ...]:
        return ("result", "results", "achieve", "improve", "outperform", "accuracy", "performance", "state-of-the-art", "beats")

    def _limitation_keywords(self) -> tuple[str, ...]:
        return ("limitation", "limitations", "future work", "not yet", "remain", "open question", "cannot", "lack")

    def _novelty_keywords(self) -> tuple[str, ...]:
        return ("novel", "first", "introduce", "new", "we present", "we propose")

    def _relation_keywords(self) -> tuple[str, ...]:
        return ("compare", "baseline", "benchmark", "similar", "contrast", "conflict", "improve", "outperform", "same benchmark")

    def _require_session(self, session_id: str):
        session = self._session_repository.get_by_id(session_id)
        if session is None:
            raise EntityNotFoundError("Session", session_id)
        return session

    def _require_paper(self, paper_id: str) -> Paper:
        paper = self._paper_repository.get_by_id(paper_id)
        if paper is None:
            raise EntityNotFoundError("Paper", paper_id)
        return paper
