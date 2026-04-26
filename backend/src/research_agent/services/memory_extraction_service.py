"""Thin memory extraction service for parsed ingest material."""

from __future__ import annotations

from dataclasses import dataclass
import re

from research_agent.domain.enums import RelationType
from research_agent.domain.models import Chunk, OpenQuestionMemory, Paper, PaperMemory, RelationMemory, SourceRef
from research_agent.domain.ports import ChunkRepositoryPort, MemoryRepositoryPort, PaperRepositoryPort, SessionRepositoryPort
from research_agent.domain.policies import merge_open_question_memory, merge_paper_memory, merge_relation_memory
from research_agent.domain.value_objects import ConfidenceScore
from research_agent.adapters.openviking import NoopOpenVikingMemoryGateway, OpenVikingMemoryGateway
from research_agent.runtime.ingest_extraction import IngestPaperSummaryDraft
from research_agent.services.ingest_analysis_service import IngestAnalysisService, MemoryAnalysisResult
from research_agent.services.errors import EntityNotFoundError


_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class MemoryExtractionResult:
    """Structured memory records created from a parsed source."""

    paper_memory: PaperMemory
    paper_operation: str
    relation_memory: RelationMemory | None
    relation_operation: str | None
    open_question_memory: OpenQuestionMemory
    open_question_operation: str
    paper_summary: IngestPaperSummaryDraft
    context_summary: str


class MemoryExtractionService:
    """Create the first structured memories from parsed document content."""

    def __init__(
        self,
        session_repository: SessionRepositoryPort,
        paper_repository: PaperRepositoryPort,
        chunk_repository: ChunkRepositoryPort,
        memory_repository: MemoryRepositoryPort,
        analysis_service: IngestAnalysisService | None = None,
        openviking_gateway: OpenVikingMemoryGateway | None = None,
    ) -> None:
        self._session_repository = session_repository
        self._paper_repository = paper_repository
        self._chunk_repository = chunk_repository
        self._memory_repository = memory_repository
        self._analysis_service = analysis_service or IngestAnalysisService(
            session_repository=session_repository,
            paper_repository=paper_repository,
            chunk_repository=chunk_repository,
            memory_repository=memory_repository,
        )
        self._openviking_gateway = openviking_gateway or NoopOpenVikingMemoryGateway()

    def extract_and_store_memories(self, session_id: str, paper_id: str) -> MemoryExtractionResult:
        """Generate and persist paper, relation, and open-question memories."""

        paper = self._require_paper(paper_id)
        analysis = self._analysis_service.analyze(session_id=session_id, paper_id=paper_id)
        paper_memory_incoming = analysis.paper_memory
        paper_existing = self._latest_paper_memory(paper_id)
        paper_decision = merge_paper_memory(paper_existing, paper_memory_incoming)
        paper_memory = self._memory_repository.upsert_paper_memory(paper_decision.record)

        relation_memory: RelationMemory | None = None
        relation_operation: str | None = None
        related_paper = self._select_related_paper(session_id=session_id, current_paper_id=paper_id)
        if analysis.relation_memory is not None:
            relation_incoming = analysis.relation_memory
            relation_existing = self._latest_relation_memory(paper.id, related_paper.id) if related_paper is not None else None
            relation_decision = merge_relation_memory(relation_existing, relation_incoming)
            relation_memory = self._memory_repository.upsert_relation_memory(relation_decision.record)
            relation_operation = relation_decision.operation

        open_question_incoming = analysis.open_question_memory
        open_question_existing = self._latest_open_question_memory(paper_id)
        open_question_decision = merge_open_question_memory(open_question_existing, open_question_incoming)
        open_question_memory = self._memory_repository.upsert_open_question_memory(open_question_decision.record)
        self._openviking_gateway.mirror_ingest_result(
            session_id=session_id,
            paper=paper,
            analysis=analysis,
        )

        return MemoryExtractionResult(
            paper_memory=paper_memory,
            paper_operation=paper_decision.operation,
            relation_memory=relation_memory,
            relation_operation=relation_operation,
            open_question_memory=open_question_memory,
            open_question_operation=open_question_decision.operation,
            paper_summary=analysis.paper_summary,
            context_summary=analysis.context_summary,
        )

    def _build_paper_memory(
        self,
        paper: Paper,
        artifact_id: str,
        chunks: list[Chunk],
        context_text: str,
    ) -> PaperMemory:
        sentences = self._sentences(context_text)
        source_refs = self._build_source_refs(paper.id, artifact_id, chunks, context_text)
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
            evidence = [f"导入时已将来源内容与《{related_paper.title}》进行关系比较。"]
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
        unresolved_question = self._limitation_to_question(limitation) if limitation else f"《{paper.title}》中还有哪些结论尚未充分验证？"
        why_open = self._collect_sentences(sentences, self._limitation_keywords(), limit=3)
        if not why_open:
            why_open = [f"已解析 {len(chunks)} 个文本分块，但未抽取到明确局限性。"]
        possible_followup = self._build_followups(why_open)
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

    def _latest_paper_memory(self, paper_id: str) -> PaperMemory | None:
        memories = list(self._memory_repository.list_paper_memories_for_papers([paper_id]))
        if not memories:
            return None
        return max(memories, key=lambda memory: memory.updated_at)

    def _latest_relation_memory(self, source_paper: str, target_paper: str) -> RelationMemory | None:
        memories = [
            memory
            for memory in self._memory_repository.list_all_relation_memories()
            if memory.source_paper == source_paper and memory.target_paper == target_paper
        ]
        if not memories:
            return None
        return max(memories, key=lambda memory: memory.updated_at)

    def _latest_open_question_memory(self, paper_id: str) -> OpenQuestionMemory | None:
        memories = [
            memory
            for memory in self._memory_repository.list_all_open_question_memories()
            if paper_id in memory.related_papers
        ]
        if not memories:
            return None
        return max(memories, key=lambda memory: memory.updated_at)

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

    def _build_followups(self, why_open: list[str]) -> list[str]:
        text = " ".join(why_open).lower()
        followups: list[str] = []
        if "robust" in text or "shift" in text:
            followups.append("Evaluate robustness under distribution shift.")
        if "scal" in text or "large" in text:
            followups.append("Run larger-scale experiments.")
        if "ablation" in text:
            followups.append("Add ablation studies for the main components.")
        if "future work" in text or "not yet" in text:
            followups.append("作者发布后续结果后，再回到这篇论文核对结论。")
        if not followups:
            followups.append("获得更多证据后，重新阅读原文并更新记忆。")
        return followups

    def _limitation_to_question(self, limitation: str) -> str:
        lowered = limitation.lower()
        if "robust" in lowered or "shift" in lowered:
            return "Does the method remain stable under distribution shift?"
        if "scal" in lowered or "large" in lowered:
            return "How does the method behave at larger scale?"
        if "ablation" in lowered:
            return "Which components are essential according to the ablation story?"
        return f"What remains unresolved about {limitation.rstrip('.')}?"

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
