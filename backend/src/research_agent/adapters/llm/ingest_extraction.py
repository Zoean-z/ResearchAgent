"""Model-backed adapter boundary for ingest extraction decisions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Literal, Protocol
from urllib import request as urllib_request

from pydantic import BaseModel, Field

from research_agent.runtime.ingest_extraction import (
    IngestExtractionCandidate,
    IngestExtractionDecision,
    IngestExtractionRequest,
    IngestOpenQuestionMemoryDraft,
    IngestPaperMemoryDraft,
    IngestPaperSummaryDraft,
    IngestRelationMemoryDraft,
)
from research_agent.domain.enums import RelationType


class StructuredIngestPaperDraft(BaseModel):
    """Structured paper-memory draft returned by a model transport."""

    problem: str | None = Field(default=None)
    method: str | None = Field(default=None)
    key_results: tuple[str, ...] = Field(default_factory=tuple)
    limitations: tuple[str, ...] = Field(default_factory=tuple)
    novelty_claim: str | None = Field(default=None)
    evidence_candidate_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class StructuredIngestRelationDraft(BaseModel):
    """Structured relation-memory draft returned by a model transport."""

    relation_type: str = Field(description="One of the project relation types")
    summary: str = Field(min_length=1)
    evidence_candidate_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class StructuredIngestOpenQuestionDraft(BaseModel):
    """Structured open-question draft returned by a model transport."""

    unresolved_question: str = Field(min_length=1)
    why_open: tuple[str, ...] = Field(default_factory=tuple)
    possible_followup: tuple[str, ...] = Field(default_factory=tuple)
    evidence_candidate_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class StructuredIngestPaperSummaryDraft(BaseModel):
    """Structured paper summary returned by a model transport."""

    what_it_is_about: str = Field(min_length=1)
    problem_solved: str = Field(min_length=1)
    new_ideas: tuple[str, ...] = Field(default_factory=tuple)
    limitations: tuple[str, ...] = Field(default_factory=tuple)
    suggestions_or_questions: tuple[str, ...] = Field(default_factory=tuple)
    evidence_candidate_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class StructuredIngestExtractionPrompt(BaseModel):
    """Structured prompt payload sent to an ingest-extraction model transport."""

    session_id: str = Field(description="Session currently ingesting the paper")
    paper_title: str = Field(description="Paper title")
    paper_abstract: str | None = Field(default=None, description="Paper abstract if available")
    related_paper_titles: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Titles of related papers already known in the session or globally",
    )
    window_kind: Literal["broad", "expanded"] = Field(description="Current candidate window width")
    context_summary: str = Field(description="Compact broad evidence summary")
    candidate_passages: tuple[dict[str, object], ...] = Field(description="Bounded evidence candidates")


class StructuredIngestExtractionChoice(BaseModel):
    """Structured ingest extraction result returned by a model transport."""

    paper: StructuredIngestPaperDraft
    relation: StructuredIngestRelationDraft | None = None
    open_question: StructuredIngestOpenQuestionDraft
    paper_summary: StructuredIngestPaperSummaryDraft
    needs_more_context: bool = Field(default=False)
    context_hints: tuple[str, ...] = Field(default_factory=tuple)
    rationale: str = Field(default="model_selected_ingest_analysis")


class StructuredIngestExtractionTransport(Protocol):
    """Transport that can obtain a structured ingest-analysis choice from a model."""

    def extract(self, prompt: StructuredIngestExtractionPrompt) -> StructuredIngestExtractionChoice:
        """Return the structured ingest-analysis choice."""


HttpPost = Callable[[str, dict[str, str], bytes, float], bytes]


class UnavailableStructuredIngestExtractionTransport:
    """Default transport placeholder until a provider-specific client is configured."""

    def extract(self, prompt: StructuredIngestExtractionPrompt) -> StructuredIngestExtractionChoice:  # pragma: no cover - defensive default
        raise RuntimeError("No structured ingest extraction transport is configured.")


@dataclass(frozen=True, slots=True)
class StaticStructuredIngestExtractionTransport:
    """Deterministic transport used by tests to simulate model responses."""

    choice: StructuredIngestExtractionChoice

    def extract(self, prompt: StructuredIngestExtractionPrompt) -> StructuredIngestExtractionChoice:
        return self.choice


def _default_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> bytes:
    request = urllib_request.Request(url=url, data=body, headers=headers, method="POST")
    with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


class DeepSeekStructuredIngestExtractionTransport:
    """DeepSeek chat-completions transport for candidate-first ingest extraction."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: float = 30.0,
        http_post: HttpPost | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = self._normalize_model_name(model)
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._http_post = http_post or _default_http_post

    @property
    def normalized_model(self) -> str:
        return self._model

    def extract(self, prompt: StructuredIngestExtractionPrompt) -> StructuredIngestExtractionChoice:
        if not self._api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured.")

        payload = {
            "model": self._model,
            "messages": self._messages_for(prompt),
            "response_format": {"type": "json_object"},
            "max_tokens": 1024,
            "temperature": 0.0,
            "stream": False,
        }
        raw_response = self._http_post(
            f"{self._base_url}/chat/completions",
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json.dumps(payload).encode("utf-8"),
            self._timeout_seconds,
        )
        return self._parse_response(raw_response)

    def _messages_for(self, prompt: StructuredIngestExtractionPrompt) -> list[dict[str, str]]:
        system_prompt = (
            "你是 memory-routed paper agent 的 ingest 抽取器。"
            "只输出 JSON。"
            "优先使用 broad candidate window；只有在证据明显不足时才请求更多上下文。"
            "优先采用标题、摘要、引言、结论和局限性等正文证据，附录、表格、图注和参考文献密集片段只能作为次要证据。"
            "忽略表格行碎片、图注、页码和行号噪音，除非没有任何正文证据。"
            "把能支持的结论改写成简洁自然的中文，不要直接复制原始表格行或断行碎片。"
            "同时生成简短的 paper_summary，说明论文讲什么、解决什么问题、提出什么新想法、还剩什么局限，以及后续能问什么。"
            "paper、relation、open_question、paper_summary、rationale、context_hints 里所有人类可见字段都默认写中文；"
            "只有在术语或标题无法翻译时才保留英文原文。"
            "返回的 JSON 必须包含 paper、relation、open_question、paper_summary、needs_more_context、context_hints 和 rationale。"
        )
        user_prompt = json.dumps(
            {
                "session_id": prompt.session_id,
                "paper_title": prompt.paper_title,
                "paper_abstract": prompt.paper_abstract,
                "related_paper_titles": prompt.related_paper_titles,
                "window_kind": prompt.window_kind,
                "context_summary": prompt.context_summary,
                "candidate_passages": prompt.candidate_passages,
                "instructions": (
                    "把论文抽成 paper、relation 和 open_question 三类 memory 草稿。"
                    "同时输出一段简短、可直接给用户看的 paper summary。"
                    "paper_summary 尽量基于标题、摘要和正文主线；只有在附录或表格明显更好时才用它们。"
                    "优先改写成中文短句，不要复述表格行、数字表格、页码或行号碎片。"
                    "如果表格是唯一证据，也要改写成简短中文句子。"
                    "paper_summary.what_it_is_about、problem_solved、new_ideas、limitations、suggestions_or_questions 以及生成的 memory 文本都必须是中文。"
                    "如果候选窗口不够，设置 needs_more_context=true，并给出简洁中文上下文提示。"
                    "否则就返回当前候选里能支持的最佳结构化结果。"
                ),
            },
            ensure_ascii=False,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_response(self, raw_response: bytes) -> StructuredIngestExtractionChoice:
        payload = json.loads(raw_response.decode("utf-8"))
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("DeepSeek ingest-extraction response contained no choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError("DeepSeek ingest-extraction response contained empty content.")
        return StructuredIngestExtractionChoice.model_validate(json.loads(content))

    def _normalize_model_name(self, model: str) -> str:
        normalized = model.strip()
        aliases = {
            "deepseekv4flash": "deepseek-v4-flash",
            "deepseek_v4_flash": "deepseek-v4-flash",
            "deepseek-v4flash": "deepseek-v4-flash",
            "deepseekv4pro": "deepseek-v4-pro",
            "deepseek_v4_pro": "deepseek-v4-pro",
            "deepseek-v4pro": "deepseek-v4-pro",
        }
        return aliases.get(normalized.lower(), normalized)


class ModelBackedIngestExtractionClient:
    """Ingest extractor that delegates analysis to a model adapter."""

    def __init__(
        self,
        *,
        transport: StructuredIngestExtractionTransport,
        client_name: str = "model_adapter",
    ) -> None:
        self._transport = transport
        self._client_name = client_name

    @property
    def client_name(self) -> str:
        return getattr(self, "_last_client_name", self._client_name)

    def extract(self, request: IngestExtractionRequest) -> IngestExtractionDecision:
        prompt = self._build_prompt(request)
        choice = self._transport.extract(prompt)
        decision = self._decision_from_choice(choice, request)
        self._last_client_name = self._client_name
        return decision

    def _build_prompt(self, request: IngestExtractionRequest) -> StructuredIngestExtractionPrompt:
        return StructuredIngestExtractionPrompt(
            session_id=request.session_id,
            paper_title=request.paper.title,
            paper_abstract=request.paper.abstract,
            related_paper_titles=tuple(paper.title for paper in request.related_papers),
            window_kind=request.window.kind,
            context_summary=request.window.context_summary,
            candidate_passages=tuple(
                {
                    "candidate_id": candidate.candidate_id,
                    "chunk_id": candidate.chunk_id,
                    "page": candidate.page,
                    "section": candidate.section,
                    "content_role": candidate.content_role,
                    "excerpt": candidate.excerpt,
                    "relevance_reason": candidate.relevance_reason,
                }
                for candidate in request.window.candidate_passages
            ),
        )

    def _decision_from_choice(
        self,
        choice: StructuredIngestExtractionChoice,
        request: IngestExtractionRequest,
    ) -> IngestExtractionDecision:
        return IngestExtractionDecision(
            paper=IngestPaperMemoryDraft(
                problem=choice.paper.problem,
                method=choice.paper.method,
                key_results=choice.paper.key_results,
                limitations=choice.paper.limitations,
                novelty_claim=choice.paper.novelty_claim,
                evidence_candidate_ids=choice.paper.evidence_candidate_ids,
                confidence=choice.paper.confidence,
            ),
            relation=self._relation_from_choice(choice.relation),
            open_question=IngestOpenQuestionMemoryDraft(
                unresolved_question=choice.open_question.unresolved_question,
                why_open=choice.open_question.why_open,
                possible_followup=choice.open_question.possible_followup,
                evidence_candidate_ids=choice.open_question.evidence_candidate_ids,
                confidence=choice.open_question.confidence,
            ),
            paper_summary=IngestPaperSummaryDraft(
                what_it_is_about=choice.paper_summary.what_it_is_about,
                problem_solved=choice.paper_summary.problem_solved,
                new_ideas=choice.paper_summary.new_ideas,
                limitations=choice.paper_summary.limitations,
                suggestions_or_questions=choice.paper_summary.suggestions_or_questions,
                evidence_candidate_ids=choice.paper_summary.evidence_candidate_ids,
                confidence=choice.paper_summary.confidence,
            ),
            needs_more_context=choice.needs_more_context,
            context_hints=choice.context_hints,
            rationale=choice.rationale,
        )

    def _relation_from_choice(
        self,
        relation: StructuredIngestRelationDraft | None,
    ) -> IngestRelationMemoryDraft | None:
        if relation is None:
            return None
        RelationType(relation.relation_type)
        return IngestRelationMemoryDraft(
            relation_type=relation.relation_type,
            summary=relation.summary,
            evidence_candidate_ids=relation.evidence_candidate_ids,
            confidence=relation.confidence,
        )


__all__ = [
    "DeepSeekStructuredIngestExtractionTransport",
    "ModelBackedIngestExtractionClient",
    "StaticStructuredIngestExtractionTransport",
    "StructuredIngestExtractionChoice",
    "StructuredIngestExtractionPrompt",
    "StructuredIngestExtractionTransport",
    "StructuredIngestPaperSummaryDraft",
    "UnavailableStructuredIngestExtractionTransport",
]
