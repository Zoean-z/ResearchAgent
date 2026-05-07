"""Model-backed adapter boundary for ingest extraction decisions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Literal, Protocol
from urllib import request as urllib_request

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from research_agent.utils import resolve_api_key

from research_agent.runtime.ingest_extraction import (
    IngestEvidenceFieldDraft,
    IngestExtractionCandidate,
    IngestExtractionDecision,
    IngestExtractionRequest,
    IngestOpenQuestionMemoryDraft,
    IngestPaperMemoryDraft,
    IngestPaperSummaryDraft,
    IngestRelationMemoryDraft,
    IngestUnderstandingDraft,
)


class StructuredIngestPaperDraft(BaseModel):
    """Structured paper-memory draft returned by a legacy model transport."""

    model_config = ConfigDict(extra="ignore")

    problem: str | None = Field(default=None)
    method: str | None = Field(default=None)
    key_results: tuple[str, ...] = Field(default_factory=tuple)
    limitations: tuple[str, ...] = Field(default_factory=tuple)
    novelty_claim: str | None = Field(default=None)
    evidence_candidate_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, value: object) -> object:
        return _normalize_textual_draft_payload(
            value,
            text_fields=("problem", "method", "novelty_claim"),
            collection_fields=("key_results", "limitations"),
            evidence_fields=("evidence_candidate_ids",),
        )


class StructuredIngestRelationDraft(BaseModel):
    """Structured relation-memory draft returned by a legacy model transport."""

    model_config = ConfigDict(extra="ignore")

    relation_type: str = Field(description="One of the project relation types")
    summary: str = Field(min_length=1)
    evidence_candidate_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, value: object) -> object:
        return _normalize_textual_draft_payload(
            value,
            text_fields=("relation_type", "summary"),
            collection_fields=(),
            evidence_fields=("evidence_candidate_ids",),
        )


class StructuredIngestOpenQuestionDraft(BaseModel):
    """Structured open-question draft returned by a legacy model transport."""

    model_config = ConfigDict(extra="ignore")

    unresolved_question: str = Field(min_length=1)
    why_open: tuple[str, ...] = Field(default_factory=tuple)
    possible_followup: tuple[str, ...] = Field(default_factory=tuple)
    evidence_candidate_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, value: object) -> object:
        return _normalize_textual_draft_payload(
            value,
            text_fields=("unresolved_question",),
            collection_fields=("why_open", "possible_followup"),
            evidence_fields=("evidence_candidate_ids",),
        )


class StructuredIngestPaperSummaryDraft(BaseModel):
    """Structured paper summary returned by a legacy model transport."""

    model_config = ConfigDict(extra="ignore")

    what_it_is_about: str = Field(min_length=1)
    problem_solved: str = Field(min_length=1)
    new_ideas: tuple[str, ...] = Field(default_factory=tuple)
    limitations: tuple[str, ...] = Field(default_factory=tuple)
    suggestions_or_questions: tuple[str, ...] = Field(default_factory=tuple)
    evidence_candidate_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, value: object) -> object:
        return _normalize_textual_draft_payload(
            value,
            text_fields=("what_it_is_about", "problem_solved"),
            collection_fields=("new_ideas", "limitations", "suggestions_or_questions"),
            evidence_fields=("evidence_candidate_ids",),
        )


class StructuredIngestEvidenceFieldDraft(BaseModel):
    """A single evidence-bound field returned by a model transport."""

    model_config = ConfigDict(extra="ignore")

    text: str | None = Field(default=None)
    evidence_chunk_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_status: Literal["strong", "weak"] = Field(default="strong")

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, value: object) -> object:
        return _normalize_evidence_field(value)


class StructuredIngestUnderstandingDraft(BaseModel):
    """Structured model output for paper understanding."""

    model_config = ConfigDict(extra="ignore")

    topic: StructuredIngestEvidenceFieldDraft | None = Field(default=None)
    problem: StructuredIngestEvidenceFieldDraft | None = Field(default=None)
    method: StructuredIngestEvidenceFieldDraft | None = Field(default=None)
    novelty_claims: tuple[StructuredIngestEvidenceFieldDraft, ...] = Field(default_factory=tuple)
    key_results: tuple[StructuredIngestEvidenceFieldDraft, ...] = Field(default_factory=tuple)
    experiment_design: StructuredIngestEvidenceFieldDraft | None = Field(default=None)
    limitations: tuple[StructuredIngestEvidenceFieldDraft, ...] = Field(default_factory=tuple)
    open_questions: tuple[StructuredIngestEvidenceFieldDraft, ...] = Field(default_factory=tuple)
    evidence_chunk_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for field_name in ("topic", "problem", "method", "experiment_design"):
            normalized[field_name] = _normalize_evidence_field(normalized.get(field_name))
        for field_name in ("novelty_claims", "key_results", "limitations", "open_questions"):
            normalized[field_name] = _normalize_evidence_field_collection(normalized.get(field_name))
        normalized["evidence_chunk_ids"] = _normalize_evidence_ids(normalized.get("evidence_chunk_ids"))
        return normalized


class StructuredIngestExtractionPrompt(BaseModel):
    """Structured prompt payload sent to an ingest-extraction model transport."""

    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(description="Session currently ingesting the paper")
    paper_title: str = Field(description="Paper title")
    paper_abstract: str | None = Field(default=None, description="Paper abstract if available")
    related_paper_titles: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Titles of related papers already known in the session or globally",
    )
    window_kind: Literal["broad", "expanded"] = Field(description="Current candidate window width")
    extraction_stage: Literal["full_text", "batch", "merge"] = Field(description="Model extraction stage")
    batch_index: int | None = Field(default=None, description="Current batch index for hierarchical extraction")
    batch_count: int | None = Field(default=None, description="Total hierarchical batches")
    batch_label: str | None = Field(default=None, description="Batch label if hierarchical extraction is active")
    context_summary: str = Field(description="Compact broad evidence summary")
    candidate_passages: tuple[dict[str, object], ...] = Field(description="Bounded evidence candidates")
    batch_summaries: tuple[dict[str, object], ...] = Field(default_factory=tuple, description="Previous batch extraction summaries when merging")


class StructuredIngestExtractionChoice(BaseModel):
    """Structured ingest extraction result returned by a model transport."""

    model_config = ConfigDict(extra="ignore")

    understanding: StructuredIngestUnderstandingDraft | None = Field(default=None)
    paper: StructuredIngestPaperDraft | None = None
    relation: StructuredIngestRelationDraft | None = None
    open_question: StructuredIngestOpenQuestionDraft | None = None
    paper_summary: StructuredIngestPaperSummaryDraft | None = None
    needs_more_context: bool = Field(default=False)
    context_hints: tuple[str, ...] = Field(default_factory=tuple)
    rationale: str = Field(default="model_selected_ingest_analysis")

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "understanding" in normalized:
            normalized["understanding"] = _normalize_response_payload(normalized.get("understanding"))
        if "paper" in normalized:
            normalized["paper"] = _normalize_textual_draft_payload(
                normalized.get("paper"),
                text_fields=("problem", "method", "novelty_claim"),
                collection_fields=("key_results", "limitations"),
                evidence_fields=("evidence_candidate_ids",),
            )
        if "relation" in normalized:
            normalized["relation"] = _normalize_textual_draft_payload(
                normalized.get("relation"),
                text_fields=("relation_type", "summary"),
                collection_fields=(),
                evidence_fields=("evidence_candidate_ids",),
            )
        if "open_question" in normalized:
            normalized["open_question"] = _normalize_textual_draft_payload(
                normalized.get("open_question"),
                text_fields=("unresolved_question",),
                collection_fields=("why_open", "possible_followup"),
                evidence_fields=("evidence_candidate_ids",),
            )
        if "paper_summary" in normalized:
            normalized["paper_summary"] = _normalize_textual_draft_payload(
                normalized.get("paper_summary"),
                text_fields=("what_it_is_about", "problem_solved"),
                collection_fields=("new_ideas", "limitations", "suggestions_or_questions"),
                evidence_fields=("evidence_candidate_ids",),
            )
        normalized["context_hints"] = _normalize_string_collection(normalized.get("context_hints"))
        return normalized


class StructuredIngestExtractionTransport(Protocol):
    """Transport that can obtain a structured ingest-analysis choice from a model."""

    def extract(self, prompt: StructuredIngestExtractionPrompt) -> StructuredIngestExtractionChoice:
        """Return the structured ingest-analysis choice."""


HttpPost = Callable[[str, dict[str, str], bytes, float], bytes]
ApiKeyProvider = Callable[[], str | None]


@dataclass(frozen=True, slots=True)
class StructuredIngestExtractionParseError(RuntimeError):
    """Raised when the ingest extraction payload cannot be parsed or validated."""

    extractor_stage: Literal["response_parse", "content_parse", "schema_validation", "extractor_failed"]
    raw_response_preview: str | None = None
    normalized_payload_preview: str | None = None
    validation_error: str | None = None
    failed_field: str | None = None


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


def _normalize_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
        return candidate or None
    if isinstance(value, dict):
        for key in ("text", "value", "summary", "content"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                candidate = candidate.strip()
                if candidate:
                    return candidate
        return None
    text = getattr(value, "text", None)
    if isinstance(text, str):
        candidate = text.strip()
        if candidate:
            return candidate
    value_text = getattr(value, "value", None)
    if isinstance(value_text, str):
        candidate = value_text.strip()
        if candidate:
            return candidate
    return None


def _normalize_string_collection(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    items = value if isinstance(value, (list, tuple, set)) else (value,)
    normalized: list[str] = []
    for item in items:
        text = _normalize_text(item)
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)


def _normalize_evidence_ids(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        candidate = value.strip()
        return (candidate,) if candidate else ()
    if isinstance(value, dict):
        source = value.get("evidence_chunk_ids")
        if source is None:
            source = value.get("evidence_candidate_ids")
        if source is None:
            source = value.get("chunk_ids")
        return _normalize_evidence_ids(source)
    if isinstance(value, (list, tuple, set)):
        ids: list[str] = []
        for item in value:
            if isinstance(item, str):
                candidate = item.strip()
                if candidate and candidate not in ids:
                    ids.append(candidate)
        return tuple(ids)
    source = getattr(value, "evidence_chunk_ids", None)
    if source is None:
        source = getattr(value, "evidence_candidate_ids", None)
    if source is None:
        source = getattr(value, "chunk_ids", None)
    return _normalize_evidence_ids(source)


def _normalize_evidence_field(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return {
            "text": text,
            "evidence_chunk_ids": (),
            "confidence": 0.5,
            "evidence_status": "weak",
        }

    if isinstance(value, dict):
        text = _normalize_text(value)
        if not text:
            return None
        evidence_chunk_ids = _normalize_evidence_ids(value)
        confidence = value.get("confidence", 0.5)
        evidence_status = "weak" if not evidence_chunk_ids else "strong"
        if evidence_chunk_ids and "evidence_status" in value and isinstance(value["evidence_status"], str):
            candidate_status = value["evidence_status"].strip().lower()
            if candidate_status in {"strong", "weak"}:
                evidence_status = candidate_status  # pragma: no branch - direct normalization
        return {
            "text": text,
            "evidence_chunk_ids": evidence_chunk_ids,
            "confidence": confidence,
            "evidence_status": evidence_status,
        }

    text = _normalize_text(value)
    if text is None:
        return None
    evidence_chunk_ids = _normalize_evidence_ids(value)
    confidence = getattr(value, "confidence", 0.5)
    evidence_status = "weak" if not evidence_chunk_ids else "strong"
    candidate_status = getattr(value, "evidence_status", None)
    if evidence_chunk_ids and isinstance(candidate_status, str):
        normalized_status = candidate_status.strip().lower()
        if normalized_status in {"strong", "weak"}:
            evidence_status = normalized_status
    return {
        "text": text,
        "evidence_chunk_ids": evidence_chunk_ids,
        "confidence": confidence,
        "evidence_status": evidence_status,
    }


def _normalize_evidence_field_collection(value: object) -> tuple[dict[str, object], ...]:
    if value is None:
        return ()
    items = value if isinstance(value, (list, tuple, set)) else (value,)
    normalized: list[dict[str, object]] = []
    for item in items:
        field = _normalize_evidence_field(item)
        if field is not None:
            normalized.append(field)
    return tuple(normalized)


def _normalize_response_payload(value: object) -> object:
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    for field_name in ("topic", "problem", "method", "experiment_design"):
        normalized[field_name] = _normalize_evidence_field(normalized.get(field_name))
    for field_name in ("novelty_claims", "key_results", "limitations", "open_questions"):
        normalized[field_name] = _normalize_evidence_field_collection(normalized.get(field_name))
    normalized["evidence_chunk_ids"] = _normalize_evidence_ids(normalized.get("evidence_chunk_ids"))
    return normalized


def _failed_field_from_validation_error(error: ValidationError) -> str | None:
    errors = error.errors()
    if not errors:
        return None
    location = errors[0].get("loc")
    if not isinstance(location, tuple):
        return None
    parts = [str(part) for part in location if part not in {None, ""}]
    return ".".join(parts) if parts else None


def _normalize_textual_draft_payload(value: object, *, text_fields: tuple[str, ...], collection_fields: tuple[str, ...], evidence_fields: tuple[str, ...]) -> object:
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    for field_name in text_fields:
        normalized[field_name] = _normalize_text(normalized.get(field_name))
    for field_name in collection_fields:
        normalized[field_name] = _normalize_string_collection(normalized.get(field_name))
    for field_name in evidence_fields:
        normalized[field_name] = _normalize_evidence_ids(normalized.get(field_name))
    return normalized


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
        api_key_provider: ApiKeyProvider | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = self._normalize_model_name(model)
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._http_post = http_post or _default_http_post
        self._api_key_provider = api_key_provider or (lambda: resolve_api_key(self._api_key))

    @property
    def normalized_model(self) -> str:
        return self._model

    def extract(self, prompt: StructuredIngestExtractionPrompt) -> StructuredIngestExtractionChoice:
        api_key = self._api_key_provider()
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured.")

        payload = {
            "model": self._model,
            "messages": self._messages_for(prompt),
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "max_tokens": 1536,
            "temperature": 0.0,
            "stream": False,
        }
        raw_response = self._http_post(
            f"{self._base_url}/chat/completions",
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            self._timeout_seconds,
        )
        return self._parse_response(raw_response)

    def _messages_for(self, prompt: StructuredIngestExtractionPrompt) -> list[dict[str, str]]:
        if prompt.extraction_stage == "merge":
            system_prompt = """You are the ingest extractor for the memory-routed paper agent.
Return JSON only.
This is the merge stage: combine several batch-level understandings into one paper-level understanding.
Do not invent paper-level claims that are not supported by the batch summaries or candidate passages.
Every non-empty field must carry evidence_chunk_ids. If a field cannot be supported, return null or an empty list for that field.
The model output must include understanding, needs_more_context, context_hints, and rationale.
Use Chinese for user-visible fields unless the original technical term is best preserved in English.
The understanding object must contain: topic, problem, method, novelty_claims, key_results, experiment_design, limitations, open_questions.
Preferred shape:
{
  "understanding": {
    "topic": {"text": "...", "evidence_chunk_ids": []},
    "problem": {"text": "...", "evidence_chunk_ids": []},
    "method": {"text": "...", "evidence_chunk_ids": []},
    "novelty_claims": [{"text": "...", "evidence_chunk_ids": []}],
    "key_results": [{"text": "...", "evidence_chunk_ids": []}],
    "experiment_design": {"text": "...", "evidence_chunk_ids": []},
    "limitations": [{"text": "...", "evidence_chunk_ids": []}],
    "open_questions": [{"text": "...", "evidence_chunk_ids": []}]
  }
}
Use text for field objects, and keep list fields as arrays even if they contain only one item.
Do not output markdown or prose outside JSON."""
        else:
            system_prompt = """You are the ingest extractor for the memory-routed paper agent.
Return JSON only.
This request already contains cleaned evidence chunks, plus their chunk ids, page, section, and content role.
Understand the paper from the evidence itself; do not use keyword rules or template fallbacks.
For every non-empty field, attach evidence_chunk_ids. If the evidence is insufficient, return null or an empty list for that field.
The model output must include understanding, needs_more_context, context_hints, and rationale.
Use Chinese for user-visible fields unless the original technical term is best preserved in English.
The understanding object must contain: topic, problem, method, novelty_claims, key_results, experiment_design, limitations, open_questions.
If extraction_stage=batch, focus on the current batch only. If extraction_stage=full_text, synthesize the whole paper.
Preferred shape:
{
  "understanding": {
    "topic": {"text": "...", "evidence_chunk_ids": []},
    "problem": {"text": "...", "evidence_chunk_ids": []},
    "method": {"text": "...", "evidence_chunk_ids": []},
    "novelty_claims": [{"text": "...", "evidence_chunk_ids": []}],
    "key_results": [{"text": "...", "evidence_chunk_ids": []}],
    "experiment_design": {"text": "...", "evidence_chunk_ids": []},
    "limitations": [{"text": "...", "evidence_chunk_ids": []}],
    "open_questions": [{"text": "...", "evidence_chunk_ids": []}]
  }
}
Use text for field objects, and keep list fields as arrays even if they contain only one item.
Do not output markdown or prose outside JSON."""
        user_prompt = json.dumps(
            {
                "session_id": prompt.session_id,
                "paper_title": prompt.paper_title,
                "paper_abstract": prompt.paper_abstract,
                "related_paper_titles": prompt.related_paper_titles,
                "window_kind": prompt.window_kind,
                "extraction_stage": prompt.extraction_stage,
                "batch_index": prompt.batch_index,
                "batch_count": prompt.batch_count,
                "batch_label": prompt.batch_label,
                "context_summary": prompt.context_summary,
                "candidate_passages": prompt.candidate_passages,
                "batch_summaries": prompt.batch_summaries,
                "instructions": """Extract an evidence-bound paper understanding.
Output only a JSON object.
For each field, keep the exact supporting chunk ids in evidence_chunk_ids.
If the field is not supported by the evidence, return null for single-value fields or an empty tuple/list for multi-value fields.
Do not generate relation memory here. Do not produce user-facing prose outside JSON.
For merge stage, merge only what is supported by the batch summaries and candidate passages.""",
            },
            ensure_ascii=False,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_response(self, raw_response: bytes) -> StructuredIngestExtractionChoice:
        raw_response_text = raw_response.decode("utf-8", errors="replace")
        raw_response_preview = raw_response_text[:4000]
        try:
            payload = json.loads(raw_response_text)
        except json.JSONDecodeError as exc:
            raise StructuredIngestExtractionParseError(
                extractor_stage="response_parse",
                raw_response_preview=raw_response_preview,
                validation_error=str(exc),
            ) from exc

        choices = payload.get("choices") or []
        if not choices:
            raise StructuredIngestExtractionParseError(
                extractor_stage="response_parse",
                raw_response_preview=raw_response_preview,
                validation_error="DeepSeek ingest-extraction response contained no choices.",
            )
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise StructuredIngestExtractionParseError(
                extractor_stage="response_parse",
                raw_response_preview=raw_response_preview,
                validation_error="DeepSeek ingest-extraction response contained empty content.",
            )

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise StructuredIngestExtractionParseError(
                extractor_stage="content_parse",
                raw_response_preview=raw_response_preview,
                validation_error=str(exc),
            ) from exc

        normalized = _normalize_response_payload(parsed)
        normalized_preview = json.dumps(normalized, ensure_ascii=False, default=str)[:4000]
        try:
            return StructuredIngestExtractionChoice.model_validate(normalized)
        except ValidationError as exc:
            raise StructuredIngestExtractionParseError(
                extractor_stage="schema_validation",
                raw_response_preview=raw_response_preview,
                normalized_payload_preview=normalized_preview,
                validation_error=str(exc),
                failed_field=_failed_field_from_validation_error(exc),
            ) from exc

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
        decision = self._decision_from_choice(choice)
        self._last_client_name = self._client_name
        return decision

    def _build_prompt(self, request: IngestExtractionRequest) -> StructuredIngestExtractionPrompt:
        return StructuredIngestExtractionPrompt(
            session_id=request.session_id,
            paper_title=request.paper.title,
            paper_abstract=request.paper.abstract,
            related_paper_titles=tuple(paper.title for paper in request.related_papers),
            window_kind=request.window.kind,
            extraction_stage=request.extraction_stage,
            batch_index=request.batch_index,
            batch_count=request.batch_count,
            batch_label=request.batch_label,
            context_summary=request.window.context_summary,
            candidate_passages=tuple(
                {
                    "candidate_id": candidate.candidate_id,
                    "chunk_id": candidate.chunk_id,
                    "source_chunk_ids": candidate.source_chunk_ids,
                    "page": candidate.page,
                    "section": candidate.section,
                    "content_role": candidate.content_role,
                    "cleaned_text": candidate.cleaned_text,
                    "excerpt": candidate.excerpt,
                    "relevance_reason": candidate.relevance_reason,
                    "quality_flags": candidate.quality_flags,
                    "removed_reason": candidate.removed_reason,
                }
                for candidate in request.window.candidate_passages
            ),
        )

    def _decision_from_choice(self, choice: StructuredIngestExtractionChoice) -> IngestExtractionDecision:
        understanding = self._understanding_from_choice(choice)
        paper = self._paper_from_understanding(understanding, choice)
        open_question = self._open_question_from_understanding(understanding, choice)
        paper_summary = self._summary_from_understanding(understanding, choice)
        return IngestExtractionDecision(
            understanding=understanding,
            paper=paper,
            relation=None,
            open_question=open_question,
            paper_summary=paper_summary,
            needs_more_context=choice.needs_more_context,
            context_hints=choice.context_hints,
            rationale=choice.rationale,
        )

    def _understanding_from_choice(self, choice: StructuredIngestExtractionChoice) -> IngestUnderstandingDraft:
        if choice.understanding is not None:
            understanding = choice.understanding
            return IngestUnderstandingDraft(
                topic=self._field_draft(understanding.topic),
                problem=self._field_draft(understanding.problem),
                method=self._field_draft(understanding.method),
                novelty_claims=tuple(field for field in (self._field_draft(item) for item in understanding.novelty_claims) if field is not None),
                key_results=tuple(field for field in (self._field_draft(item) for item in understanding.key_results) if field is not None),
                experiment_design=self._field_draft(understanding.experiment_design),
                limitations=tuple(field for field in (self._field_draft(item) for item in understanding.limitations) if field is not None),
                open_questions=tuple(field for field in (self._field_draft(item) for item in understanding.open_questions) if field is not None),
                evidence_chunk_ids=understanding.evidence_chunk_ids,
                confidence=understanding.confidence,
            )

        return IngestUnderstandingDraft(
            topic=self._field_draft(self._legacy_text(choice.paper_summary.what_it_is_about if choice.paper_summary else None) if choice.paper_summary else self._legacy_text(choice.paper.problem if choice.paper else None)),
            problem=self._field_draft(self._legacy_text(choice.paper_summary.problem_solved if choice.paper_summary else None) if choice.paper_summary else self._legacy_text(choice.paper.problem if choice.paper else None)),
            method=self._field_draft(self._legacy_text(choice.paper.method if choice.paper else None)),
            novelty_claims=tuple(
                field
                for field in (
                    self._field_draft(self._legacy_text(item))
                    for item in ((choice.paper_summary.new_ideas if choice.paper_summary else ()) or (choice.paper.novelty_claim,) if choice.paper and choice.paper.novelty_claim else ())
                )
                if field is not None
            ),
            key_results=tuple(
                field
                for field in (
                    self._field_draft(self._legacy_text(item))
                    for item in (choice.paper.key_results if choice.paper else ())
                )
                if field is not None
            ),
            experiment_design=None,
            limitations=tuple(
                field
                for field in (
                    self._field_draft(self._legacy_text(item))
                    for item in ((choice.paper_summary.limitations if choice.paper_summary else ()) or (choice.paper.limitations if choice.paper else ()))
                )
                if field is not None
            ),
            open_questions=tuple(
                field
                for field in (
                    self._field_draft(self._legacy_text(item))
                    for item in ((choice.paper_summary.suggestions_or_questions if choice.paper_summary else ()) or (choice.open_question.possible_followup if choice.open_question else ()) or (choice.open_question.why_open if choice.open_question else ()) or ((choice.open_question.unresolved_question,) if choice.open_question and choice.open_question.unresolved_question else ()))
                )
                if field is not None
            ),
            evidence_chunk_ids=self._merge_evidence_ids(
                *(choice.paper_summary.evidence_candidate_ids if choice.paper_summary else ()),
                *(choice.paper.evidence_candidate_ids if choice.paper else ()),
                *(choice.open_question.evidence_candidate_ids if choice.open_question else ()),
            ),
            confidence=float(
                (choice.understanding.confidence if choice.understanding is not None else 0.5)
            ),
        )

    def _paper_from_understanding(
        self,
        understanding: IngestUnderstandingDraft,
        choice: StructuredIngestExtractionChoice,
    ) -> IngestPaperMemoryDraft:
        key_results = tuple(field.text for field in understanding.key_results if field and field.text)
        limitations = tuple(field.text for field in understanding.limitations if field and field.text)
        novelty_claim = next((field.text for field in understanding.novelty_claims if field and field.text), None)
        problem = understanding.problem.text if understanding.problem and understanding.problem.text else None
        method = understanding.method.text if understanding.method and understanding.method.text else None
        evidence_candidate_ids = understanding.evidence_chunk_ids
        if not evidence_candidate_ids and choice.paper is not None:
            evidence_candidate_ids = choice.paper.evidence_candidate_ids
        return IngestPaperMemoryDraft(
            problem=problem,
            method=method,
            key_results=key_results,
            limitations=limitations,
            novelty_claim=novelty_claim,
            evidence_candidate_ids=evidence_candidate_ids,
            confidence=understanding.confidence,
        )

    def _open_question_from_understanding(
        self,
        understanding: IngestUnderstandingDraft,
        choice: StructuredIngestExtractionChoice,
    ) -> IngestOpenQuestionMemoryDraft:
        unresolved_question = choice.open_question.unresolved_question if choice.open_question is not None else None
        if unresolved_question is None:
            unresolved_question = next((field.text for field in understanding.open_questions if field and field.text), None)
        if unresolved_question is None:
            unresolved_question = "无法基于当前论文内容稳定生成该字段。"
        why_open = tuple(item for item in choice.open_question.why_open if item) if choice.open_question is not None else ()
        if not why_open:
            why_open = tuple(field.text for field in understanding.open_questions if field and field.text)
        if not why_open:
            why_open = (unresolved_question,)
        possible_followup = tuple(item for item in choice.open_question.possible_followup if item) if choice.open_question is not None else ()
        if not possible_followup:
            possible_followup = why_open
        evidence_chunk_ids = understanding.evidence_chunk_ids
        if not evidence_chunk_ids and choice.open_question is not None:
            evidence_chunk_ids = choice.open_question.evidence_candidate_ids
        return IngestOpenQuestionMemoryDraft(
            unresolved_question=unresolved_question,
            why_open=why_open,
            possible_followup=possible_followup,
            evidence_candidate_ids=evidence_chunk_ids,
            confidence=understanding.confidence,
        )

    def _summary_from_understanding(
        self,
        understanding: IngestUnderstandingDraft,
        choice: StructuredIngestExtractionChoice,
    ) -> IngestPaperSummaryDraft:
        topic = understanding.topic.text if understanding.topic and understanding.topic.text else None
        problem = understanding.problem.text if understanding.problem and understanding.problem.text else None
        novelty_claims = tuple(field.text for field in understanding.novelty_claims if field and field.text)
        key_results = tuple(field.text for field in understanding.key_results if field and field.text)
        limitations = tuple(field.text for field in understanding.limitations if field and field.text)
        open_questions = tuple(field.text for field in understanding.open_questions if field and field.text)
        experiment_design = understanding.experiment_design.text if understanding.experiment_design and understanding.experiment_design.text else None
        evidence_chunk_ids = understanding.evidence_chunk_ids
        if not evidence_chunk_ids and choice.paper_summary is not None:
            evidence_chunk_ids = choice.paper_summary.evidence_candidate_ids
        if topic is None and choice.paper_summary is not None:
            topic = choice.paper_summary.what_it_is_about
        if problem is None and choice.paper_summary is not None:
            problem = choice.paper_summary.problem_solved
        if not novelty_claims and choice.paper_summary is not None:
            novelty_claims = tuple(choice.paper_summary.new_ideas)
        if not limitations and choice.paper_summary is not None:
            limitations = tuple(choice.paper_summary.limitations)
        if not open_questions and choice.paper_summary is not None:
            open_questions = tuple(choice.paper_summary.suggestions_or_questions)
        if experiment_design and experiment_design not in open_questions:
            open_questions = open_questions + (experiment_design,)
        return IngestPaperSummaryDraft(
            what_it_is_about=topic or "??????????????????",
            problem_solved=problem or "??????????????????",
            new_ideas=novelty_claims or (topic,) if topic else ("??????????????????",),
            limitations=limitations or ("??????????????????",),
            suggestions_or_questions=open_questions or ("??????????????????",),
            evidence_candidate_ids=evidence_chunk_ids,
            confidence=understanding.confidence,
        )

    def _legacy_text(self, value: object) -> str | None:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    def _field_draft(self, field: object) -> IngestEvidenceFieldDraft | None:
        if field is None:
            return None
        if isinstance(field, IngestEvidenceFieldDraft):
            return field
        if isinstance(field, str):
            text = field.strip()
            if not text:
                return None
            return IngestEvidenceFieldDraft(text=text, evidence_chunk_ids=(), evidence_status="weak")
        if isinstance(field, dict):
            text = _normalize_text(field)
            if not text:
                return None
            evidence_chunk_ids = _normalize_evidence_ids(field)
            confidence = float(field.get("confidence", 0.5))
            evidence_status = "weak" if not evidence_chunk_ids else field.get("evidence_status", "strong")
            if evidence_chunk_ids and isinstance(evidence_status, str):
                evidence_status = evidence_status.strip().lower()
            if evidence_chunk_ids and evidence_status not in {"strong", "weak"}:
                evidence_status = "strong" if evidence_chunk_ids else "weak"
            return IngestEvidenceFieldDraft(
                text=text,
                evidence_chunk_ids=evidence_chunk_ids,
                confidence=confidence,
                evidence_status=evidence_status,
            )
        text = _normalize_text(field)
        if text is None:
            return None
        evidence_chunk_ids = _normalize_evidence_ids(field)
        confidence = float(getattr(field, "confidence", 0.5))
        evidence_status = "weak" if not evidence_chunk_ids else getattr(field, "evidence_status", "strong")
        if evidence_chunk_ids and isinstance(evidence_status, str):
            evidence_status = evidence_status.strip().lower()
        if evidence_chunk_ids and evidence_status not in {"strong", "weak"}:
            evidence_status = "strong" if evidence_chunk_ids else "weak"
        return IngestEvidenceFieldDraft(
            text=text,
            evidence_chunk_ids=evidence_chunk_ids,
            confidence=confidence,
            evidence_status=evidence_status,
        )

    def _merge_evidence_ids(self, *values: object) -> tuple[str, ...]:
        ids: list[str] = []
        for value in values:
            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str) and item not in ids:
                        ids.append(item)
        return tuple(ids)

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


__all__ = [
    "DeepSeekStructuredIngestExtractionTransport",
    "ModelBackedIngestExtractionClient",
    "StaticStructuredIngestExtractionTransport",
    "StructuredIngestExtractionChoice",
    "StructuredIngestExtractionPrompt",
    "StructuredIngestExtractionTransport",
    "StructuredIngestEvidenceFieldDraft",
    "StructuredIngestPaperSummaryDraft",
    "StructuredIngestUnderstandingDraft",
    "UnavailableStructuredIngestExtractionTransport",
]
