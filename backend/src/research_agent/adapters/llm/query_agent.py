"""Model-backed adapter boundary for bounded query agent decisions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from typing import Any, Callable, Literal, Protocol
from urllib import request as urllib_request
from urllib import error as urllib_error

import json
import time

from pydantic import BaseModel, Field

from research_agent.runtime.agent_protocol import AgentObservation, AgentTurnDecision, AgentTurnRequest
from research_agent.runtime.query_turn import QueryTurnClient, QueryTurnDecision, QueryTurnState
from research_agent.tools.protocol import QueryToolName, get_query_tool_definition
from research_agent.utils import resolve_api_key, to_json_safe

logger = logging.getLogger(__name__)
FINALIZATION_MAX_TOKENS = 1536
FINALIZATION_EVIDENCE_VIEW_LIMIT = 6
FINALIZATION_OBSERVATION_LIMIT = 3
FINALIZATION_STRING_LIMIT = 240


class StructuredQueryAgentPrompt(BaseModel):
    """Structured prompt payload sent to a query-agent model transport."""

    query: str = Field(description="Original follow-up query")
    allowed_tools: tuple[str, ...] = Field(description="Only these tools may be selected")
    final_answer_allowed: bool = Field(description="Whether the model may answer directly")
    completed_tools: tuple[str, ...] = Field(default_factory=tuple, description="Already executed tools in this run")
    state_summary: str = Field(description="Compact host-generated summary of current retrieval state")
    recent_conversation_context: dict[str, Any] | None = Field(
        default=None,
        description="Compact recent conversation context injected by the host",
    )
    tool_descriptions: dict[str, str] = Field(
        default_factory=dict,
        description="Short descriptions of each allowed tool",
    )
    observations: tuple[AgentObservation, ...] = Field(
        default_factory=tuple,
        description="Host observations visible to the model",
    )


class StructuredQueryAgentChoice(BaseModel):
    """Structured agent choice returned by a model transport."""

    action_type: Literal["tool_call", "final_answer"] = Field(description="Next bounded action")
    tool_name: str | None = Field(default=None, description="Selected next tool name when action_type is tool_call")
    final_answer: str | None = Field(default=None, description="Final answer when action_type is final_answer")
    rationale: str = Field(default="", description="Why this action should happen next")
    tool_parameters: dict[str, object] = Field(default_factory=dict, description="Optional tool-call parameters returned by the model")
    arguments: dict[str, object] | None = Field(default=None, description="Legacy optional alias for tool_parameters")


class StructuredQueryAgentTransport(Protocol):
    """Transport that can obtain a structured next-action choice from a model."""

    def choose_next_action(self, prompt: StructuredQueryAgentPrompt) -> StructuredQueryAgentChoice:
        """Return the structured next action choice."""

    def generate_final_answer(self, request: AgentTurnRequest) -> str | None:
        """Return a plain-text final answer for the finalization stage."""


@dataclass(frozen=True, slots=True)
class DeepSeekHttpResponse:
    """HTTP response snapshot returned by the DeepSeek transport."""

    status_code: int
    body: bytes


class DeepSeekRateLimitError(RuntimeError):
    """DeepSeek API returned a transient rate-limit or server error."""

    def __init__(self, status_code: int, body_text: str) -> None:
        self.status_code = status_code
        self.body_text = body_text
        super().__init__(f"DeepSeek API returned HTTP {status_code}")


@dataclass(frozen=True, slots=True)
class QueryAgentFailureDetail:
    """Structured failure detail captured from a bad query-agent model response."""

    failure_stage_detail: str
    status_code: int | None = None
    repair_attempted: bool = False
    raw_response_preview: str | None = None
    content_preview: str | None = None


class DeepSeekQueryAgentResponseError(RuntimeError):
    """Structured non-transient DeepSeek response failure for query decisions."""

    def __init__(
        self,
        message: str,
        *,
        failure_detail: QueryAgentFailureDetail,
    ) -> None:
        super().__init__(message)
        self.failure_detail = failure_detail


@dataclass(frozen=True, slots=True)
class FinalizationResponse:
    """Parsed plain-text finalization response from DeepSeek."""

    answer_text: str
    finish_reason: str | None
    reasoning_content: str | None
    reasoning_content_length: int
    content_length: int


HttpPost = Callable[[str, dict[str, str], bytes, float], DeepSeekHttpResponse]
ApiKeyProvider = Callable[[], str | None]


class UnavailableStructuredQueryAgentTransport:
    """Default transport placeholder until a provider-specific client is configured."""

    def choose_next_action(self, prompt: StructuredQueryAgentPrompt) -> StructuredQueryAgentChoice:  # pragma: no cover - defensive default
        raise RuntimeError("No structured query agent transport is configured.")

    def generate_final_answer(self, request: AgentTurnRequest) -> str | None:  # pragma: no cover - defensive default
        raise RuntimeError("No structured query agent transport is configured.")


@dataclass(frozen=True, slots=True)
class StaticStructuredQueryAgentTransport:
    """Deterministic transport used by tests to simulate a model response."""

    action_type: Literal["tool_call", "final_answer"]
    tool_name: str | None = None
    final_answer: str | None = None
    rationale: str = "model_selected_next_action"
    arguments: dict[str, object] | None = None

    def choose_next_action(self, prompt: StructuredQueryAgentPrompt) -> StructuredQueryAgentChoice:
        return StructuredQueryAgentChoice(
            action_type=self.action_type,
            tool_name=self.tool_name,
            final_answer=self.final_answer,
            rationale=self.rationale,
            tool_parameters=self.arguments or {},
            arguments=self.arguments,
        )

    def generate_final_answer(self, request: AgentTurnRequest) -> str | None:
        return self.final_answer


def _default_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> DeepSeekHttpResponse:
    request = urllib_request.Request(url=url, data=body, headers=headers, method="POST")
    try:
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            return DeepSeekHttpResponse(status_code=getattr(response, "status", 200), body=response.read())
    except urllib_error.HTTPError as error:
        return DeepSeekHttpResponse(status_code=error.code, body=error.read())


class DeepSeekStructuredQueryAgentTransport:
    """DeepSeek chat-completions transport for bounded query-agent decisions."""

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

    def choose_next_action(self, prompt: StructuredQueryAgentPrompt) -> StructuredQueryAgentChoice:
        api_key = self._api_key_provider()
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured.")

        last_error: Exception | None = None
        messages = self._messages_for(prompt)
        for attempt in range(3):
            repair_attempted = attempt > 0
            payload = {
                "model": self._model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "max_tokens": 2048,
                "temperature": 0.0,
                "stream": False,
            }
            raw_response = self._http_post(
                f"{self._base_url}/chat/completions",
                {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json.dumps(payload).encode("utf-8"),
                self._timeout_seconds,
            )
            self._log_raw_response(raw_response=raw_response, prompt=prompt, attempt=attempt)
            try:
                return self._parse_response(raw_response, repair_attempted=repair_attempted)
            except DeepSeekRateLimitError as error:
                last_error = error
                if attempt < 2:
                    backoff = 5 * (attempt + 1)
                    logger.warning("DeepSeek rate-limited (HTTP %d), retrying in %ds (attempt %d/3)", error.status_code, backoff, attempt + 1)
                    time.sleep(backoff)
                    continue
                raise
            except DeepSeekQueryAgentResponseError as error:
                last_error = error
                logger.warning(
                    "DeepSeek query-agent response error attempt=%d/3 stage=%s content_preview=%s",
                    attempt + 1,
                    error.failure_detail.failure_stage_detail,
                    (error.failure_detail.content_preview or "")[:200],
                )
                if attempt < 2:
                    messages = self._repair_messages_for(
                        prompt=prompt,
                        previous_response=self._raw_response_text(raw_response.body),
                        failure_reason=str(error),
                    )
                    continue
                raise DeepSeekQueryAgentResponseError(
                    f"DeepSeek query-agent response could not be repaired after 2 retries: {error}",
                    failure_detail=QueryAgentFailureDetail(
                        failure_stage_detail=error.failure_detail.failure_stage_detail,
                        status_code=error.failure_detail.status_code,
                        repair_attempted=True,
                        raw_response_preview=error.failure_detail.raw_response_preview,
                        content_preview=error.failure_detail.content_preview,
                    ),
                ) from error
        if last_error is not None:
            raise RuntimeError(f"DeepSeek query-agent response could not be parsed: {last_error}") from last_error
        raise RuntimeError("DeepSeek query-agent response could not be parsed.")

    def generate_final_answer(self, request: AgentTurnRequest) -> str:
        api_key = self._api_key_provider()
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured.")

        last_error: Exception | None = None
        compact_observations = self._compact_finalization_observations(request.observations)
        messages = self._finalization_messages_for(
            request,
            compact_observations=compact_observations,
        )
        for attempt in range(3):
            payload = {
                "model": self._model,
                "messages": messages,
                "max_tokens": FINALIZATION_MAX_TOKENS,
                "temperature": 0.0,
                "stream": False,
            }
            raw_response = self._http_post(
                f"{self._base_url}/chat/completions",
                {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json.dumps(payload).encode("utf-8"),
                self._timeout_seconds,
            )
            self._log_raw_response(raw_response=raw_response, prompt=self._request_prompt_for_logging(request), attempt=attempt)
            try:
                parsed_response = self._parse_plain_text_response(raw_response)
                self._log_finalization_summary(
                    request=request,
                    compact_observations=compact_observations,
                    messages=messages,
                    raw_response=raw_response,
                    parsed_response=parsed_response,
                    attempt=attempt,
                )
                answer_text = parsed_response.answer_text.strip()
                if not answer_text:
                    raise RuntimeError("DeepSeek query-finalization response contained empty content.")
                if parsed_response.finish_reason == "length" or self._looks_truncated_final_answer(answer_text):
                    raise RuntimeError("DeepSeek query-finalization response appears truncated.")
                return answer_text
            except DeepSeekRateLimitError as error:
                last_error = error
                if attempt < 2:
                    backoff = 5 * (attempt + 1)
                    logger.warning("DeepSeek rate-limited (HTTP %d), retrying in %ds (attempt %d/3)", error.status_code, backoff, attempt + 1)
                    time.sleep(backoff)
                    continue
                raise
            except (json.JSONDecodeError, ValueError, RuntimeError) as error:
                last_error = error
                logger.warning(
                    "DeepSeek query-finalization error attempt=%d/3 error=%s",
                    attempt + 1,
                    str(error)[:200],
                )
                if attempt < 2:
                    messages = self._repair_finalization_messages_for(
                        request=request,
                        compact_observations=compact_observations,
                        previous_response=self._raw_response_text(raw_response.body),
                        failure_reason=str(error),
                    )
                    continue
                raise RuntimeError(
                    f"DeepSeek query-finalization response could not be repaired after 2 retries: {error}"
                ) from error
        if last_error is not None:
            raise RuntimeError(f"DeepSeek query-finalization response could not be parsed: {last_error}") from last_error
        raise RuntimeError("DeepSeek query-finalization response could not be parsed.")

    def _messages_for(self, prompt: StructuredQueryAgentPrompt) -> list[dict[str, str]]:
        return self._base_messages(prompt, include_repair_guidance=False)

    def _finalization_messages_for(
        self,
        request: AgentTurnRequest,
        *,
        compact_observations: Sequence[dict[str, Any]],
    ) -> list[dict[str, str]]:
        system_prompt = (
            "You are the final answer stage for a memory-routed paper system. "
            "Return only the final user-facing answer. "
            "Do not output analysis, reasoning steps, JSON, code fences, tool names, or runtime commentary. "
            "Do not start with phrases like 'we were asked', 'from the observations', 'let's inspect', or 'therefore'. "
            "Default final answer language is Chinese unless the user explicitly asks for English. "
            "Use recent conversation context only to resolve follow-up references such as 'this paper' or 'it'. "
            "Do not mention prompts, routing, or host internals."
        )
        user_prompt = json.dumps(
            {
                "query": request.query,
                "recent_conversation_context": request.recent_conversation_context,
                "evidence_view": compact_observations,
                "instruction": (
                    "Return only the final answer text. Do not use JSON. "
                    "Base the answer on the evidence view and the original query. "
                    "If recent conversation context is present, use it to resolve follow-up references like 'this paper' or 'it'. "
                    "Do not write analysis first and do not end with a fragment. "
                    "If the evidence is sufficient, answer directly and concisely."
                ),
            },
            ensure_ascii=True,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _repair_finalization_messages_for(
        self,
        *,
        request: AgentTurnRequest,
        compact_observations: Sequence[dict[str, Any]],
        previous_response: str,
        failure_reason: str,
    ) -> list[dict[str, str]]:
        messages = self._finalization_messages_for(
            request,
            compact_observations=compact_observations,
        )
        messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "previous_response": previous_response,
                        "failure_reason": failure_reason,
                        "instruction": (
                            "Previous model response was empty, incomplete, or could not be parsed. "
                            "Return a concise complete final answer in natural language. Do not use JSON, analysis, or preamble."
                        ),
                    },
                    ensure_ascii=True,
                ),
            }
        )
        return messages

    def _compact_finalization_observations(self, observations: Sequence[AgentObservation]) -> list[dict[str, Any]]:
        return [
            self._compact_finalization_observation(observation)
            for observation in observations[:FINALIZATION_EVIDENCE_VIEW_LIMIT]
        ]

    def _compact_finalization_observation(self, observation: AgentObservation) -> dict[str, Any]:
        compacted: dict[str, Any] = {
            "kind": observation.kind,
            "summary": self._trim_answer_text(observation.summary, FINALIZATION_STRING_LIMIT),
        }
        payload = observation.payload
        if payload is None:
            return compacted
        compact_payload = self._compact_finalization_payload(observation.kind, payload)
        if compact_payload is not None:
            compacted["payload"] = compact_payload
        return compacted

    def _compact_finalization_payload(self, kind: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if kind == "session_papers":
            papers = payload.get("papers")
            if not isinstance(papers, list):
                return self._compact_generic_value(payload) if payload else None
            return {
                "tool_name": payload.get("tool_name"),
                "papers": [self._compact_session_paper(item) for item in papers[:FINALIZATION_OBSERVATION_LIMIT] if isinstance(item, dict)],
            }
        if kind == "paper_memory_bundle":
            bundle = payload.get("bundle")
            if not isinstance(bundle, dict):
                return self._compact_generic_value(payload) if payload else None
            return {
                "tool_name": payload.get("tool_name"),
                "bundle": self._compact_paper_memory_bundle(bundle),
            }
        if kind in {"source_chunk_search", "source_reread_chunks"}:
            chunks = payload.get("chunks")
            if not isinstance(chunks, list):
                return self._compact_generic_value(payload) if payload else None
            return {
                "tool_name": payload.get("tool_name"),
                "chunk_ids": [item.get("chunk_id") for item in chunks[:FINALIZATION_OBSERVATION_LIMIT] if isinstance(item, dict) and item.get("chunk_id")],
                "selection_reasons": [
                    self._trim_answer_text(item.get("selection_reason", ""), FINALIZATION_STRING_LIMIT)
                    for item in chunks[:FINALIZATION_OBSERVATION_LIMIT]
                    if isinstance(item, dict) and item.get("selection_reason")
                ],
            }
        return self._compact_generic_value(payload) if payload else None

    def _compact_session_paper(self, paper: dict[str, Any]) -> dict[str, Any]:
        return {
            "paper_id": paper.get("paper_id"),
            "title": self._trim_answer_text(str(paper.get("title") or ""), FINALIZATION_STRING_LIMIT) or None,
            "file_name": self._trim_answer_text(str(paper.get("file_name") or ""), FINALIZATION_STRING_LIMIT) or None,
        }

    def _compact_paper_memory_bundle(self, bundle: dict[str, Any]) -> dict[str, Any]:
        paper = bundle.get("paper") if isinstance(bundle.get("paper"), dict) else {}
        paper_memory = bundle.get("paper_memory") if isinstance(bundle.get("paper_memory"), dict) else None
        open_questions = bundle.get("open_questions") if isinstance(bundle.get("open_questions"), list) else []
        relations = bundle.get("relations") if isinstance(bundle.get("relations"), list) else []
        evidence_source_chunks = bundle.get("evidence_source_chunks") if isinstance(bundle.get("evidence_source_chunks"), list) else []
        return {
            "paper": {
                "paper_id": paper.get("paper_id"),
                "title": self._trim_answer_text(str(paper.get("title") or ""), FINALIZATION_STRING_LIMIT) or None,
                "file_name": self._trim_answer_text(str(paper.get("file_name") or ""), FINALIZATION_STRING_LIMIT) or None,
            },
            "paper_memory": self._compact_paper_memory(paper_memory) if paper_memory is not None else None,
            "open_questions": [
                self._compact_open_question_memory(item)
                for item in open_questions[:FINALIZATION_OBSERVATION_LIMIT]
                if isinstance(item, dict)
            ],
            "relations": [
                self._compact_relation_memory(item)
                for item in relations[:FINALIZATION_OBSERVATION_LIMIT]
                if isinstance(item, dict)
            ],
            "evidence_source_chunks": [
                self._compact_source_chunk(item)
                for item in evidence_source_chunks[:FINALIZATION_OBSERVATION_LIMIT]
                if isinstance(item, dict)
            ],
        }

    def _compact_paper_memory(self, paper_memory: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": paper_memory.get("id"),
            "problem": self._trim_answer_text(str(paper_memory.get("problem") or ""), FINALIZATION_STRING_LIMIT) or None,
            "method": self._trim_answer_text(str(paper_memory.get("method") or ""), FINALIZATION_STRING_LIMIT) or None,
            "novelty_claim": self._trim_answer_text(str(paper_memory.get("novelty_claim") or ""), FINALIZATION_STRING_LIMIT) or None,
            "key_results": [self._trim_answer_text(str(item), FINALIZATION_STRING_LIMIT) for item in (paper_memory.get("key_results") or [])[:FINALIZATION_OBSERVATION_LIMIT]],
            "limitations": [self._trim_answer_text(str(item), FINALIZATION_STRING_LIMIT) for item in (paper_memory.get("limitations") or [])[:FINALIZATION_OBSERVATION_LIMIT]],
        }

    def _compact_open_question_memory(self, open_question: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": open_question.get("id"),
            "unresolved_question": self._trim_answer_text(str(open_question.get("unresolved_question") or ""), FINALIZATION_STRING_LIMIT) or None,
            "related_papers": list(open_question.get("related_papers") or ())[:FINALIZATION_OBSERVATION_LIMIT],
            "why_open": [
                self._trim_answer_text(str(item), FINALIZATION_STRING_LIMIT)
                for item in (open_question.get("why_open") or [])[:FINALIZATION_OBSERVATION_LIMIT]
            ],
            "possible_followup": [
                self._trim_answer_text(str(item), FINALIZATION_STRING_LIMIT)
                for item in (open_question.get("possible_followup") or [])[:FINALIZATION_OBSERVATION_LIMIT]
            ],
        }

    def _compact_relation_memory(self, relation: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": relation.get("id"),
            "source_paper": relation.get("source_paper"),
            "target_paper": relation.get("target_paper"),
            "relation_type": relation.get("relation_type"),
            "summary": self._trim_answer_text(str(relation.get("summary") or ""), FINALIZATION_STRING_LIMIT) or None,
        }

    def _compact_source_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        return {
            "chunk_id": chunk.get("chunk_id"),
            "paper_id": chunk.get("paper_id"),
            "page": chunk.get("page"),
            "section": chunk.get("section"),
            "excerpt": self._trim_answer_text(str(chunk.get("excerpt") or ""), FINALIZATION_STRING_LIMIT) or None,
        }

    def _compact_generic_value(self, value: Any, *, depth: int = 0) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return self._trim_answer_text(value, FINALIZATION_STRING_LIMIT)
        if isinstance(value, dict):
            if depth >= 2:
                return self._trim_answer_text(str(value), FINALIZATION_STRING_LIMIT)
            return {
                str(key): self._compact_generic_value(item, depth=depth + 1)
                for key, item in list(value.items())[:FINALIZATION_OBSERVATION_LIMIT]
            }
        if isinstance(value, (list, tuple, set)):
            if depth >= 2:
                return self._trim_answer_text(str(list(value)), FINALIZATION_STRING_LIMIT)
            return [self._compact_generic_value(item, depth=depth + 1) for item in list(value)[:FINALIZATION_OBSERVATION_LIMIT]]
        return self._trim_answer_text(str(value), FINALIZATION_STRING_LIMIT)

    def _trim_answer_text(self, text: str | None, limit: int) -> str:
        if not text:
            return ""
        normalized = " ".join(str(text).split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[: max(limit - 1, 0)].rstrip()}…"

    def _looks_truncated_final_answer(self, answer_text: str) -> bool:
        stripped = answer_text.strip()
        if not stripped:
            return True
        lowered = stripped.lower()
        if stripped.endswith((":", "：", ",", "，", ";", "；", "、", "(", "（", "[", "【", "-", "—")):
            return True
        if stripped.endswith(("...", "…")):
            return True
        if stripped.endswith(("因此", "所以", "同时", "此外", "然后", "但", "而且", "不过")):
            return True
        if lowered.endswith(("because", "so", "and", "but", "however", "therefore", "thus", "moreover", "furthermore")):
            return True
        if lowered.endswith(("for example", "for instance", "as well", "in addition", "in summary")):
            return True
        if len(stripped) < 12:
            return False
        return False

    def _log_finalization_summary(
        self,
        *,
        request: AgentTurnRequest,
        compact_observations: Sequence[dict[str, Any]],
        messages: Sequence[dict[str, str]],
        raw_response: DeepSeekHttpResponse,
        parsed_response: FinalizationResponse,
        attempt: int,
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        prompt_chars = len(json.dumps(messages, ensure_ascii=True))
        observations_chars = len(json.dumps(compact_observations, ensure_ascii=True))
        logger.debug(
            "DeepSeek query-finalization summary attempt=%s status=%s max_tokens=%s finish_reason=%s output_length=%s prompt_chars=%s observations_chars=%s retry_triggered=%s reasoning_content_length=%s query=%s",
            attempt + 1,
            raw_response.status_code,
            FINALIZATION_MAX_TOKENS,
            parsed_response.finish_reason,
            parsed_response.content_length,
            prompt_chars,
            observations_chars,
            attempt > 0,
            parsed_response.reasoning_content_length,
            request.query,
        )

    def _repair_messages_for(
        self,
        *,
        prompt: StructuredQueryAgentPrompt,
        previous_response: str,
        failure_reason: str,
    ) -> list[dict[str, str]]:
        messages = self._base_messages(prompt, include_repair_guidance=True)
        messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "previous_response": previous_response,
                        "failure_reason": failure_reason,
                        "observations": to_json_safe([observation.model_dump(mode="python") for observation in prompt.observations]),
                        "instruction": (
                            "Previous model response was empty or could not be parsed. "
                            "You already received observations. Return valid query decision JSON based on observations; "
                            "if enough information exists, return action_type=final_answer and final_answer."
                        ),
                        "schema_instruction": (
                            "Re-output one valid JSON object that matches the schema exactly. "
                            "Do not add markdown, prose, or extra keys. "
                            'Example tool_call: {"action_type":"tool_call","rationale":"...","tool_name":"...","tool_parameters":{}}. '
                            'Example final_answer: {"action_type":"final_answer","rationale":"...","final_answer":"..."}.'
                        ),
                    },
                    ensure_ascii=True,
                ),
            }
        )
        return messages

    def _base_messages(self, prompt: StructuredQueryAgentPrompt, *, include_repair_guidance: bool) -> list[dict[str, str]]:
        schema_example = (
            "Return one JSON object only. Example tool_call: "
            '{"action_type":"tool_call","rationale":"need the paper bundle","tool_name":"get_paper_memory_bundle","tool_parameters":{"paper_id":"paper-123","source_chunk_limit":3}}. '
            "Example final_answer: "
            '{"action_type":"final_answer","rationale":"enough evidence already","final_answer":"..."}'
        )
        system_prompt = (
            "You are a query agent for a memory-routed paper system. Respond in json only. "
            "You may either request one next tool call or return a final answer. "
            "Prefer final_answer whenever the query is already answerable without retrieval. "
            "Use tool_call only when another bounded tool will materially improve the answer. "
            "If the user explicitly asks to import an arXiv paper or provides an arXiv link that should be added to the session, prefer import_arxiv_paper before answering. "
            "For greetings, acknowledgements, capability questions, or other low-context conversational turns, prefer final_answer immediately. "
            "Default final_answer language is Chinese; use English only if the user explicitly asks for English. "
            "When recent conversation context is provided, use it to resolve follow-up references such as 'this paper', 'it', or 'the previous result'. "
            "CRITICAL: When the user asks about specific content in a paper — methods, models, datasets, experiments, results, formulas, or any factual detail — you MUST use search_source_chunks or read_source_passages to verify from the original text before answering. "
            "Never guess or fabricate paper content. If you are not certain about a specific detail, search and read the source first. "
            "Memory summaries may be incomplete or imprecise; only source passages are authoritative for factual claims. "
            "Do not call retrieval tools just to be safe when there is no clear need. "
            "Never invent tools outside allowed_tools. "
            + (
                "If the previous response was invalid, output one corrected JSON object only. "
                if include_repair_guidance
                else ""
            )
            + "The response must be valid JSON and must match one of the two example shapes exactly. "
            + schema_example
            + 'Return JSON with keys "action_type", "tool_name" (for tool_call), '
            + '"tool_parameters" (business tool parameters only), "final_answer" (for final_answer), and "rationale". '
            + "Never put runtime context such as session_id in tool_parameters."
        )
        user_prompt = json.dumps(
            {
                "query": prompt.query,
                "allowed_tools": prompt.allowed_tools,
                "final_answer_allowed": prompt.final_answer_allowed,
                "completed_tools": prompt.completed_tools,
                "state_summary": prompt.state_summary,
                "recent_conversation_context": prompt.recent_conversation_context,
                "tool_descriptions": prompt.tool_descriptions,
                "observations": to_json_safe([observation.model_dump(mode="python") for observation in prompt.observations]),
                "instruction": (
                    "Choose either tool_call or final_answer. If tool_call, tool_name must be one of allowed_tools and tool_parameters may contain only business parameters. "
                    "If the user explicitly wants to import an arXiv paper into the session, call import_arxiv_paper with arxiv_id_or_url before answering. "
                    "If final_answer, provide final_answer text and do not invent extra tools. "
                    "Do not include session_id or other runtime-owned ids unless the tool explicitly asks for that business id, such as paper_id. "
                    "Default final_answer language is Chinese unless the user explicitly asks for another language. "
                    "If recent conversation context is present, use it to resolve follow-up references like 'this paper' or 'it'. "
                    "If the query asks about specific paper content (methods, models, datasets, results, formulas, experiments), you must first use search_source_chunks or read_source_passages to get the original text. Do not answer from memory alone. "
                    "If the turn is ordinary conversation and not a research retrieval request, choose final_answer. "
                    "Output valid json and follow one of the example JSON shapes exactly."
                ),
            },
            ensure_ascii=True,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _request_prompt_for_logging(self, request: AgentTurnRequest) -> StructuredQueryAgentPrompt:
        return StructuredQueryAgentPrompt(
            query=request.query,
            allowed_tools=request.allowed_actions,
            final_answer_allowed=request.final_answer_allowed,
            completed_tools=request.completed_actions,
            state_summary=request.state_summary,
            recent_conversation_context=request.recent_conversation_context,
            tool_descriptions=request.tool_descriptions,
            observations=request.observations,
        )

    def _parse_response(self, raw_response: DeepSeekHttpResponse, *, repair_attempted: bool) -> StructuredQueryAgentChoice:
        self._raise_for_status(raw_response, repair_attempted=repair_attempted)
        body_text = self._raise_for_empty_body(raw_response)
        try:
            payload = json.loads(body_text)
        except json.JSONDecodeError as error:
            raise DeepSeekQueryAgentResponseError(
                "DeepSeek query-agent response body was not valid JSON.",
                failure_detail=QueryAgentFailureDetail(
                    failure_stage_detail="parse_http_body",
                    status_code=raw_response.status_code,
                    repair_attempted=repair_attempted,
                    raw_response_preview=body_text[:800],
                ),
            ) from error
        choices = payload.get("choices") or []
        if not choices:
            raise DeepSeekQueryAgentResponseError(
                "DeepSeek query-agent response contained no choices.",
                failure_detail=QueryAgentFailureDetail(
                    failure_stage_detail="missing_choices",
                    status_code=raw_response.status_code,
                    repair_attempted=repair_attempted,
                    raw_response_preview=body_text[:800],
                ),
            )
        message = choices[0].get("message") or {}
        content = self._message_content(message)
        if not content:
            raise DeepSeekQueryAgentResponseError(
                "DeepSeek query-agent response contained empty content.",
                failure_detail=QueryAgentFailureDetail(
                    failure_stage_detail="empty_content",
                    status_code=raw_response.status_code,
                    repair_attempted=repair_attempted,
                    raw_response_preview=body_text[:800],
                ),
            )
        extracted = self._extract_json_from_content(content)
        if extracted is not None and extracted is not content:
            logger.info("Extracted JSON from model content (original %d chars, extracted %d chars)", len(content), len(extracted))
        parse_target = extracted if extracted is not None else content
        try:
            content_payload = json.loads(parse_target)
        except json.JSONDecodeError as error:
            raise DeepSeekQueryAgentResponseError(
                "DeepSeek query-agent response content was not valid JSON.",
                failure_detail=QueryAgentFailureDetail(
                    failure_stage_detail="parse_message_content",
                    status_code=raw_response.status_code,
                    repair_attempted=repair_attempted,
                    raw_response_preview=body_text[:800],
                    content_preview=content[:800],
                ),
            ) from error
        if not isinstance(content_payload, dict):
            raise DeepSeekQueryAgentResponseError(
                "DeepSeek query-agent response content must be a JSON object.",
                failure_detail=QueryAgentFailureDetail(
                    failure_stage_detail="content_not_object",
                    status_code=raw_response.status_code,
                    repair_attempted=repair_attempted,
                    raw_response_preview=body_text[:800],
                    content_preview=content[:800],
                ),
            )
        try:
            return self._normalize_choice_payload(content_payload)
        except ValueError as error:
            raise DeepSeekQueryAgentResponseError(
                str(error),
                failure_detail=QueryAgentFailureDetail(
                    failure_stage_detail="normalize_choice",
                    status_code=raw_response.status_code,
                    repair_attempted=repair_attempted,
                    raw_response_preview=body_text[:800],
                    content_preview=json.dumps(content_payload, ensure_ascii=True)[:800],
                ),
            ) from error

    def _parse_plain_text_response(self, raw_response: DeepSeekHttpResponse) -> FinalizationResponse:
        self._raise_for_status(raw_response, repair_attempted=False)
        body_text = self._raise_for_empty_body(raw_response)
        payload = json.loads(body_text)
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("DeepSeek query-finalization response contained no choices.")
        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        message = first_choice.get("message") or {}
        content = self._optional_string(message.get("content"))
        if not content:
            raise RuntimeError("DeepSeek query-finalization response contained empty content.")
        reasoning_content = self._optional_string(message.get("reasoning_content"))
        answer_text = content.strip()
        return FinalizationResponse(
            answer_text=answer_text,
            finish_reason=self._optional_string(first_choice.get("finish_reason")),
            reasoning_content=reasoning_content,
            reasoning_content_length=len(reasoning_content or ""),
            content_length=len(answer_text),
        )

    def _raw_response_text(self, raw_response: bytes) -> str:
        return raw_response.decode("utf-8", errors="replace")

    @staticmethod
    def _raise_for_status(raw_response: DeepSeekHttpResponse, *, repair_attempted: bool) -> None:
        if raw_response.status_code in {429, 503}:
            body_text = raw_response.body.decode("utf-8", errors="replace")[:500]
            logger.warning("DeepSeek API rate-limit/server error status=%s body=%s", raw_response.status_code, body_text)
            raise DeepSeekRateLimitError(raw_response.status_code, body_text)
        if raw_response.status_code >= 400:
            body_text = raw_response.body.decode("utf-8", errors="replace")[:500]
            raise DeepSeekQueryAgentResponseError(
                f"DeepSeek API returned HTTP {raw_response.status_code}: {body_text}",
                failure_detail=QueryAgentFailureDetail(
                    failure_stage_detail="http_status",
                    status_code=raw_response.status_code,
                    repair_attempted=repair_attempted,
                    raw_response_preview=body_text,
                ),
            )

    @staticmethod
    def _raise_for_empty_body(raw_response: DeepSeekHttpResponse) -> str:
        body_text = raw_response.body.decode("utf-8", errors="replace").strip()
        if not body_text:
            logger.warning("DeepSeek API returned empty response body (HTTP %d)", raw_response.status_code)
            raise DeepSeekRateLimitError(raw_response.status_code, "empty response body")
        return body_text

    def _message_content(self, message: dict[str, object]) -> str | None:
        content = self._optional_string(message.get("content"))
        if content:
            return content
        reasoning_content = self._optional_string(message.get("reasoning_content"))
        if reasoning_content:
            return reasoning_content
        return None

    @staticmethod
    def _extract_json_from_content(content: str) -> str | None:
        """Try to extract a JSON object from model content that may contain markdown or prose."""
        stripped = content.strip()
        # Strip markdown code block: ```json ... ``` or ``` ... ```
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            if first_newline != -1:
                inner = stripped[first_newline + 1:]
                if inner.rstrip().endswith("```"):
                    inner = inner.rstrip()[:-3].rstrip()
                try:
                    json.loads(inner)
                    return inner
                except (json.JSONDecodeError, ValueError):
                    stripped = inner
        # Find first { to last }
        first_brace = stripped.find("{")
        last_brace = stripped.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            candidate = stripped[first_brace:last_brace + 1]
            try:
                json.loads(candidate)
                return candidate
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    def _log_raw_response(
        self,
        *,
        raw_response: DeepSeekHttpResponse,
        prompt: StructuredQueryAgentPrompt,
        attempt: int,
    ) -> None:
        if raw_response.status_code >= 400:
            body_text = self._raw_response_text(raw_response.body)
            logger.warning(
                "DeepSeek API error attempt=%d status=%d body=%s",
                attempt + 1,
                raw_response.status_code,
                body_text[:500],
            )
        if not logger.isEnabledFor(logging.DEBUG):
            return
        body_text = self._raw_response_text(raw_response.body)
        try:
            payload = json.loads(body_text)
        except json.JSONDecodeError:
            payload = {"body_summary": body_text[:600]}
        choices = payload.get("choices") if isinstance(payload, dict) else None
        first_choice = choices[0] if isinstance(choices, list) and choices else {}
        message = first_choice.get("message") if isinstance(first_choice, dict) else {}
        reasoning_content = self._optional_string(message.get("reasoning_content")) if isinstance(message, dict) else None
        logger.debug(
            "DeepSeek query-agent raw response attempt=%s status=%s choices=%s message=%s content=%s reasoning_content=%s reasoning_content_length=%s tool_calls=%s finish_reason=%s usage=%s body_summary=%s observations=%s",
            attempt + 1,
            raw_response.status_code,
            len(choices) if isinstance(choices, list) else 0,
            message,
            self._message_content(message or {}) if isinstance(message, dict) else None,
            reasoning_content,
            len(reasoning_content or ""),
            message.get("tool_calls") if isinstance(message, dict) else None,
            first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
            payload.get("usage") if isinstance(payload, dict) else None,
            body_text[:800],
            to_json_safe([observation.model_dump(mode="python") for observation in prompt.observations]),
        )

    def _normalize_choice_payload(self, payload: dict[str, object]) -> StructuredQueryAgentChoice:
        action_type = self._normalize_action_type(payload.get("action_type"), payload)
        tool_name = self._optional_string(payload.get("tool_name"))
        final_answer = self._optional_string(payload.get("final_answer"))
        rationale = self._optional_string(payload.get("rationale")) or ""
        tool_parameters = self._normalize_tool_parameters(
            payload.get("tool_parameters"),
            payload.get("arguments"),
        )

        if action_type == "final_answer":
            if final_answer is None:
                raise ValueError("DeepSeek query-agent final_answer choice is missing final_answer text.")
            if tool_name is not None:
                raise ValueError("DeepSeek query-agent final_answer choice must not include tool_name.")
            return StructuredQueryAgentChoice(
                action_type="final_answer",
                final_answer=final_answer,
                rationale=rationale,
                tool_parameters=tool_parameters,
                arguments=payload.get("arguments") if isinstance(payload.get("arguments"), dict) else None,
            )

        if tool_name is None:
            raise ValueError("DeepSeek query-agent tool_call choice is missing tool_name.")
        if final_answer is not None:
            raise ValueError("DeepSeek query-agent tool_call choice must not include final_answer.")
        return StructuredQueryAgentChoice(
            action_type="tool_call",
            tool_name=tool_name,
            rationale=rationale,
            tool_parameters=tool_parameters,
            arguments=payload.get("arguments") if isinstance(payload.get("arguments"), dict) else None,
        )

    def _normalize_action_type(self, action_type: object, payload: dict[str, object]) -> Literal["tool_call", "final_answer"]:
        normalized_action = self._normalize_action_type_name(action_type)
        tool_name = self._optional_string(payload.get("tool_name"))
        final_answer = self._optional_string(payload.get("final_answer"))

        if normalized_action is None:
            if final_answer is not None and tool_name is None:
                return "final_answer"
            if tool_name is not None and final_answer is None:
                return "tool_call"
            raise ValueError("DeepSeek query-agent response is missing action_type and cannot be inferred safely.")

        if normalized_action == "final_answer":
            if final_answer is None:
                raise ValueError("DeepSeek query-agent final_answer choice is missing final_answer text.")
            if tool_name is not None:
                raise ValueError("DeepSeek query-agent final_answer choice must not include tool_name.")
            return "final_answer"

        if tool_name is None:
            raise ValueError("DeepSeek query-agent tool_call choice is missing tool_name.")
        if final_answer is not None:
            raise ValueError("DeepSeek query-agent tool_call choice must not include final_answer.")
        return "tool_call"

    def _normalize_action_type_name(self, action_type: object) -> Literal["tool_call", "final_answer"] | None:
        if action_type is None:
            return None
        normalized = str(action_type).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"final_answer", "answer", "final"}:
            return "final_answer"
        if normalized in {"tool_call", "tool", "call"}:
            return "tool_call"
        raise ValueError(f"DeepSeek query-agent action_type '{action_type}' is not supported.")

    def _optional_string(self, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _normalize_tool_parameters(self, tool_parameters: object, arguments: object) -> dict[str, object]:
        normalized_tool_parameters = self._optional_tool_parameters(tool_parameters)
        legacy_arguments = self._optional_tool_parameters(arguments)
        if normalized_tool_parameters is None and legacy_arguments is None:
            return {}
        if normalized_tool_parameters is None:
            return legacy_arguments or {}
        if legacy_arguments is None:
            return normalized_tool_parameters
        if normalized_tool_parameters != legacy_arguments:
            raise ValueError("DeepSeek query-agent response has conflicting tool_parameters and arguments.")
        return normalized_tool_parameters

    def _optional_tool_parameters(self, value: object) -> dict[str, object] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return dict(value)
        raise ValueError("DeepSeek query-agent tool_parameters must be a JSON object when provided.")

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


class ModelBackedQueryAgentClient:
    """Query agent that delegates next-action selection to a model adapter with fallback."""

    def __init__(
        self,
        *,
        transport: StructuredQueryAgentTransport,
        fallback: QueryTurnClient | None,
        agent_name: str = "model_adapter",
    ) -> None:
        self._transport = transport
        self._agent_name = agent_name

    @property
    def agent_name(self) -> str:
        return getattr(self, "_last_agent_name", self._agent_name)

    def decide_next_action(
        self,
        *,
        query: str,
        state: QueryTurnState,
        allowed_tools: Sequence[QueryToolName],
        final_answer_allowed: bool,
    ) -> QueryTurnDecision | None:
        prompt = self._build_prompt(
            query=query,
            state=state,
            allowed_tools=allowed_tools,
            final_answer_allowed=final_answer_allowed,
        )
        try:
            choice = self._transport.choose_next_action(prompt)
            decision = self._decision_from_choice(choice, allowed_tools, final_answer_allowed)
            if decision is None:
                raise ValueError("Model query-agent choice could not be converted to a decision.")
            self._record_success(decision.agent_name, decision.fallback_used)
            return decision
        except Exception as exc:
            self._record_failure(exc)
            return None

    @property
    def fallback_used(self) -> bool:
        return getattr(self, "_last_fallback_used", False)

    def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
        prompt = StructuredQueryAgentPrompt(
            query=request.query,
            allowed_tools=request.allowed_actions,
            final_answer_allowed=request.final_answer_allowed,
            completed_tools=request.completed_actions,
            state_summary=request.state_summary,
            recent_conversation_context=request.recent_conversation_context,
            tool_descriptions=request.tool_descriptions,
            observations=request.observations,
        )
        try:
            choice = self._transport.choose_next_action(prompt)
            decision = self._decision_from_choice(choice, tuple(QueryToolName(tool) for tool in request.allowed_actions), request.final_answer_allowed)
            if decision is None:
                raise ValueError("Model query-agent choice could not be converted to a decision.")
            self._record_success(decision.agent_name, decision.fallback_used)
            return decision.to_agent_turn_decision()
        except Exception as exc:
            self._record_failure(exc)
            return None

    def generate_final_answer(self, request: AgentTurnRequest) -> str | None:
        try:
            answer = self._transport.generate_final_answer(request)
            if answer is None:
                raise RuntimeError("DeepSeek query-finalization response contained empty content.")
            final_text = answer.strip()
            if not final_text:
                raise RuntimeError("DeepSeek query-finalization response contained empty content.")
            self._record_success(self._agent_name, False)
            return final_text
        except Exception as exc:
            self._record_failure(exc)
            raise

    @property
    def fallback_used(self) -> bool:
        return getattr(self, "_last_fallback_used", False)

    @property
    def fallback_reason(self) -> str | None:
        return getattr(self, "_last_fallback_reason", None)

    @property
    def failure_detail(self) -> dict[str, object] | None:
        return getattr(self, "_last_failure_detail", None)

    def _record_success(self, agent_name: str, fallback_used: bool) -> None:
        self._last_agent_name = agent_name
        self._last_fallback_used = fallback_used
        self._last_fallback_reason = None
        self._last_failure_detail = None

    def _record_failure(self, exc: Exception) -> None:
        self._last_agent_name = self._agent_name
        self._last_fallback_used = False
        self._last_fallback_reason = f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ').strip()}"
        self._last_failure_detail = self._failure_detail_dict(exc)

    def _failure_detail_dict(self, exc: Exception) -> dict[str, object] | None:
        detail = getattr(exc, "failure_detail", None)
        if detail is None:
            return None
        return {
            "failure_stage_detail": detail.failure_stage_detail,
            "status_code": detail.status_code,
            "repair_attempted": detail.repair_attempted,
            "raw_response_preview": detail.raw_response_preview,
            "content_preview": detail.content_preview,
        }

    def _decision_from_choice(
        self,
        choice: StructuredQueryAgentChoice,
        allowed_tools: Sequence[QueryToolName],
        final_answer_allowed: bool,
    ) -> QueryTurnDecision:
        if choice.action_type == "tool_call":
            if not choice.tool_name:
                raise ValueError("Model query-agent tool_call choice is missing tool_name.")
            chosen_tool = QueryToolName(choice.tool_name)
            if chosen_tool not in allowed_tools:
                raise ValueError(f"Chosen tool '{chosen_tool.value}' is outside the allowed tool set.")
            return QueryTurnDecision(
                action_type="tool_call",
                tool_name=chosen_tool,
                tool_parameters=choice.tool_parameters,
                rationale=choice.rationale,
                agent_name=self._agent_name,
                fallback_used=False,
            )
        if not final_answer_allowed:
            raise ValueError("Model query-agent final_answer choice is not allowed at this step.")
        if not choice.final_answer:
            raise ValueError("Model query-agent final_answer choice is missing final_answer text.")
        return QueryTurnDecision(
            action_type="final_answer",
            final_answer=choice.final_answer,
            rationale=choice.rationale,
            agent_name=self._agent_name,
            fallback_used=False,
        )

    def _build_prompt(
        self,
        *,
        query: str,
        state: QueryTurnState,
        allowed_tools: Sequence[QueryToolName],
        final_answer_allowed: bool,
        recent_conversation_context: dict[str, Any] | None = None,
    ) -> StructuredQueryAgentPrompt:
        tool_descriptions = {
            tool.value: (definition.description if (definition := get_query_tool_definition(tool)) is not None else tool.value)
            for tool in allowed_tools
        }
        return StructuredQueryAgentPrompt(
            query=query,
            allowed_tools=tuple(tool.value for tool in allowed_tools),
            final_answer_allowed=final_answer_allowed,
            completed_tools=tuple(tool.value for tool in state.completed_tools),
            state_summary=self._state_summary(state),
            recent_conversation_context=recent_conversation_context,
            tool_descriptions=tool_descriptions,
        )

    def _state_summary(self, state: QueryTurnState) -> str:
        return (
            f"completed={','.join(tool.value for tool in state.completed_tools) or 'none'}; "
            f"session_memories={len(state.session_memories)}; "
            f"global_memories={len(state.global_memories)}; "
            f"selected_memory_ids={len(state.selected_memory_ids)}; "
            f"should_reread_source={state.should_reread_source}; "
            f"selected_chunks={len(state.selected_chunks)}"
        )


__all__ = [
    "DeepSeekQueryAgentResponseError",
    "DeepSeekStructuredQueryAgentTransport",
    "ModelBackedQueryAgentClient",
    "QueryAgentFailureDetail",
    "StaticStructuredQueryAgentTransport",
    "StructuredQueryAgentChoice",
    "StructuredQueryAgentPrompt",
    "StructuredQueryAgentTransport",
    "UnavailableStructuredQueryAgentTransport",
]
