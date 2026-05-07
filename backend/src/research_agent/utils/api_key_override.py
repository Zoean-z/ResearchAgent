"""Request-scoped API key override helpers for model-backed adapters."""

from __future__ import annotations

from contextvars import ContextVar, Token

_REQUEST_API_KEY_OVERRIDE: ContextVar[str | None] = ContextVar("research_agent_request_api_key_override", default=None)


def set_request_api_key_override(api_key: str | None) -> Token[str | None]:
    """Bind the current request-scoped API key override."""

    normalized = api_key.strip() if isinstance(api_key, str) else None
    return _REQUEST_API_KEY_OVERRIDE.set(normalized or None)


def reset_request_api_key_override(token: Token[str | None]) -> None:
    """Restore the previous request-scoped API key override."""

    _REQUEST_API_KEY_OVERRIDE.reset(token)


def get_request_api_key_override() -> str | None:
    """Return the current request-scoped API key override, if any."""

    return _REQUEST_API_KEY_OVERRIDE.get()


def resolve_api_key(default_api_key: str | None) -> str | None:
    """Resolve the effective API key for the current request."""

    return get_request_api_key_override() or default_api_key
