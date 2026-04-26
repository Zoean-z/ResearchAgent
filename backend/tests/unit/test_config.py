"""Unit tests for local environment loading."""

from __future__ import annotations

import os

from research_agent.config import load_env_file


def test_load_env_file_reads_values_without_overriding_existing(monkeypatch, tmp_path) -> None:
    env_path = tmp_path / "custom.env"
    env_path.write_text(
        "\n".join(
            [
                "# comment",
                "RESEARCH_AGENT_STORAGE_BACKEND=sqlite",
                "RESEARCH_AGENT_QUERY_PLANNER_MODEL='deepseekv4flash'",
                'RESEARCH_AGENT_TEST_SECRET="secret-key"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RESEARCH_AGENT_STORAGE_BACKEND", "inmemory")
    monkeypatch.delenv("RESEARCH_AGENT_QUERY_PLANNER_MODEL", raising=False)
    monkeypatch.delenv("RESEARCH_AGENT_TEST_SECRET", raising=False)

    loaded = load_env_file(env_path)

    assert loaded == env_path
    assert os.getenv("RESEARCH_AGENT_STORAGE_BACKEND") == "inmemory"
    assert os.getenv("RESEARCH_AGENT_QUERY_PLANNER_MODEL") == "deepseekv4flash"
    assert os.getenv("RESEARCH_AGENT_TEST_SECRET") == "secret-key"


def test_load_env_file_can_override_existing(monkeypatch, tmp_path) -> None:
    env_path = tmp_path / "override.env"
    env_path.write_text("RESEARCH_AGENT_STORAGE_BACKEND=sqlite\n", encoding="utf-8")
    monkeypatch.setenv("RESEARCH_AGENT_STORAGE_BACKEND", "inmemory")

    load_env_file(env_path, override=True)

    assert os.getenv("RESEARCH_AGENT_STORAGE_BACKEND") == "sqlite"
