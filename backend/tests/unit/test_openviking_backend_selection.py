"""Tests for OpenViking backend selection."""

from __future__ import annotations

from research_agent.adapters.openviking import OpenVikingAdapterSurfaceBundle
from research_agent.api.deps import _create_openviking_surface_bundle


def test_create_openviking_surface_bundle_selects_embedded_backend(monkeypatch) -> None:
    captured: dict[str, str] = {}
    expected_bundle = OpenVikingAdapterSurfaceBundle()

    def fake_build_embedded_openviking_surface_bundle(config) -> OpenVikingAdapterSurfaceBundle:  # noqa: ANN001
        captured["path"] = config.path
        return expected_bundle

    monkeypatch.setenv("RESEARCH_AGENT_OPENVIKING_BACKEND", "embedded")
    monkeypatch.setenv("RESEARCH_AGENT_OPENVIKING_DATA_PATH", "D:/py/research-agent/data/openviking")
    monkeypatch.setattr(
        "research_agent.api.deps.build_embedded_openviking_surface_bundle",
        fake_build_embedded_openviking_surface_bundle,
    )

    result = _create_openviking_surface_bundle()

    assert result is expected_bundle
    assert captured["path"] == "D:/py/research-agent/data/openviking"


def test_create_openviking_surface_bundle_uses_embedded_backend_branch(monkeypatch) -> None:
    expected_bundle = OpenVikingAdapterSurfaceBundle()
    captured: dict[str, str] = {}

    def fake_build_embedded_openviking_surface_bundle(config) -> OpenVikingAdapterSurfaceBundle:  # noqa: ANN001
        captured["path"] = config.path
        return expected_bundle

    monkeypatch.setenv("RESEARCH_AGENT_OPENVIKING_BACKEND", "embedded")
    monkeypatch.setenv("RESEARCH_AGENT_OPENVIKING_DATA_PATH", "D:/py/research-agent/data/openviking")
    monkeypatch.setattr(
        "research_agent.api.deps.build_embedded_openviking_surface_bundle",
        fake_build_embedded_openviking_surface_bundle,
    )

    result = _create_openviking_surface_bundle()

    assert result is expected_bundle
    assert captured["path"] == "D:/py/research-agent/data/openviking"
