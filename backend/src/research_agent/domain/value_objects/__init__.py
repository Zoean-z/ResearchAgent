"""Domain value objects for paper identity, confidence, and source location."""

from research_agent.domain.value_objects.canonical_key import CanonicalKey
from research_agent.domain.value_objects.confidence_score import ConfidenceScore
from research_agent.domain.value_objects.source_locator import SourceLocator

__all__ = ["CanonicalKey", "ConfidenceScore", "SourceLocator"]
