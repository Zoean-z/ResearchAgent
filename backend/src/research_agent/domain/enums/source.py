"""Source and relation enums for documents, artifacts, and paper links."""

from __future__ import annotations

from enum import StrEnum


class SourceType(StrEnum):
    """Supported source types for session documents."""

    ARXIV = "arxiv"
    PDF = "pdf"


class ArtifactKind(StrEnum):
    """Artifact kinds supported in the scaffolded storage layer."""

    ARXIV_PDF = "arxiv_pdf"
    LOCAL_PDF = "local_pdf"


class RelationType(StrEnum):
    """Canonical relation types between papers."""

    IMPROVES_ON = "improves_on"
    SIMILAR_TO = "similar_to"
    CONFLICTS_WITH = "conflicts_with"
    COMPLEMENTS = "complements"
    USES_SAME_BENCHMARK = "uses_same_benchmark"
    COMPARES_WITH = "compares_with"
