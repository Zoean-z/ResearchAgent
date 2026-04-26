"""Thin ingest materialization service for placeholder source bindings."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from uuid import uuid4

from pypdf import PdfReader

from research_agent.domain.enums import ArtifactKind, SourceType
from research_agent.domain.models import Artifact, Chunk, Paper, SessionDocument
from research_agent.domain.ports import ArtifactRepositoryPort, ChunkRepositoryPort, PaperRepositoryPort, SessionRepositoryPort
from research_agent.domain.policies import build_canonical_key, normalize_arxiv_id
from research_agent.domain.value_objects import CanonicalKey
from research_agent.tools.registry import InternalToolRegistry
from research_agent.services.errors import EntityNotFoundError, InvalidIngestSourceError


_ARXIV_ID_PATTERN = re.compile(r"(?:abs|pdf)/(?P<arxiv_id>[^?#/]+)")


@dataclass(frozen=True, slots=True)
class IngestMaterializationResult:
    """Structured source bindings created during ingest materialization."""

    paper: Paper
    artifact: Artifact
    session_document: SessionDocument
    operation: str
    chunk_count: int


class _NullChunkRepository(ChunkRepositoryPort):
    """Fallback chunk repository used when a caller has not wired one yet."""

    def save_many(self, chunks: list[Chunk]) -> list[Chunk]:
        return chunks

    def list_by_paper_ids(self, paper_ids: list[str]) -> list[Chunk]:
        return []


class IngestMaterializationService:
    """Create durable source bindings from arXiv or local PDF sources."""

    def __init__(
        self,
        session_repository: SessionRepositoryPort,
        paper_repository: PaperRepositoryPort,
        artifact_repository: ArtifactRepositoryPort,
        chunk_repository: ChunkRepositoryPort | None = None,
        tool_registry: InternalToolRegistry | None = None,
    ) -> None:
        self._session_repository = session_repository
        self._paper_repository = paper_repository
        self._artifact_repository = artifact_repository
        self._chunk_repository = chunk_repository or _NullChunkRepository()
        self._tool_registry = tool_registry

    def materialize_arxiv_source(self, session_id: str, arxiv_url: str) -> IngestMaterializationResult:
        """Download and materialize an arXiv PDF source."""

        self._require_session(session_id)
        arxiv_id = self._extract_arxiv_id(arxiv_url)
        pdf_url = self._build_arxiv_pdf_url(arxiv_id)
        pdf_bytes = self._download_arxiv_pdf(pdf_url, arxiv_url)
        checksum = sha256(pdf_bytes).hexdigest()
        canonical_key = build_canonical_key(arxiv_id=arxiv_id)
        reader = self._load_pdf_reader_from_bytes(pdf_bytes, arxiv_url)
        page_count = len(reader.pages)
        artifact = self._artifact_repository.save(
            Artifact(
                id=str(uuid4()),
                kind=ArtifactKind.ARXIV_PDF,
                uri_or_path=arxiv_url,
                checksum=checksum,
                page_count=page_count,
            )
        )
        title, authors = self._extract_pdf_metadata(reader, f"arXiv paper {arxiv_id}")
        paper, operation = self.register_paper(
            canonical_key=canonical_key,
            title=title,
            arxiv_id=arxiv_id,
            pdf_fingerprint=checksum,
            authors=authors,
        )
        chunks = self._extract_pdf_chunks(reader, paper.id, artifact.id)
        self._chunk_repository.save_many(chunks)
        session_document = self._session_repository.save_document(
            SessionDocument(
                session_id=session_id,
                paper_id=paper.id,
                source_type=SourceType.ARXIV,
                artifact_id=artifact.id,
            )
        )
        return IngestMaterializationResult(
            paper=paper,
            artifact=artifact,
            session_document=session_document,
            operation=operation,
            chunk_count=len(chunks),
        )

    def materialize_pdf_source(self, session_id: str, file_path: str) -> IngestMaterializationResult:
        """Create placeholder records for a local PDF ingest source."""

        self._require_session(session_id)
        pdf_path = Path(file_path).expanduser()
        if not pdf_path.exists():
            raise InvalidIngestSourceError(SourceType.PDF.value, file_path, "file does not exist")

        checksum = sha256(pdf_path.read_bytes()).hexdigest()
        canonical_key = build_canonical_key(pdf_checksum=checksum)
        reader = self._load_pdf_reader(pdf_path, file_path)
        page_count = len(reader.pages)
        artifact = self._artifact_repository.save(
            Artifact(
                id=str(uuid4()),
                kind=ArtifactKind.LOCAL_PDF,
                uri_or_path=file_path,
                checksum=checksum,
                page_count=page_count,
            )
        )
        title, authors = self._extract_pdf_metadata(reader, f"local PDF {pdf_path.stem or 'source'}")
        paper, operation = self.register_paper(
            canonical_key=canonical_key,
            title=title,
            pdf_fingerprint=checksum,
            authors=authors,
        )
        chunks = self._extract_pdf_chunks(reader, paper.id, artifact.id)
        self._chunk_repository.save_many(chunks)
        session_document = self._session_repository.save_document(
            SessionDocument(
                session_id=session_id,
                paper_id=paper.id,
                source_type=SourceType.PDF,
                artifact_id=artifact.id,
            )
        )
        return IngestMaterializationResult(
            paper=paper,
            artifact=artifact,
            session_document=session_document,
            operation=operation,
            chunk_count=len(chunks),
        )

    def register_paper(
        self,
        *,
        canonical_key: CanonicalKey,
        title: str,
        authors: list[str] | None = None,
        arxiv_id: str | None = None,
        pdf_fingerprint: str | None = None,
    ) -> tuple[Paper, str]:
        """Register or match a canonical paper record."""

        if self._tool_registry is not None:
            result = self._tool_registry.register_paper(
                title=title,
                authors=authors,
                arxiv_id=arxiv_id,
                pdf_fingerprint=pdf_fingerprint,
            )
            return result.paper, result.operation
        return self._register_paper(
            canonical_key=canonical_key,
            title=title,
            authors=authors,
            arxiv_id=arxiv_id,
            pdf_fingerprint=pdf_fingerprint,
        )

    def _register_paper(
        self,
        *,
        canonical_key: CanonicalKey,
        title: str,
        authors: list[str] | None = None,
        arxiv_id: str | None = None,
        pdf_fingerprint: str | None = None,
    ) -> tuple[Paper, str]:
        existing = self._paper_repository.get_by_canonical_key(canonical_key)
        if existing is not None:
            updated = existing.model_copy(
                update={
                    "authors": existing.authors or (authors or existing.authors),
                    "arxiv_id": existing.arxiv_id or arxiv_id,
                    "pdf_fingerprint": existing.pdf_fingerprint or pdf_fingerprint,
                }
            )
            return self._paper_repository.save(updated), "matched"

        paper = self._paper_repository.save(
            Paper(
                id=str(uuid4()),
                canonical_key=canonical_key,
                title=title,
                authors=authors or [],
                arxiv_id=arxiv_id,
                pdf_fingerprint=pdf_fingerprint,
            )
        )
        return paper, "created"

    def _extract_arxiv_id(self, arxiv_url: str) -> str:
        match = _ARXIV_ID_PATTERN.search(arxiv_url)
        if match is None:
            parsed = urlparse(arxiv_url)
            candidate = Path(parsed.path).name or parsed.path.rsplit("/", 1)[-1]
            arxiv_id = candidate or sha256(arxiv_url.strip().encode("utf-8")).hexdigest()[:12]
        else:
            arxiv_id = match.group("arxiv_id")
        return normalize_arxiv_id(arxiv_id.removesuffix(".pdf"))

    def _build_arxiv_pdf_url(self, arxiv_id: str) -> str:
        normalized_id = arxiv_id.strip().removesuffix(".pdf")
        return f"https://arxiv.org/pdf/{normalized_id}.pdf"

    def _download_arxiv_pdf(self, pdf_url: str, source_value: str) -> bytes:
        request = Request(
            pdf_url,
            headers={
                "User-Agent": "research-agent/1.0 (+https://arxiv.org)",
                "Accept": "application/pdf",
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = response.read()
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            raise InvalidIngestSourceError(SourceType.ARXIV.value, source_value, str(exc)) from exc
        if not payload:
            raise InvalidIngestSourceError(SourceType.ARXIV.value, source_value, "empty PDF response")
        return payload

    def _require_session(self, session_id: str) -> None:
        if self._session_repository.get_by_id(session_id) is None:
            raise EntityNotFoundError("Session", session_id)

    def _load_pdf_reader(self, pdf_path: Path, source_value: str) -> PdfReader:
        try:
            return PdfReader(str(pdf_path))
        except Exception as exc:  # pragma: no cover - narrow error mapping not critical here
            raise InvalidIngestSourceError(SourceType.PDF.value, source_value, str(exc)) from exc

    def _load_pdf_reader_from_bytes(self, pdf_bytes: bytes, source_value: str) -> PdfReader:
        try:
            return PdfReader(BytesIO(pdf_bytes))
        except Exception as exc:  # pragma: no cover - narrow error mapping not critical here
            raise InvalidIngestSourceError(SourceType.ARXIV.value, source_value, str(exc)) from exc

    def _clean_pdf_page_text(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        lines: list[str] = []
        for raw_line in normalized.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"\s*\d{3}$", "", line).strip()
            if line:
                lines.append(line)

        if not lines:
            return ""

        cleaned = lines[0]
        for line in lines[1:]:
            if cleaned.endswith("-"):
                if line[:1].islower():
                    cleaned = cleaned[:-1] + line
                else:
                    cleaned = f"{cleaned}{line}"
            else:
                cleaned = f"{cleaned} {line}"

        cleaned = re.sub(r"(?<!\d)\b\d{3}\b", "", cleaned)
        cleaned = re.sub(r"(?<=\w)-\s+(?=[a-z])", "", cleaned)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _extract_pdf_metadata(self, reader: PdfReader, fallback_label: str) -> tuple[str, list[str]]:
        metadata = reader.metadata
        title = None
        authors: list[str] = []
        if metadata is not None:
            raw_title = getattr(metadata, "title", None) or getattr(metadata, "/Title", None)
            raw_author = getattr(metadata, "author", None) or getattr(metadata, "/Author", None)
            if isinstance(raw_title, str) and raw_title.strip():
                title = raw_title.strip()
            if isinstance(raw_author, str) and raw_author.strip():
                authors = [part.strip() for part in re.split(r"[;,]", raw_author) if part.strip()]
        if title is None:
            title = f"Imported {fallback_label}"
        return title, authors

    def _extract_pdf_chunks(self, reader: PdfReader, paper_id: str, artifact_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = self._clean_pdf_page_text(page.extract_text() or "")
            for chunk_text in self._split_text_into_chunks(text):
                chunks.append(
                    Chunk(
                        id=str(uuid4()),
                        paper_id=paper_id,
                        artifact_id=artifact_id,
                        text=chunk_text,
                        page=page_number,
                        section=f"page-{page_number}",
                    )
                )
        return chunks

    def _split_text_into_chunks(self, text: str, max_chars: int = 1200) -> list[str]:
        normalized = "\n\n".join(
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
            if paragraph.strip()
        )
        if not normalized.strip():
            return []

        chunks: list[str] = []
        current = ""
        for paragraph in normalized.split("\n\n"):
            candidate = paragraph if not current else f"{current}\n\n{paragraph}"
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = paragraph
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks
