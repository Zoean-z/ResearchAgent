from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import re
import unicodedata

from research_agent.domain.models import Chunk


def clean_text_for_model_input(text: str) -> str:
    """Normalize raw text before it enters model-facing prompts."""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_chars: list[str] = []
    for character in normalized:
        category = unicodedata.category(character)
        if character in {"\n", "\t"}:
            cleaned_chars.append(character)
            continue
        if category in {"Cc", "Cf"}:
            continue
        cleaned_chars.append(character)
    normalized = "".join(cleaned_chars)
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    lines = [line.strip() for line in normalized.split("\n")]
    normalized = "\n".join(line for line in lines if line).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized


@dataclass(frozen=True, slots=True)
class CleanedChunkRecord:
    """Per-chunk cleaning outcome retained for model input and trace debug."""

    chunk_id: str
    page: int | None
    section: str | None
    cleaned_text: str
    quality_flags: tuple[str, ...]
    removed_reason: str | None = None

    @property
    def text_hash(self) -> str:
        return sha256(self.cleaned_text.encode("utf-8")).hexdigest() if self.cleaned_text else ""


@dataclass(frozen=True, slots=True)
class ChunkCleanupReport:
    """Traceable summary of the model-input cleaning pass."""

    chunks_before: int
    chunks_after: int
    removed_duplicate_count: int
    removed_noise_count: int
    low_quality_count: int
    references_removed_count: int
    total_chars_before: int
    total_chars_after: int
    chunks: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class ChunkCleanupBundle:
    """Prepared chunks and trace data ready for extraction prompts."""

    cleaned_chunks: list[Chunk]
    records: tuple[CleanedChunkRecord, ...]
    report: ChunkCleanupReport


class IngestCleaningHelper:
    """Clean and batch raw chunks before they enter extraction prompts."""

    def prepare_clean_chunks(self, chunks: list[Chunk]) -> ChunkCleanupBundle:
        chunk_rows: list[tuple[Chunk, str, list[str]]] = []
        total_chars_before = 0
        for chunk in chunks:
            normalized = clean_text_for_model_input(chunk.text)
            total_chars_before += len(chunk.text)
            chunk_rows.append((chunk, normalized, self.split_clean_lines(normalized)))

        repeated_header_footer_lines = self.detect_repeated_header_footer_lines(chunk_rows)
        kept_chunks: list[Chunk] = []
        records: list[CleanedChunkRecord] = []
        seen_text_hashes: set[str] = set()
        removed_duplicate_count = 0
        removed_noise_count = 0
        low_quality_count = 0
        references_removed_count = 0

        for chunk, normalized_text, lines in chunk_rows:
            cleaned_text, quality_flags, removed_reason = self.clean_chunk_for_model_input(
                chunk=chunk,
                normalized_text=normalized_text,
                lines=lines,
                repeated_header_footer_lines=repeated_header_footer_lines,
            )

            kept = removed_reason is None
            if kept and cleaned_text:
                cleaned_hash = sha256(cleaned_text.encode("utf-8")).hexdigest()
                if cleaned_hash in seen_text_hashes:
                    removed_reason = "duplicate_text"
                    quality_flags = self.append_quality_flag(quality_flags, "duplicate_text")
                    kept = False
                    removed_duplicate_count += 1
                else:
                    seen_text_hashes.add(cleaned_hash)

            if "low_quality" in quality_flags:
                low_quality_count += 1
            if removed_reason == "references_section":
                references_removed_count += 1
            elif removed_reason is not None and removed_reason != "duplicate_text":
                removed_noise_count += 1

            if kept and cleaned_text:
                kept_chunks.append(
                    chunk.model_copy(
                        update={
                            "text": cleaned_text,
                        }
                    )
                )

            records.append(
                CleanedChunkRecord(
                    chunk_id=chunk.id,
                    page=chunk.page,
                    section=chunk.section,
                    cleaned_text=cleaned_text,
                    quality_flags=quality_flags,
                    removed_reason=removed_reason,
                )
            )

        total_chars_after = sum(len(chunk.text) for chunk in kept_chunks)
        report = ChunkCleanupReport(
            chunks_before=len(chunks),
            chunks_after=len(kept_chunks),
            removed_duplicate_count=removed_duplicate_count,
            removed_noise_count=removed_noise_count,
            low_quality_count=low_quality_count,
            references_removed_count=references_removed_count,
            total_chars_before=total_chars_before,
            total_chars_after=total_chars_after,
            chunks=tuple(
                {
                    "chunk_id": record.chunk_id,
                    "page": record.page,
                    "section": record.section,
                    "cleaned_text": record.cleaned_text,
                    "quality_flags": list(record.quality_flags),
                    "removed_reason": record.removed_reason,
                }
                for record in records
            ),
        )
        return ChunkCleanupBundle(cleaned_chunks=kept_chunks, records=tuple(records), report=report)

    def split_clean_lines(self, text: str) -> list[str]:
        return [line.strip() for line in text.splitlines() if line.strip()]

    def detect_repeated_header_footer_lines(self, chunk_rows: list[tuple[Chunk, str, list[str]]]) -> set[str]:
        counts: Counter[str] = Counter()
        for _, normalized_text, lines in chunk_rows:
            if not lines:
                continue
            for candidate_line in {lines[0], lines[-1]}:
                normalized_line = self.normalize_repeated_line(candidate_line)
                if normalized_line and self.looks_like_header_footer_candidate(normalized_line, normalized_text):
                    counts[normalized_line] += 1
        threshold = max(3, len(chunk_rows) // 4)
        return {line for line, count in counts.items() if count >= threshold}

    def normalize_repeated_line(self, text: str) -> str:
        return re.sub(r"\s+", " ", clean_text_for_model_input(text)).strip().lower()

    def looks_like_header_footer_candidate(self, line: str, normalized_text: str) -> bool:
        if not line or len(line) > 120:
            return False
        if re.fullmatch(r"[\d\W_]+", line):
            return True
        if len(line.split()) <= 12 and not line.endswith((".", "。", "!", "?", ":", ";")):
            return True
        if len(normalized_text.splitlines()) <= 4 and len(line) <= 80:
            return True
        return False

    def clean_chunk_for_model_input(
        self,
        *,
        chunk: Chunk,
        normalized_text: str,
        lines: list[str],
        repeated_header_footer_lines: set[str],
    ) -> tuple[str, tuple[str, ...], str | None]:
        quality_flags: list[str] = []
        if not normalized_text:
            return "", tuple(quality_flags), "empty_after_normalization"

        section = (chunk.section or "").strip().lower()
        if self.is_reference_section(section, normalized_text):
            return "", ("references_section",), "references_section"

        removed_header_footer = False
        if repeated_header_footer_lines and lines:
            filtered_lines: list[str] = []
            for line in lines:
                normalized_line = self.normalize_repeated_line(line)
                if normalized_line and normalized_line in repeated_header_footer_lines:
                    removed_header_footer = True
                    continue
                filtered_lines.append(line)
            lines = filtered_lines

        cleaned_text = "\n".join(lines).strip()
        if not cleaned_text:
            flags = ("header_footer_removed",) if removed_header_footer else ()
            return "", flags, "header_footer_only" if removed_header_footer else "empty_after_normalization"

        if removed_header_footer:
            quality_flags.append("header_footer_removed")

        pre_compression_text = cleaned_text
        if self.looks_like_table_or_noise(cleaned_text):
            quality_flags.append("table_or_noise")
            cleaned_text = self.compress_table_like_text(cleaned_text)

        if self.looks_like_low_quality_text(pre_compression_text):
            quality_flags.append("low_quality")

        if self.looks_like_too_short(cleaned_text):
            quality_flags = list(dict.fromkeys(quality_flags))
            return cleaned_text, tuple(quality_flags), "too_short"

        deduped_flags = tuple(dict.fromkeys(quality_flags))
        return cleaned_text, deduped_flags, None

    def is_reference_section(self, section: str, text: str) -> bool:
        lowered = f"{section}\n{text}".lower()
        return any(marker in lowered for marker in ("references", "bibliography", "reference"))

    def looks_like_table_or_noise(self, text: str) -> bool:
        digits = sum(character.isdigit() for character in text)
        letters = sum(character.isalpha() for character in text)
        symbols = sum(not character.isalnum() and not character.isspace() for character in text)
        if digits >= 12 and digits > max(letters // 3, 1):
            return True
        if len(text.splitlines()) <= 4 and digits >= 6 and symbols >= 6:
            return True
        if symbols >= 20 and digits >= 6:
            return True
        return False

    def compress_table_like_text(self, text: str) -> str:
        line_count = max(1, len([line for line in text.splitlines() if line.strip()]))
        digits = sum(character.isdigit() for character in text)
        symbols = sum(not character.isalnum() and not character.isspace() for character in text)
        excerpt = re.sub(r"\s+", " ", text).strip()
        if len(excerpt) > 320:
            excerpt = f"{excerpt[:320].rstrip()} ..."
        return (
            f"[table-like content compressed: chars={len(text)}, lines={line_count}, "
            f"digits={digits}, symbols={symbols}]\n{excerpt}"
        )

    def looks_like_low_quality_text(self, text: str) -> bool:
        if "�" in text:
            return True
        total_chars = sum(1 for character in text if not character.isspace())
        if total_chars == 0:
            return True
        non_alnum = sum(1 for character in text if not character.isalnum() and not character.isspace())
        if len(text) >= 24 and non_alnum / total_chars > 0.45:
            return True
        if sum(character.isalpha() for character in text) == 0 and sum(character.isdigit() for character in text) >= 4:
            return True
        return False

    def looks_like_too_short(self, text: str) -> bool:
        stripped = text.strip()
        if len(stripped) < 12:
            return True
        if len(stripped.split()) < 3 and len(stripped) < 40 and not self.contains_cjk(stripped):
            return True
        return False

    def append_quality_flag(self, flags: tuple[str, ...], flag: str) -> tuple[str, ...]:
        if flag in flags:
            return flags
        return (*flags, flag)

    def cleanup_report_payload(self, report: ChunkCleanupReport) -> dict[str, object]:
        return {
            "chunks_before": report.chunks_before,
            "chunks_after": report.chunks_after,
            "removed_duplicate_count": report.removed_duplicate_count,
            "removed_noise_count": report.removed_noise_count,
            "low_quality_count": report.low_quality_count,
            "references_removed_count": report.references_removed_count,
            "total_chars_before": report.total_chars_before,
            "total_chars_after": report.total_chars_after,
            "chunks": list(report.chunks),
        }

    def merge_short_chunk_groups(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        groups: list[list[Chunk]] = []
        current: list[Chunk] = []
        for chunk in chunks:
            if not current:
                current = [chunk]
                continue
            previous = current[-1]
            if self.should_merge_chunks(previous, chunk):
                current.append(chunk)
                continue
            groups.append(current)
            current = [chunk]
        if current:
            groups.append(current)
        return groups

    def should_merge_chunks(self, previous: Chunk, current: Chunk) -> bool:
        if previous.page != current.page:
            return False
        previous_section = (previous.section or "").strip().lower()
        current_section = (current.section or "").strip().lower()
        if previous_section != current_section:
            return False
        if self.classify_chunk_role_basic(previous, previous.text) in {"title", "abstract"}:
            return False
        if len(previous.text.strip()) >= 160 and len(current.text.strip()) >= 160:
            return False
        return len(previous.text.strip()) < 120 or len(current.text.strip()) < 120

    def classify_chunk_role_basic(
        self,
        chunk: Chunk,
        text: str | None = None,
    ) -> str:
        section = (chunk.section or "").lower()
        text_lower = (text if text is not None else chunk.text).lower()
        if "title" in section:
            return "title"
        if "abstract" in section or text_lower.startswith("abstract"):
            return "abstract"
        if any(token in section or token in text_lower for token in ("appendix", "appendices", "supplement", "supplementary")):
            return "appendix"
        if any(token in section or token in text_lower for token in ("reference", "bibliography")):
            return "reference"
        if "table" in text_lower or (len(text_lower.split()) < 18 and sum(character.isdigit() for character in text_lower) >= 4):
            return "table"
        return "main"

    def should_use_full_text(self, chunks: list[Chunk]) -> bool:
        total_chars = sum(len(chunk.text) for chunk in chunks)
        return total_chars <= 18000 and len(chunks) <= 80

    def split_clean_chunks_into_batches(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        batches: list[list[Chunk]] = []
        current: list[Chunk] = []
        current_chars = 0
        max_batch_chars = 9000
        max_batch_chunks = 10
        for chunk in chunks:
            chunk_size = len(chunk.text)
            if current and (current_chars + chunk_size > max_batch_chars or len(current) >= max_batch_chunks):
                batches.append(current)
                current = []
                current_chars = 0
            current.append(chunk)
            current_chars += chunk_size
        if current:
            batches.append(current)
        return batches

    def merge_context_chunks(self, batches: list[list[Chunk]]) -> list[Chunk]:
        context_chunks: list[Chunk] = []
        for batch in batches:
            if not batch:
                continue
            context_chunks.append(batch[0])
            if len(context_chunks) >= 10:
                break
        return context_chunks
