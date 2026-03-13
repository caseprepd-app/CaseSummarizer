"""
Standardized result type for document text extraction.

Every extraction function (FileReaders, OCRProcessor, RawTextExtractor)
returns an ExtractionResult instead of a hand-built dict.  Factory
class methods keep call sites concise:

    return ExtractionResult.success(text, "direct_read", confidence)
    return ExtractionResult.error("File not found")

Backward compatibility: ExtractionResult supports ``result["key"]``
and ``result.get("key", default)`` so existing consumers that index
into extraction results continue to work unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExtractionResult:
    """Immutable result of a text extraction attempt.

    Attributes:
        text: Extracted text (None on failure).
        method: Extraction method name (e.g. 'direct_read', 'ocr').
        confidence: Dictionary-based quality score 0-100.
        status: 'success', 'error', or 'ocr_skipped'.
        error_message: Human-readable error (None on success).
        page_count: Number of pages processed (optional).
        pages: Per-page OCR text keyed by 1-indexed page number (optional).
    """

    text: str | None
    method: str | None
    confidence: int
    status: str
    error_message: str | None = None
    page_count: int | None = None
    pages: dict[int, str] | None = field(default=None, repr=False)

    # -- dict-style access for backward compatibility -----------------

    def __getitem__(self, key: str):
        """Allow ``result['text']`` access."""
        return getattr(self, key)

    def get(self, key: str, default=None):
        """Allow ``result.get('text', '')`` access."""
        return getattr(self, key, default)

    # -- factory helpers ----------------------------------------------

    @classmethod
    def success(
        cls,
        text: str,
        method: str,
        confidence: float | int,
        *,
        page_count: int | None = None,
        pages: dict[int, str] | None = None,
    ) -> ExtractionResult:
        """Build a successful extraction result."""
        return cls(
            text=text,
            method=method,
            confidence=int(confidence),
            status="success",
            page_count=page_count,
            pages=pages,
        )

    @classmethod
    def error(
        cls,
        message: str,
        *,
        page_count: int | None = None,
        text: str | None = None,
        method: str | None = None,
        confidence: int = 0,
        status: str = "error",
    ) -> ExtractionResult:
        """Build a failed extraction result."""
        return cls(
            text=text,
            method=method,
            confidence=confidence,
            status=status,
            error_message=message,
            page_count=page_count,
        )
