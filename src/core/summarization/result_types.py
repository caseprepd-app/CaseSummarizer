"""
Result Types for Multi-Document Summarization

This module defines the data structures used to pass summarization results
through the multi-document pipeline. These are simple dataclasses that hold
the output of each processing stage.

Key Types:
    DocumentSummaryResult - Result from summarizing a single document
    MultiDocumentSummaryResult - Result from summarizing multiple documents

Usage:
    result = DocumentSummaryResult(
        filename="complaint.pdf",
        summary="The plaintiff alleges...",
        word_count=187,
        chunk_count=12,
        processing_time_seconds=45.2
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DocumentSummaryResult:
    """
    Result from summarizing a single document through progressive chunking.

    Attributes:
        filename: Original filename (e.g., "complaint.pdf").
        summary: Final progressive summary of the document.
        word_count: Number of words in the summary.
        chunk_count: Number of chunks the document was split into.
        processing_time_seconds: Time taken to process this document.
        success: Whether summarization completed successfully.
        error_message: Error description if success is False.
    """
    filename: str
    summary: str
    word_count: int
    chunk_count: int
    processing_time_seconds: float
    success: bool = True
    error_message: str | None = None

    def __post_init__(self):
        """Validate that failed results have an error message."""
        if not self.success and not self.error_message:
            self.error_message = "Unknown error during document summarization"


@dataclass
class MultiDocumentSummaryResult:
    """
    Result from hierarchical multi-document summarization.

    Contains both individual document summaries (map phase) and
    the combined meta-summary (reduce phase).

    Attributes:
        individual_summaries: Dict mapping filename to DocumentSummaryResult.
        meta_summary: Combined summary synthesizing all documents.
        total_processing_time_seconds: Total wall-clock time for all processing.
        documents_processed: Count of successfully processed documents.
        documents_failed: Count of documents that failed processing.
        document_order: List of filenames in processing order (for display).
    """
    individual_summaries: dict[str, DocumentSummaryResult] = field(default_factory=dict)
    meta_summary: str = ""
    total_processing_time_seconds: float = 0.0
    documents_processed: int = 0
    documents_failed: int = 0
    document_order: list[str] = field(default_factory=list)

    def get_summary_for_document(self, filename: str) -> str | None:
        """Get the summary for a specific document by filename."""
        result = self.individual_summaries.get(filename)
        return result.summary if result else None

    def get_all_summaries_formatted(self) -> str:
        """
        Get all individual summaries formatted for display.

        Returns summaries in document_order with headers.
        """
        parts = []
        for filename in self.document_order:
            result = self.individual_summaries.get(filename)
            if result and result.success:
                parts.append(f"--- {filename} ---\n{result.summary}")
            elif result:
                parts.append(f"--- {filename} ---\n[Error: {result.error_message}]")
        return "\n\n".join(parts)

    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        total = self.documents_processed + self.documents_failed
        if total == 0:
            return 0.0
        return (self.documents_processed / total) * 100
