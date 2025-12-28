"""
Document Service for LocalScribe.

Provides a clean interface for document processing operations.
Wraps the extraction and preprocessing core modules.

Usage:
    from src.services import DocumentService

    service = DocumentService()
    results = service.process_documents(file_paths)
    combined = service.combine_texts(results)
"""

from pathlib import Path
from typing import Callable

from src.config import DEBUG_MODE
from src.logging_config import debug_log
from src.core.extraction import RawTextExtractor
from src.core.preprocessing import PreprocessingPipeline, create_default_pipeline
from src.core.sanitization import CharacterSanitizer


class DocumentService:
    """
    Service layer for document processing.

    Coordinates extraction, sanitization, and preprocessing of documents.
    Provides a simplified interface for the UI layer.
    """

    def __init__(self):
        """Initialize the document service with default components."""
        self.extractor = RawTextExtractor()
        self.sanitizer = CharacterSanitizer()
        self.preprocessor = create_default_pipeline()

    def process_documents(
        self,
        file_paths: list[str],
        progress_callback: Callable[[int, int], None] | None = None
    ) -> list[dict]:
        """
        Process multiple documents through extraction, sanitization, and preprocessing.

        Args:
            file_paths: List of paths to documents
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of result dicts with 'file_path', 'text', 'confidence', etc.
        """
        results = []
        total = len(file_paths)

        for i, file_path in enumerate(file_paths):
            if progress_callback:
                progress_callback(i, total)

            result = self.process_single_document(file_path)
            results.append(result)

            if DEBUG_MODE:
                debug_log(f"[DocumentService] Processed {i + 1}/{total}: {Path(file_path).name}")

        if progress_callback:
            progress_callback(total, total)

        return results

    def process_single_document(self, file_path: str) -> dict:
        """
        Process a single document.

        Args:
            file_path: Path to the document

        Returns:
            Result dict with 'file_path', 'text', 'raw_text', 'confidence', etc.
        """
        # Extract raw text
        extraction_result = self.extractor.extract(file_path)

        raw_text = extraction_result.get('text', '')
        confidence = extraction_result.get('confidence', 0)

        # Sanitize (fix encoding issues, normalize whitespace)
        sanitized_text, sanitize_stats = self.sanitizer.sanitize(raw_text)

        # Preprocess (remove headers/footers, line numbers, etc.)
        preprocessed_text = self.preprocessor.process(sanitized_text)

        return {
            'file_path': file_path,
            'file_name': Path(file_path).name,
            'raw_text': raw_text,
            'text': preprocessed_text,
            'confidence': confidence,
            'word_count': len(preprocessed_text.split()),
            'sanitize_stats': sanitize_stats,
        }

    def combine_texts(
        self,
        results: list[dict],
        separator: str = "\n\n---\n\n"
    ) -> str:
        """
        Combine processed texts from multiple documents.

        Args:
            results: List of result dicts from process_documents
            separator: String to insert between documents

        Returns:
            Combined text string
        """
        texts = [r.get('text', '') for r in results if r.get('text')]
        return separator.join(texts)

    def get_total_word_count(self, results: list[dict]) -> int:
        """
        Get total word count across all processed documents.

        Args:
            results: List of result dicts from process_documents

        Returns:
            Total word count
        """
        return sum(r.get('word_count', 0) for r in results)
