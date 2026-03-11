"""
Smart Preprocessing Pipeline Module

Provides text preprocessing for legal documents before downstream processing.
Each preprocessor is a standalone class that can be enabled/disabled independently.

Pipeline Architecture:
- BasePreprocessor: Abstract base class defining the preprocessor interface
- PreprocessingPipeline: Orchestrates multiple preprocessors in sequence
- Individual preprocessors: LineNumberRemover, HeaderFooterRemover, etc.

Usage:
    from src.core.preprocessing import PreprocessingPipeline

    pipeline = PreprocessingPipeline()
    cleaned_text = pipeline.process(raw_text)

    # Or customize which preprocessors to use:
    pipeline = PreprocessingPipeline(preprocessors=[
        LineNumberRemover(),
        HeaderFooterRemover(),
    ])
    cleaned_text = pipeline.process(raw_text)
"""

from src.core.preprocessing.base import BasePreprocessor, PreprocessingPipeline
from src.core.preprocessing.coreference_resolver import CoreferenceResolver
from src.core.preprocessing.header_footer_remover import HeaderFooterRemover
from src.core.preprocessing.index_page_remover import IndexPageRemover
from src.core.preprocessing.line_number_remover import LineNumberRemover
from src.core.preprocessing.page_boundary_cleaner import PageBoundaryCleaner
from src.core.preprocessing.title_page_remover import TitlePageRemover
from src.core.preprocessing.transcript_cleaner import TranscriptCleaner

# Mapping from setting key to preprocessor class name
# Note: Coreference resolution runs in UnifiedChunker before chunking,
# so it only affects search and key excerpts, not vocabulary extraction.
_SETTING_TO_PREPROCESSOR = {
    "preprocess_title_pages": "Title Page Remover",
    "preprocess_index_pages": "Index Page Remover",
    "preprocess_headers_footers": "Header/Footer Remover",
    "preprocess_line_numbers": "Line Number Remover",
    "preprocess_page_boundaries": "Page Boundary Cleaner",
    "preprocess_transcript_artifacts": "Transcript Cleaner",
}


def create_default_pipeline(settings: dict | None = None) -> PreprocessingPipeline:
    """
    Create a preprocessing pipeline with all standard preprocessors.

    Order matters:
    1. TitlePageRemover - Removes cover/title pages first
    2. IndexPageRemover - Removes index/concordance pages (and all after)
    3. HeaderFooterRemover - Removes repetitive headers/footers
    4. LineNumberRemover - Removes line numbers from margins
    5. PageBoundaryCleaner - Cleans collapsed page boundary artifacts
    6. TranscriptCleaner - Removes page numbers, certification, index pages

    Note: Coreference resolution runs in UnifiedChunker (before chunking),
    affecting only search and key excerpts.

    Args:
        settings: Optional dict of preprocessing toggle settings.
            Keys are setting names (e.g., 'preprocess_line_numbers'),
            values are booleans. If None, all preprocessors are enabled.

    Returns:
        Configured PreprocessingPipeline instance
    """
    preprocessors = [
        TitlePageRemover(),
        IndexPageRemover(),
        HeaderFooterRemover(),
        LineNumberRemover(),
        PageBoundaryCleaner(),
        TranscriptCleaner(),
    ]

    # Apply settings toggles if provided
    if settings:
        for setting_key, preprocessor_name in _SETTING_TO_PREPROCESSOR.items():
            if setting_key in settings:
                enabled = bool(settings[setting_key])
                for p in preprocessors:
                    if p.name == preprocessor_name:
                        p.enabled = enabled
                        break

    return PreprocessingPipeline(preprocessors=preprocessors)


__all__ = [
    "BasePreprocessor",
    "CoreferenceResolver",
    "HeaderFooterRemover",
    "IndexPageRemover",
    "LineNumberRemover",
    "PageBoundaryCleaner",
    "PreprocessingPipeline",
    "TitlePageRemover",
    "TranscriptCleaner",
    "create_default_pipeline",
]
