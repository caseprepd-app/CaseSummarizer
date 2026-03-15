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
# Note: title_page_handling is handled separately (3-way dropdown, not bool).
_SETTING_TO_PREPROCESSOR = {
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

    The title_page_handling setting controls TitlePageRemover:
    - "exclude_all": enabled in this pipeline (removes title pages before vocab)
    - "vocab_only": disabled here; workers.py applies removal before chunking
    - "include_all": disabled everywhere

    Args:
        settings: Optional dict of preprocessing settings.
            title_page_handling accepts "exclude_all", "vocab_only", "include_all".
            Other keys are booleans. If None, all preprocessors use defaults.

    Returns:
        Configured PreprocessingPipeline instance
    """
    title_page_remover = TitlePageRemover()

    # title_page_handling controls TitlePageRemover; only enable for "exclude_all"
    if settings:
        handling = settings.get("title_page_handling", "vocab_only")
        title_page_remover.enabled = handling == "exclude_all"
    else:
        title_page_remover.enabled = False  # default: vocab_only (workers.py handles search)

    preprocessors = [
        title_page_remover,
        IndexPageRemover(),
        HeaderFooterRemover(),
        LineNumberRemover(),
        PageBoundaryCleaner(),
        TranscriptCleaner(),
    ]

    # Apply boolean toggle settings (title_page_handling is handled above)
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
