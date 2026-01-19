"""
Smart Preprocessing Pipeline Module

Provides text preprocessing for legal documents before AI summarization.
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
        QAConverter(),
    ])
    cleaned_text = pipeline.process(raw_text)
"""

from src.core.preprocessing.base import BasePreprocessor, PreprocessingPipeline
from src.core.preprocessing.header_footer_remover import HeaderFooterRemover
from src.core.preprocessing.index_page_remover import IndexPageRemover
from src.core.preprocessing.line_number_remover import LineNumberRemover
from src.core.preprocessing.qa_converter import QAConverter
from src.core.preprocessing.title_page_remover import TitlePageRemover
from src.core.preprocessing.transcript_cleaner import TranscriptCleaner


# Default pipeline with all preprocessors
def create_default_pipeline() -> PreprocessingPipeline:
    """
    Create a preprocessing pipeline with all standard preprocessors.

    Order matters:
    1. TitlePageRemover - Removes cover/title pages first
    2. IndexPageRemover - Removes index/concordance pages (and all after)
    3. HeaderFooterRemover - Removes repetitive headers/footers
    4. LineNumberRemover - Removes line numbers from margins
    5. TranscriptCleaner - Removes page numbers, certification, index pages
    6. QAConverter - Converts Q./A. notation to readable format

    Returns:
        Configured PreprocessingPipeline instance
    """
    return PreprocessingPipeline(
        preprocessors=[
            TitlePageRemover(),
            IndexPageRemover(),
            HeaderFooterRemover(),
            LineNumberRemover(),
            TranscriptCleaner(),
            QAConverter(),
        ]
    )


__all__ = [
    "BasePreprocessor",
    "HeaderFooterRemover",
    "IndexPageRemover",
    "LineNumberRemover",
    "PreprocessingPipeline",
    "QAConverter",
    "TitlePageRemover",
    "TranscriptCleaner",
    "create_default_pipeline",
]
