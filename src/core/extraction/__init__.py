"""
Extraction Package (Session 45 Update)

This package handles document processing and vocabulary extraction:
- Step 1: Extract raw text from files (PDF/TXT/RTF)
- Step 2: Apply basic normalization (de-hyphenation, page removal, etc.)
- LLM Extraction: Combined people + vocabulary extraction using Ollama

Session 45 Updates:
- LLMPerson dataclass for people/organizations with roles
- Combined prompt extracts both people and vocabulary in one pass
- Support for UnifiedChunk objects from unified chunker
"""

from src.core.extraction.raw_text_extractor import RawTextExtractor
from src.core.extraction.llm_extractor import (
    LLMVocabExtractor,
    LLMPerson,
    LLMTerm,
    LLMExtractionResult,
)

__all__ = [
    "RawTextExtractor",
    "LLMVocabExtractor",
    "LLMPerson",
    "LLMTerm",
    "LLMExtractionResult",
]
