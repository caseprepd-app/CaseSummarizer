"""
Extraction Package

This package handles document processing:
- Step 1: Extract raw text from files (PDF/TXT/RTF)
- Step 2: Apply basic normalization (de-hyphenation, page removal, etc.)
"""

from src.core.extraction.raw_text_extractor import RawTextExtractor

__all__ = [
    "RawTextExtractor",
]
