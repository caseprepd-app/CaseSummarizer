"""
Character sanitization module for cleaning extracted text.

This module handles the critical Step 2.5 in the document processing pipeline:
cleaning up problematic characters introduced during extraction and OCR that
would otherwise cause issues with embeddings, cross-encoder, and other models.
"""

from .character_sanitizer import CharacterSanitizer

__all__ = ["CharacterSanitizer"]
