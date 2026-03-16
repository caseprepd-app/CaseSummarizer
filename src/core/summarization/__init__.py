"""
Summarization Package for CasePrepd.

Extractive key passages via K-means clustering on pre-computed embeddings.
"""

from .key_sentences import KeySentence, extract_key_passages

__all__ = [
    "KeySentence",
    "extract_key_passages",
]
