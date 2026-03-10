"""
Summarization Package for CasePrepd.

LLM-based summarization has been removed. Only extractive key sentences
(via K-means clustering on embeddings) remain active.

Previous LLM summarization code moved to src/deprecated/.
"""

from .key_sentences import KeySentence, extract_key_sentences

__all__ = [
    "KeySentence",
    "extract_key_sentences",
]
