"""
Unified Chunking Module (Session 45)

Provides semantic chunking with token enforcement for:
- LLM extraction (names + vocabulary)
- Q&A indexing (FAISS + BM25)

Single chunking pass serves all downstream consumers.
"""

from src.core.chunking.unified_chunker import (
    UnifiedChunk,
    UnifiedChunker,
    create_unified_chunker,
)

__all__ = [
    "UnifiedChunk",
    "UnifiedChunker",
    "create_unified_chunker",
]
