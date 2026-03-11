"""
Unified Chunking Module

Provides recursive sentence chunking with token enforcement for:
- Vocabulary extraction (NER-based)
- Search indexing (FAISS + BM25)

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
