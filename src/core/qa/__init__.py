"""
Search Package for CasePrepd - Unified API for Semantic Search.

This is the main entry point for all search functionality. Import everything
search-related from this package:

    from src.core.qa import (
        # Orchestration
        QAOrchestrator, QAResult,
        # Vector Store
        VectorStoreBuilder, QARetriever,
        # Retrieval Algorithms
        HybridRetriever, ChunkMerger,
    )

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  src.core.qa (this package) - Unified Search API              │
    ├─────────────────────────────────────────────────────────────┤
    │  QAOrchestrator → QARetriever                              │
    │                                          ↓                  │
    │  src.core.vector_store: VectorStoreBuilder                 │
    │                                          ↓                  │
    │  src.core.retrieval: HybridRetriever (BM25+ + FAISS)       │
    └─────────────────────────────────────────────────────────────┘

Components by layer:
- Orchestration: QAOrchestrator, QAResult
- Storage: VectorStoreBuilder (creates indexes), QARetriever (queries indexes)
- Retrieval: HybridRetriever, ChunkMerger (BM25+ and FAISS algorithms)
"""

# Core search orchestration
from src.core.qa.default_questions_manager import (
    DefaultQuestion,
    DefaultQuestionsManager,
    get_default_questions_manager,
)
from src.core.qa.qa_orchestrator import QAOrchestrator, QAResult

# Hybrid retrieval (re-exported for unified API)
from src.core.retrieval import (
    AlgorithmRetrievalResult,
    BaseRetrievalAlgorithm,
    ChunkMerger,
    DocumentChunk,
    HybridRetriever,
    MergedChunk,
    RetrievedChunk,
)

# Vector store and retrieval (re-exported for unified API)
from src.core.vector_store import (
    QARetriever,
    VectorStoreBuilder,
)

__all__ = [
    "AlgorithmRetrievalResult",
    "BaseRetrievalAlgorithm",
    "ChunkMerger",
    "DefaultQuestion",
    "DefaultQuestionsManager",
    "DocumentChunk",
    # Retrieval
    "HybridRetriever",
    "MergedChunk",
    # Core orchestration
    "QAOrchestrator",
    "QAResult",
    "QARetriever",
    "RetrievedChunk",
    # Vector store
    "VectorStoreBuilder",
    "get_default_questions_manager",
]
