"""
Search Package for CasePrepd - Unified API for Semantic Search.

This is the main entry point for all search functionality. Import everything
search-related from this package:

    from src.core.semantic import (
        # Orchestration
        SemanticOrchestrator, SemanticResult,
        # Vector Store
        VectorStoreBuilder, SemanticRetriever,
        # Retrieval Algorithms
        HybridRetriever, ChunkMerger,
    )

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  src.core.semantic (this package) - Unified Search API         │
    ├─────────────────────────────────────────────────────────────┤
    │  SemanticOrchestrator → SemanticRetriever                     │
    │                                          ↓                  │
    │  src.core.vector_store: VectorStoreBuilder                 │
    │                                          ↓                  │
    │  src.core.retrieval: HybridRetriever (BM25+ + FAISS)       │
    └─────────────────────────────────────────────────────────────┘

Components by layer:
- Orchestration: SemanticOrchestrator, SemanticResult
- Storage: VectorStoreBuilder (creates indexes), SemanticRetriever (queries indexes)
- Retrieval: HybridRetriever, ChunkMerger (BM25+ and FAISS algorithms)
"""

# Core search orchestration
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
from src.core.semantic.default_questions_manager import (
    DefaultQuestion,
    DefaultQuestionsManager,
    get_default_questions_manager,
)
from src.core.semantic.semantic_orchestrator import SemanticOrchestrator, SemanticResult

# Vector store and retrieval (re-exported for unified API)
from src.core.vector_store import (
    SemanticRetriever,
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
    "SemanticOrchestrator",
    "SemanticResult",
    "SemanticRetriever",
    "RetrievedChunk",
    # Vector store
    "VectorStoreBuilder",
    "get_default_questions_manager",
]
