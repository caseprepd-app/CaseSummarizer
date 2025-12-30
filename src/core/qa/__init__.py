"""
Q&A Package for LocalScribe - Unified API for Question Answering.

This is the main entry point for all Q&A functionality. Import everything
Q&A-related from this package:

    from src.core.qa import (
        # Orchestration
        QAOrchestrator, QAResult, AnswerGenerator, AnswerMode,
        # Vector Store
        VectorStoreBuilder, QARetriever, QuestionFlowManager,
        # Retrieval Algorithms
        HybridRetriever, ChunkMerger,
    )

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  src.core.qa (this package) - Unified Q&A API               │
    ├─────────────────────────────────────────────────────────────┤
    │  QAOrchestrator → AnswerGenerator → QARetriever            │
    │                                          ↓                  │
    │  src.core.vector_store: VectorStoreBuilder, QuestionFlow   │
    │                                          ↓                  │
    │  src.core.retrieval: HybridRetriever (BM25+ + FAISS)       │
    └─────────────────────────────────────────────────────────────┘

Components by layer:
- Orchestration: QAOrchestrator, AnswerGenerator, AnswerMode, QAResult
- Storage: VectorStoreBuilder (creates indexes), QARetriever (queries indexes)
- Questions: QuestionFlowManager (branching question trees)
- Retrieval: HybridRetriever, ChunkMerger (BM25+ and FAISS algorithms)
"""

# Core Q&A orchestration
from src.core.qa.answer_generator import AnswerGenerator, AnswerMode
from src.core.qa.qa_orchestrator import QAOrchestrator, QAResult
from src.core.qa.default_questions_manager import (
    DefaultQuestionsManager,
    DefaultQuestion,
    get_default_questions_manager,
)

# Vector store and retrieval (re-exported for unified API)
from src.core.vector_store import (
    VectorStoreBuilder,
    QARetriever,
    QuestionFlowManager,
    QuestionAnswer,
    FlowState,
)

# Hybrid retrieval (re-exported for unified API)
from src.core.retrieval import (
    HybridRetriever,
    ChunkMerger,
    MergedChunk,
    BaseRetrievalAlgorithm,
    RetrievedChunk,
    AlgorithmRetrievalResult,
    DocumentChunk,
)

__all__ = [
    # Core orchestration
    "QAOrchestrator",
    "QAResult",
    "AnswerGenerator",
    "AnswerMode",
    # Default questions management (Session 63c)
    "DefaultQuestionsManager",
    "DefaultQuestion",
    "get_default_questions_manager",
    # Vector store
    "VectorStoreBuilder",
    "QARetriever",
    "QuestionFlowManager",
    "QuestionAnswer",
    "FlowState",
    # Retrieval
    "HybridRetriever",
    "ChunkMerger",
    "MergedChunk",
    "BaseRetrievalAlgorithm",
    "RetrievedChunk",
    "AlgorithmRetrievalResult",
    "DocumentChunk",
]
