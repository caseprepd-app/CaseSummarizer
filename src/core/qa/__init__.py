"""
Q&A Package for CasePrepd - Unified API for Question Answering.

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
    FlowState,
    QARetriever,
    QuestionAnswer,
    QuestionFlowManager,
    VectorStoreBuilder,
)

__all__ = [
    "AlgorithmRetrievalResult",
    "AnswerGenerator",
    "AnswerMode",
    "BaseRetrievalAlgorithm",
    "ChunkMerger",
    "DefaultQuestion",
    # Default questions management (Session 63c)
    "DefaultQuestionsManager",
    "DocumentChunk",
    "FlowState",
    # Retrieval
    "HybridRetriever",
    "MergedChunk",
    # Core orchestration
    "QAOrchestrator",
    "QAResult",
    "QARetriever",
    "QuestionAnswer",
    "QuestionFlowManager",
    "RetrievedChunk",
    # Vector store
    "VectorStoreBuilder",
    "get_default_questions_manager",
]
