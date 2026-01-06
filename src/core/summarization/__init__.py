"""
Summarization Package for LocalScribe - Unified API for Document Summarization.

This is the main entry point for all summarization functionality. Import
everything summarization-related from this package:

    from src.core.summarization import (
        # Core components
        ProgressiveSummarizer,
        # Document-level
        ProgressiveDocumentSummarizer, DocumentSummaryResult,
        # Multi-document
        MultiDocumentOrchestrator, MultiDocumentSummaryResult,
    )

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  src.core.summarization - Unified Summarization API        │
    ├─────────────────────────────────────────────────────────────┤
    │  MultiDocumentOrchestrator (coordinates multiple docs)      │
    │            ↓                                                │
    │  ProgressiveDocumentSummarizer (single doc wrapper)        │
    │            ↓                                                │
    │  ProgressiveSummarizer → UnifiedChunker                    │
    │            ↓                                                │
    │  Ollama Model → Chunk Summaries → Final Summary            │
    └─────────────────────────────────────────────────────────────┘

Map-Reduce Flow:
1. Map Phase: Each document → ProgressiveSummarizer → DocumentSummaryResult
   (chunking → chunk summaries → progressive document summary)

2. Reduce Phase: Document summaries → MetaSummaryGenerator → Final narrative
"""

# Result types
# Core summarization (re-exported from src root for unified API)
from src.progressive_summarizer import ProgressiveSummarizer

# Document summarizers
from .document_summarizer import (
    DocumentSummarizer,
    ProgressiveDocumentSummarizer,
)

# Multi-document orchestration
from .multi_document_orchestrator import MultiDocumentOrchestrator
from .result_types import (
    DocumentSummaryResult,
    MultiDocumentSummaryResult,
)

__all__ = [
    # Document summarizer
    "DocumentSummarizer",
    # Result types
    "DocumentSummaryResult",
    # Multi-document orchestration
    "MultiDocumentOrchestrator",
    "MultiDocumentSummaryResult",
    "ProgressiveDocumentSummarizer",
    # Core summarization engine
    "ProgressiveSummarizer",
]
