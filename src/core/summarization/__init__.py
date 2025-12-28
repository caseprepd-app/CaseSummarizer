"""
Summarization Package for LocalScribe - Unified API for Document Summarization.

This is the main entry point for all summarization functionality. Import
everything summarization-related from this package:

    from src.core.summarization import (
        # Core components
        ProgressiveSummarizer, ChunkingEngine,
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
    │  ProgressiveSummarizer → ChunkingEngine                    │
    │            ↓                                                │
    │  Ollama Model → Chunk Summaries → Final Summary            │
    └─────────────────────────────────────────────────────────────┘

Map-Reduce Flow:
1. Map Phase: Each document → ProgressiveSummarizer → DocumentSummaryResult
   (chunking → chunk summaries → progressive document summary)

2. Reduce Phase: Document summaries → MetaSummaryGenerator → Final narrative
"""

# Result types
from .result_types import (
    DocumentSummaryResult,
    MultiDocumentSummaryResult,
)

# Document summarizers
from .document_summarizer import (
    DocumentSummarizer,
    ProgressiveDocumentSummarizer,
)

# Multi-document orchestration
from .multi_document_orchestrator import MultiDocumentOrchestrator

# Core summarization (re-exported from src root for unified API)
from src.progressive_summarizer import ProgressiveSummarizer
from src.chunking_engine import Chunk, ChunkingEngine

__all__ = [
    # Core summarization engine
    'ProgressiveSummarizer',
    'ChunkingEngine',
    'Chunk',
    # Result types
    'DocumentSummaryResult',
    'MultiDocumentSummaryResult',
    # Document summarizer
    'DocumentSummarizer',
    'ProgressiveDocumentSummarizer',
    # Multi-document orchestration
    'MultiDocumentOrchestrator',
]
