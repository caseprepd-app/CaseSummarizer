"""
Case Briefing Generator Package for LocalScribe.

This package implements LLM-First structured extraction using a Map-Reduce
pattern to generate Case Briefing Sheets for court reporters.

Architecture:
- DocumentChunker: Section-aware document splitting
- ChunkExtractor: Per-chunk LLM extraction with JSON schema
- DataAggregator: Merge and deduplicate extracted data
- NarrativeSynthesizer: Generate "WHAT HAPPENED" narrative
- BriefingOrchestrator: Coordinates the full pipeline
- BriefingFormatter: Format final Case Briefing Sheet output

The Map-Reduce pattern:
1. MAP: Process each chunk in parallel, extracting structured data
2. REDUCE: Aggregate and deduplicate across chunks
3. SYNTHESIZE: Generate narrative from aggregated data
4. FORMAT: Produce the final briefing sheet

This replaces the previous Q&A system (BM25+ retrieval-based) with
direct LLM extraction for better quality structured output.

Usage:
    from src.core.briefing import BriefingOrchestrator, BriefingFormatter

    orchestrator = BriefingOrchestrator()
    result = orchestrator.generate_briefing([
        {"filename": "complaint.pdf", "text": "..."},
        {"filename": "answer.pdf", "text": "..."},
    ])

    formatter = BriefingFormatter()
    formatted = formatter.format(result)
    print(formatted.text)
"""

from .chunker import DocumentChunker, BriefingChunk
from .extractor import ChunkExtractor, ChunkExtraction
from .aggregator import DataAggregator, AggregatedBriefingData, PersonEntry
from .synthesizer import NarrativeSynthesizer, SynthesisResult
from .orchestrator import BriefingOrchestrator, BriefingResult
from .formatter import BriefingFormatter, FormattedBriefing

__all__ = [
    # Phase 1: Chunking and Extraction
    "DocumentChunker",
    "BriefingChunk",
    "ChunkExtractor",
    "ChunkExtraction",
    # Phase 2: Aggregation and Synthesis
    "DataAggregator",
    "AggregatedBriefingData",
    "PersonEntry",
    "NarrativeSynthesizer",
    "SynthesisResult",
    # Phase 3: Orchestration and Formatting
    "BriefingOrchestrator",
    "BriefingResult",
    "BriefingFormatter",
    "FormattedBriefing",
]
