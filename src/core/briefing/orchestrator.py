"""
Briefing Orchestrator for Case Briefing Generator.

Coordinates the full pipeline from raw documents to formatted Case Briefing Sheet:
1. CHUNK: Section-aware document splitting
2. MAP: Per-chunk LLM extraction (parallelizable in Phase 4)
3. REDUCE: Aggregate and deduplicate
4. SYNTHESIZE: Generate narrative
5. FORMAT: Produce final output

This is the main entry point for generating Case Briefing Sheets.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.core.ai.ollama_model_manager import OllamaModelManager
from src.logging_config import debug_log

from .aggregator import AggregatedBriefingData, DataAggregator
from .chunker import BriefingChunk, DocumentChunker
from .extractor import ChunkExtraction, ChunkExtractor
from .synthesizer import NarrativeSynthesizer, SynthesisResult


@dataclass
class BriefingResult:
    """
    Complete result of the briefing generation process.

    Contains all data needed to format and display the Case Briefing Sheet,
    plus timing and diagnostic information.

    Attributes:
        aggregated_data: Merged data from all documents
        narrative: Generated "WHAT HAPPENED" text
        success: Whether generation completed successfully
        error_message: Error description if failed
        timing: Dict of timing info per phase (ms)
        chunk_count: Number of chunks processed
        extraction_count: Successful extractions
    """

    aggregated_data: AggregatedBriefingData | None = None
    narrative: SynthesisResult | None = None
    success: bool = True
    error_message: str = ""
    timing: dict = field(default_factory=dict)
    chunk_count: int = 0
    extraction_count: int = 0

    @property
    def total_time_ms(self) -> float:
        """Total processing time in milliseconds."""
        return sum(self.timing.values())

    @property
    def total_time_seconds(self) -> float:
        """Total processing time in seconds."""
        return self.total_time_ms / 1000


# Progress callback signature: (phase: str, current: int, total: int, message: str)
ProgressCallback = Callable[[str, int, int, str], None]


class BriefingOrchestrator:
    """
    Main coordinator for Case Briefing Sheet generation.

    Manages the full Map-Reduce pipeline:
    1. Document chunking (section-aware)
    2. Per-chunk extraction (MAP)
    3. Data aggregation (REDUCE)
    4. Narrative synthesis
    5. Output formatting

    Example:
        orchestrator = BriefingOrchestrator()

        documents = [
            {"filename": "complaint.pdf", "text": "..."},
            {"filename": "answer.pdf", "text": "..."},
        ]

        result = orchestrator.generate_briefing(documents)

        if result.success:
            print(result.narrative.narrative)
            print(result.aggregated_data.case_type)
    """

    def __init__(
        self,
        ollama_manager: OllamaModelManager | None = None,
        target_chunk_chars: int = 1800,
        max_chunk_chars: int = 2500,
    ):
        """
        Initialize the orchestrator.

        Args:
            ollama_manager: Shared OllamaModelManager (creates new if None)
            target_chunk_chars: Target chunk size for splitting
            max_chunk_chars: Maximum chunk size
        """
        self.ollama_manager = ollama_manager or OllamaModelManager()

        # Initialize pipeline components
        self.chunker = DocumentChunker(
            target_chars=target_chunk_chars,
            max_chars=max_chunk_chars,
        )
        self.extractor = ChunkExtractor(ollama_manager=self.ollama_manager)
        self.aggregator = DataAggregator()
        self.synthesizer = NarrativeSynthesizer(ollama_manager=self.ollama_manager)

        debug_log("[BriefingOrchestrator] Initialized pipeline components")

    def generate_briefing(
        self,
        documents: list[dict],
        progress_callback: ProgressCallback | None = None,
    ) -> BriefingResult:
        """
        Generate a complete Case Briefing Sheet from documents.

        This is the main entry point for the briefing generation pipeline.

        Args:
            documents: List of document dicts with keys:
                - filename: Original filename
                - text: Extracted document text
                - doc_type: (optional) Pre-classified type
                - date: (optional) Document date for weighting
            progress_callback: Optional callback for UI updates
                Signature: (phase, current, total, message)

        Returns:
            BriefingResult with all generated data and timing info
        """
        result = BriefingResult()
        start_time = time.time()

        try:
            # Validate input
            if not documents:
                result.success = False
                result.error_message = "No documents provided"
                return result

            valid_docs = [d for d in documents if d.get("text", "").strip()]
            if not valid_docs:
                result.success = False
                result.error_message = "All documents are empty"
                return result

            debug_log(f"[BriefingOrchestrator] Starting briefing for {len(valid_docs)} documents")
            self._notify_progress(progress_callback, "init", 0, 4, "Starting briefing generation...")

            # Phase 1: Chunking
            chunks = self._phase_chunk(valid_docs, progress_callback, result)
            if not chunks:
                result.success = False
                result.error_message = "No chunks generated from documents"
                return result

            # Phase 2: Extraction (MAP)
            extractions = self._phase_extract(chunks, progress_callback, result)
            if not extractions:
                result.success = False
                result.error_message = "No data extracted from chunks"
                return result

            # Phase 3: Aggregation (REDUCE)
            aggregated = self._phase_aggregate(extractions, progress_callback, result)
            result.aggregated_data = aggregated

            # Phase 4: Synthesis
            narrative = self._phase_synthesize(aggregated, progress_callback, result)
            result.narrative = narrative

            # Complete
            total_time = (time.time() - start_time) * 1000
            result.timing["total"] = total_time

            self._notify_progress(
                progress_callback,
                "complete",
                4,
                4,
                f"Briefing complete in {total_time / 1000:.1f}s",
            )

            debug_log(f"[BriefingOrchestrator] Complete: {result.total_time_seconds:.1f}s")
            debug_log(f"[BriefingOrchestrator] Timing breakdown: {result.timing}")

            return result

        except Exception as e:
            result.success = False
            result.error_message = f"Briefing generation failed: {str(e)}"
            debug_log(f"[BriefingOrchestrator] Error: {e}")
            return result

    def _phase_chunk(
        self,
        documents: list[dict],
        callback: ProgressCallback | None,
        result: BriefingResult,
    ) -> list[BriefingChunk]:
        """
        Phase 1: Section-aware document chunking.

        Args:
            documents: List of document dicts
            callback: Progress callback
            result: BriefingResult to update

        Returns:
            List of BriefingChunk objects
        """
        self._notify_progress(callback, "chunking", 1, 4, "Splitting documents into chunks...")

        start = time.time()
        chunks = self.chunker.chunk_documents(documents)
        elapsed = (time.time() - start) * 1000

        result.timing["chunking"] = elapsed
        result.chunk_count = len(chunks)

        debug_log(f"[BriefingOrchestrator] Chunking: {len(chunks)} chunks in {elapsed:.0f}ms")

        return chunks

    def _phase_extract(
        self,
        chunks: list[BriefingChunk],
        callback: ProgressCallback | None,
        result: BriefingResult,
    ) -> list[ChunkExtraction]:
        """
        Phase 2: Per-chunk LLM extraction (MAP phase).

        Args:
            chunks: List of chunks to process
            callback: Progress callback
            result: BriefingResult to update

        Returns:
            List of ChunkExtraction objects
        """
        self._notify_progress(
            callback,
            "extraction",
            2,
            4,
            f"Extracting information from {len(chunks)} chunks...",
        )

        start = time.time()

        # Progress wrapper for chunk-level updates
        def chunk_progress(current: int, total: int):
            self._notify_progress(
                callback,
                "extraction",
                2,
                4,
                f"Processing chunk {current + 1}/{total}...",
            )

        extractions = self.extractor.extract_batch(chunks, progress_callback=chunk_progress)
        elapsed = (time.time() - start) * 1000

        result.timing["extraction"] = elapsed
        result.extraction_count = sum(1 for e in extractions if e.extraction_success)

        debug_log(
            f"[BriefingOrchestrator] Extraction: {result.extraction_count}/{len(chunks)} "
            f"successful in {elapsed:.0f}ms"
        )

        return extractions

    def _phase_aggregate(
        self,
        extractions: list[ChunkExtraction],
        callback: ProgressCallback | None,
        result: BriefingResult,
    ) -> AggregatedBriefingData:
        """
        Phase 3: Merge and deduplicate extracted data (REDUCE phase).

        Args:
            extractions: List of chunk extractions
            callback: Progress callback
            result: BriefingResult to update

        Returns:
            AggregatedBriefingData with merged information
        """
        self._notify_progress(callback, "aggregation", 3, 4, "Merging extracted information...")

        start = time.time()
        aggregated = self.aggregator.aggregate(extractions)
        elapsed = (time.time() - start) * 1000

        result.timing["aggregation"] = elapsed

        debug_log(
            f"[BriefingOrchestrator] Aggregation: "
            f"{len(aggregated.allegations)} allegations, "
            f"{sum(len(v) for v in aggregated.people_by_category.values())} people "
            f"in {elapsed:.0f}ms"
        )

        return aggregated

    def _phase_synthesize(
        self,
        aggregated: AggregatedBriefingData,
        callback: ProgressCallback | None,
        result: BriefingResult,
    ) -> SynthesisResult:
        """
        Phase 4: Generate narrative summary.

        Args:
            aggregated: Merged data from all documents
            callback: Progress callback
            result: BriefingResult to update

        Returns:
            SynthesisResult with narrative text
        """
        self._notify_progress(callback, "synthesis", 4, 4, "Generating narrative summary...")

        start = time.time()
        narrative = self.synthesizer.synthesize(aggregated)
        elapsed = (time.time() - start) * 1000

        result.timing["synthesis"] = elapsed

        debug_log(
            f"[BriefingOrchestrator] Synthesis: {narrative.word_count} words "
            f"({narrative.method}) in {elapsed:.0f}ms"
        )

        return narrative

    def _notify_progress(
        self,
        callback: ProgressCallback | None,
        phase: str,
        current: int,
        total: int,
        message: str,
    ) -> None:
        """Send progress update if callback provided."""
        if callback:
            try:
                callback(phase, current, total, message)
            except Exception as e:
                debug_log(f"[BriefingOrchestrator] Progress callback error: {e}")

    def is_ready(self) -> bool:
        """
        Check if the orchestrator is ready to generate briefings.

        Verifies Ollama connection is available.

        Returns:
            True if ready to process
        """
        return self.ollama_manager.is_model_loaded()

    def get_status(self) -> dict:
        """
        Get current status of the orchestrator.

        Returns:
            Dict with connection and configuration info
        """
        return {
            "ready": self.is_ready(),
            "model": self.ollama_manager.model_name,
            "target_chunk_chars": self.chunker.target_chars,
            "max_chunk_chars": self.chunker.max_chars,
        }
