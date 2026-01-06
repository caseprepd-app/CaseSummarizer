"""
Multi-Document Orchestrator - Hierarchical Map-Reduce Summarization

This module coordinates the summarization of multiple documents using a
hierarchical approach:

1. Map Phase: Each document is summarized in parallel using ProgressiveDocumentSummarizer
2. Reduce Phase: Individual summaries are combined into a meta-summary

The orchestrator uses the existing parallel infrastructure (ThreadPoolStrategy,
ParallelTaskRunner, ProgressAggregator) to maximize throughput while maintaining
responsive UI feedback.

Architecture:
    MultiDocumentOrchestrator
        ├── Map Phase: ParallelTaskRunner + ProgressiveDocumentSummarizer
        └── Reduce Phase: Meta-summary generation from individual summaries

Usage:
    from src.core.summarization import (
        MultiDocumentOrchestrator,
        ProgressiveDocumentSummarizer,
    )
    from src.core.ai import OllamaModelManager

    model_manager = OllamaModelManager()
    doc_summarizer = ProgressiveDocumentSummarizer(model_manager)

    orchestrator = MultiDocumentOrchestrator(
        document_summarizer=doc_summarizer,
        model_manager=model_manager
    )

    result = orchestrator.summarize_documents(
        documents=[{'filename': 'doc1.pdf', 'extracted_text': '...'}],
        max_words_per_document=200,
        max_meta_summary_words=500
    )
    print(result.meta_summary)
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from src.config import OLLAMA_CONTEXT_WINDOW, PARALLEL_MAX_WORKERS
from src.core.parallel import (
    ExecutorStrategy,
    ParallelTaskRunner,
    ProgressAggregator,
    ThreadPoolStrategy,
)
from src.logging_config import debug_log, error, info

from .document_summarizer import DocumentSummarizer
from .result_types import DocumentSummaryResult, MultiDocumentSummaryResult

if TYPE_CHECKING:
    from queue import Queue

    from src.core.ai.ollama_model_manager import OllamaModelManager
    from src.core.prompting import PromptAdapter


class MultiDocumentOrchestrator:
    """
    Orchestrates hierarchical multi-document summarization.

    Coordinates parallel document processing (map phase) followed by
    meta-summary generation (reduce phase). Uses the Strategy Pattern
    for parallel execution, enabling both production and test modes.

    The orchestrator:
    1. Accepts a list of documents with extracted text
    2. Summarizes each document in parallel via DocumentSummarizer
    3. Combines individual summaries into a meta-summary
    4. Reports progress throughout via callback

    When a prompt_adapter is provided, the orchestrator uses focus-aware
    prompts for meta-summary generation, emphasizing the user's areas
    of interest as extracted from their selected template.

    Attributes:
        document_summarizer: Summarizer for individual documents.
        model_manager: OllamaModelManager for meta-summary generation.
        strategy: ExecutorStrategy for parallel processing.
        prompt_adapter: Optional PromptAdapter for focus-aware meta-summary.
        preset_id: Template preset ID for focus extraction.
    """

    def __init__(
        self,
        document_summarizer: DocumentSummarizer,
        model_manager: OllamaModelManager,
        strategy: ExecutorStrategy | None = None,
        prompt_adapter: PromptAdapter | None = None,
        preset_id: str = "factual-summary",
    ):
        """
        Initialize the multi-document orchestrator.

        Args:
            document_summarizer: Summarizer for individual documents.
            model_manager: OllamaModelManager for AI text generation.
            strategy: ExecutorStrategy for parallel execution.
                     Defaults to ThreadPoolStrategy with PARALLEL_MAX_WORKERS.
            prompt_adapter: Optional PromptAdapter for generating focus-aware
                          meta-summary prompts. If None, uses default prompts.
            preset_id: Template preset ID for focus extraction. Used with
                      prompt_adapter to thread user's focus through prompts.
        """
        self.document_summarizer = document_summarizer
        self.model_manager = model_manager
        self.strategy = strategy or ThreadPoolStrategy(max_workers=PARALLEL_MAX_WORKERS)
        self.prompt_adapter = prompt_adapter
        self.preset_id = preset_id
        self._model_name: str | None = None  # Cached model name for adapter

        # Cancellation support
        self._stop_event = threading.Event()

    def stop(self):
        """Signal the orchestrator to stop processing."""
        self._stop_event.set()

    def summarize_documents(
        self,
        documents: list[dict],
        max_words_per_document: int = 200,
        max_meta_summary_words: int = 500,
        progress_callback: Callable[[int, str], None] | None = None,
        ui_queue: Queue | None = None,
    ) -> MultiDocumentSummaryResult:
        """
        Summarize multiple documents with hierarchical map-reduce approach.

        Args:
            documents: List of dicts with 'filename' and 'extracted_text' keys.
            max_words_per_document: Target words for each document summary.
            max_meta_summary_words: Target words for the final meta-summary.
            progress_callback: Optional callback(percent, message) for progress.
            ui_queue: Optional Queue for progress messages (uses ProgressAggregator).

        Returns:
            MultiDocumentSummaryResult with individual and meta summaries.
        """
        start_time = time.time()
        self._stop_event.clear()

        # Filter documents with actual text
        valid_documents = [
            doc
            for doc in documents
            if doc.get("extracted_text") and len(doc["extracted_text"].strip()) > 50
        ]

        if not valid_documents:
            return MultiDocumentSummaryResult(
                meta_summary="No valid documents to summarize.",
                total_processing_time_seconds=time.time() - start_time,
            )

        doc_count = len(valid_documents)
        info(f"[MULTI-DOC] Starting summarization of {doc_count} documents")

        if progress_callback:
            progress_callback(5, f"Preparing to summarize {doc_count} documents...")

        # === PHASE 1: MAP - Parallel document summarization ===
        individual_results = self._map_phase(
            documents=valid_documents,
            max_words=max_words_per_document,
            progress_callback=progress_callback,
            ui_queue=ui_queue,
        )

        if self._stop_event.is_set():
            return MultiDocumentSummaryResult(
                individual_summaries=individual_results,
                meta_summary="Processing cancelled.",
                total_processing_time_seconds=time.time() - start_time,
                documents_processed=sum(1 for r in individual_results.values() if r.success),
                documents_failed=sum(1 for r in individual_results.values() if not r.success),
            )

        # === PHASE 2: REDUCE - Meta-summary generation ===
        if progress_callback:
            progress_callback(85, "Generating meta-summary from all documents...")

        successful_summaries = [r for r in individual_results.values() if r.success]

        if successful_summaries:
            meta_summary = self._reduce_phase(
                summaries=successful_summaries, max_words=max_meta_summary_words
            )
        else:
            meta_summary = "No documents were successfully summarized."

        # Build final result
        total_time = time.time() - start_time
        documents_processed = len(successful_summaries)
        documents_failed = doc_count - documents_processed

        if progress_callback:
            progress_callback(100, f"Completed: {documents_processed}/{doc_count} documents")

        info(f"[MULTI-DOC] Completed: {documents_processed} docs in {total_time:.1f}s")

        return MultiDocumentSummaryResult(
            individual_summaries=individual_results,
            meta_summary=meta_summary,
            total_processing_time_seconds=total_time,
            documents_processed=documents_processed,
            documents_failed=documents_failed,
            document_order=[doc["filename"] for doc in valid_documents],
        )

    def _map_phase(
        self,
        documents: list[dict],
        max_words: int,
        progress_callback: Callable[[int, str], None] | None,
        ui_queue: Queue | None,
    ) -> dict[str, DocumentSummaryResult]:
        """
        Phase 1: Summarize each document in parallel.

        Uses ParallelTaskRunner with the configured strategy to process
        documents concurrently. Progress is reported via both callback
        and ProgressAggregator (if ui_queue provided).

        Args:
            documents: List of documents to summarize.
            max_words: Target words per document summary.
            progress_callback: Optional progress callback.
            ui_queue: Optional queue for aggregated progress.

        Returns:
            Dict mapping filename to DocumentSummaryResult.
        """
        doc_count = len(documents)
        results: dict[str, DocumentSummaryResult] = {}

        # Set up progress aggregation if queue provided
        aggregator = None
        if ui_queue:
            aggregator = ProgressAggregator(ui_queue, throttle_ms=100)
            aggregator.set_total(doc_count)

        def summarize_single_document(doc: dict) -> DocumentSummaryResult:
            """Process a single document (runs in thread pool)."""
            filename = doc["filename"]
            text = doc["extracted_text"]

            if aggregator:
                aggregator.update(filename, f"Summarizing {filename}...")

            # Create per-document progress callback
            def doc_progress(percent: int, message: str):
                if aggregator:
                    aggregator.update(filename, message)
                # Scale document progress to overall progress (5-80%)
                if progress_callback:
                    completed = aggregator.completed if aggregator else 0
                    doc_contribution = (percent / 100) / doc_count
                    overall = int(5 + (completed / doc_count + doc_contribution) * 75)
                    progress_callback(overall, message)

            # Check for cancellation
            def should_stop() -> bool:
                return self._stop_event.is_set()

            result = self.document_summarizer.summarize(
                text=text,
                filename=filename,
                max_words=max_words,
                progress_callback=doc_progress,
                stop_check=should_stop,
            )

            if aggregator:
                aggregator.complete(filename)

            return result

        def on_task_complete(task_id: str, result: DocumentSummaryResult):
            """Callback when a document finishes."""
            results[result.filename] = result
            debug_log(
                f"[MULTI-DOC] Completed: {result.filename} "
                f"({result.word_count} words, {result.chunk_count} chunks)"
            )

        # Create task runner with strategy
        runner = ParallelTaskRunner(strategy=self.strategy, on_task_complete=on_task_complete)

        # Prepare items: (task_id, payload)
        items = [(doc["filename"], doc) for doc in documents]

        # Execute parallel processing
        task_results = runner.run(summarize_single_document, items)

        # Ensure all results are captured (some may have failed)
        for task_result in task_results:
            if not task_result.success:
                # Create failed result for documents that errored
                results[task_result.task_id] = DocumentSummaryResult(
                    filename=task_result.task_id,
                    summary="",
                    word_count=0,
                    chunk_count=0,
                    processing_time_seconds=0,
                    success=False,
                    error_message=str(task_result.error),
                )

        return results

    def _reduce_phase(self, summaries: list[DocumentSummaryResult], max_words: int) -> str:
        """
        Phase 2: Combine individual summaries into meta-summary.

        Generates a coherent meta-summary that synthesizes the key
        information from all document summaries.

        Args:
            summaries: List of successful DocumentSummaryResults.
            max_words: Target word count for meta-summary.

        Returns:
            Meta-summary string.
        """
        if not summaries:
            return ""

        # Format summaries for the prompt
        formatted_summaries = self._format_summaries_for_prompt(summaries)

        # Check if combined summaries fit in context window
        estimated_tokens = len(formatted_summaries) // 4
        context_available = OLLAMA_CONTEXT_WINDOW - 500  # Reserve for prompt and output

        if estimated_tokens > context_available:
            # Too large - need to chunk the summaries
            return self._generate_chunked_meta_summary(summaries, max_words)
        else:
            # Fits in context - direct generation
            return self._generate_direct_meta_summary(
                formatted_summaries, max_words, len(summaries)
            )

    def _format_summaries_for_prompt(self, summaries: list[DocumentSummaryResult]) -> str:
        """
        Format individual summaries for inclusion in meta-summary prompt.

        Args:
            summaries: List of DocumentSummaryResult objects.

        Returns:
            Formatted string with document headers and summaries.
        """
        parts = []
        for summary in summaries:
            parts.append(f"--- {summary.filename} ---\n{summary.summary}")
        return "\n\n".join(parts)

    def _generate_direct_meta_summary(
        self, formatted_summaries: str, max_words: int, doc_count: int
    ) -> str:
        """
        Generate meta-summary when all summaries fit in context window.

        If a prompt_adapter is configured, uses focus-aware prompts that
        emphasize the user's areas of interest. Otherwise, uses default
        prompts for generic legal document summarization.

        Args:
            formatted_summaries: Combined formatted summaries.
            max_words: Target word count.
            doc_count: Number of documents summarized.

        Returns:
            Meta-summary string.
        """
        # Use focus-aware prompts if adapter is configured
        if self.prompt_adapter:
            # Get model name for adapter (cached)
            if not self._model_name:
                self._model_name = getattr(self.model_manager, "loaded_model_name", "phi-3-mini")

            prompt = self.prompt_adapter.create_meta_summary_prompt(
                preset_id=self.preset_id,
                model_name=self._model_name,
                formatted_summaries=formatted_summaries,
                max_words=max_words,
                doc_count=doc_count,
            )
        else:
            # Fallback: use default prompts
            min_words = max(100, int(max_words * 0.7))

            prompt = f"""You are a legal document analyst reviewing summaries of {doc_count} documents from a single case.

Individual document summaries:

{formatted_summaries}

Create a comprehensive meta-summary ({min_words}-{max_words} words) that:
1. Synthesizes the overall case narrative and timeline
2. Identifies all key parties, their roles, and relationships
3. Highlights the primary claims, defenses, and legal issues
4. Notes significant evidence, testimony, or procedural events
5. Identifies any patterns, contradictions, or critical findings

Present the information in a logical, chronological order where appropriate.

Meta-Summary:"""

        max_tokens = int(max_words * 2.0)

        try:
            meta_summary = self.model_manager.generate_text(prompt=prompt, max_tokens=max_tokens)
            return meta_summary.strip()
        except Exception as e:
            error(f"[MULTI-DOC] Meta-summary generation failed: {e}")
            return f"Meta-summary generation failed: {e}"

    def _generate_chunked_meta_summary(
        self, summaries: list[DocumentSummaryResult], max_words: int
    ) -> str:
        """
        Generate meta-summary for large document sets via chunking.

        When combined summaries exceed context window, this method:
        1. Groups summaries into batches that fit in context
        2. Generates intermediate summaries for each batch
        3. Combines intermediate summaries into final meta-summary

        Args:
            summaries: List of DocumentSummaryResult objects.
            max_words: Target word count for final meta-summary.

        Returns:
            Meta-summary string.
        """
        # Calculate how many summaries can fit per batch
        avg_summary_length = sum(len(s.summary) for s in summaries) // len(summaries)
        context_budget = (OLLAMA_CONTEXT_WINDOW - 500) * 4  # Convert tokens to chars
        summaries_per_batch = max(2, context_budget // (avg_summary_length + 100))

        debug_log(
            f"[MULTI-DOC] Chunking {len(summaries)} summaries into batches of {summaries_per_batch}"
        )

        # Process batches
        intermediate_summaries = []
        for i in range(0, len(summaries), summaries_per_batch):
            batch = summaries[i : i + summaries_per_batch]
            batch_formatted = self._format_summaries_for_prompt(batch)

            batch_summary = self._generate_direct_meta_summary(
                formatted_summaries=batch_formatted,
                max_words=max_words // 2,  # Shorter intermediate summaries
                doc_count=len(batch),
            )
            intermediate_summaries.append(batch_summary)

        # Combine intermediate summaries into final
        if len(intermediate_summaries) == 1:
            return intermediate_summaries[0]

        combined_intermediates = "\n\n---\n\n".join(intermediate_summaries)

        final_prompt = f"""You are combining multiple partial case summaries into one comprehensive summary.

Partial summaries:

{combined_intermediates}

Create a unified meta-summary ({max_words} words) that synthesizes all the information above into a coherent narrative.

Final Meta-Summary:"""

        max_tokens = int(max_words * 2.0)

        try:
            final_summary = self.model_manager.generate_text(
                prompt=final_prompt, max_tokens=max_tokens
            )
            return final_summary.strip()
        except Exception as e:
            error(f"[MULTI-DOC] Final meta-summary failed: {e}")
            # Return concatenated intermediates as fallback
            return "\n\n".join(intermediate_summaries)
