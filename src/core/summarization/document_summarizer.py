"""
Document Summarizer - Single Document Summarization via Progressive Chunking

This module provides an abstraction for summarizing individual documents using
the progressive chunking approach. It wraps the existing ProgressiveSummarizer
to provide a clean interface for the multi-document orchestrator.

Architecture:
    DocumentSummarizer (ABC)
        └── ProgressiveDocumentSummarizer (concrete implementation)

The ProgressiveDocumentSummarizer:
1. Chunks the document into ~1000-word segments
2. Generates a summary for each chunk with rolling context
3. Maintains a progressive summary that evolves as chunks are processed
4. Returns the final progressive summary as the document summary

Usage:
    from src.core.summarization import ProgressiveDocumentSummarizer
    from src.core.ai import OllamaModelManager

    model_manager = OllamaModelManager()
    summarizer = ProgressiveDocumentSummarizer(model_manager)

    result = summarizer.summarize(
        text="Full document text...",
        filename="complaint.pdf",
        max_words=200
    )
    print(result.summary)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from src.logging_config import debug_log, error, info
from src.progressive_summarizer import ProgressiveSummarizer

from .result_types import DocumentSummaryResult

if TYPE_CHECKING:
    from src.core.ai.ollama_model_manager import OllamaModelManager
    from src.core.prompting import PromptAdapter


class DocumentSummarizer(ABC):
    """
    Abstract base class for document summarization.

    Defines the interface that all document summarizers must implement.
    This allows for different summarization strategies (progressive,
    extractive, etc.) to be swapped without changing the calling code.
    """

    @abstractmethod
    def summarize(
        self,
        text: str,
        filename: str,
        max_words: int = 200,
        progress_callback: Callable[[int, str], None] | None = None,
        stop_check: Callable[[], bool] | None = None
    ) -> DocumentSummaryResult:
        """
        Summarize a single document.

        Args:
            text: Full document text to summarize.
            filename: Original filename (for display and tracking).
            max_words: Target word count for the final summary.
            progress_callback: Optional callback(percent, message) for progress updates.
            stop_check: Optional callable that returns True if processing should stop.

        Returns:
            DocumentSummaryResult with the summary and metadata.
        """
        pass


class ProgressiveDocumentSummarizer(DocumentSummarizer):
    """
    Summarizes documents using progressive chunking with rolling context.

    This implementation:
    1. Splits the document into manageable chunks (~1000 words each)
    2. Summarizes each chunk with context from previous chunks
    3. Maintains a progressive summary that evolves through the document
    4. Returns the final progressive summary

    The progressive approach ensures that later chunks benefit from
    context established in earlier chunks, producing more coherent
    summaries for long documents.

    When a prompt_adapter is provided, the summarizer threads the user's
    focus areas through all stages of summarization, ensuring the final
    summary emphasizes what the user cares about.

    Attributes:
        model_manager: OllamaModelManager for text generation.
        config_path: Path to chunking configuration (optional).
        prompt_adapter: Optional PromptAdapter for focus-aware prompts.
        preset_id: Template preset ID for focus extraction.
    """

    def __init__(
        self,
        model_manager: OllamaModelManager,
        config_path: Path | None = None,
        prompt_adapter: "PromptAdapter | None" = None,
        preset_id: str = "factual-summary"
    ):
        """
        Initialize the progressive document summarizer.

        Args:
            model_manager: OllamaModelManager instance for text generation.
            config_path: Path to chunking_config.yaml. Uses default if None.
            prompt_adapter: Optional PromptAdapter for generating focus-aware
                          prompts. If None, uses default hardcoded prompts.
            preset_id: Template preset ID for focus extraction. Used with
                      prompt_adapter to thread user's focus through prompts.
        """
        self.model_manager = model_manager
        self.config_path = config_path
        self.prompt_adapter = prompt_adapter
        self.preset_id = preset_id
        self._model_name: str | None = None  # Cached model name for adapter

    def summarize(
        self,
        text: str,
        filename: str,
        max_words: int = 200,
        progress_callback: Callable[[int, str], None] | None = None,
        stop_check: Callable[[], bool] | None = None
    ) -> DocumentSummaryResult:
        """
        Summarize a document using progressive chunking.

        Args:
            text: Full document text.
            filename: Original filename for tracking.
            max_words: Target summary length in words.
            progress_callback: Optional progress callback(percent, message).
            stop_check: Optional callable returning True if should stop.

        Returns:
            DocumentSummaryResult with summary and processing metadata.
        """
        start_time = time.time()

        # Handle empty or very short text
        if not text or len(text.strip()) < 50:
            return DocumentSummaryResult(
                filename=filename,
                summary="",
                word_count=0,
                chunk_count=0,
                processing_time_seconds=time.time() - start_time,
                success=False,
                error_message="Document text is empty or too short to summarize"
            )

        try:
            # Create a fresh ProgressiveSummarizer for this document
            progressive = ProgressiveSummarizer(self.config_path)

            # Step 1: Chunk the document
            if progress_callback:
                progress_callback(5, f"Chunking {filename}...")

            chunks = progressive.chunk_document(text)
            chunk_count = len(chunks)

            if chunk_count == 0:
                return DocumentSummaryResult(
                    filename=filename,
                    summary="",
                    word_count=0,
                    chunk_count=0,
                    processing_time_seconds=time.time() - start_time,
                    success=False,
                    error_message="Document could not be chunked"
                )

            debug_log(f"[DOC SUMMARIZER] {filename}: {chunk_count} chunks")

            # Step 2: Prepare DataFrame for tracking
            progressive.prepare_chunks_dataframe(chunks)

            # Step 3: Get batch boundaries for progressive updates
            batch_boundaries = progressive._get_batch_boundaries(chunk_count)

            # Step 4: Process each chunk
            chunk_summaries = []
            target_chunk_words = 75  # Target words per chunk summary

            for i, chunk in enumerate(chunks):
                # Check for cancellation
                if stop_check and stop_check():
                    return DocumentSummaryResult(
                        filename=filename,
                        summary="",
                        word_count=0,
                        chunk_count=i,
                        processing_time_seconds=time.time() - start_time,
                        success=False,
                        error_message="Processing cancelled by user"
                    )

                chunk_num = i + 1
                progress_percent = int(10 + (chunk_num / chunk_count) * 80)

                if progress_callback:
                    progress_callback(
                        progress_percent,
                        f"{filename}: chunk {chunk_num}/{chunk_count}"
                    )

                # Generate chunk summary with context
                chunk_summary = self._summarize_chunk(
                    progressive=progressive,
                    chunk_num=chunk_num,
                    chunk_text=chunk.text,
                    target_words=target_chunk_words
                )

                chunk_summaries.append(chunk_summary)

                # Update DataFrame
                progressive.df.loc[
                    progressive.df['chunk_num'] == chunk_num,
                    'chunk_summary'
                ] = chunk_summary

                # Update progressive summary at batch boundaries
                if chunk_num in batch_boundaries:
                    progressive.current_progressive_summary = self._update_progressive_summary(
                        chunk_summaries,
                        filename,
                        max_words=max(50, max_words // 2)  # Progressive summary shorter than final
                    )
                    progressive.df.loc[
                        progressive.df['chunk_num'] == chunk_num,
                        'progressive_summary'
                    ] = progressive.current_progressive_summary

            # Step 5: Generate final summary from all chunk summaries
            if progress_callback:
                progress_callback(95, f"Finalizing {filename} summary...")

            final_summary = self._generate_final_summary(
                chunk_summaries=chunk_summaries,
                filename=filename,
                max_words=max_words
            )

            processing_time = time.time() - start_time
            word_count = len(final_summary.split())

            info(f"[DOC SUMMARIZER] {filename}: {word_count} words in {processing_time:.1f}s")

            return DocumentSummaryResult(
                filename=filename,
                summary=final_summary,
                word_count=word_count,
                chunk_count=chunk_count,
                processing_time_seconds=processing_time,
                success=True
            )

        except Exception as e:
            error(f"[DOC SUMMARIZER] Failed to summarize {filename}: {e}")
            return DocumentSummaryResult(
                filename=filename,
                summary="",
                word_count=0,
                chunk_count=0,
                processing_time_seconds=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

    def _summarize_chunk(
        self,
        progressive: ProgressiveSummarizer,
        chunk_num: int,
        chunk_text: str,
        target_words: int = 75
    ) -> str:
        """
        Generate a summary for a single chunk with context.

        If a prompt_adapter is configured, uses focus-aware prompts that
        emphasize the user's areas of interest. Otherwise, falls back to
        the default ProgressiveSummarizer prompts.

        Args:
            progressive: ProgressiveSummarizer instance with context.
            chunk_num: Current chunk number (1-indexed).
            chunk_text: Text of the chunk to summarize.
            target_words: Target word count for chunk summary.

        Returns:
            Chunk summary string.
        """
        # Use focus-aware prompts if adapter is configured
        if self.prompt_adapter:
            # Get context from progressive summarizer
            global_context = progressive.current_progressive_summary or "Document analysis just started."
            local_context = self._get_local_context(progressive, chunk_num)

            # Get model name for adapter (cached)
            if not self._model_name:
                self._model_name = getattr(
                    self.model_manager, 'loaded_model_name', 'phi-3-mini'
                )

            prompt = self.prompt_adapter.create_chunk_prompt(
                preset_id=self.preset_id,
                model_name=self._model_name,
                global_context=global_context,
                local_context=local_context,
                chunk_text=chunk_text,
                max_words=target_words
            )
        else:
            # Fallback: use default prompts from ProgressiveSummarizer
            prompt = progressive.create_summarization_prompt(
                chunk_num=chunk_num,
                chunk_text=chunk_text,
                summary_target_words=target_words
            )

        # Generate summary via Ollama
        # Use 1.5x tokens per word with buffer
        max_tokens = int(target_words * 2.0)

        summary = self.model_manager.generate_text(
            prompt=prompt,
            max_tokens=max_tokens
        )

        return summary.strip()

    def _get_local_context(
        self,
        progressive: ProgressiveSummarizer,
        chunk_num: int
    ) -> str:
        """
        Get local context (previous chunk summary) for chunk prompt.

        Args:
            progressive: ProgressiveSummarizer with chunk data.
            chunk_num: Current chunk number (1-indexed).

        Returns:
            Previous chunk's summary or placeholder text.
        """
        if chunk_num <= 1 or progressive.df is None:
            return "This is the first section of the document."

        # Get previous chunk's summary from DataFrame
        prev_row = progressive.df[progressive.df['chunk_num'] == chunk_num - 1]
        if not prev_row.empty:
            prev_summary = prev_row['chunk_summary'].iloc[0]
            if prev_summary and str(prev_summary).strip():
                return str(prev_summary)

        return "Previous section summary not available."

    def _update_progressive_summary(
        self,
        chunk_summaries: list[str],
        filename: str,
        max_words: int = 100
    ) -> str:
        """
        Update the progressive summary from accumulated chunk summaries.

        This is called at batch boundaries to maintain a rolling
        overview of the document processed so far.

        Args:
            chunk_summaries: List of chunk summaries so far.
            filename: Document filename for context.
            max_words: Target word count for progressive summary.

        Returns:
            Updated progressive summary string.
        """
        if not chunk_summaries:
            return ""

        # Combine recent chunk summaries
        combined = "\n\n".join(chunk_summaries[-5:])  # Last 5 chunks

        prompt = f"""You are summarizing a legal document in progress.

Below are summaries of the most recent sections from "{filename}":

{combined}

Create a brief progressive summary ({max_words} words max) that captures:
1. The main topic or claim
2. Key parties mentioned
3. Important facts or events

Progressive Summary:"""

        max_tokens = int(max_words * 2.0)
        summary = self.model_manager.generate_text(prompt=prompt, max_tokens=max_tokens)

        return summary.strip()

    def _generate_final_summary(
        self,
        chunk_summaries: list[str],
        filename: str,
        max_words: int = 200
    ) -> str:
        """
        Generate the final document summary from all chunk summaries.

        Combines all chunk summaries into a coherent final summary
        that represents the entire document. If a prompt_adapter is
        configured, uses focus-aware prompts to emphasize user's interests.

        Args:
            chunk_summaries: List of all chunk summaries.
            filename: Document filename for context.
            max_words: Target word count for final summary.

        Returns:
            Final document summary string.
        """
        if not chunk_summaries:
            return ""

        # Combine all chunk summaries
        combined = "\n\n".join(chunk_summaries)

        # Use focus-aware prompts if adapter is configured
        if self.prompt_adapter:
            # Get model name for adapter (cached)
            if not self._model_name:
                self._model_name = getattr(
                    self.model_manager, 'loaded_model_name', 'phi-3-mini'
                )

            prompt = self.prompt_adapter.create_document_final_prompt(
                preset_id=self.preset_id,
                model_name=self._model_name,
                chunk_summaries=combined,
                filename=filename,
                max_words=max_words
            )
        else:
            # Fallback: use default prompts
            min_words = max(50, int(max_words * 0.7))

            prompt = f"""You are a legal document analyst creating a comprehensive summary.

Below are section-by-section summaries of "{filename}":

{combined}

Create a unified summary ({min_words}-{max_words} words) that:
1. Synthesizes the overall document narrative
2. Identifies all key parties, claims, and outcomes
3. Highlights the most significant facts and findings
4. Presents information in logical order

Document Summary:"""

        max_tokens = int(max_words * 2.0)
        summary = self.model_manager.generate_text(prompt=prompt, max_tokens=max_tokens)

        return summary.strip()
