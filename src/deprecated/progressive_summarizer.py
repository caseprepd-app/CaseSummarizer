"""
Progressive Summarization Engine

This module implements Fast Mode summarization with batched progressive updates.
It processes document chunks sequentially, maintaining:
1. Individual chunk summaries
2. A progressive (rolling) document summary
3. Contextual information for the AI model
4. Batch boundaries for progressive summary updates
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.core.chunking import UnifiedChunk, create_unified_chunker
from src.core.config import load_yaml

logger = logging.getLogger(__name__)


@dataclass
class SummarizationContext:
    """Context information passed to summarization function."""

    global_context: str  # Rolling summary of ALL preceding chunks (1-2 sentences)
    local_context: str  # Summary of the immediately preceding chunk only (1-2 sentences)
    chunk_text: str  # Current chunk text
    chunk_num: int  # Current chunk number
    total_chunks: int  # Total number of chunks
    section_name: str | None = None  # Detected section name


@dataclass
class SummarizationResult:
    """Result from summarizing a single chunk."""

    chunk_num: int
    chunk_summary: str  # Summary of this chunk
    progressive_summary: str  # Updated document summary up to this point
    processing_time_sec: float
    context_used: dict[str, str] = field(default_factory=dict)


class ProgressiveSummarizer:
    """
    Progressive summarization engine for document chunks.

    Implements Fast Mode with batched progressive updates:
    - Processes chunks sequentially
    - Generates summary for each chunk with context
    - Updates progressive document summary every N chunks
    - Uses pandas DataFrame for organization
    """

    def __init__(self, config_path: Path | None = None):
        """
        Initialize progressive summarizer.

        Args:
            config_path: Path to chunking_config.yaml. If None, uses default.
        """
        if config_path is None:
            from src.config import CONFIG_FILES_DIR

            config_path = CONFIG_FILES_DIR / "chunking_config.yaml"

        self.config = self._load_config(config_path)

        # CHUNKING SITE 2 of 2: Summarization (per-document, fresh instance)
        #
        # A fresh ProgressiveSummarizer (and thus a fresh UnifiedChunker) is created
        # for each document by ProgressiveDocumentSummarizer.summarize(). This is
        # intentional -- each instance tracks its own rolling summary state via the
        # DataFrame below.
        #
        # This re-chunks the same preprocessed_text that Q&A already chunked (Site 1
        # in workers.py), including re-running coreference resolution. That duplication
        # is acceptable: summary is rarely enabled, and the bottleneck is LLM calls
        # (minutes to hours), not chunking (seconds). Sharing chunks across the Q&A
        # thread and the later summary worker would add cross-thread plumbing for
        # negligible gain.
        self.unified_chunker = create_unified_chunker()

        # DataFrame to track chunks and summaries
        self.df = pd.DataFrame(
            columns=[
                "chunk_num",
                "chunk_text",
                "chunk_summary",
                "progressive_summary",
                "section_detected",
                "word_count",
                "processing_time_sec",
            ]
        )

        # Progressive summary state
        self.current_progressive_summary = ""
        self.last_progressive_update_chunk = 0

    def _load_config(self, config_path: Path) -> dict:
        """Load configuration from YAML file."""
        return load_yaml(config_path, log_prefix="[ProgressiveSummarizer]")

    def _get_batch_boundaries(self, total_chunks: int) -> list[int]:
        """
        Calculate at which chunk numbers to update the progressive summary.

        Uses adaptive batching if enabled, otherwise fixed batching.

        Returns:
            List of chunk numbers at which to update (e.g., [5, 10, 15, ...])
        """
        fast_mode = self.config.get("fast_mode", {})
        if not fast_mode.get("enabled", True):
            # Deep mode: update after every chunk
            return list(range(1, total_chunks + 1))

        if fast_mode.get("section_aware_batching", True):
            # Section-aware batching (preferred)
            logger.debug("Using section-aware batching for progressive updates")
            return self._calculate_section_aware_boundaries()
        elif fast_mode.get("adaptive_batching", True):
            # Adaptive batching (fallback)
            logger.debug("Using adaptive batching for progressive updates")
            return self._calculate_adaptive_boundaries(total_chunks)
        else:
            # Fixed batching
            base_freq = fast_mode.get("base_batch_frequency", 5)
            logger.debug("Using fixed batching (every %s chunks)", base_freq)
            return [*list(range(base_freq, total_chunks + 1, base_freq)), total_chunks]

    def _calculate_section_aware_boundaries(self) -> list[int]:
        """
        Calculate batch boundaries based on detected sections in DataFrame.

        Called after chunks are created, before summarization.
        """
        if self.df.empty:
            return []

        config = self.config.get("fast_mode", {})
        min_chunks = config.get("section_batch_min_chunks", 3)
        _max_chunks = config.get("section_batch_max_chunks", 15)  # Reserved for future use

        boundaries = []
        current_section = None
        section_start = 0

        for _idx, row in self.df.iterrows():
            section = row["section_detected"]
            chunk_num = row["chunk_num"]

            # Check if section changed
            if section != current_section and section_start > 0:
                section_length = chunk_num - section_start
                # Only update if section has enough chunks
                if section_length >= min_chunks:
                    boundaries.append(chunk_num - 1)

            if section != current_section:
                current_section = section
                section_start = chunk_num

        # Add final boundary
        if len(self.df) > 0:
            boundaries.append(len(self.df))

        boundaries = sorted(set(boundaries))
        logger.debug("Section-aware boundaries: %s", boundaries)
        return boundaries

    def _calculate_adaptive_boundaries(self, total_chunks: int) -> list[int]:
        """
        Calculate batch boundaries with adaptive frequency.

        Frequency varies based on document position:
        - Early chunks: More frequent updates (establishing context)
        - Middle chunks: Less frequent (context established)
        - Late chunks: More frequent (important conclusions)
        """
        config = self.config.get("fast_mode", {})
        early = config.get("early_document", {})
        middle = config.get("middle_document", {})
        late = config.get("late_document", {})

        early_threshold = early.get("threshold_chunks", 20)
        early_freq = early.get("batch_frequency", 5)

        middle_threshold = middle.get("threshold_chunks", 80)
        middle_freq = middle.get("batch_frequency", 10)

        late_freq = late.get("batch_frequency", 5)

        boundaries = []

        # Early document
        for chunk in range(early_freq, min(early_threshold, total_chunks) + 1, early_freq):
            boundaries.append(chunk)

        # Middle document
        if total_chunks > early_threshold:
            for chunk in range(
                early_threshold + middle_freq, min(middle_threshold, total_chunks) + 1, middle_freq
            ):
                boundaries.append(chunk)

        # Late document
        if total_chunks > middle_threshold:
            for chunk in range(middle_threshold + late_freq, total_chunks + 1, late_freq):
                boundaries.append(chunk)

        # Ensure final chunk is included
        if total_chunks not in boundaries:
            boundaries.append(total_chunks)

        boundaries = sorted(set(boundaries))
        logger.debug("Adaptive boundaries: %s", boundaries)
        return boundaries

    def chunk_document(self, text: str) -> list[UnifiedChunk]:
        """
        Chunk a document using the unified chunker.

        Args:
            text: Full document text

        Returns:
            List of UnifiedChunk objects (300-1000 tokens each, research-based)
        """
        start_time = time.time()
        logger.debug("Starting document chunking...")
        # use_cache=False: each summarizer instance has independent state (DataFrame,
        # rolling summary), so cached chunks from another instance would be misleading.
        chunks = self.unified_chunker.chunk_text(text, use_cache=False)
        elapsed = time.time() - start_time
        logger.debug("%s took %.2fs", f"Document chunking ({len(chunks)} chunks)", elapsed)
        logger.info("Document chunked into %s chunks", len(chunks))
        return chunks

    def prepare_chunks_dataframe(self, chunks: list[UnifiedChunk]) -> pd.DataFrame:
        """
        Prepare DataFrame with chunk information.

        This is done before summarization so we can calculate batch boundaries.

        Args:
            chunks: List of UnifiedChunk objects

        Returns:
            DataFrame with chunk data
        """
        data = []
        for chunk in chunks:
            data.append(
                {
                    "chunk_num": chunk.chunk_num,
                    "chunk_text": chunk.text,
                    "chunk_summary": "",  # Will be filled during summarization
                    "progressive_summary": "",  # Will be filled during summarization
                    "section_detected": chunk.section_name,
                    "word_count": chunk.word_count,
                    "processing_time_sec": 0.0,
                }
            )

        self.df = pd.DataFrame(data)
        logger.info("Prepared DataFrame with %s chunks", len(self.df))
        return self.df

    def get_context_for_chunk(self, chunk_num: int) -> tuple[str, str]:
        """
        Get context information for a specific chunk.

        Args:
            chunk_num: Chunk number (1-indexed)

        Returns:
            (global_context, local_context) tuple of strings
        """
        config = self.config.get("summarization", {})
        # global_max_sentences reserved for future progressive summary truncation
        _global_max_sentences = config.get("progressive_summary_max_sentences", 2)
        local_max_sentences = config.get("local_context_max_sentences", 2)

        # Get global context (progressive summary from previous chunk)
        if chunk_num == 1:
            global_context = "[This is the beginning of the document.]"
        else:
            prev_rows = self.df.loc[self.df["chunk_num"] == chunk_num - 1, "progressive_summary"]
            previous_progressive = prev_rows.iloc[0] if not prev_rows.empty else ""
            if previous_progressive:
                global_context = f"[Document overview: {previous_progressive}]"
            else:
                global_context = "[Global context not yet available]"

        # Get local context (summary of previous chunk)
        if chunk_num == 1:
            local_context = "[No preceding content.]"
        else:
            prev_summary_rows = self.df.loc[self.df["chunk_num"] == chunk_num - 1, "chunk_summary"]
            previous_summary = prev_summary_rows.iloc[0] if not prev_summary_rows.empty else ""
            if previous_summary:
                # Truncate to max sentences if needed
                from src.core.utils.sentence_splitter import split_sentences

                sentences = split_sentences(previous_summary)
                if len(sentences) > local_max_sentences:
                    previous_summary = " ".join(sentences[:local_max_sentences])
                local_context = f"[Previous: {previous_summary}]"
            else:
                local_context = "[Previous chunk summary not yet available]"

        return global_context, local_context

    def create_summarization_prompt(
        self, chunk_num: int, chunk_text: str = "", summary_target_words: int = 75
    ) -> str:
        """
        Create the prompt for AI model to summarize a chunk with context.

        Uses the chunked_prompt_template.txt file for consistent formatting.

        Args:
            chunk_num: Chunk number (1-indexed)
            chunk_text: The chunk text to summarize (optional, for preview/validation)
            summary_target_words: Target word count for the summary

        Returns:
            Prompt string with context and placeholders
        """
        # Get context
        global_context, local_context = self.get_context_for_chunk(chunk_num)

        # Load template
        from src.config import BUNDLED_CONFIG_DIR

        template_path = BUNDLED_CONFIG_DIR / "chunked_prompt_template.txt"

        try:
            with open(template_path, encoding="utf-8") as f:
                template = f.read()
        except Exception as e:
            logger.error("Failed to load chunked prompt template: %s", e, exc_info=True)
            # Fallback to inline template
            template = """You are a legal document analyst. Below is a chunk from a longer document.

{global_context}
{local_context}

Now analyze and summarize the following section, focusing on key facts, decisions, or developments:

{chunk_text}

Summary:"""

        # Calculate word ranges for target
        min_words = max(1, int(summary_target_words * 0.7))
        max_words = int(summary_target_words * 1.3)

        # Format template with context
        prompt = template.format(
            global_context=global_context,
            local_context=local_context,
            chunk_text=chunk_text,
            min_words=min_words,
            max_words_range=f"{min_words}-{max_words}",
            max_words=summary_target_words,
        )

        return prompt

    def save_debug_dataframe(self, output_dir: Path | None = None) -> Path:
        """
        Save the processing DataFrame to CSV for debugging.

        Args:
            output_dir: Directory to save to. Defaults to project debug folder.

        Returns:
            Path to saved CSV file
        """
        if output_dir is None:
            from src.config import LOGS_DIR

            output_dir = LOGS_DIR / "debug"

        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"summarization_{timestamp}.csv"

        # Create display version (truncate long text for readability)
        df_display = self.df.copy()
        df_display["chunk_text"] = df_display["chunk_text"].str[:100] + "..."
        df_display["chunk_summary"] = df_display["chunk_summary"].str[:75] + "..."
        df_display["progressive_summary"] = df_display["progressive_summary"].str[:75] + "..."

        df_display.to_csv(filename, index=False, encoding="utf-8-sig")
        logger.info("Saved debug DataFrame to %s", filename)

        # Clean up old files (keep only recent ones)
        config = self.config.get("processing", {})
        keep_count = config.get("debug_files_to_keep", 5)
        self._cleanup_old_debug_files(output_dir, keep_count)

        return filename

    def _cleanup_old_debug_files(self, debug_dir: Path, keep_count: int):
        """Remove old debug CSV files, keeping only the most recent."""
        csv_files = sorted(debug_dir.glob("summarization_*.csv"), key=lambda p: p.stat().st_mtime)
        if len(csv_files) > keep_count:
            for old_file in csv_files[:-keep_count]:
                try:
                    old_file.unlink()
                    logger.debug("Cleaned up old debug file: %s", old_file)
                except Exception as e:
                    logger.error("Failed to remove old debug file %s: %s", old_file, e)

    def get_progress_string(self, chunk_num: int, total_chunks: int) -> str:
        """
        Create a progress string for display.

        Format: "Processing chunk 23/100 (23%) - Section: 'Plaintiff Testimony'"

        Args:
            chunk_num: Current chunk number
            total_chunks: Total number of chunks

        Returns:
            Progress string
        """
        percentage = int((chunk_num / total_chunks) * 100) if total_chunks > 0 else 0

        # Get section name
        if chunk_num <= len(self.df):
            filtered = self.df.loc[self.df["chunk_num"] == chunk_num, "section_detected"]
            section = filtered.iloc[0] if not filtered.empty else ""
            section_str = f" - Section: '{section}'" if section else ""
        else:
            section_str = ""

        return f"Processing chunk {chunk_num}/{total_chunks} ({percentage}%){section_str}"

    def generate_summary_metadata(self, summary_data: list[dict]) -> dict:
        """
        Analyzes summary data to extract overall metadata.

        Args:
            summary_data: List of dicts, each with 'title', 'summary', 'keywords'.

        Returns:
            A dictionary containing the extracted metadata.
        """
        if not summary_data:
            return {
                "overall_sentiment": "Neutral",
                "key_themes": [],
                "document_count": 0,
                "average_summary_length": 0,
                "most_frequent_keyword": None,
            }

        all_keywords = []
        total_summary_length = 0

        for item in summary_data:
            if item.get("keywords"):
                all_keywords.extend(item["keywords"])
            if "summary" in item:
                total_summary_length += len(item["summary"].split())

        document_count = len(summary_data)
        average_summary_length = total_summary_length // document_count if document_count > 0 else 0

        # Key Themes (unique keywords)
        key_themes = sorted(set(all_keywords))

        # Most Frequent Keyword
        most_frequent_keyword = None
        if all_keywords:
            from collections import Counter

            keyword_counts = Counter(all_keywords)
            most_frequent_keyword = keyword_counts.most_common(1)[0][0]

        return {
            "overall_sentiment": "Mixed (Placeholder)",  # Placeholder for future NLP sentiment analysis
            "key_themes": key_themes,
            "document_count": document_count,
            "average_summary_length": average_summary_length,
            "most_frequent_keyword": most_frequent_keyword,
        }
