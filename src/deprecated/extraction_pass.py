"""
Extraction Pass Processor for Two-Pass Summarization.

Pass 1 sends each chunk to the LLM with a focused extraction prompt that
captures claims, facts, relief, and testimony. The extracted facts are then
injected into Pass 2 (the normal summarization pass) so that headline
information cannot fade during progressive summarization.

This module is only active when enhanced_mode is enabled (GPU users by default).
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.core.ai.ollama_model_manager import OllamaModelManager
    from src.core.utils.chunk_scoring import ChunkScores

EXTRACTION_PROMPT_TEMPLATE = """Extract key legal information from this text. Be brief and specific.

CLAIMS: Any claims, allegations, or causes of action stated
FACTS: Key dates, amounts, events, injuries, diagnoses
RELIEF: Any damages, injunctions, or remedies sought
TESTIMONY: Key statements, admissions, or denials

Text:
{chunk_text}

Extraction (50-75 words):"""


class ExtractionPassProcessor:
    """
    Pass 1 processor for two-pass enhanced summarization.

    Sends each chunk to the LLM with a focused extraction prompt to capture
    structured legal facts before the summarization pass.

    Attributes:
        model_manager: OllamaModelManager for text generation.
        chunk_scores: Optional ChunkScores for skipping redundant chunks.
        stop_check: Optional callable returning True to cancel processing.
    """

    def __init__(
        self,
        model_manager: "OllamaModelManager",
        chunk_scores: "ChunkScores | None" = None,
        stop_check: Callable[[], bool] | None = None,
    ):
        """
        Initialize the extraction pass processor.

        Args:
            model_manager: OllamaModelManager instance for text generation.
            chunk_scores: Optional ChunkScores to skip redundant chunks.
            stop_check: Optional callable that returns True if should stop.
        """
        self.model_manager = model_manager
        self.chunk_scores = chunk_scores
        self.stop_check = stop_check

    def extract_from_chunks(
        self,
        chunks: list,
        status_reporter: Callable[[int, str], None] | None = None,
    ) -> list[str]:
        """
        Run extraction pass on all chunks.

        For each chunk, sends a focused extraction prompt to the LLM.
        Chunks flagged as redundant by chunk_scores are skipped (empty string).

        Args:
            chunks: List of chunk objects with .text attribute, or strings.
            status_reporter: Optional callback(percent, message) for progress.

        Returns:
            List of extraction results (one per chunk). Empty string for
            skipped or cancelled chunks.
        """
        total = len(chunks)
        results = []

        for i, chunk in enumerate(chunks):
            # Check for cancellation
            if self.stop_check and self.stop_check():
                logger.debug("[Pass 1] Cancelled at chunk %d/%d", i + 1, total)
                results.extend([""] * (total - len(results)))
                break

            # Skip redundant chunks
            if self.chunk_scores and i < len(self.chunk_scores.skip) and self.chunk_scores.skip[i]:
                reason = (
                    self.chunk_scores.skip_reason[i]
                    if i < len(self.chunk_scores.skip_reason)
                    else "redundant"
                )
                logger.debug("[Pass 1] Skipped chunk %d - %s", i + 1, reason)
                results.append("")
                continue

            # Report progress
            if status_reporter:
                pct = int((i / total) * 100)
                status_reporter(pct, f"[Pass 1] Extracting key facts from chunk {i + 1}/{total}")

            # Get chunk text
            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)

            # Build and send extraction prompt
            prompt = EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
            extraction = self.model_manager.generate_text(
                prompt=prompt,
                max_tokens=150,  # ~75 words * 2 tokens/word
            )

            results.append(extraction.strip())
            logger.debug(
                "[Pass 1] Chunk %d/%d: %d chars extracted",
                i + 1,
                total,
                len(extraction.strip()),
            )

        logger.info(
            "[Pass 1] Extraction complete: %d/%d chunks processed",
            sum(1 for r in results if r),
            total,
        )
        return results
