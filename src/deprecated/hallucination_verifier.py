"""
Hallucination Verification Module for CasePrepd Q&A.

Uses LettuceDetect to analyze answer text against source context,
returning span-level hallucination probabilities for color-coded display.

Architecture:
    - Wraps HallucinationDetector from lettucedetect library
    - Returns VerificationResult with spans and overall reliability
    - Lazy-loads model to avoid startup overhead (~68MB, loads once per session)
    - Supports bundled models for Windows installer (no network calls)

Usage:
    verifier = HallucinationVerifier()
    result = verifier.verify(
        answer="The temperature was 98.6°F",
        context="Q: You took his temperature? A: Yes.",
        question="What were the vital signs?"
    )
    # result.spans contains color-coding info
    # result.overall_reliability indicates answer quality
    # result.answer_rejected is True if reliability < 50%

Model Loading:
    1. First checks for bundled model at models/tinylettuce-ettin-68m-en/
    2. If bundled model exists, uses local_files_only=True (no network)
    3. If not found, downloads from HuggingFace (development mode)
"""

import logging
from dataclasses import dataclass

from src.config import (
    HALLUCINATION_MODEL_LOCAL_PATH,
    HF_CACHE_DIR,
)
from src.core.qa.verification_config import (
    ANSWER_REJECTION_THRESHOLD,
    VERIFIER_MODEL_PATH,
)

logger = logging.getLogger(__name__)


@dataclass
class VerifiedSpan:
    """A span of text with its hallucination probability."""

    text: str
    start: int
    end: int
    hallucination_prob: float  # 0.0 = verified, 1.0 = hallucinated


@dataclass
class VerificationResult:
    """Result of hallucination verification for an answer."""

    spans: list[VerifiedSpan]
    overall_reliability: float  # 1.0 - weighted average hallucination prob
    answer_rejected: bool  # True if overall reliability < threshold


class HallucinationVerifier:
    """
    Verifies Q&A answers for hallucination using LettuceDetect.

    Uses KRLabsOrg/tinylettuce-ettin-68m-en model (~68MB, 75% F1).
    Runs on CPU, processes ~5-10 answers per second.

    The model is lazy-loaded on first use to avoid startup delay.
    """

    def __init__(self):
        """Initialize verifier."""
        self._detector = None  # Lazy load
        self._model_path = VERIFIER_MODEL_PATH

    def _load_detector(self) -> None:
        """
        Load the LettuceDetect model (called once on first use).

        Loading strategy:
        1. Set HuggingFace cache directory to models/.hf_cache
        2. If bundled model exists at models/tinylettuce-ettin-68m-en/,
           use local_files_only=True to prevent network calls
        3. Otherwise, download from HuggingFace (development mode)
        """
        from src.core.utils.model_loader import resolve_model_path, set_hf_cache_env

        set_hf_cache_env(HF_CACHE_DIR)

        model_path, local_only = resolve_model_path(
            HALLUCINATION_MODEL_LOCAL_PATH, self._model_path
        )

        from lettucedetect.models.inference import HallucinationDetector

        self._detector = HallucinationDetector(
            method="transformer", model_path=model_path, local_files_only=local_only
        )

        logger.debug("Model loaded successfully")

    def verify(self, answer: str, context: str, question: str) -> VerificationResult:
        """
        Verify answer text against source context.

        Args:
            answer: Generated answer to verify
            context: Retrieved context (citation text from documents)
            question: Original question asked

        Returns:
            VerificationResult with spans, reliability score, and rejection flag
        """
        if not answer or not answer.strip():
            return VerificationResult(spans=[], overall_reliability=0.0, answer_rejected=True)

        # Lazy load detector on first use
        if self._detector is None:
            try:
                self._load_detector()
            except Exception as e:
                logger.debug("Failed to load model: %s", e)
                return VerificationResult(
                    spans=[
                        VerifiedSpan(text=answer, start=0, end=len(answer), hallucination_prob=0.5)
                    ],
                    overall_reliability=0.5,
                    answer_rejected=False,
                )

        try:
            # Warn if combined input approaches LettuceDetect's 4096-token limit
            # (~4 chars/token conservative estimate)
            combined_len = len(context or "") + len(question) + len(answer)
            char_limit = 4096 * 4
            if combined_len > char_limit * 0.8:
                logger.warning(
                    "Hallucination verifier input is %.0f%% of 4096-token limit "
                    "(%d chars / ~%d limit) — detection may miss truncated content",
                    (combined_len / char_limit) * 100,
                    combined_len,
                    char_limit,
                )

            # Get span predictions from LettuceDetect
            # Returns list of dicts with: text, start, end, confidence
            predictions = self._detector.predict(
                context=[context] if context else [""],
                question=question,
                answer=answer,
                output_format="spans",
            )

            logger.debug("Raw predictions: %s", predictions)

            # Convert to VerifiedSpan objects and fill gaps
            spans = self._build_complete_spans(answer, predictions)

            # Calculate overall reliability
            overall_reliability = self._calculate_reliability(spans, answer)
            answer_rejected = overall_reliability < ANSWER_REJECTION_THRESHOLD

            logger.debug(
                "Reliability: %.2f%%, Rejected: %s",
                overall_reliability * 100,
                answer_rejected,
            )

            return VerificationResult(
                spans=spans,
                overall_reliability=overall_reliability,
                answer_rejected=answer_rejected,
            )

        except Exception as e:
            logger.debug("Error during verification: %s", e)

            # On error, return the full answer as unverified (uncertain)
            return VerificationResult(
                spans=[
                    VerifiedSpan(
                        text=answer,
                        start=0,
                        end=len(answer),
                        hallucination_prob=0.35,  # Slightly uncertain
                    )
                ],
                overall_reliability=0.65,
                answer_rejected=False,
            )

    def _build_complete_spans(self, answer: str, predictions: list[dict]) -> list[VerifiedSpan]:
        """
        Build complete span list covering the entire answer text.

        LettuceDetect only returns spans for hallucinated portions.
        This method fills in the gaps with "verified" spans (prob=0.0).

        Args:
            answer: Full answer text
            predictions: Raw predictions from LettuceDetect

        Returns:
            List of VerifiedSpan covering entire answer
        """
        if not predictions:
            # No hallucinations detected - entire answer is verified
            return [VerifiedSpan(text=answer, start=0, end=len(answer), hallucination_prob=0.0)]

        spans = []
        current_pos = 0

        # Sort predictions by start position
        sorted_preds = sorted(predictions, key=lambda p: p.get("start", 0))

        for pred in sorted_preds:
            start = pred.get("start", 0)
            end = pred.get("end", start)
            confidence = pred.get("confidence", 0.5)
            text = pred.get("text", answer[start:end])

            # Fill gap before this hallucinated span with verified text
            if start > current_pos:
                gap_text = answer[current_pos:start]
                if gap_text.strip():  # Only add non-empty gaps
                    spans.append(
                        VerifiedSpan(
                            text=gap_text,
                            start=current_pos,
                            end=start,
                            hallucination_prob=0.0,  # Verified
                        )
                    )

            # Add the hallucinated span
            spans.append(
                VerifiedSpan(text=text, start=start, end=end, hallucination_prob=confidence)
            )

            current_pos = end

        # Fill any remaining text after last hallucination
        if current_pos < len(answer):
            remaining_text = answer[current_pos:]
            if remaining_text.strip():
                spans.append(
                    VerifiedSpan(
                        text=remaining_text,
                        start=current_pos,
                        end=len(answer),
                        hallucination_prob=0.0,  # Verified
                    )
                )

        return spans

    def _calculate_reliability(self, spans: list[VerifiedSpan], answer: str) -> float:
        """
        Calculate overall reliability score for the answer.

        Uses character-weighted average: longer spans have more influence.
        Reliability = 1.0 - weighted_average_hallucination_prob

        Args:
            spans: List of verified spans with probabilities
            answer: Original answer text (for fallback)

        Returns:
            Overall reliability score (0.0 to 1.0, higher = better)
        """
        if not spans:
            return 0.0

        total_chars = 0
        weighted_prob_sum = 0.0

        for span in spans:
            char_count = len(span.text)
            total_chars += char_count
            weighted_prob_sum += span.hallucination_prob * char_count

        if total_chars == 0:
            return 0.0

        avg_hallucination_prob = weighted_prob_sum / total_chars
        reliability = 1.0 - avg_hallucination_prob

        return max(0.0, min(1.0, reliability))  # Clamp to [0, 1]
