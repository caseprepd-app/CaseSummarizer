"""
Summary Post-Processor for CasePrepd

Handles post-processing of AI-generated summaries, including:
- Length enforcement (recursive condensation if over target)
- Future: sentiment analysis, keyword extraction, etc.

This module is AI-backend-agnostic - it works with any text generation function.
"""

from collections.abc import Callable

from src.config import (
    PROMPTS_DIR,
    SUMMARY_LENGTH_TOLERANCE,
    SUMMARY_MAX_CONDENSE_ATTEMPTS,
    USER_PROMPTS_DIR,
)
from src.core.prompting import PromptTemplateManager, get_prompt_config
from src.logging_config import debug_log


class SummaryPostProcessor:
    """
    Post-processes AI-generated summaries to ensure quality and compliance.

    Primary responsibility: Length enforcement via recursive condensation.

    This class is designed to be AI-backend-agnostic. It accepts a text generation
    function as a dependency, allowing it to work with Ollama, OpenAI, or any
    future backend.

    Example:
        post_processor = SummaryPostProcessor(
            generate_text_fn=ollama_manager.generate_text,
            prompt_template_manager=ollama_manager.prompt_template_manager
        )
        enforced_summary = post_processor.enforce_length(summary, target_words=200)
    """

    def __init__(
        self,
        generate_text_fn: Callable[[str, int], str],
        prompt_template_manager: PromptTemplateManager | None = None,
        tolerance: float | None = None,
        max_attempts: int | None = None,
        model_id: str | None = None,
    ):
        """
        Initialize the post-processor.

        Args:
            generate_text_fn: Function that generates text from a prompt.
                              Signature: fn(prompt: str, max_tokens: int) -> str
            prompt_template_manager: Manager for loading prompt templates.
                                    If None, creates a new instance.
            tolerance: Length tolerance (0.20 = 20% overage allowed).
                      Defaults to config value.
            max_attempts: Maximum condensation attempts.
                         Defaults to config value.
            model_id: Model identifier for loading correct prompt templates.
        """
        self.generate_text = generate_text_fn
        self.prompt_template_manager = prompt_template_manager or PromptTemplateManager(
            PROMPTS_DIR, USER_PROMPTS_DIR
        )
        self.prompt_config = get_prompt_config()
        self.model_id = model_id

        # Use config values as defaults, allow override
        self.tolerance = tolerance if tolerance is not None else SUMMARY_LENGTH_TOLERANCE
        self.max_attempts = (
            max_attempts if max_attempts is not None else SUMMARY_MAX_CONDENSE_ATTEMPTS
        )

    def enforce_length(
        self, summary: str, target_words: int, max_attempts: int | None = None
    ) -> str:
        """
        Enforce summary length by recursively condensing if over target.

        Uses configured tolerance: if summary exceeds target by more than the
        tolerance percentage, it will be condensed. Continues until within
        tolerance or max attempts reached.

        Args:
            summary: The summary text to check/condense
            target_words: Target word count
            max_attempts: Override max attempts (optional)

        Returns:
            str: Summary within target length or best effort after max attempts
        """
        attempts = max_attempts if max_attempts is not None else self.max_attempts
        max_acceptable_words = int(target_words * (1 + self.tolerance))

        actual_words = len(summary.split())
        attempt = 0

        debug_log(
            f"[LENGTH ENFORCE] Target: {target_words} words, "
            f"Max acceptable: {max_acceptable_words} words, "
            f"Actual: {actual_words} words"
        )

        while actual_words > max_acceptable_words and attempt < attempts:
            attempt += 1
            debug_log(
                f"[LENGTH ENFORCE] Attempt {attempt}/{attempts}: "
                f"Summary is {actual_words} words (>{max_acceptable_words}). Condensing..."
            )

            summary = self._condense_summary(summary, target_words)
            actual_words = len(summary.split())

            debug_log(f"[LENGTH ENFORCE] After condensation: {actual_words} words")

        if actual_words > max_acceptable_words:
            debug_log(
                f"[LENGTH ENFORCE] WARNING: After {attempts} attempts, "
                f"summary is still {actual_words} words. Returning best effort."
            )
        else:
            debug_log(
                f"[LENGTH ENFORCE] Success: {actual_words} words "
                f"(within {self.tolerance * 100:.0f}% tolerance of {target_words})"
            )

        return summary

    def _condense_summary(self, summary: str, target_words: int) -> str:
        """
        Condense an over-length summary to target word count.

        Uses a dedicated condensation prompt template that preserves key facts
        while reducing verbosity.

        Args:
            summary: The summary text to condense
            target_words: Target word count

        Returns:
            str: Condensed summary
        """
        model_id = self.model_id or "phi-3-mini"
        min_words, max_words_range = self.prompt_config.get_word_count_range(target_words)

        try:
            # Load condensation template (underscore prefix = internal use)
            template = self.prompt_template_manager.load_template(model_id, "_condense-summary")
            prompt = self.prompt_template_manager.format_template(
                template=template,
                min_words=min_words,
                max_words=target_words,
                max_words_range=max_words_range,
                case_text=summary,  # The summary becomes the input text
            )
        except FileNotFoundError:
            # Fallback: use a simple inline condensation prompt
            debug_log("[CONDENSE] Condensation template not found, using fallback prompt")
            prompt = f"""Condense the following legal case summary to approximately {target_words} words.
Preserve all key facts (parties, claims, damages, dates, status).
Remove redundant information and use more concise phrasing.

SUMMARY TO CONDENSE:
{summary}

CONDENSED SUMMARY:"""

        # Generate condensed version
        tokens_per_word = self.prompt_config.tokens_per_word
        buffer_multiplier = self.prompt_config.token_buffer_multiplier
        max_tokens = int(max_words_range * tokens_per_word * buffer_multiplier)

        return self.generate_text(prompt, max_tokens)

    def is_within_tolerance(self, summary: str, target_words: int) -> bool:
        """
        Check if a summary is within the acceptable length tolerance.

        Args:
            summary: The summary text to check
            target_words: Target word count

        Returns:
            bool: True if within tolerance, False if over
        """
        actual_words = len(summary.split())
        max_acceptable = int(target_words * (1 + self.tolerance))
        return actual_words <= max_acceptable

    def get_word_count(self, text: str) -> int:
        """
        Get word count of text.

        Args:
            text: Text to count words in

        Returns:
            int: Word count
        """
        return len(text.split())
