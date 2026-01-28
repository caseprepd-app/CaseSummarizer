"""
Transcript Cleaner Preprocessor

Comprehensive cleaning for court transcript PDFs.

Handles transcript-specific patterns not covered by other preprocessors:
- Page numbers (sequential throughout document)
- Inline concordance citations (embedded page:line references)
- Aggressive whitespace normalization

NOTE: Line numbers, headers/footers, title pages, and index pages are handled
by their respective dedicated preprocessors (LineNumberRemover, HeaderFooterRemover,
TitlePageRemover, IndexPageRemover). This module only handles patterns unique to
transcripts that aren't covered by those specialized preprocessors.

Usage:
    Automatically included in the default preprocessing pipeline.
    Runs after LineNumberRemover, before QAConverter.
"""

import json
import logging
import re
from pathlib import Path

from src.core.preprocessing.base import BasePreprocessor, PreprocessingResult

logger = logging.getLogger(__name__)


def _load_transcript_patterns() -> dict:
    """
    Load transcript cleaning patterns from config file.

    Returns:
        Dict with pattern configurations, or empty dict if file not found.
    """
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "transcript_patterns.json"
    if not config_path.exists():
        logger.debug("Config not found: %s, using defaults", config_path)
        return {}
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Error loading config: %s, using defaults", e)
        return {}


class TranscriptCleaner(BasePreprocessor):
    """
    Cleans court transcript text by removing page numbers, certification
    blocks, index pages, and normalizing whitespace.

    This preprocessor is designed to be safe - if patterns don't match,
    the original text is returned unchanged.

    Example:
        cleaner = TranscriptCleaner()
        result = cleaner.process(raw_transcript_text)
        cleaned_text = result.text
    """

    name = "Transcript Cleaner"

    def process(self, text: str) -> PreprocessingResult:
        """
        Apply all transcript cleaning operations.

        Order:
        1. Remove page numbers (sequential throughout document)
        2. Strip inline concordance citations
        3. Normalize whitespace (final cleanup)

        NOTE: Index page removal is handled by IndexPageRemover, which has
        superior logic for handling PDFs where newlines were stripped during
        extraction. TranscriptCleaner's index detection was removed to avoid
        redundancy.

        Args:
            text: Raw transcript text

        Returns:
            PreprocessingResult with cleaned text and metadata
        """
        if not text:
            return PreprocessingResult(text=text, changes_made=0)

        original_len = len(text)
        metadata = {}

        # Step 1: Remove page numbers
        text, page_nums_removed = self._remove_page_numbers(text)
        metadata["page_numbers_removed"] = page_nums_removed

        # Step 2: Strip inline concordance citations
        # Catches patterns like "told (5) 959:14;1003:24" embedded in body text
        text, inline_citations_removed = self._strip_inline_citations(text)
        metadata["inline_citations_removed"] = inline_citations_removed

        # Step 3: Normalize whitespace
        text = self._normalize_whitespace(text)

        changes = original_len - len(text)
        metadata["chars_removed"] = changes

        if changes > 0:
            logger.debug(
                "Removed %s chars: page_nums=%s, inline_citations=%s",
                changes,
                page_nums_removed,
                inline_citations_removed,
            )

        return PreprocessingResult(
            text=text,
            changes_made=changes,
            metadata=metadata,
        )

    def _remove_page_numbers(self, text: str, min_pages: int = 3) -> tuple[str, int]:
        """
        Remove sequential page numbers from transcript.

        Page numbers differ from line numbers:
        - Can range into hundreds or thousands
        - Appear in sequential order throughout the document
        - Typically appear once each (at page boundaries)

        Detection:
        1. Find standalone numbers appearing exactly once
        2. Check if they form a sequential pattern in document order
        3. Remove if sequential pattern confirmed

        Args:
            text: Input text
            min_pages: Minimum sequential numbers to confirm detection

        Returns:
            Tuple of (cleaned_text, count_of_removed_numbers)
        """
        lines = text.split("\n")
        standalone_number_pattern = re.compile(r"^\s*(\d+)\s*$")

        # Find all standalone numbers and their positions
        potential_page_numbers: list[tuple[int, int]] = []
        for i, line in enumerate(lines):
            match = standalone_number_pattern.match(line)
            if match:
                num = int(match.group(1))
                potential_page_numbers.append((i, num))

        if len(potential_page_numbers) < min_pages:
            return text, 0

        # Group by value to find single-occurrence numbers
        occurrences_by_value: dict = {}
        for line_idx, num in potential_page_numbers:
            if num not in occurrences_by_value:
                occurrences_by_value[num] = []
            occurrences_by_value[num].append(line_idx)

        # Page numbers appear exactly once
        single_occurrence = {
            num: positions[0]
            for num, positions in occurrences_by_value.items()
            if len(positions) == 1
        }

        if len(single_occurrence) < min_pages:
            return text, 0

        # Check if numbers are sequential and positions increase
        sorted_nums = sorted(single_occurrence.keys())
        sorted_positions = [single_occurrence[n] for n in sorted_nums]

        # Positions must increase (page 1 before page 2)
        if not all(
            sorted_positions[i] < sorted_positions[i + 1] for i in range(len(sorted_positions) - 1)
        ):
            return text, 0

        # Numbers must be roughly sequential (allow gaps up to 3)
        max_gap = 3
        if not all(
            sorted_nums[i + 1] - sorted_nums[i] <= max_gap for i in range(len(sorted_nums) - 1)
        ):
            return text, 0

        # Remove identified page number lines
        page_line_indices: set[int] = set(single_occurrence.values())
        cleaned_lines = [line for i, line in enumerate(lines) if i not in page_line_indices]

        return "\n".join(cleaned_lines), len(page_line_indices)

    def _strip_inline_citations(self, text: str) -> tuple[str, int]:
        """
        Strip inline concordance citations embedded in body text.

        Patterns are loaded from config/transcript_patterns.json.
        Falls back to hardcoded defaults if config unavailable.

        Pattern examples:
        - "told (5) 959:14;1003:24" -> "told"
        - "Antonio 59:12;60:21;62:12" -> "Antonio"

        Args:
            text: Input text

        Returns:
            Tuple of (cleaned_text, count_of_patterns_removed)
        """
        config = _load_transcript_patterns()
        patterns = config.get("inline_citation_patterns", [])

        # Fallback defaults if config not available
        if not patterns:
            patterns = [
                {
                    "name": "word_with_count_refs",
                    "pattern": r"\b(\w+)\s*\(\d+\)\s*(?:\d{2,4}:\d{1,2}[;,\s]*)+",
                    "replacement": r"\1",
                    "enabled": True,
                },
                {
                    "name": "name_with_refs",
                    "pattern": r"\b([A-Z][a-z]+)\s+\d{2,4}:\d{1,2}(?:[;,]\s*\d{2,4}:\d{1,2})*",
                    "replacement": r"\1",
                    "enabled": True,
                },
            ]

        total_count = 0
        cleaned_text = text

        for pattern_config in patterns:
            if not pattern_config.get("enabled", True):
                continue

            try:
                pattern = re.compile(pattern_config["pattern"])
                replacement = pattern_config.get("replacement", r"\1")
                cleaned_text, count = pattern.subn(replacement, cleaned_text)
                if count > 0:
                    logger.debug(
                        "Pattern '%s' stripped %s match(es)",
                        pattern_config.get("name", "unnamed"),
                        count,
                    )
                    total_count += count
            except re.error as e:
                logger.debug(
                    "Invalid regex in pattern '%s': %s",
                    pattern_config.get("name", "unnamed"),
                    e,
                )

        return cleaned_text, total_count

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize excessive whitespace in transcript.

        Operations:
        1. Replace 3+ newlines with exactly 2 (one blank line max)
        2. Normalize multiple spaces within lines
        3. Strip leading/trailing whitespace

        Args:
            text: Input text

        Returns:
            Text with normalized whitespace
        """
        # Reduce multiple blank lines to one
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Normalize spaces within lines
        lines = text.split("\n")
        normalized_lines = []
        for line in lines:
            # Multiple spaces -> single space
            normalized_line = re.sub(r" {2,}", " ", line)
            normalized_lines.append(normalized_line.strip())

        text = "\n".join(normalized_lines)

        return text.strip()
