"""
Transcript Cleaner Preprocessor

Comprehensive cleaning for court transcript PDFs.

Handles transcript-specific patterns not covered by other preprocessors:
- Page numbers (sequential throughout document)
- Certification blocks (end-of-document attestation)
- Index pages (concordance at end)
- Aggressive whitespace normalization

NOTE: Line numbers, headers/footers, and title pages are handled by their
respective dedicated preprocessors (LineNumberRemover, HeaderFooterRemover,
TitlePageRemover). This module only handles patterns unique to transcripts.

Usage:
    Automatically included in the default preprocessing pipeline.
    Runs after LineNumberRemover, before QAConverter.
"""

import json
import re
from pathlib import Path

from src.core.preprocessing.base import BasePreprocessor, PreprocessingResult
from src.logging_config import debug_log


def _load_transcript_patterns() -> dict:
    """
    Load transcript cleaning patterns from config file.

    Returns:
        Dict with pattern configurations, or empty dict if file not found.
    """
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "transcript_patterns.json"
    if not config_path.exists():
        debug_log(f"[TranscriptCleaner] Config not found: {config_path}, using defaults")
        return {}
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        debug_log(f"[TranscriptCleaner] Error loading config: {e}, using defaults")
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
        2. Remove certification block (attestation at end)
        3. Remove index pages (concordance at end)
        4. Normalize whitespace (final cleanup)

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

        # Step 2: Certification block removal DISABLED (Session 84)
        # Was causing false positives - "I HEREBY CERTIFY" appears in affidavits,
        # certificates of merit, etc. throughout legal documents, not just at end.
        cert_removed = False
        metadata["certification_removed"] = cert_removed

        # Step 2.5: Strip inline concordance citations
        # Catches patterns like "told (5) 959:14;1003:24" embedded in body text
        text, inline_citations_removed = self._strip_inline_citations(text)
        metadata["inline_citations_removed"] = inline_citations_removed

        # Step 3: Remove index pages
        text, index_lines_removed = self._remove_index_pages(text)
        metadata["index_lines_removed"] = index_lines_removed

        # Step 4: Normalize whitespace
        text = self._normalize_whitespace(text)

        changes = original_len - len(text)
        metadata["chars_removed"] = changes

        if changes > 0:
            debug_log(
                f"[TranscriptCleaner] Removed {changes} chars: "
                f"page_nums={page_nums_removed}, cert={cert_removed}, "
                f"inline_citations={inline_citations_removed}, "
                f"index_lines={index_lines_removed}"
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
                    debug_log(
                        f"[TranscriptCleaner] Pattern '{pattern_config.get('name', 'unnamed')}' "
                        f"stripped {count} match(es)"
                    )
                    total_count += count
            except re.error as e:
                debug_log(
                    f"[TranscriptCleaner] Invalid regex in pattern "
                    f"'{pattern_config.get('name', 'unnamed')}': {e}"
                )

        return cleaned_text, total_count

    def _remove_certification_block(self, text: str) -> tuple[str, bool]:
        """
        Remove certification/attestation block from transcript end.

        Court reporters certify accuracy at the end of transcripts:
        - "CERTIFIED TO BE A TRUE AND ACCURATE TRANSCRIPT"
        - "I HEREBY CERTIFY..."
        - Often preceded by asterisks

        Args:
            text: Input text

        Returns:
            Tuple of (cleaned_text, was_removed)
        """
        certification_patterns = [
            r"CERTIFIED\s+TO\s+BE\s+A\s+TRUE",
            r"CERTIFICATE\s+OF\s+TRANSCRIPT",
            r"REPORTER\'?S?\s+CERTIFICATE",
            r"I\s+HEREBY\s+CERTIFY",
            r"CERTIFICATION\s*$",
            r"C\s*E\s*R\s*T\s*I\s*F\s*I\s*C\s*A\s*T\s*E",  # S P A C E D
            r"\*{10,}",  # Row of 10+ asterisks (separator)
        ]

        combined_pattern = "|".join(f"({p})" for p in certification_patterns)
        match = re.search(combined_pattern, text, re.IGNORECASE)

        if not match:
            return text, False

        # Find start of the line containing the match
        match_start = match.start()
        line_start = text.rfind("\n", 0, match_start)
        line_start = 0 if line_start == -1 else line_start + 1

        return text[:line_start].rstrip(), True

    def _remove_index_pages(self, text: str, min_cluster_size: int = 5) -> tuple[str, int]:
        """
        Remove index/concordance pages from transcript end.

        Patterns and thresholds loaded from config/transcript_patterns.json.

        Detection:
        1. Identify lines matching index entry patterns
        2. Look for clusters of consecutive index-like lines
        3. Remove entire clusters (not isolated matches)

        Args:
            text: Input text
            min_cluster_size: Minimum consecutive entries to confirm index

        Returns:
            Tuple of (cleaned_text, lines_removed)
        """
        config = _load_transcript_patterns()
        index_config = config.get("index_line_detection", {})
        thresholds = config.get("index_detection_thresholds", {})

        # Load thresholds from config or use defaults
        min_cluster = thresholds.get("min_cluster_size", min_cluster_size)
        min_refs_for_index = thresholds.get("min_refs_for_index_line", 3)
        min_refs_with_count = thresholds.get("min_refs_with_word_count", 2)
        digit_letter_ratio = thresholds.get("digit_to_letter_ratio", 1.5)

        lines = text.split("\n")

        # Pattern: WORD(s) [separator] PAGE:LINE (simple format)
        single_ref = re.compile(
            r"^[\s]*"
            r"[A-Za-z][A-Za-z\s\.\,\-\']*"  # Term
            r"[\s]*[-\u2013\u2014/]?[\s]*"  # Separator
            r"\d{1,4}"  # Page
            r"[\s]*[:/][\s]*"  # Page-line separator
            r"\d{1,2}"  # Line
            r"[\s]*$"
        )

        # Pattern: Multiple references (traditional format)
        multi_ref = re.compile(
            r"^[\s]*"
            r"[A-Za-z][A-Za-z\s\.\,\-\']*"
            r"[\s]*[-\u2013\u2014/]?[\s]*"
            r"("
            r"\d{1,4}[\s]*[:/][\s]*\d{1,2}"
            r"[\s]*[,;]?[\s]*"
            r")+"
            r"[\s]*$"
        )

        # Load detection patterns from config or use defaults
        dense_refs_pattern = index_config.get("dense_refs", {}).get("pattern", r"\d{3,4}:\d{1,2}")
        word_count_pattern = index_config.get("word_with_count", {}).get(
            "pattern", r"[A-Za-z]+\s*\(\d+\)"
        )
        ref_chain_pattern = index_config.get("ref_chain", {}).get(
            "pattern", r"\d{3,4}:\d{1,2}\s*[;,]\s*\d{3,4}:\d{1,2}"
        )

        dense_refs = re.compile(dense_refs_pattern)
        word_with_count = re.compile(word_count_pattern)
        ref_chain = re.compile(ref_chain_pattern)

        def is_concordance_index_line(line: str) -> bool:
            """
            Check if line looks like concordance/word index content.

            Detection is purely regex-based - no keyword dependencies.
            Identifies lines with high density of page:line references.
            """
            stripped = line.strip()
            if not stripped:
                return False

            # Count page:line references (like 1260:12)
            ref_count = len(dense_refs.findall(line))

            # Count word(count) patterns (like "Counsel (59)")
            word_count_matches = len(word_with_count.findall(line))

            # Check for reference chains (semicolon/comma separated refs)
            has_ref_chain = bool(ref_chain.search(line))

            # Index lines have high density of page:line refs
            # Thresholds loaded from config
            if ref_count >= min_refs_for_index:
                return True

            # Lines with refs AND word counts are likely index
            if ref_count >= min_refs_with_count and word_count_matches >= 1:
                return True

            # Lines with reference chains are index
            if has_ref_chain and ref_count >= min_refs_with_count:
                return True

            # Lines that are mostly numbers/punctuation with few words
            # Calculate ratio of digits+punctuation to letters
            digits_punct = sum(1 for c in stripped if c.isdigit() or c in ":;,()[]")
            letters = sum(1 for c in stripped if c.isalpha())
            if letters > 0 and digits_punct / letters > digit_letter_ratio and ref_count >= 1:
                return True

            return False

        # Mark each line as index-like or not
        is_index_line: list[bool] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                is_index_line.append(False)
            elif single_ref.match(line) or multi_ref.match(line) or is_concordance_index_line(line):
                is_index_line.append(True)
            else:
                is_index_line.append(False)

        # Find clusters of index lines
        clusters_to_remove: list[tuple[int, int]] = []
        i = 0

        while i < len(is_index_line):
            if is_index_line[i]:
                cluster_start = i
                index_count = 0
                gap = 0

                j = i
                while j < len(is_index_line) and gap < 3:
                    if is_index_line[j]:
                        index_count += 1
                        gap = 0
                    else:
                        gap += 1
                    j += 1

                cluster_end = j - gap

                if index_count >= min_cluster:
                    clusters_to_remove.append((cluster_start, cluster_end))
                    i = cluster_end
                else:
                    i += 1
            else:
                i += 1

        if not clusters_to_remove:
            return text, 0

        # Build set of lines to remove
        lines_to_remove: set[int] = set()
        for start, end in clusters_to_remove:
            for idx in range(start, end):
                lines_to_remove.add(idx)

        cleaned_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]

        debug_log(
            f"[TranscriptCleaner] Removed {len(lines_to_remove)} index lines "
            f"from {len(clusters_to_remove)} cluster(s)"
        )

        return "\n".join(cleaned_lines), len(lines_to_remove)

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
