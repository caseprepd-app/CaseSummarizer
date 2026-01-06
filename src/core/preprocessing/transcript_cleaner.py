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

import re
from typing import List, Set, Tuple

from src.core.preprocessing.base import BasePreprocessor, PreprocessingResult
from src.logging_config import debug_log


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

        # Step 2: Remove certification block
        text, cert_removed = self._remove_certification_block(text)
        metadata["certification_removed"] = cert_removed

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
                f"index_lines={index_lines_removed}"
            )

        return PreprocessingResult(
            text=text,
            changes_made=changes,
            metadata=metadata,
        )

    def _remove_page_numbers(self, text: str, min_pages: int = 3) -> Tuple[str, int]:
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
        potential_page_numbers: List[Tuple[int, int]] = []
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
        page_line_indices: Set[int] = set(single_occurrence.values())
        cleaned_lines = [line for i, line in enumerate(lines) if i not in page_line_indices]

        return "\n".join(cleaned_lines), len(page_line_indices)

    def _remove_certification_block(self, text: str) -> Tuple[str, bool]:
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

    def _remove_index_pages(self, text: str, min_cluster_size: int = 5) -> Tuple[str, int]:
        """
        Remove index/concordance pages from transcript end.

        Index entries have formats like:
        - "Fabuloso - 34:5" (word - page:line)
        - "objection 34:5, 45:6" (multiple references)

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
        lines = text.split("\n")

        # Pattern: WORD(s) [separator] PAGE:LINE
        single_ref = re.compile(
            r"^[\s]*"
            r"[A-Za-z][A-Za-z\s\.\,\-\']*"  # Term
            r"[\s]*[-\u2013\u2014/]?[\s]*"  # Separator
            r"\d{1,4}"  # Page
            r"[\s]*[:/][\s]*"  # Page-line separator
            r"\d{1,2}"  # Line
            r"[\s]*$"
        )

        # Pattern: Multiple references
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

        # Mark each line as index-like or not
        is_index_line: List[bool] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                is_index_line.append(False)
            elif single_ref.match(line) or multi_ref.match(line):
                is_index_line.append(True)
            else:
                is_index_line.append(False)

        # Find clusters of index lines
        clusters_to_remove: List[Tuple[int, int]] = []
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

                if index_count >= min_cluster_size:
                    clusters_to_remove.append((cluster_start, cluster_end))
                    i = cluster_end
                else:
                    i += 1
            else:
                i += 1

        if not clusters_to_remove:
            return text, 0

        # Build set of lines to remove
        lines_to_remove: Set[int] = set()
        for start, end in clusters_to_remove:
            for idx in range(start, end):
                lines_to_remove.add(idx)

        cleaned_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]

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
