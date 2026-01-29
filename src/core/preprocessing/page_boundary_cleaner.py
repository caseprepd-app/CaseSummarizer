"""
Page Boundary Cleaner Preprocessor

Cleans artifacts that appear when PDF extraction collapses page boundaries
into a single line of text. This happens when extractors don't preserve
newlines between pages, causing line numbers, page numbers, headers, and
footer initials to merge into body text.

Example of the problem:
    "...appearance of this 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
     19 20 21 22 23 24 sn Proceedings 29 1 Court."

After cleaning:
    "...appearance of this Court."

Three-pass approach:
    Pass 1: Detect and remove collapsed line-number runs (1 2 3 ... N)
    Pass 2: Detect sequential page numbers scattered through the document
    Pass 3: Clean debris around page-number anchors (initials, headers)

Design: Pass-through if no patterns detected. Logs what was removed.
"""

import logging
import re

from src.core.preprocessing.base import BasePreprocessor, PreprocessingResult

logger = logging.getLogger(__name__)

# Speaker markers that signal real content (stop cleaning here)
SPEAKER_MARKERS = re.compile(
    r"(?:THE\s+COURT|MR\.|MS\.|MRS\.|Q\.|A\.|BY\s+MR\.|BY\s+MS\.)",
    re.IGNORECASE,
)

# Known header words found at page tops in legal transcripts
HEADER_WORDS = {
    "proceedings",
    "court",
    "direct",
    "cross",
    "redirect",
    "recross",
    "examination",
    "continued",
    "voir",
    "dire",
    "colloquy",
    "summation",
    "argument",
}


class PageBoundaryCleaner(BasePreprocessor):
    """
    Cleans page-boundary artifacts from collapsed PDF text.

    Runs after LineNumberRemover in the pipeline. Uses page-number
    detection as anchors to find and clean surrounding debris
    (reporter initials, header words).

    Safe: returns text unchanged if no patterns detected.
    """

    name = "Page Boundary Cleaner"

    def process(self, text: str) -> PreprocessingResult:
        """
        Clean page boundary artifacts from text.

        Args:
            text: Input text, potentially with collapsed page boundaries

        Returns:
            PreprocessingResult with cleaned text and metadata
        """
        if not text:
            return PreprocessingResult(text=text, changes_made=0)

        changes = 0
        metadata = {}

        # Pass 1: Remove collapsed line-number runs
        text, run_count = self._remove_line_number_runs(text)
        changes += run_count
        metadata["line_number_runs_removed"] = run_count

        # Pass 2: Detect sequential page numbers
        page_numbers = self._detect_page_numbers(text)
        metadata["page_numbers_detected"] = len(page_numbers)

        # Pass 3: Clean debris around page-number anchors
        if page_numbers:
            text, debris_count = self._clean_page_boundaries(text, page_numbers)
            changes += debris_count
            metadata["boundary_debris_removed"] = debris_count

        if changes > 0:
            logger.info(
                "Page boundary cleaner: %d changes (runs=%d, page_nums=%d, debris=%d)",
                changes,
                run_count,
                len(page_numbers),
                metadata.get("boundary_debris_removed", 0),
            )

        return PreprocessingResult(text=text, changes_made=changes, metadata=metadata)

    def _remove_line_number_runs(self, text: str) -> tuple[str, int]:
        """
        Remove collapsed line-number runs (e.g., '1 2 3 4 ... 24').

        Detects runs of 5+ consecutive integers starting from 1.
        Uses consistency of the max value N across runs as confirmation.

        Args:
            text: Input text

        Returns:
            (cleaned_text, number_of_runs_removed)
        """
        # Find runs of consecutive integers starting from 1
        # Pattern: sequence of space-separated numbers where each is prev+1
        # We scan for "1 2 3" as an anchor, then extend forward
        run_pattern = re.compile(r"(?<!\d)1\s+2\s+3\s+")

        runs_found = []
        for match in run_pattern.finditer(text):
            start = match.start()
            # Extend the run forward from 4 onward
            pos = match.end()
            last_num = 3

            while pos < len(text):
                # Try to match next consecutive number
                next_num = last_num + 1
                next_str = str(next_num)
                # Skip whitespace
                ws_end = pos
                while ws_end < len(text) and text[ws_end] in " \t":
                    ws_end += 1

                if text[ws_end : ws_end + len(next_str)] == next_str:
                    # Check the char after the number isn't a digit
                    after_pos = ws_end + len(next_str)
                    if after_pos < len(text) and text[after_pos].isdigit():
                        break
                    last_num = next_num
                    pos = after_pos
                else:
                    break

            if last_num >= 5:
                # Valid run: at least 1-5
                runs_found.append((start, pos, last_num))

        if not runs_found:
            return text, 0

        # Check consistency: do runs end at the same N?
        end_values = [r[2] for r in runs_found]
        if len(set(end_values)) <= 2:
            # Consistent — these are line-number runs
            logger.debug(
                "Found %d line-number runs ending at %s",
                len(runs_found),
                sorted(set(end_values)),
            )
        elif len(runs_found) >= 3:
            # Multiple runs, allow some variation
            logger.debug(
                "Found %d line-number runs with varying endpoints: %s",
                len(runs_found),
                sorted(set(end_values)),
            )
        else:
            # Only 1-2 runs with no consistency — could be false positive
            # Still remove if run is long enough (>= 10 numbers)
            runs_found = [r for r in runs_found if r[2] >= 10]
            if not runs_found:
                return text, 0

        # Remove runs (process from end to preserve indices)
        result = text
        for start, end, _ in reversed(runs_found):
            result = result[:start] + result[end:]

        return result, len(runs_found)

    def _detect_page_numbers(self, text: str) -> list[int]:
        """
        Detect sequential page numbers scattered through text.

        Page numbers are isolated numbers that form a document-wide
        sequential series (e.g., 29, 30, 31, ... 85). They differ
        from line numbers: line numbers appear as dense runs starting
        from 1; page numbers appear isolated with content between them.

        Args:
            text: Input text

        Returns:
            List of character positions where page numbers were found
        """
        # Find all standalone numbers (surrounded by spaces/boundaries)
        number_pattern = re.compile(r"(?<!\d)(\d{1,4})(?!\d)")
        candidates = []

        for match in number_pattern.finditer(text):
            num = int(match.group(1))
            # Page numbers are typically > 1 and < 5000
            if 2 <= num <= 5000:
                candidates.append((match.start(), num))

        if len(candidates) < 3:
            return []

        # Look for sequential subsequences
        # Build a graph of candidates where each points to the next
        # sequential number that appears later in the text
        best_chain = []

        for i, (pos_i, num_i) in enumerate(candidates):
            chain = [(pos_i, num_i)]
            expected = num_i + 1

            for j in range(i + 1, len(candidates)):
                pos_j, num_j = candidates[j]
                if num_j == expected:
                    chain.append((pos_j, num_j))
                    expected = num_j + 1

            if len(chain) > len(best_chain):
                best_chain = chain

        # Need at least 3 sequential page numbers to confirm
        if len(best_chain) < 3:
            return []

        # Verify spacing: page numbers should have substantial content
        # between them (at least 50 chars on average)
        positions = [pos for pos, _ in best_chain]
        avg_gap = sum(positions[i + 1] - positions[i] for i in range(len(positions) - 1)) / max(
            len(positions) - 1, 1
        )

        if avg_gap < 50:
            # Too close together — probably not page numbers
            return []

        logger.debug(
            "Detected %d sequential page numbers (%d-%d), avg gap %.0f chars",
            len(best_chain),
            best_chain[0][1],
            best_chain[-1][1],
            avg_gap,
        )

        return [pos for pos, _ in best_chain]

    def _clean_page_boundaries(self, text: str, page_positions: list[int]) -> tuple[str, int]:
        """
        Clean debris around page-number anchors.

        For each page number position:
        - Before: remove isolated 1-3 char lowercase tokens (initials)
        - The page number itself: remove it
        - After: remove known header words and standalone numbers
        - Stop at content boundaries (speaker markers, sentences, 5+ char words)

        Args:
            text: Input text
            page_positions: Character positions of detected page numbers

        Returns:
            (cleaned_text, count_of_debris_items_removed)
        """
        # Re-find the actual number matches at these positions
        number_pattern = re.compile(r"(?<!\d)(\d{1,4})(?!\d)")
        removals = []  # (start, end) ranges to remove
        debris_count = 0

        for page_pos in page_positions:
            # Find the number at this position
            match = number_pattern.match(text, page_pos)
            if not match:
                continue

            page_num_start = match.start()
            page_num_end = match.end()

            # === Clean BEFORE the page number (footer zone) ===
            before_start = page_num_start
            scan_pos = page_num_start - 1

            # Skip whitespace backward
            while scan_pos >= 0 and text[scan_pos] in " \t":
                scan_pos -= 1

            # Look for footer debris: short lowercase tokens (initials)
            # and header words (like "Proceedings") that got collapsed
            while scan_pos >= 0:
                # Find token boundary
                token_end = scan_pos + 1
                while scan_pos >= 0 and text[scan_pos].isalpha():
                    scan_pos -= 1
                token_start = scan_pos + 1
                token = text[token_start:token_end]

                if not token:
                    break

                if len(token) <= 3 and token.islower():
                    # Reporter initials (e.g., 'sn', 'jk')
                    before_start = token_start
                    debris_count += 1
                elif token.lower() in HEADER_WORDS:
                    # Header word collapsed before page number
                    before_start = token_start
                    debris_count += 1
                else:
                    break

                # Skip more whitespace
                while scan_pos >= 0 and text[scan_pos] in " \t":
                    scan_pos -= 1

            # === Clean AFTER the page number (header zone) ===
            after_end = page_num_end
            scan_pos = page_num_end

            # Skip whitespace forward
            while scan_pos < len(text) and text[scan_pos] in " \t":
                scan_pos += 1

            # Look for header words, standalone numbers, and short tokens
            while scan_pos < len(text):
                # Check for speaker marker (stop cleaning)
                remaining = text[scan_pos:]
                if SPEAKER_MARKERS.match(remaining):
                    break

                # Try to read a word/number token
                token_start = scan_pos
                if text[scan_pos].isdigit():
                    # Number token
                    while scan_pos < len(text) and text[scan_pos].isdigit():
                        scan_pos += 1
                    token = text[token_start:scan_pos]
                    # Standalone small numbers in header zone — remove
                    num_val = int(token)
                    if num_val <= 30:
                        after_end = scan_pos
                        debris_count += 1
                    else:
                        break
                elif text[scan_pos].isalpha():
                    # Word token
                    while scan_pos < len(text) and text[scan_pos].isalpha():
                        scan_pos += 1
                    token = text[token_start:scan_pos].lower()

                    if token in HEADER_WORDS:
                        after_end = scan_pos
                        debris_count += 1
                    elif len(token) >= 5:
                        # Long word — likely real content, stop
                        break
                    else:
                        # Short word not in header set — could be content
                        break
                else:
                    break

                # Skip whitespace for next token
                while scan_pos < len(text) and text[scan_pos] in " \t":
                    scan_pos += 1

            removals.append((before_start, after_end))

        if not removals:
            return text, 0

        # Merge overlapping removals and apply (from end to preserve indices)
        removals.sort(key=lambda r: r[0])
        merged = [removals[0]]
        for start, end in removals[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        result = text
        for start, end in reversed(merged):
            result = result[:start] + " " + result[end:]

        return result, debris_count
