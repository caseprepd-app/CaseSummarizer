"""
Line Number Remover Preprocessor

Removes line numbers that appear in margins of legal transcripts.
Common in deposition transcripts and court filings.

Patterns handled:
- "1  ", "2  ", ..., "25  " at line start (transcript format)
- "  1", "  2", ..., "  25" at line end (some PDF exports)
- "|1", "|2", etc. (some legal document formats)

Does NOT remove:
- Numbers that are part of content (dates, case numbers, citations)
- Page numbers (handled by RawTextExtractor)
"""

import re

from src.core.preprocessing.base import BasePreprocessor, PreprocessingResult


class LineNumberRemover(BasePreprocessor):
    """
    Removes line numbers from legal transcript margins.

    Legal transcripts often have line numbers 1-25 in the left margin
    for reference during depositions. These add noise to AI summaries.

    Example input:
        1  Q.  Good morning, Mr. Smith.
        2  A.  Good morning.
        3  Q.  State your name for the record.

    Example output:
        Q.  Good morning, Mr. Smith.
        A.  Good morning.
        Q.  State your name for the record.
    """

    name = "Line Number Remover"

    # Pattern for line numbers at start of line (1-25, common in transcripts)
    # Matches: "1  ", "12  ", "25  " but not "100  " or numbers mid-sentence
    # The \s{2,} requires at least 2 spaces after the number (typical transcript format)
    LINE_START_PATTERN = re.compile(r'^(\s*)([1-9]|1[0-9]|2[0-5])\s{2,}', re.MULTILINE)

    # Pattern for line numbers at end of line (less common, some PDF exports)
    # Matches: "  1\n", "  25\n" at very end of line
    LINE_END_PATTERN = re.compile(r'\s{2,}([1-9]|1[0-9]|2[0-5])$', re.MULTILINE)

    # Pattern for pipe-prefixed line numbers (|1, |2, etc.)
    PIPE_PATTERN = re.compile(r'^\|([1-9]|1[0-9]|2[0-5])\s*', re.MULTILINE)

    def process(self, text: str) -> PreprocessingResult:
        """
        Remove line numbers from text margins.

        Args:
            text: Input text potentially containing line numbers

        Returns:
            PreprocessingResult with cleaned text and change count
        """
        if not text:
            return PreprocessingResult(text=text, changes_made=0)

        changes = 0
        result = text

        # Remove line numbers at start of lines
        # Preserve any leading whitespace before the number
        def replace_start(match):
            nonlocal changes
            changes += 1
            # Keep leading whitespace, remove number and extra spaces
            return match.group(1)

        result = self.LINE_START_PATTERN.sub(replace_start, result)

        # Remove line numbers at end of lines
        result, end_count = self.LINE_END_PATTERN.subn('', result)
        changes += end_count

        # Remove pipe-prefixed line numbers
        result, pipe_count = self.PIPE_PATTERN.subn('', result)
        changes += pipe_count

        return PreprocessingResult(
            text=result,
            changes_made=changes,
            metadata={
                'start_line_numbers': changes - end_count - pipe_count,
                'end_line_numbers': end_count,
                'pipe_line_numbers': pipe_count,
            }
        )
