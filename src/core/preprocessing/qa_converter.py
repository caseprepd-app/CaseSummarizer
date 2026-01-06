"""
Q&A Converter Preprocessor

Converts abbreviated Q./A. notation to readable "Question:"/"Answer:" format.
Improves AI summary quality by making deposition transcripts more readable.

Handles various formats:
- "Q." / "A."
- "Q:" / "A:"
- "Q " / "A " (space only)
- All followed by the actual question/answer text
"""

import re

from src.core.preprocessing.base import BasePreprocessor, PreprocessingResult


class QAConverter(BasePreprocessor):
    """
    Converts Q./A. notation to readable format.

    Deposition transcripts use abbreviated notation:
        Q.  Where were you on January 5th?
        A.  I was at home.

    This converts to:
        Question: Where were you on January 5th?
        Answer: I was at home.

    Benefits:
    - Clearer for AI models to understand context
    - Better summary quality
    - More readable output
    """

    name = "Q/A Converter"

    # Pattern to match Q./A. at start of lines
    # Captures: optional whitespace + Q/A + separator (. or : or space) + optional space
    # The pattern is flexible to handle various formats:
    # - "Q.  text" (standard)
    # - "Q: text"
    # - "Q   text" (multiple spaces)
    # - "  Q. text" (indented)
    QA_PATTERN = re.compile(r"^(\s*)([QA])([.:])?\s+", re.MULTILINE)

    # Alternative pattern for "BY MR./MS." examination markers
    # These indicate a new examiner is asking questions
    BY_PATTERN = re.compile(r"^(\s*)(BY\s+(?:MR\.|MS\.|MRS\.)\s+[A-Z][A-Za-z]+):?\s*", re.MULTILINE)

    def process(self, text: str) -> PreprocessingResult:
        """
        Convert Q./A. notation to Question:/Answer: format.

        Args:
            text: Input text with Q./A. notation

        Returns:
            PreprocessingResult with converted text
        """
        if not text:
            return PreprocessingResult(text=text, changes_made=0)

        changes = 0
        result = text

        def replace_qa(match):
            nonlocal changes
            changes += 1
            indent = match.group(1)  # Preserve indentation
            letter = match.group(2).upper()

            if letter == "Q":
                return f"{indent}Question: "
            else:  # A
                return f"{indent}Answer: "

        result = self.QA_PATTERN.sub(replace_qa, result)

        # Optionally format "BY MR./MS." markers
        # These can stay as-is or be formatted - keeping them helps context
        by_count = len(self.BY_PATTERN.findall(result))

        return PreprocessingResult(
            text=result,
            changes_made=changes,
            metadata={
                "questions_converted": sum(
                    1 for m in re.finditer(r"^(\s*)Q[.:]\s", text, re.MULTILINE)
                ),
                "answers_converted": sum(
                    1 for m in re.finditer(r"^(\s*)A[.:]\s", text, re.MULTILINE)
                ),
                "examiner_markers": by_count,
            },
        )
