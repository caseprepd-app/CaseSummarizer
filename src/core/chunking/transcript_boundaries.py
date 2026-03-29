"""
Transcript-Aware Speaker-Turn Boundary Injector

Ensures paragraph breaks (\\n\\n) at speaker-turn changes so the sentence
splitter respects them as natural boundaries. This keeps one speaker's
content together in a single chunk where possible.

No-op on non-transcript documents (no markers = no changes).

Patterns handled:
- BY MR./MS. NAME: — examiner changes
- THE COURT:, THE WITNESS:, MR./MS. NAME: — speaker turns
- DIRECT EXAMINATION, CROSS EXAMINATION, etc. — section headers
"""

import re

logger = __import__("logging").getLogger(__name__)

# Speaker turn patterns that should get paragraph breaks before them.
# Each pattern matches at the start of a line (after optional whitespace).
_SPEAKER_PATTERNS = [
    # Examiner changes: "BY MR. SMITH:", "BY MS. JONES:"
    re.compile(r"^(\s*BY\s+(?:MR|MS|MRS|DR)\.\s+\w+\s*:)", re.MULTILINE | re.IGNORECASE),
    # Named speakers: "MR. SMITH:", "MS. JONES:", "DR. BROWN:"
    re.compile(r"^(\s*(?:MR|MS|MRS|DR)\.\s+\w+\s*:)", re.MULTILINE | re.IGNORECASE),
    # Role speakers: "THE COURT:", "THE WITNESS:", "THE CLERK:"
    re.compile(
        r"^(\s*THE\s+(?:COURT|WITNESS|CLERK|REPORTER|BAILIFF)\s*:)", re.MULTILINE | re.IGNORECASE
    ),
    # Section headers: "DIRECT EXAMINATION", "CROSS EXAMINATION", etc.
    re.compile(
        r"^(\s*(?:DIRECT|CROSS|REDIRECT|RECROSS|RE-DIRECT|RE-CROSS)\s+EXAMINATION)",
        re.MULTILINE | re.IGNORECASE,
    ),
]

# Quick check: if none of these markers appear, skip the full regex pass
_QUICK_CHECK_MARKERS = ["BY MR", "BY MS", "THE COURT", "THE WITNESS", "EXAMINATION"]


def _has_transcript_markers(text: str) -> bool:
    """
    Check whether text contains transcript markers using multi-position sampling.

    Sampling strategy (avoids scanning entire multi-MB documents):
    - Documents <= 20K chars: check the entire text.
    - Documents > 20K chars: check first 10K + a 10K window from the middle.

    Python slicing is bounds-safe (returns empty/short string if indices
    exceed length), so no IndexError is possible here.
    """
    if not text:
        return False

    length = len(text)

    if length <= 20_000:
        # Short enough to check everything
        sample = text.upper()
        return any(marker in sample for marker in _QUICK_CHECK_MARKERS)

    # Sample first 10K and middle 10K
    first_sample = text[:10_000].upper()
    if any(marker in first_sample for marker in _QUICK_CHECK_MARKERS):
        return True

    mid_start = (length // 2) - 5_000
    mid_sample = text[mid_start : mid_start + 10_000].upper()
    return any(marker in mid_sample for marker in _QUICK_CHECK_MARKERS)


def inject_speaker_boundaries(text: str) -> str:
    """
    Inject paragraph breaks at speaker-turn boundaries.

    Ensures \\n\\n before each speaker turn so the sentence splitter
    treats them as natural chunk boundaries.

    Uses multi-position sampling (first 10K + middle 10K) so transcripts
    with long preambles are still detected.

    No-op on non-transcript documents (returns text unchanged).

    Args:
        text: Full document text

    Returns:
        Text with paragraph breaks at speaker turns
    """
    if not _has_transcript_markers(text):
        return text

    changes = 0
    for pattern in _SPEAKER_PATTERNS:
        # Insert \n\n before each match if not already preceded by one
        def _ensure_break(match):
            nonlocal changes
            start = match.start()
            # Check if already preceded by a paragraph break
            preceding = text[max(0, start - 2) : start]
            if "\n\n" in preceding:
                return match.group(0)
            changes += 1
            return "\n\n" + match.group(0).lstrip()

        text = pattern.sub(_ensure_break, text)

    if changes > 0:
        logger.debug("Injected %d speaker-turn boundaries", changes)

    return text
