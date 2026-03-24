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


def inject_speaker_boundaries(text: str) -> str:
    """
    Inject paragraph breaks at speaker-turn boundaries.

    Ensures \\n\\n before each speaker turn so the sentence splitter
    treats them as natural chunk boundaries.

    No-op on non-transcript documents (returns text unchanged).

    Args:
        text: Full document text

    Returns:
        Text with paragraph breaks at speaker turns
    """
    # Quick check: skip if no transcript markers found
    text_upper = text[:20000].upper()
    if not any(marker in text_upper for marker in _QUICK_CHECK_MARKERS):
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
