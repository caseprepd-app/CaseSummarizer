"""
Legal-aware sentence boundary detection using NUPunkt.

NUPunkt was chosen over naive regex splitting because legal text is full of
abbreviations that cause false sentence breaks: "v.", "U.S.C.", "F.Supp.2d",
"Inc.", "No.", etc. NUPunkt ships with 4,000+ built-in legal abbreviations
and achieves 91.1% precision on legal text (vs 59% for pySBD).

If nupunkt is not installed, falls back to a simple regex splitter with a
log.warning on first use.

Inputs/Outputs:
    split_sentences(text: str) -> list[str]
    split_sentence_spans(text: str) -> list[tuple[str, tuple[int, int]]]
"""

import logging
import re

logger = logging.getLogger(__name__)

try:
    from nupunkt import sent_spans_with_text as _nupunkt_spans
    from nupunkt import sent_tokenize as _nupunkt_tokenize

    _nupunkt_available = True
except ImportError:
    _nupunkt_available = False

_fallback_warned = False

# Fallback regex: split on sentence-ending punctuation followed by whitespace
_RE_FALLBACK_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using NUPunkt (legal-aware) or regex fallback.

    Args:
        text: Input text to split.

    Returns:
        List of sentence strings (empty strings filtered out).
    """
    if not text or not text.strip():
        return []

    global _fallback_warned

    if _nupunkt_available:
        sentences = list(_nupunkt_tokenize(text))
    else:
        if not _fallback_warned:
            logger.warning(
                "nupunkt not installed — falling back to regex sentence splitter. "
                "Install with: pip install nupunkt"
            )
            _fallback_warned = True
        sentences = _RE_FALLBACK_SPLIT.split(text)

    return [s.strip() for s in sentences if s.strip()]


def split_sentence_spans(text: str) -> list[tuple[str, tuple[int, int]]]:
    """
    Split text into sentences with character offset spans.

    Each result is (sentence_text, (start_char, end_char)) where offsets
    refer to positions in the original text.

    Args:
        text: Input text to split.

    Returns:
        List of (sentence, (start, end)) tuples.
    """
    if not text or not text.strip():
        return []

    if _nupunkt_available:
        # nupunkt returns (text, (start, end)) tuples directly
        raw_spans = list(_nupunkt_spans(text))
        return [(sent.strip(), (start, end)) for sent, (start, end) in raw_spans if sent.strip()]

    # Fallback: compute spans manually from split_sentences
    sentences = split_sentences(text)
    spans = []
    search_start = 0

    for sentence in sentences:
        idx = text.find(sentence, search_start)
        if idx == -1:
            idx = search_start

        end = idx + len(sentence)
        spans.append((sentence, (idx, end)))
        search_start = end

    return spans
