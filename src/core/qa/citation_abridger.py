"""
Chunk-boundary-aware citation abridging.

Replaces blind character truncation with intelligent abridging that
keeps whole retrieval chunks (already sorted by relevance score)
until the character budget is exhausted.

Inputs:
    context: str — concatenated chunks separated by "\\n\\n---\\n\\n"
    max_chars: int — character budget

Outputs:
    str — abridged citation with an optional footer noting omitted chunks
"""

import logging

logger = logging.getLogger(__name__)

SEPARATOR = "\n\n---\n\n"


def abridge_citation(context: str, max_chars: int) -> str:
    """
    Keep whole top-scored chunks that fit within max_chars.

    Chunks are already ordered by combined_score (highest first) from the
    hybrid retriever, so dropping from the tail removes the least relevant.

    Args:
        context: Raw citation text with chunks joined by SEPARATOR.
        max_chars: Maximum characters for the abridged output.

    Returns:
        Abridged citation string, possibly with a footer noting omissions.
    """
    if len(context) <= max_chars:
        return context

    chunks = context.split(SEPARATOR)

    kept: list[str] = []
    running = 0

    for i, chunk_text in enumerate(chunks):
        sep_cost = len(SEPARATOR) if kept else 0
        chunk_cost = sep_cost + len(chunk_text)
        remaining = len(chunks) - (i + 1)
        footer = f"\n\n[... {remaining} lower-relevance excerpt(s) omitted]"
        footer_cost = len(footer) if remaining > 0 else 0

        if running + chunk_cost + footer_cost <= max_chars:
            kept.append(chunk_text)
            running += chunk_cost
        else:
            break

    # Always keep at least the first chunk
    if not kept:
        kept.append(chunks[0])

    dropped = len(chunks) - len(kept)
    result = SEPARATOR.join(kept)
    if dropped > 0:
        result += f"\n\n[... {dropped} lower-relevance excerpt(s) omitted]"
        logger.debug(
            "Citation abridged: kept %d/%d chunks (%d chars)",
            len(kept),
            len(chunks),
            len(result),
        )

    return result
