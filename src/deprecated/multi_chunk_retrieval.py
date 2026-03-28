"""
Deprecated multi-chunk context assembly code.

Moved here when CasePrepd switched to single-chunk retrieval (Mar 2026).
With no LLM to synthesize across chunks, retrieving multiple chunks was
wasted work — citation_excerpt only used the top-1 chunk, and averaging
multiple source scores diluted the relevance.

Kept for reference in case multi-chunk retrieval is needed in the future
(e.g., if LLM answer generation is re-added).
"""

CONTEXT_SEPARATOR = "\n\n---\n\n"


def get_top_chunk(context: str) -> str:
    """
    Extract the first (highest-relevance) chunk from separated context.

    Args:
        context: Chunks joined by CONTEXT_SEPARATOR.

    Returns:
        First chunk text.
    """
    chunks = context.split(CONTEXT_SEPARATOR)
    return chunks[0] if chunks else ""


def calculate_average_relevance(sources: list) -> float:
    """
    Calculate average relevance score across multiple sources.

    Deprecated: averaging dilutes the top chunk's score with weaker chunks.
    Replaced by using the single top source's score directly.

    Args:
        sources: List of SourceInfo objects

    Returns:
        Average relevance score (0-1)
    """
    if not sources:
        return 0.0
    avg = sum(s.relevance_score for s in sources) / len(sources)
    return round(avg, 2)


def get_effective_semantic_context_window() -> int:
    """
    Get semantic search context window (fixed token budget).

    Deprecated: token budget management removed with single-chunk retrieval.
    A single chunk never exceeds the context window.

    Returns:
        Context window size in tokens
    """
    from src.config import SEMANTIC_CONTEXT_WINDOW

    return SEMANTIC_CONTEXT_WINDOW
