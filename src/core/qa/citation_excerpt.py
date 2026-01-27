"""
Focused citation excerpt extraction using embedding similarity.

Selects the best ~250-char window from the top retrieval chunk using
the same embedding model that FAISS used for retrieval. This is
semantically consistent: FAISS found the chunk by embedding similarity
to the question, and we use the same method to find the best sub-chunk.

Algorithm:
1. Split context by separator -> take the top chunk (highest relevance)
2. Strip the [filename]: prefix (redundant with Source: line)
3. Create overlapping ~250-char windows with 50% overlap, snapped to words
4. Embed question + all windows using the same embedding model
5. Pick window with highest cosine similarity to question
6. Snap to sentence boundaries where possible
7. Add ... markers if excerpt doesn't start/end at chunk boundaries

Inputs:
    context: str - concatenated chunks separated by "\\n\\n---\\n\\n"
    question: str - the question that was asked
    embeddings: HuggingFaceEmbeddings - same model used for FAISS
    max_chars: int - target excerpt length (~250)

Outputs:
    str - focused excerpt from the most relevant passage
"""

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

SEPARATOR = "\n\n---\n\n"


def extract_citation_excerpt(
    context: str,
    question: str,
    embeddings,
    max_chars: int = 250,
) -> str:
    """
    Extract a focused ~max_chars excerpt from context using embedding similarity.

    Args:
        context: Raw citation text with chunks joined by SEPARATOR.
        question: The question that was asked.
        embeddings: HuggingFaceEmbeddings model (same as FAISS).
        max_chars: Target excerpt length in characters.

    Returns:
        Focused excerpt string, or empty string if context is empty.
    """
    if not context or not context.strip():
        return ""

    chunk = _get_top_chunk(context)
    chunk = _strip_source_prefix(chunk)
    chunk = chunk.strip()

    if not chunk:
        return ""

    # Short chunk: return as-is
    if len(chunk) <= max_chars:
        return chunk

    # Fallback if embeddings unavailable
    if embeddings is None:
        logger.debug("No embeddings available, using sentence-truncation fallback")
        return _truncate_to_sentence(chunk, max_chars)

    windows = _build_windows(chunk, max_chars)
    if not windows:
        return _truncate_to_sentence(chunk, max_chars)

    start, window_text = _select_best_window(windows, question, embeddings)
    return _format_excerpt(chunk, start, start + len(window_text))


def _get_top_chunk(context: str) -> str:
    """
    Extract the first (highest-relevance) chunk from separated context.

    Args:
        context: Chunks joined by SEPARATOR.

    Returns:
        First chunk text.
    """
    chunks = context.split(SEPARATOR)
    return chunks[0] if chunks else ""


def _strip_source_prefix(chunk: str) -> str:
    """
    Remove [filename, section]: prefix from chunk text.

    These prefixes are redundant with the Source: line shown in the UI.
    Matches patterns like: [complaint.pdf, page 3]:

    Args:
        chunk: Raw chunk text possibly starting with source prefix.

    Returns:
        Chunk text with prefix removed.
    """
    # Match [anything]: at the start, possibly with leading whitespace
    stripped = re.sub(r"^\s*\[[^\]]+\]\s*:\s*", "", chunk)
    return stripped


def _build_windows(text: str, window_size: int) -> list[tuple[int, str]]:
    """
    Create overlapping windows snapped to word boundaries.

    Uses 50% overlap (stride = window_size // 2) to ensure good coverage.

    Args:
        text: Full chunk text to window over.
        window_size: Target window size in characters.

    Returns:
        List of (start_position, window_text) tuples.
    """
    stride = max(window_size // 2, 1)
    windows = []
    text_len = len(text)

    pos = 0
    while pos < text_len:
        end = min(pos + window_size, text_len)

        # Snap start to word boundary (don't break mid-word)
        actual_start = pos
        if pos > 0 and pos < text_len and text[pos] != " ":
            next_space = text.find(" ", pos)
            if next_space != -1 and next_space < pos + 30:
                actual_start = next_space + 1

        # Snap end to word boundary
        actual_end = end
        if end < text_len and text[end - 1] != " ":
            last_space = text.rfind(" ", actual_start, end)
            if last_space > actual_start:
                actual_end = last_space

        window_text = text[actual_start:actual_end].strip()
        if window_text:
            windows.append((actual_start, window_text))

        pos += stride
        # Stop if we've covered the end
        if end >= text_len:
            break

    return windows


def _select_best_window(
    windows: list[tuple[int, str]],
    question: str,
    embeddings,
) -> tuple[int, str]:
    """
    Select the window most similar to the question using embedding cosine similarity.

    Args:
        windows: List of (start_position, window_text) tuples.
        question: The question to match against.
        embeddings: HuggingFaceEmbeddings model.

    Returns:
        (start_position, window_text) of the best-matching window.
    """
    q_emb = np.array(embeddings.embed_query(question))
    w_texts = [text for _, text in windows]
    w_embs = np.array(embeddings.embed_documents(w_texts))

    # Cosine similarity: (w_embs @ q_emb) / (||w_embs|| * ||q_emb||)
    q_norm = np.linalg.norm(q_emb)
    w_norms = np.linalg.norm(w_embs, axis=1)
    norms = w_norms * q_norm
    similarities = w_embs @ q_emb / np.maximum(norms, 1e-10)

    best_idx = int(np.argmax(similarities))
    logger.debug(
        "Best window %d/%d (similarity=%.3f)",
        best_idx + 1,
        len(windows),
        similarities[best_idx],
    )
    return windows[best_idx]


def _format_excerpt(full_text: str, start: int, end: int) -> str:
    """
    Format excerpt with sentence snapping and ellipsis markers.

    Tries to snap to sentence boundaries (. ! ?) when nearby.
    Adds ... at start/end if the excerpt doesn't cover chunk boundaries.

    Args:
        full_text: The complete chunk text.
        start: Start position of the selected window.
        end: End position of the selected window.

    Returns:
        Formatted excerpt with ellipsis markers as needed.
    """
    # Try to snap start to a sentence boundary (look back up to 40 chars)
    snapped_start = start
    if start > 0:
        search_region = full_text[max(0, start - 40) : start]
        # Find last sentence-ending punctuation followed by space
        for i in range(len(search_region) - 1, -1, -1):
            if search_region[i] in ".!?" and i + 1 < len(search_region):
                snapped_start = max(0, start - 40) + i + 1
                break

    # Try to snap end to a sentence boundary (look forward up to 40 chars)
    snapped_end = end
    if end < len(full_text):
        search_region = full_text[end : min(len(full_text), end + 40)]
        for i, ch in enumerate(search_region):
            if ch in ".!?":
                snapped_end = end + i + 1
                break

    excerpt = full_text[snapped_start:snapped_end].strip()

    # Add ellipsis markers
    if snapped_start > 0:
        excerpt = "..." + excerpt
    if snapped_end < len(full_text):
        excerpt = excerpt + "..."

    return excerpt


def _truncate_to_sentence(text: str, max_chars: int) -> str:
    """
    Fallback: truncate text to max_chars at a sentence boundary.

    Used when embeddings are unavailable.

    Args:
        text: Text to truncate.
        max_chars: Maximum character count.

    Returns:
        Truncated text ending at a sentence boundary if possible.
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]

    # Try to find a sentence boundary
    for punct in [". ", "! ", "? "]:
        last_idx = truncated.rfind(punct)
        if last_idx > max_chars // 2:
            return truncated[: last_idx + 1].strip() + "..."

    # Fall back to word boundary
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        return truncated[:last_space].strip() + "..."

    return truncated.strip() + "..."
