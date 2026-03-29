"""
Focused citation excerpt extraction using embedding similarity.

Selects the best ~250-char window from the retrieved chunk using
the same embedding model that FAISS used for retrieval. This is
semantically consistent: FAISS found the chunk by embedding similarity
to the question, and we use the same method to find the best sub-chunk.

Algorithm:
1. Create overlapping ~250-char windows with 50% overlap, snapped to words
2. Embed question + all windows using the same embedding model
3. Pick window with highest cosine similarity to question
4. Snap to sentence boundaries where possible
5. Add ... markers if excerpt doesn't start/end at chunk boundaries

Inputs:
    context: str - raw chunk text from single-chunk retrieval
    question: str - the question that was asked
    embeddings: HuggingFaceEmbeddings - same model used for FAISS
    max_chars: int - target excerpt length (~250)

Outputs:
    str - focused excerpt from the most relevant passage
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def extract_citation_excerpt(
    context: str,
    question: str,
    embeddings,
    max_chars: int = 250,
) -> str:
    """
    Extract a focused ~max_chars excerpt from context using embedding similarity.

    Args:
        context: Raw chunk text (single chunk, no separators or source prefix).
        question: The question that was asked.
        embeddings: HuggingFaceEmbeddings model (same as FAISS).
        max_chars: Target excerpt length in characters.

    Returns:
        Focused excerpt string, or empty string if context is empty.
    """
    if not context or not context.strip():
        return ""

    chunk = context.strip()

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

    start, end, window_text = _select_best_window(windows, question, embeddings)
    return _format_excerpt(chunk, start, end, max_chars=max_chars)


def _build_windows(text: str, window_size: int) -> list[tuple[int, int, str]]:
    """
    Create overlapping windows snapped to word boundaries.

    Uses 50% overlap (stride = window_size // 2) to ensure good coverage.

    Args:
        text: Full chunk text to window over.
        window_size: Target window size in characters.

    Returns:
        List of (start_position, end_position, window_text) tuples.
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
        if end < text_len and text[end] != " ":
            last_space = text.rfind(" ", actual_start, end)
            if last_space > actual_start:
                actual_end = last_space

        window_text = text[actual_start:actual_end].strip()
        if window_text:
            windows.append((actual_start, actual_end, window_text))

        pos += stride
        # Stop if we've covered the end
        if end >= text_len:
            break

    return windows


def _select_best_window(
    windows: list[tuple[int, int, str]],
    question: str,
    embeddings,
) -> tuple[int, int, str]:
    """
    Select the window most similar to the question using embedding cosine similarity.

    Args:
        windows: List of (start_position, end_position, window_text) tuples.
        question: The question to match against.
        embeddings: HuggingFaceEmbeddings model.

    Returns:
        (start_position, end_position, window_text) of the best-matching window.
    """
    q_emb = np.array(embeddings.embed_query(question))
    w_texts = [text for _, _, text in windows]
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


def _format_excerpt(full_text: str, start: int, end: int, max_chars: int = 0) -> str:
    """
    Format excerpt with sentence snapping and ellipsis markers.

    Adds ... at start/end if the excerpt doesn't cover chunk boundaries.
    Falls back to unsnapped positions if snapping exceeds 2x max_chars.
    """
    from src.core.utils.sentence_splitter import split_sentence_spans

    spans = split_sentence_spans(full_text)

    snapped_start = start
    snapped_end = end

    if spans:
        for _sent, (s, _e) in spans:
            if s <= start:
                snapped_start = s
            else:
                break
        for _sent, (_s, e) in spans:
            if e >= end:
                snapped_end = e
                break

    # Guard against oversized excerpts from long legal sentences
    original_len = end - start
    if max_chars > 0 and (snapped_end - snapped_start) > max_chars * 2:
        snapped_start = start
        snapped_end = end
        logger.debug(
            "Sentence snap exceeded 2x max_chars (%d), falling back to unsnapped (%d chars)",
            max_chars * 2,
            original_len,
        )

    excerpt = full_text[snapped_start:snapped_end].strip()

    if snapped_start > 0:
        excerpt = "..." + excerpt
    if snapped_end < len(full_text):
        excerpt = excerpt + "..."

    return excerpt


def _truncate_to_sentence(text: str, max_chars: int) -> str:
    """
    Fallback: truncate text to max_chars at a sentence boundary.

    Used when embeddings are unavailable.
    """
    if len(text) <= max_chars:
        return text

    logger.warning(
        "Citation excerpt truncated from %d to %d chars (embeddings unavailable).",
        len(text),
        max_chars,
    )
    from src.core.utils.sentence_splitter import split_sentences

    sentences = split_sentences(text)
    result = ""
    for sentence in sentences:
        candidate = result + (" " if result else "") + sentence
        if len(candidate) > max_chars:
            break
        result = candidate

    if result and len(result) > max_chars // 2:
        return result.strip() + "..."

    # Fall back to word boundary if no sentence fits
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        return truncated[:last_space].strip() + "..."

    return truncated.strip() + "..."
