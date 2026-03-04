"""
Token budget utilities for Q&A prompt assembly.

Provides accurate token counting via tiktoken and progressive sub-chunking
to fit context within the LLM's context window.

When context exceeds the budget:
- Pass 1 (aggressive): 3 windows at 60% of context
- Pass 2 (conservative): 5 windows at 80% of context
Best sub-chunk chosen by FAISS cosine similarity to the question.
"""

import logging
import threading

import numpy as np
import tiktoken

from src.config import UNIFIED_CHUNK_ENCODING

logger = logging.getLogger(__name__)

_encoder = None
_encoder_lock = threading.Lock()


def _get_encoder():
    """Lazy-load tiktoken encoder (cl100k_base), thread-safe."""
    global _encoder
    if _encoder is not None:
        return _encoder
    with _encoder_lock:
        if _encoder is None:
            _encoder = tiktoken.get_encoding(UNIFIED_CHUNK_ENCODING)
    return _encoder


def count_tokens(text: str) -> int:
    """
    Count tokens using tiktoken (cl100k_base).

    Args:
        text: Input text

    Returns:
        Token count
    """
    return len(_get_encoder().encode(text))


def compute_context_budget(
    context_window: int,
    prompt_template_tokens: int,
    question_tokens: int,
    max_output_tokens: int,
    safety_margin: int = 16,
) -> int:
    """
    How many tokens the context can use inside the prompt.

    Args:
        context_window: Total LLM context window in tokens
        prompt_template_tokens: Tokens used by prompt template (excluding context/question)
        question_tokens: Tokens used by the question text
        max_output_tokens: Tokens reserved for LLM output
        safety_margin: Extra buffer for tokenizer variance

    Returns:
        Maximum tokens available for context (minimum 64)
    """
    budget = (
        context_window
        - prompt_template_tokens
        - question_tokens
        - max_output_tokens
        - safety_margin
    )
    return max(64, budget)


def select_best_subchunk(context, question, max_tokens, embeddings):
    """
    Progressive sub-chunking to fit context within token budget.

    Pass 1 (aggressive): 3 windows at 60% (saves 40% of tokens).
    If best sub-chunk scores < 90% of full chunk, fallback.
    Pass 2 (conservative): 5 windows at 80% (saves 20% of tokens).

    Args:
        context: Full context string (too large for budget)
        question: User's question (for FAISS similarity)
        max_tokens: Token budget for context
        embeddings: HuggingFaceEmbeddings for similarity scoring

    Returns:
        Best sub-chunk string that fits within max_tokens
    """
    AGGRESSIVE_RATIO = 0.60
    AGGRESSIVE_WINDOWS = 3
    CONSERVATIVE_RATIO = 0.80
    CONSERVATIVE_WINDOWS = 5
    FALLBACK_THRESHOLD = 0.90

    # Build aggressive windows (0-60%, 20-80%, 40-100%)
    aggressive_chunks = _build_windows(context, AGGRESSIVE_RATIO, AGGRESSIVE_WINDOWS)

    # Score full chunk + aggressive sub-chunks in one batch
    all_texts = [context] + aggressive_chunks
    scores = _score_against_question(all_texts, question, embeddings)
    full_score = scores[0]
    aggressive_scores = scores[1:]

    best_agg_idx = int(np.argmax(aggressive_scores))
    best_agg_score = aggressive_scores[best_agg_idx]

    logger.debug(
        "Aggressive pass: full=%.3f, best sub-chunk=%.3f (%.0f%% of full)",
        full_score,
        best_agg_score,
        best_agg_score / max(full_score, 1e-10) * 100,
    )

    # Check if aggressive sub-chunk is close enough to full chunk
    if full_score > 0 and (best_agg_score / full_score) >= FALLBACK_THRESHOLD:
        result = aggressive_chunks[best_agg_idx]
        return _ensure_fits(result, max_tokens)

    # Fallback: conservative pass
    logger.debug("Aggressive too lossy, trying conservative (80%% windows)")
    conservative_chunks = _build_windows(context, CONSERVATIVE_RATIO, CONSERVATIVE_WINDOWS)
    con_scores = _score_against_question(conservative_chunks, question, embeddings)

    best_con_idx = int(np.argmax(con_scores))
    logger.debug(
        "Conservative pass: best=%.3f (%.0f%% of full)",
        con_scores[best_con_idx],
        con_scores[best_con_idx] / max(full_score, 1e-10) * 100,
    )

    result = conservative_chunks[best_con_idx]
    return _ensure_fits(result, max_tokens)


def _build_windows(text, ratio, num_windows):
    """
    Build overlapping sub-chunks from text.

    Args:
        text: Source text to split
        ratio: Fraction of text each window covers (0.0-1.0)
        num_windows: Number of overlapping windows to create

    Returns:
        List of sub-chunk strings
    """
    total_len = len(text)
    window_len = int(total_len * ratio)
    if num_windows == 1:
        return [text[:window_len]]
    step = (total_len - window_len) / (num_windows - 1)
    windows = []
    for i in range(num_windows):
        start = int(i * step)
        end = min(start + window_len, total_len)
        windows.append(text[start:end])
    return windows


def _score_against_question(texts, question, embeddings):
    """
    Compute cosine similarity of each text against question.

    Args:
        texts: List of text strings to score
        question: Question to compare against
        embeddings: HuggingFaceEmbeddings model

    Returns:
        numpy array of cosine similarity scores
    """
    q_emb = np.array(embeddings.embed_query(question))
    t_embs = np.array(embeddings.embed_documents(texts))
    # Normalized embeddings -> dot product = cosine similarity
    return t_embs @ q_emb


def _ensure_fits(text, max_tokens):
    """
    Truncate text to fit token budget, breaking at paragraph separator.

    Args:
        text: Text to truncate
        max_tokens: Maximum allowed tokens

    Returns:
        Text that fits within the token budget
    """
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    logger.warning(
        "Context truncated from %d to %d tokens to fit token budget.",
        len(tokens),
        max_tokens,
    )
    truncated = enc.decode(tokens[:max_tokens])
    # Try to break at a chunk separator for cleaner truncation
    last_sep = truncated.rfind("\n\n---\n\n")
    if last_sep > len(truncated) // 2:
        truncated = truncated[:last_sep]
    return truncated
