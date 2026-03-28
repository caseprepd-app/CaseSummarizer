"""
Dedicated tests for src/core/semantic/citation_excerpt.py.

Covers excerpt extraction with embedding similarity, fallback truncation,
short passages, unicode handling, and empty inputs.
"""

from unittest.mock import MagicMock, patch

from src.core.semantic.citation_excerpt import (
    _build_windows,
    extract_citation_excerpt,
)


def _make_embeddings(best_window_idx=0):
    """Create a mock embeddings object that selects a specific window."""
    emb = MagicMock()
    emb.embed_query.return_value = [1.0, 0.0, 0.0]

    def embed_documents(texts):
        """Return embeddings where best_window_idx has highest similarity."""
        result = []
        for i, _ in enumerate(texts):
            if i == best_window_idx:
                result.append([1.0, 0.0, 0.0])  # perfect match
            else:
                result.append([0.0, 1.0, 0.0])  # orthogonal
        return result

    emb.embed_documents.side_effect = embed_documents
    return emb


def test_extract_citation_basic():
    """Passage with a relevant portion returns a focused excerpt."""
    passage = "The defendant filed a motion to dismiss. " * 10
    question = "What motion was filed?"

    with patch(
        "src.core.utils.sentence_splitter.split_sentence_spans",
        return_value=[(passage, (0, len(passage)))],
    ):
        result = extract_citation_excerpt(passage, question, _make_embeddings(), max_chars=100)

    assert len(result) > 0
    assert isinstance(result, str)


def test_extract_citation_no_match_returns_string():
    """Query that doesn't match still returns a string, never crashes."""
    passage = "The weather was sunny and warm all day long. " * 10
    question = "What is the speed of light?"

    with patch(
        "src.core.utils.sentence_splitter.split_sentence_spans",
        return_value=[(passage, (0, len(passage)))],
    ):
        result = extract_citation_excerpt(passage, question, _make_embeddings(), max_chars=100)

    assert isinstance(result, str)
    assert len(result) > 0


def test_short_passage_returned_as_is():
    """Passage shorter than max_chars is returned without modification."""
    short = "The court ruled in favor of the plaintiff."

    result = extract_citation_excerpt(short, "Who won?", _make_embeddings(), max_chars=250)

    assert result == short


def test_long_passage_extracts_focused():
    """Long passage with relevant section in middle: excerpt is not just the start."""
    filler = "This is irrelevant padding text about nothing. " * 20
    relevant = "The jury awarded $2.5 million in damages to the plaintiff."
    passage = filler + relevant + filler

    # Make the embeddings select the window closest to the relevant part
    windows = _build_windows(passage, 250)
    best_idx = 0
    for i, (start, end, text) in enumerate(windows):
        if "jury awarded" in text:
            best_idx = i
            break

    with patch(
        "src.core.utils.sentence_splitter.split_sentence_spans",
        return_value=[(passage, (0, len(passage)))],
    ):
        result = extract_citation_excerpt(
            passage, "damages", _make_embeddings(best_idx), max_chars=250
        )

    # Excerpt should not just be the beginning of the passage
    assert "jury awarded" in result or "..." in result


def test_unicode_in_passage():
    """Accented characters and em-dashes are preserved in excerpt."""
    passage = (
        "The caf\u00e9 r\u00e9sum\u00e9 included details about the "
        "defendant\u2019s prior convictions \u2014 including theft."
    )

    result = extract_citation_excerpt(passage, "convictions", _make_embeddings(), max_chars=250)

    # Short passage returned as-is; verify unicode survived
    assert "\u00e9" in result
    assert "\u2014" in result


def test_empty_passage():
    """Empty passage returns empty string."""
    assert extract_citation_excerpt("", "question", _make_embeddings()) == ""


def test_whitespace_only_passage():
    """Whitespace-only passage returns empty string."""
    assert extract_citation_excerpt("   \n\t  ", "q", _make_embeddings()) == ""


def test_empty_question_still_works():
    """Empty question does not crash; an excerpt is still returned."""
    passage = "The court issued its ruling on Monday. " * 10

    with patch(
        "src.core.utils.sentence_splitter.split_sentence_spans",
        return_value=[(passage, (0, len(passage)))],
    ):
        result = extract_citation_excerpt(passage, "", _make_embeddings(), max_chars=100)

    assert isinstance(result, str)
    assert len(result) > 0


def test_build_windows_coverage():
    """Windows cover the full text with 50% overlap."""
    text = "word " * 100  # 500 chars
    windows = _build_windows(text, window_size=100)

    assert len(windows) >= 3
    # First window starts near 0
    assert windows[0][0] <= 5
    # Last window reaches near the end
    last_start, last_end, last_text = windows[-1]
    assert last_end >= len(text.strip()) - 30


def test_none_embeddings_fallback():
    """When embeddings is None, sentence-truncation fallback is used."""
    passage = "First sentence. Second sentence. Third sentence. " * 10

    with patch(
        "src.core.utils.sentence_splitter.split_sentences",
        return_value=["First sentence.", "Second sentence."],
    ):
        result = extract_citation_excerpt(passage, "query", embeddings=None, max_chars=50)

    assert isinstance(result, str)
    assert len(result) <= 60  # max_chars + ellipsis
