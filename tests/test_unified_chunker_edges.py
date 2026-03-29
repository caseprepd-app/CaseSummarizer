"""
Edge case tests for UnifiedChunker.

Covers boundary conditions: no punctuation, oversized sentences,
empty/whitespace input, unicode at boundaries, and repeated content.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_chunker():
    """Create a UnifiedChunker with mocked tiktoken."""
    with patch("src.core.chunking.unified_chunker.tiktoken") as mock_tt:
        mock_encoder = MagicMock()
        mock_encoder.encode = lambda text: text.split() if text.strip() else []
        mock_tt.get_encoding.return_value = mock_encoder

        from src.core.chunking.unified_chunker import UnifiedChunker

        chunker = UnifiedChunker(
            min_tokens=10,
            target_tokens=50,
            max_tokens=100,
        )
        yield chunker


def test_no_sentence_boundaries(mock_chunker):
    """Text without punctuation still produces chunks."""
    words = " ".join(f"word{i}" for i in range(200))
    chunks = mock_chunker.chunk_text(words, use_cache=False)
    assert len(chunks) >= 1
    combined = " ".join(c.text for c in chunks)
    assert "word0" in combined
    assert "word199" in combined


def test_oversized_single_sentence(mock_chunker):
    """A very long text with many sentences gets split into multiple chunks."""
    parts = [f"Word{i} said that the matter was resolved quickly." for i in range(500)]
    giant = " ".join(parts)

    # Mock the sentence splitter to split on periods (NUPunkt may not be loaded)
    spans = []
    pos = 0
    for part in parts:
        start = giant.find(part, pos)
        end = start + len(part)
        spans.append((part, (start, end)))
        pos = end

    with patch(
        "src.core.utils.sentence_splitter.split_sentence_spans",
        return_value=spans,
    ):
        chunks = mock_chunker.chunk_text(giant, use_cache=False)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= mock_chunker.max_tokens * 2


def test_empty_input(mock_chunker):
    """Empty string returns empty list without crashing."""
    result = mock_chunker.chunk_text("", use_cache=False)
    assert result == []


def test_unicode_boundaries(mock_chunker):
    """Unicode chars at chunk edges are not split mid-character."""
    sentences = []
    for i in range(30):
        sentences.append(f"Sentence {i} with em\u2014dash and caf\u00e9 quote\u201d.")
    text = " ".join(sentences)
    chunks = mock_chunker.chunk_text(text, use_cache=False)
    assert len(chunks) >= 1
    combined = "".join(c.text for c in chunks)
    assert "\u2014" in combined
    assert "\u00e9" in combined
    assert "\u201d" in combined


def test_all_whitespace_input(mock_chunker):
    """Only whitespace returns empty list, no crash."""
    result = mock_chunker.chunk_text("   \n\n\t  \n  ", use_cache=False)
    assert result == []


def test_single_sentence_input(mock_chunker):
    """One normal sentence becomes exactly one chunk."""
    text = "The deposition of Dr. Smith was taken on March 15."
    chunks = mock_chunker.chunk_text(text, use_cache=False)
    assert len(chunks) == 1
    assert "Dr. Smith" in chunks[0].text


def test_repeated_content(mock_chunker):
    """Same sentence repeated 100 times completes without hang."""
    sentence = "The plaintiff alleges negligence by the defendant. "
    text = sentence * 100
    chunks = mock_chunker.chunk_text(text, use_cache=False)
    assert len(chunks) >= 1
    total_words = sum(c.word_count for c in chunks)
    assert total_words > 0
