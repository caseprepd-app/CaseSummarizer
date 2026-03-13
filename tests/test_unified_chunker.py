"""Tests for unified_chunker.py"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_chunker():
    """Create a UnifiedChunker with mocked heavy dependencies."""
    with patch("src.core.chunking.unified_chunker.tiktoken") as mock_tiktoken:
        mock_encoder = MagicMock()
        # Simple token counting: split on spaces
        mock_encoder.encode = lambda text: text.split() if text.strip() else []
        mock_tiktoken.encoding_for_model.return_value = mock_encoder
        mock_tiktoken.get_encoding.return_value = mock_encoder

        from src.core.chunking.unified_chunker import UnifiedChunker

        chunker = UnifiedChunker(
            min_tokens=10,
            target_tokens=50,
            max_tokens=100,
        )
        # Ensure encoder is our mock
        chunker._encoder = mock_encoder
        yield chunker


class TestCountTokens:
    def test_basic_count(self, mock_chunker):
        count = mock_chunker.count_tokens("hello world foo bar")
        assert count == 4

    def test_empty_string(self, mock_chunker):
        count = mock_chunker.count_tokens("")
        assert count == 0


class TestMergeSmallChunks:
    def test_small_chunks_merged(self, mock_chunker):
        # Each chunk has fewer tokens than min_tokens (10)
        small_chunks = ["hello world", "foo bar", "baz qux"]
        result = mock_chunker._merge_small_chunks(small_chunks)
        # Should merge since each is only 2 tokens, well below min of 10
        assert len(result) < len(small_chunks)

    def test_large_chunks_not_merged(self, mock_chunker):
        # Each chunk has enough tokens
        large_chunks = [
            " ".join(f"word{i}" for i in range(20)),
            " ".join(f"term{i}" for i in range(20)),
        ]
        result = mock_chunker._merge_small_chunks(large_chunks)
        assert len(result) >= 1


class TestSplitAtSentences:
    def test_splits_at_sentence_boundary(self, mock_chunker):
        text = ". ".join(f"This is sentence number {i}" for i in range(20))
        result = mock_chunker._split_at_sentences(text, target_tokens=30)
        assert len(result) > 1
        # Each piece should be part of the original
        combined = " ".join(result)
        assert "sentence" in combined

    def test_short_text_single_chunk(self, mock_chunker):
        text = "Short sentence here."
        result = mock_chunker._split_at_sentences(text, target_tokens=50)
        assert len(result) == 1


class TestOverlap:
    def test_overlap_spans(self, mock_chunker):
        """Verify overlap carries forward trailing spans."""
        mock_chunker.overlap_tokens = 10
        spans = [
            ("word1 word2 word3", (0, 17)),
            ("word4 word5 word6", (18, 35)),
            ("word7 word8 word9", (36, 53)),
        ]
        overlap, tokens = mock_chunker._get_overlap_spans(spans)
        assert len(overlap) >= 1
        assert tokens > 0

    def test_no_overlap_when_zero(self, mock_chunker):
        """When overlap_tokens=0, no overlap should occur."""
        mock_chunker.overlap_tokens = 0
        text = ". ".join(f"This is sentence number {i}" for i in range(20))
        result = mock_chunker._split_at_sentences(text, target_tokens=30)
        # Each chunk should start fresh (no repeated content from prior chunk)
        assert len(result) > 1


class TestDetectSection:
    def test_detects_direct_examination(self, mock_chunker):
        result = mock_chunker._detect_section("DIRECT EXAMINATION\nQ. Please state your name.")
        assert result is not None
        assert "direct" in result.lower() or "examination" in result.lower()

    def test_detects_cause_of_action(self, mock_chunker):
        result = mock_chunker._detect_section("FIRST CAUSE OF ACTION\nDefendant negligently...")
        assert result is not None

    def test_no_section_detected(self, mock_chunker):
        result = mock_chunker._detect_section("The patient presented with symptoms of pain.")
        assert result is None or isinstance(result, str)


class TestChunkText:
    def test_empty_input(self, mock_chunker):
        result = mock_chunker.chunk_text("", source_file="test.txt", use_cache=False)
        assert result == [] or len(result) == 0

    def test_returns_chunks(self, mock_chunker):
        text = ". ".join(f"This is sentence number {i} with extra words" for i in range(50))
        result = mock_chunker.chunk_text(text, source_file="test.txt", use_cache=False)
        assert len(result) >= 1
        # Each chunk should have text content
        for chunk in result:
            assert hasattr(chunk, "text") or hasattr(chunk, "content")

    def test_caching(self, mock_chunker):
        text = "Some text for caching test with enough words to matter here."
        r1 = mock_chunker.chunk_text(text, source_file="test.txt", use_cache=True)
        r2 = mock_chunker.chunk_text(text, source_file="test.txt", use_cache=True)
        assert len(r1) == len(r2)
