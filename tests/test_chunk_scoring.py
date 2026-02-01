"""
Tests for src.core.utils.chunk_scoring — redundancy detection via cosine similarity.

Covers:
- Empty input
- Single chunk (never flagged)
- Identical vectors (second flagged)
- Below-threshold similarity (not flagged)
- Above-threshold similarity (flagged with correct reason)
- First occurrence always kept
- Multiple duplicates
- Zero vectors handled gracefully
- Custom threshold
"""

import numpy as np

from src.core.utils.chunk_scoring import ChunkScores, detect_redundant_chunks


class TestChunkScoresDataclass:
    """Test the ChunkScores dataclass defaults."""

    def test_default_empty_lists(self):
        """Empty ChunkScores should have empty lists."""
        scores = ChunkScores()
        assert scores.skip == []
        assert scores.skip_reason == []


class TestDetectRedundantChunksEdgeCases:
    """Test edge cases for detect_redundant_chunks."""

    def test_empty_input(self):
        """Empty embedding list returns empty ChunkScores."""
        result = detect_redundant_chunks([])
        assert result.skip == []
        assert result.skip_reason == []

    def test_single_chunk_never_flagged(self):
        """A single chunk cannot be redundant."""
        result = detect_redundant_chunks([[1.0, 0.0, 0.0]])
        assert result.skip == [False]
        assert result.skip_reason == [""]

    def test_two_chunks_no_similarity(self):
        """Orthogonal vectors should not be flagged."""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        result = detect_redundant_chunks(embeddings)
        assert result.skip == [False, False]

    def test_zero_vector_handled(self):
        """Zero vectors should not crash (norms protected)."""
        embeddings = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
        result = detect_redundant_chunks(embeddings)
        assert len(result.skip) == 2
        # Should not raise


class TestDetectRedundantChunksIdentical:
    """Test detection of identical/near-identical chunks."""

    def test_identical_vectors_second_flagged(self):
        """Two identical vectors: second is flagged, first is kept."""
        vec = [0.5, 0.3, 0.8, 0.1]
        result = detect_redundant_chunks([vec, vec])
        assert result.skip[0] is False, "First chunk should never be flagged"
        assert result.skip[1] is True, "Second identical chunk should be flagged"
        assert "redundant with chunk 1" in result.skip_reason[1]

    def test_three_identical_vectors(self):
        """Three identical vectors: only first is kept."""
        vec = [1.0, 2.0, 3.0]
        result = detect_redundant_chunks([vec, vec, vec])
        assert result.skip == [False, True, True]
        assert "redundant with chunk 1" in result.skip_reason[1]
        assert "redundant with chunk 1" in result.skip_reason[2]

    def test_first_occurrence_always_kept(self):
        """When duplicates appear, only the first is kept regardless of position."""
        unique = [1.0, 0.0, 0.0]
        duplicate = [0.0, 1.0, 0.0]
        embeddings = [unique, duplicate, unique, duplicate]
        result = detect_redundant_chunks(embeddings)
        assert result.skip[0] is False, "First unique kept"
        assert result.skip[1] is False, "First duplicate kept"
        assert result.skip[2] is True, "Second unique flagged"
        assert result.skip[3] is True, "Second duplicate flagged"


class TestDetectRedundantChunksThreshold:
    """Test threshold behavior."""

    def test_below_threshold_not_flagged(self):
        """Vectors with similarity 0.97 should NOT be flagged at default threshold 0.98."""
        # Create two vectors with known cosine similarity ~0.97
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # Rotate slightly to get ~0.97 similarity
        angle = np.arccos(0.97)
        v2 = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)

        # Verify similarity is what we expect
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        assert 0.96 < cos_sim < 0.98, f"Test setup error: similarity is {cos_sim}"

        result = detect_redundant_chunks([v1.tolist(), v2.tolist()])
        assert result.skip[1] is False, (
            f"Similarity {cos_sim:.4f} is below 0.98 — should not be flagged"
        )

    def test_above_threshold_flagged(self):
        """Vectors with similarity 0.99 should be flagged at default threshold 0.98."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        angle = np.arccos(0.99)
        v2 = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)

        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        assert cos_sim > 0.98, f"Test setup error: similarity is {cos_sim}"

        result = detect_redundant_chunks([v1.tolist(), v2.tolist()])
        assert result.skip[1] is True, f"Similarity {cos_sim:.4f} exceeds 0.98 — should be flagged"

    def test_custom_lower_threshold(self):
        """A lower threshold (0.90) should flag more chunks."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        angle = np.arccos(0.95)
        v2 = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)

        result = detect_redundant_chunks([v1.tolist(), v2.tolist()], threshold=0.90)
        assert result.skip[1] is True, "0.95 similarity should be flagged at 0.90 threshold"

    def test_custom_higher_threshold(self):
        """A higher threshold (1.0) should only flag exact duplicates."""
        v1 = [1.0, 0.0, 0.0]
        angle = np.arccos(0.99)
        v2 = [float(np.cos(angle)), float(np.sin(angle)), 0.0]

        result = detect_redundant_chunks([v1, v2], threshold=1.0)
        assert result.skip[1] is False, "0.99 similarity should not be flagged at 1.0 threshold"


class TestDetectRedundantChunksReasonMessages:
    """Test that skip_reason messages are informative."""

    def test_reason_contains_chunk_number(self):
        """Skip reason should reference the earlier chunk (1-indexed)."""
        vec = [1.0, 0.5, 0.3]
        result = detect_redundant_chunks([vec, vec])
        assert "chunk 1" in result.skip_reason[1]

    def test_reason_contains_similarity(self):
        """Skip reason should contain the similarity score."""
        vec = [1.0, 0.5, 0.3]
        result = detect_redundant_chunks([vec, vec])
        assert "sim=" in result.skip_reason[1]

    def test_non_flagged_reason_is_empty(self):
        """Non-flagged chunks should have empty reason strings."""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        result = detect_redundant_chunks(embeddings)
        assert result.skip_reason[0] == ""
        assert result.skip_reason[1] == ""
