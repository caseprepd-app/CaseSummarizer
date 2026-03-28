"""
Tests for CrossEncoderReranker.

Covers the reranking step that refines hybrid retrieval results using
a cross-encoder model (Alibaba-NLP/gte-reranker-modernbert-base).

Tests use mocked CrossEncoder models to avoid the ~300MB download.

Tests:
- Empty input returns empty output immediately (no model load)
- Model load failure falls back to original top-k chunks
- Sigmoid score normalization math (logits → probabilities)
- MIN_RELEVANCE_SCORE filtering removes low-relevance chunks
- rerank stores reranker metadata on each chunk
- top_k limits the number of returned chunks
- Combined score on returned chunks is updated to the sigmoid score
- is_available returns False when model load fails
"""

from unittest.mock import MagicMock, patch

import numpy as np


def _make_chunk(text="The plaintiff filed suit.", filename="doc.pdf", chunk_num=0, score=0.5):
    """Create a minimal MergedChunk mock for reranker tests."""
    from src.core.retrieval.chunk_merger import MergedChunk

    return MergedChunk(
        chunk_id=f"{filename}_{chunk_num}",
        text=text,
        combined_score=score,
        sources=["FAISS"],
        filename=filename,
        chunk_num=chunk_num,
        section_name="Body",
        metadata={},
    )


class TestRerankerEmptyInput:
    """Tests for CrossEncoderReranker.rerank with empty input."""

    def test_empty_chunks_returns_empty_list_without_loading_model(self):
        """rerank([]) returns [] without attempting to load the cross-encoder."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        with patch.object(reranker, "_load_model") as mock_load:
            result = reranker.rerank("Who filed?", chunks=[], top_k=5)

            assert result == []
            assert reranker._model is None  # Model was never loaded
            mock_load.assert_not_called()  # No unnecessary model load


class TestRerankerModelLoadFailure:
    """Tests for CrossEncoderReranker.rerank when model loading fails."""

    def test_falls_back_to_first_top_k_chunks_on_load_failure(self):
        """When the model cannot load, rerank returns the first top_k chunks unchanged."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        chunks = [
            _make_chunk("Chunk A", chunk_num=0),
            _make_chunk("Chunk B", chunk_num=1),
            _make_chunk("Chunk C", chunk_num=2),
        ]

        reranker = CrossEncoderReranker()

        with patch.object(reranker, "_load_model", side_effect=RuntimeError("no model")):
            result = reranker.rerank("Query?", chunks=chunks, top_k=2)

        assert len(result) == 2
        assert result[0].chunk_num == 0
        assert result[1].chunk_num == 1


class TestRerankerSigmoidNormalization:
    """Tests for CrossEncoderReranker sigmoid score normalization."""

    def _reranker_with_mock_model(self, raw_scores):
        """Return a CrossEncoderReranker whose model.predict returns raw_scores."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()
        reranker._model.predict.return_value = np.array(raw_scores)
        return reranker

    def test_logit_zero_maps_to_sigmoid_half(self):
        """Raw score 0.0 → sigmoid(0) = 0.5, which is above MIN_RELEVANCE_SCORE."""
        reranker = self._reranker_with_mock_model([0.0])
        chunks = [_make_chunk("Chunk.", chunk_num=0)]

        result = reranker.rerank("Q?", chunks=chunks, top_k=5)

        assert len(result) == 1
        assert abs(result[0].combined_score - 0.5) < 1e-4

    def test_large_positive_logit_maps_to_near_one(self):
        """High raw logit (e.g. 10) → sigmoid close to 1.0."""
        reranker = self._reranker_with_mock_model([10.0])
        chunks = [_make_chunk("Highly relevant chunk.", chunk_num=0)]

        result = reranker.rerank("Q?", chunks=chunks, top_k=5)

        assert len(result) == 1
        assert result[0].combined_score > 0.99

    def test_large_negative_logit_maps_to_near_zero_and_filtered(self):
        """Very negative logit → sigmoid near 0 → chunk is filtered out."""
        reranker = self._reranker_with_mock_model([-10.0])
        chunks = [_make_chunk("Irrelevant chunk.", chunk_num=0)]

        result = reranker.rerank("Q?", chunks=chunks, top_k=5)

        # Sigmoid(-10) ≈ 0.000045, far below MIN_RELEVANCE_SCORE (0.3)
        assert len(result) == 0

    def test_chunks_sorted_by_normalized_score_descending(self):
        """Returned chunks are sorted highest normalized score first."""
        reranker = self._reranker_with_mock_model([1.0, 5.0, -0.5])
        chunks = [
            _make_chunk("Low relevance.", chunk_num=0),
            _make_chunk("Highest relevance.", chunk_num=1),
            _make_chunk("Medium relevance.", chunk_num=2),
        ]

        result = reranker.rerank("Q?", chunks=chunks, top_k=3)

        # Must be sorted descending by combined_score
        scores = [r.combined_score for r in result]
        assert scores == sorted(scores, reverse=True)


class TestRerankerMinRelevanceFiltering:
    """Tests for CrossEncoderReranker MIN_RELEVANCE_SCORE threshold."""

    def test_chunks_below_threshold_are_excluded(self):
        """Chunks with sigmoid score below 0.3 are filtered from results."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()
        # sigmoid(-1.0) ≈ 0.269, which is below 0.3
        reranker._model.predict.return_value = np.array([-1.0, 2.0])

        chunks = [
            _make_chunk("Below threshold.", chunk_num=0),
            _make_chunk("Above threshold.", chunk_num=1),
        ]

        result = reranker.rerank("Q?", chunks=chunks, top_k=5)

        assert len(result) == 1
        assert result[0].chunk_num == 1

    def test_all_chunks_filtered_returns_empty_list(self):
        """If all chunks are below threshold, rerank returns []."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()
        reranker._model.predict.return_value = np.array([-5.0, -4.0])

        chunks = [
            _make_chunk("Chunk A.", chunk_num=0),
            _make_chunk("Chunk B.", chunk_num=1),
        ]

        result = reranker.rerank("Q?", chunks=chunks, top_k=5)
        assert result == []

    def test_min_relevance_score_constant(self):
        """MIN_RELEVANCE_SCORE is 0.3 (sigmoid(0) = 0.5 gives clear positive signal)."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        assert CrossEncoderReranker.MIN_RELEVANCE_SCORE == 0.3


class TestRerankerMetadata:
    """Tests for reranker metadata written to chunk.metadata."""

    def _reranker_with_scores(self, scores):
        """Return a reranker whose mock model returns the given scores."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()
        reranker._model.predict.return_value = np.array(scores)
        return reranker

    def test_reranker_score_raw_stored(self):
        """chunk.metadata['reranker_score_raw'] holds the original logit."""
        reranker = self._reranker_with_scores([2.0])
        chunks = [_make_chunk("Good chunk.", chunk_num=0, score=0.4)]

        result = reranker.rerank("Q?", chunks=chunks, top_k=5)

        assert len(result) == 1
        assert "reranker_score_raw" in result[0].metadata
        assert abs(result[0].metadata["reranker_score_raw"] - 2.0) < 1e-4

    def test_reranker_score_normalized_stored(self):
        """chunk.metadata['reranker_score'] holds the sigmoid-normalized score."""
        reranker = self._reranker_with_scores([2.0])
        chunks = [_make_chunk("Good chunk.", chunk_num=0)]

        result = reranker.rerank("Q?", chunks=chunks, top_k=5)

        assert "reranker_score" in result[0].metadata
        expected_sigmoid = 1 / (1 + np.exp(-2.0))
        assert abs(result[0].metadata["reranker_score"] - expected_sigmoid) < 1e-4

    def test_original_hybrid_score_preserved_in_metadata(self):
        """chunk.metadata['original_hybrid_score'] stores the pre-reranking combined_score."""
        reranker = self._reranker_with_scores([2.0])
        chunk = _make_chunk("Chunk.", chunk_num=0, score=0.65)
        original_score = chunk.combined_score

        result = reranker.rerank("Q?", chunks=[chunk], top_k=5)

        assert result[0].metadata["original_hybrid_score"] == original_score

    def test_rerank_position_stored(self):
        """chunk.metadata['rerank_position'] records the 1-based rank after reranking."""
        # Use equal scores so both are at the mean (not filtered by mean cutoff)
        reranker = self._reranker_with_scores([3.0, 3.0])
        chunks = [
            _make_chunk("Best.", chunk_num=0, score=0.9),
            _make_chunk("Second.", chunk_num=1, score=0.7),
        ]

        result = reranker.rerank("Q?", chunks=chunks, top_k=5)

        assert result[0].metadata["rerank_position"] == 1
        assert result[1].metadata["rerank_position"] == 2

    def test_combined_score_updated_to_reranker_score(self):
        """chunk.combined_score is replaced with the normalized reranker score."""
        reranker = self._reranker_with_scores([2.0])
        chunk = _make_chunk("Chunk.", chunk_num=0, score=0.3)

        result = reranker.rerank("Q?", chunks=[chunk], top_k=5)

        expected_sigmoid = float(1 / (1 + np.exp(-2.0)))
        assert abs(result[0].combined_score - expected_sigmoid) < 1e-4


class TestRerankerTopK:
    """Tests for CrossEncoderReranker top_k limiting."""

    def _reranker_with_scores(self, scores):
        """Return a reranker whose mock model returns the given scores."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()
        reranker._model.predict.return_value = np.array(scores)
        return reranker

    def test_top_k_limits_results(self):
        """rerank returns at most top_k chunks."""
        # All 4 chunks score above threshold (sigmoid(1)≈0.73)
        reranker = self._reranker_with_scores([1.0, 1.0, 1.0, 1.0])
        chunks = [_make_chunk(f"Chunk {i}.", chunk_num=i) for i in range(4)]

        result = reranker.rerank("Q?", chunks=chunks, top_k=2)

        assert len(result) <= 2

    def test_top_k_one_returns_single_best_chunk(self):
        """top_k=1 returns only the highest-scoring chunk."""
        reranker = self._reranker_with_scores([1.0, 3.0, 0.5])
        chunks = [_make_chunk(f"Chunk {i}.", chunk_num=i) for i in range(3)]

        result = reranker.rerank("Q?", chunks=chunks, top_k=1)

        assert len(result) == 1
        assert result[0].chunk_num == 1  # highest raw score was chunk_num=1


class TestRerankerIsAvailable:
    """Tests for CrossEncoderReranker.is_available."""

    def test_is_available_returns_false_when_load_fails(self):
        """is_available returns False when the model file is not present."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        with patch.object(reranker, "_load_model", side_effect=OSError("file not found")):
            assert reranker.is_available() is False

    def test_is_available_returns_true_when_model_loads(self):
        """is_available returns True when _load_model succeeds."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()  # Simulate already-loaded model

        assert reranker.is_available() is True


class TestMeanCutoffReranking:
    """Tests for dynamic mean-cutoff reranking behavior (Task 2B)."""

    def _reranker_with_scores(self, scores):
        """Return a reranker whose mock model returns the given scores."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()
        reranker._model.predict.return_value = np.array(scores)
        return reranker

    def test_single_chunk_always_returned(self):
        """A single chunk above MIN_RELEVANCE_SCORE should always be returned."""
        reranker = self._reranker_with_scores([2.0])
        chunks = [_make_chunk("Only chunk.", chunk_num=0)]

        result = reranker.rerank("Q?", chunks=chunks, top_k=5)

        # sigmoid(2.0)≈0.88, mean=0.88, chunk is at mean → included
        assert len(result) == 1

    def test_all_above_mean_returns_all(self):
        """When all chunks have equal scores, all are at the mean and returned."""
        reranker = self._reranker_with_scores([2.0, 2.0, 2.0])
        chunks = [_make_chunk(f"Chunk {i}.", chunk_num=i) for i in range(3)]

        result = reranker.rerank("Q?", chunks=chunks, top_k=5)

        assert len(result) == 3

    def test_below_mean_chunks_filtered(self):
        """Chunks below the mean sigmoid score should be filtered out."""
        # sigmoid(5)≈0.993, sigmoid(-0.5)≈0.378, mean≈0.686
        # Only chunk 0 (0.993) is above mean
        reranker = self._reranker_with_scores([5.0, -0.5])
        chunks = [
            _make_chunk("Great chunk.", chunk_num=0),
            _make_chunk("Mediocre chunk.", chunk_num=1),
        ]

        result = reranker.rerank("Q?", chunks=chunks, top_k=5)

        assert len(result) == 1
        assert result[0].chunk_num == 0

    def test_all_below_min_relevance_returns_empty(self):
        """When all chunks are below MIN_RELEVANCE_SCORE, return empty."""
        reranker = self._reranker_with_scores([-5.0, -6.0, -7.0])
        chunks = [_make_chunk(f"Chunk {i}.", chunk_num=i) for i in range(3)]

        result = reranker.rerank("Q?", chunks=chunks, top_k=5)

        assert result == []

    def test_mean_cutoff_respects_top_k_cap(self):
        """Mean cutoff should still respect the top_k cap."""
        # All equal high scores: all pass mean cutoff
        reranker = self._reranker_with_scores([3.0, 3.0, 3.0, 3.0])
        chunks = [_make_chunk(f"Chunk {i}.", chunk_num=i) for i in range(4)]

        result = reranker.rerank("Q?", chunks=chunks, top_k=2)

        assert len(result) <= 2

    def test_empty_input_returns_empty(self):
        """Empty chunks input should return empty list."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        result = reranker.rerank("Q?", chunks=[], top_k=5)

        assert result == []

    def test_mixed_scores_returns_above_mean_only(self):
        """With mixed scores, only above-mean chunks that pass floor are returned."""
        # sigmoid(4)≈0.982, sigmoid(3)≈0.953, sigmoid(0)=0.5, sigmoid(-2)≈0.119
        # mean ≈ (0.982+0.953+0.5+0.119)/4 ≈ 0.639
        # Above mean: chunks 0,1 (0.982,0.953). Chunk 2 (0.5) below mean.
        # Chunk 3 (0.119) below MIN_RELEVANCE_SCORE
        reranker = self._reranker_with_scores([4.0, 3.0, 0.0, -2.0])
        chunks = [_make_chunk(f"Chunk {i}.", chunk_num=i) for i in range(4)]

        result = reranker.rerank("Q?", chunks=chunks, top_k=5)

        assert len(result) == 2
        assert result[0].chunk_num == 0
        assert result[1].chunk_num == 1
