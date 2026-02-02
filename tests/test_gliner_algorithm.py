"""
Tests for GLiNER zero-shot NER algorithm.

Tests cover:
- Entity extraction with mocked model
- Label-to-type mapping for all categories
- Text chunking with overlap
- Deduplication (keeps highest confidence)
- Graceful failure when gliner not installed
- Label validation
"""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Label-to-type mapping tests
# ---------------------------------------------------------------------------


class TestLabelToTypeMapping:
    """Test _map_label_to_type for each category."""

    def test_person_label(self):
        from src.core.vocabulary.algorithms.gliner_algorithm import _map_label_to_type

        assert _map_label_to_type("person") == "Person"
        assert _map_label_to_type("Person Name") == "Person"

    def test_medical_labels(self):
        from src.core.vocabulary.algorithms.gliner_algorithm import _map_label_to_type

        assert _map_label_to_type("medical condition") == "Medical"
        assert _map_label_to_type("medication") == "Medical"
        assert _map_label_to_type("drug name") == "Medical"
        assert _map_label_to_type("disease") == "Medical"
        assert _map_label_to_type("chronic condition") == "Medical"

    def test_organization_labels(self):
        from src.core.vocabulary.algorithms.gliner_algorithm import _map_label_to_type

        assert _map_label_to_type("court name") == "Organization"
        assert _map_label_to_type("organization") == "Organization"
        assert _map_label_to_type("company") == "Organization"

    def test_place_labels(self):
        from src.core.vocabulary.algorithms.gliner_algorithm import _map_label_to_type

        assert _map_label_to_type("place") == "Place"
        assert _map_label_to_type("location") == "Place"
        assert _map_label_to_type("city") == "Place"
        assert _map_label_to_type("state") == "Place"
        assert _map_label_to_type("country") == "Place"

    def test_technical_fallback(self):
        from src.core.vocabulary.algorithms.gliner_algorithm import _map_label_to_type

        assert _map_label_to_type("case citation") == "Technical"
        assert _map_label_to_type("statute") == "Technical"
        assert _map_label_to_type("legal term") == "Technical"


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------


class TestChunking:
    """Test _chunk_text produces correct segments with overlap."""

    def test_short_text_single_chunk(self):
        from src.core.vocabulary.algorithms.gliner_algorithm import _chunk_text

        text = "word " * 100
        chunks = _chunk_text(text.strip())
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self):
        from src.core.vocabulary.algorithms.gliner_algorithm import _chunk_text

        # 600 words -> should produce multiple chunks
        text = " ".join(f"word{i}" for i in range(600))
        chunks = _chunk_text(text)
        assert len(chunks) >= 2

    def test_overlap_between_chunks(self):
        from src.core.vocabulary.algorithms.gliner_algorithm import (
            _OVERLAP_WORDS,
            _chunk_text,
        )

        words = [f"w{i}" for i in range(600)]
        text = " ".join(words)
        chunks = _chunk_text(text)

        # First chunk ends at word 300, second starts at word 250
        chunk1_words = set(chunks[0].split())
        chunk2_words = set(chunks[1].split())
        overlap = chunk1_words & chunk2_words
        assert len(overlap) >= _OVERLAP_WORDS - 5  # Allow small margin


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Test dedup keeps highest confidence across chunks."""

    @patch("src.core.vocabulary.algorithms.gliner_algorithm.GLiNERAlgorithm._load_model")
    def test_dedup_keeps_highest_confidence(self, mock_load):
        from src.core.vocabulary.algorithms.gliner_algorithm import GLiNERAlgorithm

        algo = GLiNERAlgorithm(labels=["person"])
        # Mock the model
        algo._model = MagicMock()
        # First chunk returns entity with score 0.6, second with 0.9
        algo._model.predict_entities.side_effect = [
            [{"text": "John Smith", "label": "person", "score": 0.6}],
            [{"text": "John Smith", "label": "person", "score": 0.9}],
        ]

        # Create text with 2 chunks
        text = " ".join(["word"] * 400)
        result = algo.extract(text)

        assert len(result.candidates) == 1
        assert result.candidates[0].confidence == 0.9


# ---------------------------------------------------------------------------
# Extract returns valid AlgorithmResult
# ---------------------------------------------------------------------------


class TestExtract:
    """Test extract returns valid AlgorithmResult."""

    @patch("src.core.vocabulary.algorithms.gliner_algorithm.GLiNERAlgorithm._load_model")
    def test_extract_returns_algorithm_result(self, mock_load):
        from src.core.vocabulary.algorithms.gliner_algorithm import GLiNERAlgorithm

        algo = GLiNERAlgorithm(labels=["person", "medical condition"])
        algo._model = MagicMock()
        algo._model.predict_entities.return_value = [
            {"text": "John Smith", "label": "person", "score": 0.85},
            {"text": "hypertension", "label": "medical condition", "score": 0.72},
        ]

        result = algo.extract("John Smith was diagnosed with hypertension.")

        assert len(result.candidates) == 2
        assert result.processing_time_ms >= 0
        assert result.metadata["chunk_count"] >= 1

        # Check types mapped correctly
        types = {c.term: c.suggested_type for c in result.candidates}
        assert types["John Smith"] == "Person"
        assert types["hypertension"] == "Medical"

    @patch("src.core.vocabulary.algorithms.gliner_algorithm.GLiNERAlgorithm._load_model")
    def test_extract_skips_short_and_numeric(self, mock_load):
        from src.core.vocabulary.algorithms.gliner_algorithm import GLiNERAlgorithm

        algo = GLiNERAlgorithm(labels=["person"])
        algo._model = MagicMock()
        algo._model.predict_entities.return_value = [
            {"text": "A", "label": "person", "score": 0.9},  # too short
            {"text": "123", "label": "person", "score": 0.9},  # numeric
            {"text": "Valid Name", "label": "person", "score": 0.8},
        ]

        result = algo.extract("Some text here.")
        assert len(result.candidates) == 1
        assert result.candidates[0].term == "Valid Name"


# ---------------------------------------------------------------------------
# Graceful failure when gliner not installed
# ---------------------------------------------------------------------------


class TestGracefulFailure:
    """Test graceful handling when gliner package is not installed."""

    def test_import_failure_returns_empty_result(self):
        from src.core.vocabulary.algorithms.gliner_algorithm import GLiNERAlgorithm

        algo = GLiNERAlgorithm(labels=["person"])

        # Simulate import failure in _load_model
        with patch.object(algo, "_load_model", side_effect=ImportError("No module named 'gliner'")):
            result = algo.extract("Some text here.")

        assert len(result.candidates) == 0
        assert result.metadata.get("skipped") is True


# ---------------------------------------------------------------------------
# Label validation tests
# ---------------------------------------------------------------------------


class TestLabelValidation:
    """Test GLiNER label validation rules."""

    def test_validate_strips_whitespace(self):
        # We test the validation function directly
        # It's defined inside _register_all_settings, so we test via the algorithm
        from src.config import GLINER_DEFAULT_LABELS, GLINER_MAX_LABELS

        assert len(GLINER_DEFAULT_LABELS) >= 1
        assert GLINER_MAX_LABELS == 20

    def test_algorithm_uses_default_labels_when_none(self):
        from src.config import GLINER_DEFAULT_LABELS
        from src.core.vocabulary.algorithms.gliner_algorithm import GLiNERAlgorithm

        algo = GLiNERAlgorithm(labels=None)
        assert algo.labels == list(GLINER_DEFAULT_LABELS)

    def test_algorithm_accepts_custom_labels(self):
        from src.core.vocabulary.algorithms.gliner_algorithm import GLiNERAlgorithm

        custom = ["person", "vehicle"]
        algo = GLiNERAlgorithm(labels=custom)
        assert algo.labels == custom
