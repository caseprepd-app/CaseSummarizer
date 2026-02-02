"""
Tests for GLiNER zero-shot NER algorithm.

Tests cover:
- Entity extraction with mocked model
- Label-to-type mapping for all categories
- Text chunking with sentence-aware overlap
- Deduplication (keeps highest confidence)
- Graceful failure when gliner not installed
- Label validation
- Background model warm-up
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
        assert _map_label_to_type("anatomical body part") == "Medical"
        assert _map_label_to_type("medical procedure") == "Medical"

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

        assert _map_label_to_type("specialized scientific term") == "Technical"
        assert _map_label_to_type("chemical compound") == "Technical"
        assert _map_label_to_type("foreign phrase") == "Technical"


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------


class TestChunking:
    """Test _chunk_text produces correct sentence-aligned segments with overlap."""

    def test_short_text_single_chunk(self):
        from src.core.vocabulary.algorithms.gliner_algorithm import _chunk_text

        text = "This is a short sentence. And another one."
        chunks = _chunk_text(text)
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self):
        from src.core.vocabulary.algorithms.gliner_algorithm import _chunk_text

        # Build text with many sentences totalling ~600 words
        sentences = [f"This is sentence number {i} with some extra words." for i in range(80)]
        text = " ".join(sentences)
        chunks = _chunk_text(text)
        assert len(chunks) >= 2

    def test_overlap_between_chunks(self):
        from src.core.vocabulary.algorithms.gliner_algorithm import _chunk_text

        # Build sentences with unique words so we can detect overlap
        sentences = [f"Unique{i} is a word in sentence {i}." for i in range(80)]
        text = " ".join(sentences)
        chunks = _chunk_text(text)

        assert len(chunks) >= 2
        # Chunks should share some sentences (overlap)
        chunk1_words = set(chunks[0].split())
        chunk2_words = set(chunks[1].split())
        overlap = chunk1_words & chunk2_words
        # Should have meaningful overlap (at least a few shared words)
        assert len(overlap) >= 5

    def test_chunks_dont_split_mid_sentence(self):
        from src.core.vocabulary.algorithms.gliner_algorithm import _chunk_text

        # Build text where each sentence is clearly delimited
        sentences = [f"The anterior cruciate ligament number {i} was examined." for i in range(80)]
        text = " ".join(sentences)
        chunks = _chunk_text(text)

        # Each chunk should contain complete sentences — check that
        # "anterior cruciate ligament" is never split across chunk boundary
        for chunk in chunks:
            assert "anterior cruciate" not in chunk or "anterior cruciate ligament" in chunk


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

        # Create text with enough sentences for 2 chunks
        sentences = [f"This is sentence number {i} about things." for i in range(80)]
        text = " ".join(sentences)
        result = algo.extract(text)

        assert len(result.candidates) == 1
        assert result.candidates[0].confidence == 0.9
        assert result.candidates[0].metadata["chunk_hits"] == 2


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

        # Single-chunk text should have chunk_hits == 1
        for c in result.candidates:
            assert c.metadata["chunk_hits"] == 1

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
# Background warm-up tests
# ---------------------------------------------------------------------------


class TestWarmUp:
    """Test background model warm-up."""

    def test_warm_up_makes_model_available(self):
        from src.core.vocabulary.algorithms.gliner_algorithm import GLiNERAlgorithm

        algo = GLiNERAlgorithm(labels=["person"])

        # Mock _load_model to set _model
        def fake_load():
            algo._model = MagicMock()

        with patch.object(algo, "_load_model", side_effect=fake_load):
            algo.warm_up()
            # Wait for background thread to finish
            algo._model_ready.wait(timeout=5)

        assert algo._model is not None
        assert algo._load_error is None

    def test_extract_works_without_warm_up(self):
        from src.core.vocabulary.algorithms.gliner_algorithm import GLiNERAlgorithm

        algo = GLiNERAlgorithm(labels=["person"])

        def fake_load():
            algo._model = MagicMock()
            algo._model.predict_entities.return_value = [
                {"text": "Test Entity", "label": "person", "score": 0.8},
            ]

        with patch.object(algo, "_load_model", side_effect=fake_load):
            result = algo.extract("Some text here.")

        assert len(result.candidates) == 1

    def test_warm_up_failure_returns_empty_result(self):
        from src.core.vocabulary.algorithms.gliner_algorithm import GLiNERAlgorithm

        algo = GLiNERAlgorithm(labels=["person"])

        with patch.object(algo, "_load_model", side_effect=RuntimeError("load failed")):
            algo.warm_up()
            algo._model_ready.wait(timeout=5)

        assert algo._load_error == "load failed"


# ---------------------------------------------------------------------------
# Label validation tests
# ---------------------------------------------------------------------------


class TestLabelValidation:
    """Test GLiNER label validation rules."""

    def test_default_labels_exist(self):
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


# ---------------------------------------------------------------------------
# File-based label loading tests
# ---------------------------------------------------------------------------


class TestLoadGlinerLabels:
    """Test load_gliner_labels reads from file with validation."""

    def test_loads_from_file(self, tmp_path):
        from unittest.mock import patch as _patch

        labels_file = tmp_path / "gliner_labels.txt"
        labels_file.write_text(
            "# comment\nanatomical body part\nmedication\n\n# another comment\nforeign phrase\n",
            encoding="utf-8",
        )

        with _patch("src.config.GLINER_LABELS_FILE", labels_file):
            from src.config import load_gliner_labels

            result = load_gliner_labels()

        assert result == ["anatomical body part", "medication", "foreign phrase"]

    def test_skips_invalid_labels(self, tmp_path):
        from unittest.mock import patch as _patch

        labels_file = tmp_path / "gliner_labels.txt"
        labels_file.write_text(
            "a\n"  # too short
            "123\n"  # no letters
            "valid label\n"
            "valid label\n"  # duplicate
            "",
            encoding="utf-8",
        )

        with _patch("src.config.GLINER_LABELS_FILE", labels_file):
            from src.config import load_gliner_labels

            result = load_gliner_labels()

        assert result == ["valid label"]

    def test_truncates_over_maximum(self, tmp_path):
        from unittest.mock import patch as _patch

        labels_file = tmp_path / "gliner_labels.txt"
        lines = [f"label number {i}" for i in range(25)]
        labels_file.write_text("\n".join(lines), encoding="utf-8")

        with _patch("src.config.GLINER_LABELS_FILE", labels_file):
            from src.config import load_gliner_labels

            result = load_gliner_labels()

        assert len(result) == 20

    def test_falls_back_to_defaults_on_empty_file(self, tmp_path):
        from unittest.mock import patch as _patch

        labels_file = tmp_path / "gliner_labels.txt"
        labels_file.write_text("# only comments\n", encoding="utf-8")

        with _patch("src.config.GLINER_LABELS_FILE", labels_file):
            from src.config import GLINER_DEFAULT_LABELS, load_gliner_labels

            result = load_gliner_labels()

        assert result == list(GLINER_DEFAULT_LABELS)
