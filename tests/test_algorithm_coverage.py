"""
Tests for vocabulary extraction algorithms with zero prior coverage.

Covers:
- RAKEAlgorithm (rake_algorithm.py)
- ScispaCyAlgorithm (scispacy_algorithm.py)
"""

from unittest.mock import MagicMock, patch

# ============================================================================
# RAKEAlgorithm
# ============================================================================


class TestRAKEAlgorithm:
    """Tests for RAKEAlgorithm keyword extraction."""

    def test_init_defaults(self):
        from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

        algo = RAKEAlgorithm()
        assert algo.name == "RAKE"
        assert algo.min_length == 1
        assert algo.max_length == 3
        assert algo.max_candidates == 150
        assert algo.min_score == 2.0

    def test_extract_returns_algorithm_result(self):
        from src.core.vocabulary.algorithms.base import AlgorithmResult
        from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

        algo = RAKEAlgorithm()
        text = "The plaintiff filed a motion for summary judgment in the federal court."
        result = algo.extract(text)
        assert isinstance(result, AlgorithmResult)
        assert result.processing_time_ms >= 0
        assert isinstance(result.candidates, list)

    def test_extract_finds_phrases(self):
        from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

        algo = RAKEAlgorithm(min_score=1.0, min_frequency=1)
        text = (
            "The percutaneous coronary intervention was performed by Dr. Smith. "
            "The percutaneous coronary intervention required a drug-eluting stent. "
            "The drug-eluting stent was placed in the left anterior descending artery."
        )
        result = algo.extract(text)
        terms = [c.term.lower() for c in result.candidates]
        # Should find multi-word phrases
        assert len(result.candidates) > 0
        assert any("stent" in t or "coronary" in t or "artery" in t for t in terms)

    def test_candidates_have_required_fields(self):
        from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

        algo = RAKEAlgorithm(min_score=1.0)
        text = "Summary judgment motion filed. Summary judgment granted."
        result = algo.extract(text)
        for c in result.candidates:
            assert hasattr(c, "term")
            assert hasattr(c, "source_algorithm")
            assert hasattr(c, "confidence")
            assert c.source_algorithm == "RAKE"
            assert 0 <= c.confidence <= 1.0

    def test_metadata_in_result(self):
        from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

        algo = RAKEAlgorithm()
        result = algo.extract("Test phrase for extraction. Test phrase repeated.")
        assert "raw_phrases_found" in result.metadata
        assert "filtered_candidates" in result.metadata

    def test_empty_text(self):
        from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

        result = RAKEAlgorithm().extract("")
        assert result.candidates == []

    def test_max_candidates_respected(self):
        from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

        algo = RAKEAlgorithm(max_candidates=5, min_score=0.1)
        text = " ".join([f"unique phrase number {i} here" for i in range(100)])
        result = algo.extract(text)
        assert len(result.candidates) <= 5

    def test_preprocess_removes_line_numbers(self):
        from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

        algo = RAKEAlgorithm()
        text = "  1  The witness testified about the incident.\n  2  The plaintiff was present."
        cleaned = algo._preprocess_text(text)
        assert "  1  " not in cleaned

    def test_clean_phrase_strips_junk_prefix(self):
        from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

        algo = RAKEAlgorithm()
        assert algo._clean_phrase("the motion") == "motion"
        assert algo._clean_phrase("a hearing") == "hearing"

    def test_clean_phrase_rejects_short(self):
        from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

        algo = RAKEAlgorithm()
        assert algo._clean_phrase("x") == ""
        assert algo._clean_phrase("") == ""

    def test_clean_phrase_rejects_numbers_only(self):
        from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

        algo = RAKEAlgorithm()
        assert algo._clean_phrase("12345") == ""

    def test_clean_phrase_rejects_long(self):
        from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

        algo = RAKEAlgorithm()
        assert algo._clean_phrase("a" * 51) == ""

    def test_count_phrase_occurrences(self):
        from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

        algo = RAKEAlgorithm()
        text = "the plaintiff filed a motion. the plaintiff objected."
        count = algo._count_phrase_occurrences("plaintiff", text)
        assert count == 2

    def test_get_config(self):
        from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

        config = RAKEAlgorithm(min_score=3.0, max_candidates=100).get_config()
        assert config["min_score"] == 3.0
        assert config["max_candidates"] == 100

    def test_rake_property_lazy_loads(self):
        from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

        algo = RAKEAlgorithm()
        assert algo._rake is None
        _ = algo.rake
        assert algo._rake is not None

    def test_clean_phrase_preserves_original_casing(self):
        """_clean_phrase should NOT title-case — preserve original document casing."""
        from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

        algo = RAKEAlgorithm()
        # Lowercase input stays lowercase
        assert algo._clean_phrase("motion in limine") == "motion in limine"
        # Uppercase input stays uppercase
        assert algo._clean_phrase("SUMMARY JUDGMENT") == "SUMMARY JUDGMENT"
        # Mixed case stays mixed
        assert algo._clean_phrase("DNA evidence") == "DNA evidence"

    def test_clean_phrase_does_not_title_case(self):
        """Explicitly verify .title() is not applied."""
        from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

        algo = RAKEAlgorithm()
        # "ibuprofen" should NOT become "Ibuprofen"
        assert algo._clean_phrase("ibuprofen") == "ibuprofen"
        # "MRI scan" should NOT become "Mri Scan"
        assert algo._clean_phrase("MRI scan") == "MRI scan"

    def test_stopwords_used(self):
        """RAKE should use app's STOPWORDS, not NLTK corpus."""
        from src.core.utils.tokenizer import STOPWORDS
        from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

        algo = RAKEAlgorithm()
        assert algo.rake.stopwords == STOPWORDS


# ============================================================================
# ScispaCyAlgorithm
# ============================================================================


class TestScispaCyAlgorithm:
    """Tests for ScispaCyAlgorithm (MedicalNER)."""

    def test_init_defaults(self):
        from src.core.vocabulary.algorithms.scispacy_algorithm import ScispaCyAlgorithm

        algo = ScispaCyAlgorithm()
        assert algo.name == "MedicalNER"
        assert algo.max_candidates == 200
        assert algo._nlp is None

    def test_extract_graceful_when_model_missing(self):
        """Should return empty result when scispaCy model is not installed."""
        from src.core.vocabulary.algorithms.scispacy_algorithm import ScispaCyAlgorithm

        algo = ScispaCyAlgorithm()
        # Force model load to fail by patching spacy.load inside _load_nlp
        with patch("spacy.load", side_effect=OSError("Model not found")):
            algo._nlp = None
            result = algo.extract("Some medical text with aspirin.")
            assert result.candidates == []
            assert result.metadata.get("skipped") is True

    def test_extract_with_mocked_model(self):
        """With a mocked spaCy model, should return CandidateTerms."""
        from src.core.vocabulary.algorithms.base import AlgorithmResult
        from src.core.vocabulary.algorithms.scispacy_algorithm import ScispaCyAlgorithm

        algo = ScispaCyAlgorithm()

        # Mock the NLP model
        mock_ent1 = MagicMock()
        mock_ent1.text = "aspirin"
        mock_ent1.label_ = "CHEMICAL"
        mock_ent1.start_char = 10

        mock_ent2 = MagicMock()
        mock_ent2.text = "myocardial infarction"
        mock_ent2.label_ = "DISEASE"
        mock_ent2.start_char = 30

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent1, mock_ent2]

        algo._nlp = MagicMock(return_value=mock_doc)

        result = algo.extract("The patient was given aspirin for myocardial infarction.")
        assert isinstance(result, AlgorithmResult)
        assert len(result.candidates) == 2
        assert result.candidates[0].term == "aspirin"
        assert result.candidates[0].suggested_type == "Medical"
        assert result.candidates[1].term == "myocardial infarction"

    def test_deduplicates_entities(self):
        from src.core.vocabulary.algorithms.scispacy_algorithm import ScispaCyAlgorithm

        algo = ScispaCyAlgorithm()

        mock_ent1 = MagicMock(text="aspirin", label_="CHEMICAL", start_char=0)
        mock_ent2 = MagicMock(text="Aspirin", label_="CHEMICAL", start_char=50)

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent1, mock_ent2]
        algo._nlp = MagicMock(return_value=mock_doc)

        result = algo.extract("aspirin was given. Aspirin helped.")
        assert len(result.candidates) == 1

    def test_skips_short_entities(self):
        from src.core.vocabulary.algorithms.scispacy_algorithm import ScispaCyAlgorithm

        algo = ScispaCyAlgorithm()
        mock_ent = MagicMock(text="x", label_="CHEMICAL", start_char=0)
        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        algo._nlp = MagicMock(return_value=mock_doc)

        result = algo.extract("x was given.")
        assert len(result.candidates) == 0

    def test_skips_pure_numbers(self):
        from src.core.vocabulary.algorithms.scispacy_algorithm import ScispaCyAlgorithm

        algo = ScispaCyAlgorithm()
        mock_ent = MagicMock(text="12345", label_="CHEMICAL", start_char=0)
        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        algo._nlp = MagicMock(return_value=mock_doc)

        result = algo.extract("12345 was found.")
        assert len(result.candidates) == 0

    def test_max_candidates_respected(self):
        from src.core.vocabulary.algorithms.scispacy_algorithm import ScispaCyAlgorithm

        algo = ScispaCyAlgorithm(max_candidates=2)
        ents = [
            MagicMock(text=f"drug_{i}", label_="CHEMICAL", start_char=i * 10) for i in range(10)
        ]
        mock_doc = MagicMock()
        mock_doc.ents = ents
        algo._nlp = MagicMock(return_value=mock_doc)

        result = algo.extract("lots of drugs")
        assert len(result.candidates) <= 2

    def test_truncates_long_text(self):
        from src.core.vocabulary.algorithms.scispacy_algorithm import ScispaCyAlgorithm

        algo = ScispaCyAlgorithm()
        mock_doc = MagicMock()
        mock_doc.ents = []
        algo._nlp = MagicMock(return_value=mock_doc)

        long_text = "a" * 2_000_000
        result = algo.extract(long_text)
        # Should have called nlp with truncated text
        call_args = algo._nlp.call_args[0][0]
        assert len(call_args) <= 1_024 * 1_024
        assert result.metadata.get("text_truncated") is True

    def test_preserves_original_entity_casing(self):
        """Entities should keep their original casing, not be title-cased."""
        from src.core.vocabulary.algorithms.scispacy_algorithm import ScispaCyAlgorithm

        algo = ScispaCyAlgorithm()

        # Mixed-case entities from model output
        mock_ent1 = MagicMock(text="Ibuprofen", label_="CHEMICAL", start_char=0)
        mock_ent2 = MagicMock(text="type 2 diabetes", label_="DISEASE", start_char=20)
        mock_ent3 = MagicMock(text="NSAID", label_="CHEMICAL", start_char=40)

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent1, mock_ent2, mock_ent3]
        algo._nlp = MagicMock(return_value=mock_doc)

        result = algo.extract("Ibuprofen for type 2 diabetes. NSAID treatment.")
        terms = [c.term for c in result.candidates]
        # Each should be preserved exactly as the model returned it
        assert "Ibuprofen" in terms, "Title-case from model preserved"
        assert "type 2 diabetes" in terms, "Lowercase from model preserved"
        assert "NSAID" in terms, "Uppercase acronym from model preserved"
        # Verify none got title-cased by our code
        assert "Type 2 Diabetes" not in terms, "Should NOT title-case"
        assert "Nsaid" not in terms, "Should NOT title-case acronyms"

    def test_get_config(self):
        from src.core.vocabulary.algorithms.scispacy_algorithm import ScispaCyAlgorithm

        config = ScispaCyAlgorithm(max_candidates=50).get_config()
        assert config["max_candidates"] == 50
