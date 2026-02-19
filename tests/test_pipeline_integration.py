"""
Pipeline integration tests.

Verifies that data flows correctly between pipeline stages:
extraction -> preprocessing -> vocabulary extraction.

These tests use real code paths (not mocked) with realistic legal text
to catch handoff issues between stages.
"""


# ============================================================================
# Realistic legal document text for pipeline testing
# ============================================================================

DEPOSITION_TEXT = """
SUPREME COURT OF THE STATE OF NEW YORK
COUNTY OF KINGS
Index No. 123456/2024

JOHN SMITH,
                Plaintiff,
    -against-

ABC CORPORATION,
                Defendant.

DEPOSITION OF JANE DOE
January 15, 2024

Q. Please state your name for the record.
A. My name is Jane Doe.

Q. And what is your occupation?
A. I am a registered nurse at Brooklyn Methodist Hospital.

Q. Can you describe what happened on the day of the incident?
A. On March 5, 2023, the plaintiff John Smith was admitted to the
emergency department with complaints of chest pain. Dr. Robert Chen
examined him and ordered an electrocardiogram. The ECG showed signs
of acute myocardial infarction.

Q. What treatment was administered?
A. Dr. Chen prescribed nitroglycerin and aspirin immediately. The
patient was then transferred to the cardiac catheterization laboratory
where Dr. Sarah Williams performed a percutaneous coronary intervention.
A drug-eluting stent was placed in the left anterior descending artery.

Q. Were there any complications?
A. Unfortunately, yes. Post-procedure, Mr. Smith developed a hematoma
at the femoral access site. The defendant ABC Corporation manufactured
the catheter used during the procedure. We later discovered the catheter
had a manufacturing defect that contributed to the complication.
"""


class TestExtractionToPreprocessing:
    """Test that raw text extraction output is valid preprocessing input."""

    def test_raw_text_extractor_produces_valid_output(self, tmp_path):
        """RawTextExtractor output should have keys preprocessing expects."""
        from src.core.extraction import RawTextExtractor

        # Create a real txt file
        txt_file = tmp_path / "deposition.txt"
        txt_file.write_text(DEPOSITION_TEXT, encoding="utf-8")

        extractor = RawTextExtractor(jurisdiction="ny")
        result = extractor.process_document(str(txt_file))

        # Verify the result dict has the keys downstream consumers need
        assert "extracted_text" in result
        assert "filename" in result
        assert isinstance(result["extracted_text"], str)
        assert len(result["extracted_text"]) > 100
        assert "John Smith" in result["extracted_text"]

    def test_preprocessing_accepts_extraction_output(self, tmp_path):
        """Preprocessing pipeline should accept raw extracted text."""
        from src.core.extraction import RawTextExtractor
        from src.core.preprocessing import create_default_pipeline

        # Extract from real file
        txt_file = tmp_path / "deposition.txt"
        txt_file.write_text(DEPOSITION_TEXT, encoding="utf-8")

        extractor = RawTextExtractor(jurisdiction="ny")
        result = extractor.process_document(str(txt_file))
        raw_text = result["extracted_text"]

        # Run through preprocessing
        pipeline = create_default_pipeline()
        preprocessed = pipeline.process(raw_text)

        # Preprocessed output should be a string with substantive content
        assert isinstance(preprocessed, str)
        assert len(preprocessed) > 50
        # Key entities should survive preprocessing
        assert "Smith" in preprocessed or "plaintiff" in preprocessed.lower()


class TestPreprocessingToVocabulary:
    """Test that preprocessed text produces valid vocabulary output."""

    def test_vocabulary_extractor_on_real_text(self):
        """VocabularyExtractor should find realistic terms from legal text."""
        from src.core.vocabulary import VocabularyExtractor

        extractor = VocabularyExtractor(
            exclude_list_path=None,
            medical_terms_path=None,
        )

        vocab_data = extractor.extract(DEPOSITION_TEXT, doc_count=1)

        # Should return a list of dicts
        assert isinstance(vocab_data, list)
        assert len(vocab_data) > 0

        # Each entry should have the required columns
        for entry in vocab_data[:5]:
            assert "Term" in entry
            assert "Is Person" in entry

        # Should find at least some of the key entities/terms
        all_terms = [v["Term"].lower() for v in vocab_data]
        found_entities = sum(
            1 for name in ["smith", "doe", "chen", "williams"] if any(name in t for t in all_terms)
        )
        assert found_entities >= 1, f"Expected to find person names, got terms: {all_terms[:10]}"


class TestFullPipelineIntegration:
    """End-to-end test: file -> extraction -> preprocessing -> vocabulary."""

    def test_txt_through_full_pipeline(self, tmp_path):
        """A .txt file should produce vocabulary terms through the full pipeline."""
        from src.core.extraction import RawTextExtractor
        from src.core.preprocessing import create_default_pipeline
        from src.core.vocabulary import VocabularyExtractor

        # Step 1: Extract
        txt_file = tmp_path / "case.txt"
        txt_file.write_text(DEPOSITION_TEXT, encoding="utf-8")

        extractor = RawTextExtractor(jurisdiction="ny")
        result = extractor.process_document(str(txt_file))
        raw_text = result["extracted_text"]
        assert len(raw_text) > 100

        # Step 2: Preprocess
        pipeline = create_default_pipeline()
        preprocessed = pipeline.process(raw_text)
        assert len(preprocessed) > 50

        # Step 3: Vocabulary extraction
        vocab_extractor = VocabularyExtractor(
            exclude_list_path=None,
            medical_terms_path=None,
        )
        vocab_data = vocab_extractor.extract(preprocessed, doc_count=1)

        # Verify end-to-end results
        assert isinstance(vocab_data, list)
        assert len(vocab_data) >= 1

        # The pipeline should preserve enough text for vocabulary to find terms
        terms = [v["Term"] for v in vocab_data]
        assert len(terms) > 0, "Pipeline produced no vocabulary terms"
