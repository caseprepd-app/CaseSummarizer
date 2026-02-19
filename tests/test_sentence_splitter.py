"""Tests for the NUPunkt-powered sentence splitter utility."""

import pytest

from src.core.utils.sentence_splitter import split_sentence_spans, split_sentences


class TestSplitSentences:
    """Tests for split_sentences()."""

    def test_empty_string(self):
        assert split_sentences("") == []

    def test_whitespace_only(self):
        assert split_sentences("   ") == []

    def test_single_sentence(self):
        result = split_sentences("The court ruled in favor of the plaintiff.")
        assert len(result) == 1
        assert "court ruled" in result[0]

    def test_two_simple_sentences(self):
        result = split_sentences("First sentence. Second sentence.")
        assert len(result) == 2

    @pytest.mark.xfail(reason="NUPunkt does not handle legal abbreviation 'v.'", strict=False)
    def test_legal_abbreviation_v(self):
        """'v.' in case names should NOT cause a split."""
        text = "The ruling in Smith v. Jones was unanimous. The appeal was denied."
        result = split_sentences(text)
        # Should be 2 sentences, not 3 (v. should not split)
        assert len(result) == 2
        assert "Smith v. Jones" in result[0]

    @pytest.mark.xfail(reason="NUPunkt does not handle 'U.S.C.' abbreviation", strict=False)
    def test_legal_abbreviation_usc(self):
        """'U.S.C.' should NOT cause a split."""
        text = "Under 42 U.S.C. § 1983, the claim was valid. The court agreed."
        result = split_sentences(text)
        assert len(result) == 2
        assert "U.S.C." in result[0]

    @pytest.mark.xfail(reason="NUPunkt does not handle 'Dr.', 'Mr.' abbreviations", strict=False)
    def test_title_abbreviations(self):
        """'Dr.', 'Mr.', etc. should NOT cause splits."""
        text = "Dr. Smith testified on Monday. Mr. Jones disagreed."
        result = split_sentences(text)
        assert len(result) == 2
        assert "Dr. Smith" in result[0]
        assert "Mr. Jones" in result[1]

    @pytest.mark.xfail(reason="NUPunkt does not handle 'Inc.', 'Corp.' abbreviations", strict=False)
    def test_corporate_abbreviations(self):
        """'Inc.', 'Corp.' should NOT cause splits."""
        text = "Acme Inc. filed the motion. BigCo Corp. responded."
        result = split_sentences(text)
        assert len(result) == 2

    def test_exclamation_and_question(self):
        result = split_sentences("Objection! Was the witness credible?")
        assert len(result) == 2

    def test_preserves_content(self):
        """All original text content should be preserved across sentences."""
        text = "First point here. Second point there. Third point everywhere."
        result = split_sentences(text)
        rejoined = " ".join(result)
        assert "First" in rejoined
        assert "Second" in rejoined
        assert "Third" in rejoined


class TestSplitSentenceSpans:
    """Tests for split_sentence_spans()."""

    def test_empty_string(self):
        assert split_sentence_spans("") == []

    def test_spans_cover_text(self):
        text = "First sentence. Second sentence."
        spans = split_sentence_spans(text)
        assert len(spans) >= 1
        # Each span should have (sentence, (start, end))
        for sent, (start, end) in spans:
            assert isinstance(sent, str)
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert start >= 0
            assert end > start

    def test_span_offsets_match_text(self):
        """Character offsets should point to correct positions in original text."""
        text = "The court ruled. The jury agreed."
        spans = split_sentence_spans(text)
        for sent, (start, end) in spans:
            # The extracted text at those offsets should contain the sentence
            extracted = text[start:end].strip()
            assert sent.strip() in extracted or extracted in sent.strip()

    def test_spans_non_overlapping(self):
        text = "One. Two. Three."
        spans = split_sentence_spans(text)
        for i in range(len(spans) - 1):
            _, (_, end_i) = spans[i]
            _, (start_next, _) = spans[i + 1]
            assert start_next >= end_i


class TestFallbackBehavior:
    """Tests that the regex fallback works when nupunkt is unavailable."""

    def test_fallback_splits_on_punctuation(self):
        """When nupunkt is mocked as unavailable, regex fallback should work."""
        import src.core.utils.sentence_splitter as mod

        # Save originals
        orig_available = mod._nupunkt_available
        orig_warned = mod._fallback_warned

        try:
            mod._nupunkt_available = False
            mod._fallback_warned = True  # Suppress warning

            result = mod.split_sentences("First sentence. Second sentence.")
            assert len(result) == 2
            assert "First" in result[0]
            assert "Second" in result[1]
        finally:
            mod._nupunkt_available = orig_available
            mod._fallback_warned = orig_warned
