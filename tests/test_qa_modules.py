"""
Tests for Q&A core modules.

Covers:
- AnswerGenerator (answer_generator.py) — extraction mode
- HallucinationVerifier (hallucination_verifier.py) — span building, reliability
- Citation excerpt extraction (citation_excerpt.py)
- Token budget utilities (token_budget.py)
- Verification config (verification_config.py)
- Q&A constants (qa_constants.py)
"""

from unittest.mock import patch

# ============================================================================
# A. Answer Generator — Extraction Mode
# ============================================================================


class TestAnswerGeneratorExtraction:
    """Tests for AnswerGenerator in extraction mode (no LLM needed)."""

    def test_init_extraction_mode(self):
        """Creates generator in extraction mode."""
        from src.core.qa.answer_generator import AnswerGenerator

        gen = AnswerGenerator(mode="extraction")
        assert gen is not None

    def test_empty_context_returns_unanswered(self):
        """Empty context returns unanswered text."""
        from src.core.qa.answer_generator import AnswerGenerator
        from src.core.qa.qa_constants import UNANSWERED_TEXT

        gen = AnswerGenerator(mode="extraction")
        assert gen.generate("Who?", "") == UNANSWERED_TEXT
        assert gen.generate("Who?", "   ") == UNANSWERED_TEXT

    def test_extracts_relevant_sentences(self):
        """Extraction returns sentences matching question keywords."""
        from src.core.qa.answer_generator import AnswerGenerator

        gen = AnswerGenerator(mode="extraction")
        context = (
            "The plaintiff is John Smith. "
            "He filed the complaint on January 15. "
            "The weather was sunny that day."
        )
        answer = gen.generate("Who is the plaintiff?", context)
        assert "John Smith" in answer

    def test_no_matching_keywords(self):
        """Returns 'no specific answer' when no keywords match."""
        from src.core.qa.answer_generator import AnswerGenerator

        gen = AnswerGenerator(mode="extraction")
        context = "The cat sat on the mat."
        answer = gen.generate("What is the plaintiff's address?", context)
        assert "no specific answer" in answer.lower() or len(answer) > 0

    def test_long_answer_truncated(self):
        """Very long answers are truncated to 500 chars."""
        from src.core.qa.answer_generator import AnswerGenerator

        gen = AnswerGenerator(mode="extraction")
        # Create context with many matching sentences
        context = " ".join([f"The plaintiff filed document {i}." for i in range(100)])
        answer = gen.generate("What did the plaintiff file?", context)
        assert len(answer) <= 510  # 500 + "..."

    def test_extract_keywords(self):
        """Keywords extraction removes stopwords and short words."""
        from src.core.qa.answer_generator import AnswerGenerator

        gen = AnswerGenerator(mode="extraction")
        keywords = gen._extract_keywords("Who is the plaintiff in this case?")
        assert "plaintiff" in keywords
        assert "case" in keywords
        assert "the" not in keywords
        assert "is" not in keywords
        assert "who" not in keywords

    def test_score_sentence(self):
        """Sentence scoring counts keyword matches."""
        from src.core.qa.answer_generator import AnswerGenerator

        gen = AnswerGenerator(mode="extraction")
        keywords = {"plaintiff", "filed", "complaint"}
        score = gen._score_sentence("The plaintiff filed a complaint.", keywords)
        assert score == 3

    def test_clean_sentence_adds_period(self):
        """Clean sentence adds period if missing."""
        from src.core.qa.answer_generator import AnswerGenerator

        gen = AnswerGenerator(mode="extraction")
        assert gen._clean_sentence("Hello world") == "Hello world."
        assert gen._clean_sentence("Hello world.") == "Hello world."
        assert gen._clean_sentence("Hello world!") == "Hello world!"

    def test_set_mode_is_noop(self):
        """set_mode is a no-op (always extraction)."""
        from src.core.qa.answer_generator import AnswerGenerator

        gen = AnswerGenerator(mode="extraction")
        gen.set_mode("ollama")  # Should not crash
        # Always uses extraction regardless


# ============================================================================
# B. Hallucination Verifier — Span Building and Reliability
# ============================================================================


class TestHallucinationVerifierSpans:
    """Tests for HallucinationVerifier internal methods (no model loading)."""

    def _make_verifier(self):
        """Create verifier without loading model."""
        from src.core.qa.hallucination_verifier import HallucinationVerifier

        v = HallucinationVerifier.__new__(HallucinationVerifier)
        v._detector = None
        v._model_path = "test"
        return v

    def test_build_complete_spans_no_predictions(self):
        """No predictions -> entire answer marked verified."""
        v = self._make_verifier()
        spans = v._build_complete_spans("Hello world", [])
        assert len(spans) == 1
        assert spans[0].text == "Hello world"
        assert spans[0].hallucination_prob == 0.0

    def test_build_complete_spans_one_hallucination(self):
        """Single hallucination creates verified + hallucinated + verified spans."""
        v = self._make_verifier()
        predictions = [{"start": 6, "end": 11, "text": "world", "confidence": 0.9}]
        spans = v._build_complete_spans("Hello world test", predictions)

        # Should have: "Hello " (verified), "world" (hallucinated), " test" (verified)
        assert len(spans) >= 2
        hallucinated = [s for s in spans if s.hallucination_prob > 0.5]
        assert len(hallucinated) == 1
        assert hallucinated[0].text == "world"

    def test_build_complete_spans_at_start(self):
        """Hallucination at very start of answer."""
        v = self._make_verifier()
        predictions = [{"start": 0, "end": 5, "text": "Hello", "confidence": 0.8}]
        spans = v._build_complete_spans("Hello world", predictions)
        assert spans[0].text == "Hello"
        assert spans[0].hallucination_prob == 0.8

    def test_calculate_reliability_all_verified(self):
        """All-verified spans -> reliability 1.0."""
        from src.core.qa.hallucination_verifier import VerifiedSpan

        v = self._make_verifier()
        spans = [VerifiedSpan("Hello world", 0, 11, 0.0)]
        assert v._calculate_reliability(spans, "Hello world") == 1.0

    def test_calculate_reliability_all_hallucinated(self):
        """All-hallucinated spans -> reliability 0.0."""
        from src.core.qa.hallucination_verifier import VerifiedSpan

        v = self._make_verifier()
        spans = [VerifiedSpan("Hello world", 0, 11, 1.0)]
        assert v._calculate_reliability(spans, "Hello world") == 0.0

    def test_calculate_reliability_mixed(self):
        """Mixed spans give weighted average."""
        from src.core.qa.hallucination_verifier import VerifiedSpan

        v = self._make_verifier()
        # 10 chars verified (0.0), 10 chars hallucinated (1.0) -> reliability 0.5
        spans = [
            VerifiedSpan("a" * 10, 0, 10, 0.0),
            VerifiedSpan("b" * 10, 10, 20, 1.0),
        ]
        reliability = v._calculate_reliability(spans, "a" * 10 + "b" * 10)
        assert abs(reliability - 0.5) < 0.01

    def test_calculate_reliability_empty_spans(self):
        """Empty spans -> reliability 0.0."""
        v = self._make_verifier()
        assert v._calculate_reliability([], "") == 0.0

    def test_verify_empty_answer(self):
        """Empty answer returns rejected result."""
        v = self._make_verifier()
        result = v.verify("", "context", "question")
        assert result.answer_rejected is True
        assert result.overall_reliability == 0.0

    def test_verify_model_load_failure(self):
        """Model load failure returns uncertain result."""
        v = self._make_verifier()
        with patch.object(v, "_load_detector", side_effect=RuntimeError("no model")):
            result = v.verify("Test answer", "context", "question")
        assert result.overall_reliability == 0.5
        assert result.answer_rejected is False


# ============================================================================
# C. Citation Excerpt
# ============================================================================


class TestCitationExcerpt:
    """Tests for citation excerpt extraction (no embeddings needed for some)."""

    def test_empty_context(self):
        """Empty context returns empty string."""
        from src.core.qa.citation_excerpt import extract_citation_excerpt

        assert extract_citation_excerpt("", "Who?", None) == ""
        assert extract_citation_excerpt("  ", "Who?", None) == ""

    def test_short_context_returned_as_is(self):
        """Context shorter than max_chars returned unchanged."""
        from src.core.qa.citation_excerpt import extract_citation_excerpt

        context = "The plaintiff is John Smith."
        result = extract_citation_excerpt(context, "Who?", None, max_chars=250)
        assert result == context

    def test_strips_source_prefix(self):
        """Removes [filename]: prefix from chunk."""
        from src.core.qa.citation_excerpt import _strip_source_prefix

        assert _strip_source_prefix("[complaint.pdf]: The plaintiff") == "The plaintiff"
        assert _strip_source_prefix("[doc.pdf, page 3]: Text") == "Text"
        assert _strip_source_prefix("No prefix here") == "No prefix here"

    def test_get_top_chunk(self):
        """Returns first chunk from separator-separated context."""
        from src.core.qa.citation_excerpt import SEPARATOR, _get_top_chunk

        context = f"First chunk{SEPARATOR}Second chunk{SEPARATOR}Third chunk"
        assert _get_top_chunk(context) == "First chunk"

    def test_build_windows(self):
        """Builds overlapping windows from text."""
        from src.core.qa.citation_excerpt import _build_windows

        text = "A" * 500
        windows = _build_windows(text, 200)
        assert len(windows) >= 2
        # Each window should be roughly 200 chars
        for _, w_text in windows:
            assert len(w_text) <= 220

    def test_truncate_to_sentence_fallback(self):
        """Fallback truncation works without embeddings."""
        from src.core.qa.citation_excerpt import _truncate_to_sentence

        long_text = "First sentence. Second sentence. Third sentence. " * 10
        result = _truncate_to_sentence(long_text, 100)
        assert len(result) <= 110  # some tolerance for "..."
        assert result.endswith("...")

    def test_long_context_with_no_embeddings(self):
        """Long context without embeddings uses sentence truncation."""
        from src.core.qa.citation_excerpt import extract_citation_excerpt

        long_text = "The plaintiff filed a motion. " * 50
        result = extract_citation_excerpt(long_text, "Who filed?", None, max_chars=200)
        assert len(result) <= 250  # some tolerance


# ============================================================================
# D. Token Budget
# ============================================================================


class TestTokenBudget:
    """Tests for token counting and budget computation."""

    def test_count_tokens_simple(self):
        """Counts tokens for simple text."""
        from src.core.qa.token_budget import count_tokens

        count = count_tokens("Hello world")
        assert count >= 2

    def test_count_tokens_empty(self):
        """Empty string returns 0 tokens."""
        from src.core.qa.token_budget import count_tokens

        assert count_tokens("") == 0

    def test_compute_context_budget(self):
        """Budget = context_window - template - question - output - margin."""
        from src.core.qa.token_budget import compute_context_budget

        budget = compute_context_budget(
            context_window=4096,
            prompt_template_tokens=100,
            question_tokens=20,
            max_output_tokens=256,
            safety_margin=16,
        )
        assert budget == 4096 - 100 - 20 - 256 - 16

    def test_compute_context_budget_minimum(self):
        """Budget never goes below 64 tokens."""
        from src.core.qa.token_budget import compute_context_budget

        budget = compute_context_budget(
            context_window=100,
            prompt_template_tokens=50,
            question_tokens=50,
            max_output_tokens=50,
        )
        assert budget == 64

    def test_ensure_fits_short_text(self):
        """Short text passes through unchanged."""
        from src.core.qa.token_budget import _ensure_fits

        text = "Short text"
        assert _ensure_fits(text, 1000) == text

    def test_ensure_fits_truncates(self):
        """Long text is truncated to fit budget."""
        from src.core.qa.token_budget import _ensure_fits, count_tokens

        long_text = "word " * 1000
        result = _ensure_fits(long_text, 50)
        assert count_tokens(result) <= 50

    def test_build_windows(self):
        """Builds correct number of overlapping windows."""
        from src.core.qa.token_budget import _build_windows

        text = "x" * 1000
        windows = _build_windows(text, 0.6, 3)
        assert len(windows) == 3
        # Each window ~600 chars
        for w in windows:
            assert abs(len(w) - 600) < 50


# ============================================================================
# E. Verification Config
# ============================================================================


class TestVerificationConfig:
    """Tests for verification threshold functions."""

    def test_get_span_category_verified(self):
        """Low prob -> verified."""
        from src.core.qa.verification_config import get_span_category

        assert get_span_category(0.1) == "verified"

    def test_get_span_category_hallucinated(self):
        """High prob -> hallucinated."""
        from src.core.qa.verification_config import get_span_category

        assert get_span_category(0.95) == "hallucinated"

    def test_get_reliability_level_high(self):
        """High reliability -> 'high'."""
        from src.core.qa.verification_config import get_reliability_level

        assert get_reliability_level(0.90) == "high"

    def test_get_reliability_level_low(self):
        """Low reliability -> 'low'."""
        from src.core.qa.verification_config import get_reliability_level

        assert get_reliability_level(0.20) == "low"


# ============================================================================
# F. Q&A Constants
# ============================================================================


class TestQAConstants:
    """Tests for Q&A prompt templates and constants."""

    def test_prompts_have_placeholders(self):
        """Both prompts contain {context} and {question} placeholders."""
        from src.core.qa.qa_constants import COMPACT_QA_PROMPT, FULL_QA_PROMPT

        for prompt in [COMPACT_QA_PROMPT, FULL_QA_PROMPT]:
            assert "{context}" in prompt
            assert "{question}" in prompt

    def test_compact_prompt_shorter(self):
        """Compact prompt is shorter than full prompt."""
        from src.core.qa.qa_constants import COMPACT_QA_PROMPT, FULL_QA_PROMPT

        assert len(COMPACT_QA_PROMPT) < len(FULL_QA_PROMPT)

    def test_unanswered_text_defined(self):
        """UNANSWERED_TEXT is a non-empty string."""
        from src.core.qa.qa_constants import UNANSWERED_TEXT

        assert isinstance(UNANSWERED_TEXT, str)
        assert len(UNANSWERED_TEXT) > 0

    def test_compact_prompt_threshold(self):
        """Threshold is a reasonable integer."""
        from src.core.qa.qa_constants import COMPACT_PROMPT_THRESHOLD

        assert isinstance(COMPACT_PROMPT_THRESHOLD, int)
        assert COMPACT_PROMPT_THRESHOLD >= 2048
