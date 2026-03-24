"""
Tests for semantic search core modules.

Covers:
- Citation excerpt extraction (citation_excerpt.py) — window building, truncation
- Token budget utilities (token_budget.py) — counting, budget math
- Semantic constants (semantic_constants.py) — prompt templates, thresholds
"""


# ============================================================================
# A. Citation Excerpt
# ============================================================================


class TestCitationExcerpt:
    """Tests for citation excerpt extraction (no embeddings needed for some)."""

    def test_empty_context(self):
        """Empty context returns empty string."""
        from src.core.semantic.citation_excerpt import extract_citation_excerpt

        assert extract_citation_excerpt("", "Who?", None) == ""
        assert extract_citation_excerpt("  ", "Who?", None) == ""

    def test_short_context_returned_as_is(self):
        """Context shorter than max_chars returned unchanged."""
        from src.core.semantic.citation_excerpt import extract_citation_excerpt

        context = "The plaintiff is John Smith."
        result = extract_citation_excerpt(context, "Who?", None, max_chars=250)
        assert result == context

    def test_strips_source_prefix(self):
        """Removes [filename]: prefix from chunk."""
        from src.core.semantic.citation_excerpt import _strip_source_prefix

        assert _strip_source_prefix("[complaint.pdf]: The plaintiff") == "The plaintiff"
        assert _strip_source_prefix("[doc.pdf, page 3]: Text") == "Text"
        assert _strip_source_prefix("No prefix here") == "No prefix here"

    def test_get_top_chunk(self):
        """Returns first chunk from separator-separated context."""
        from src.core.semantic.citation_excerpt import SEPARATOR, _get_top_chunk

        context = f"First chunk{SEPARATOR}Second chunk{SEPARATOR}Third chunk"
        assert _get_top_chunk(context) == "First chunk"

    def test_build_windows(self):
        """Builds overlapping windows from text."""
        from src.core.semantic.citation_excerpt import _build_windows

        text = "A" * 500
        windows = _build_windows(text, 200)
        assert len(windows) >= 2
        # Each window should be roughly 200 chars
        for _, _, w_text in windows:
            assert len(w_text) <= 220

    def test_truncate_to_sentence_fallback(self):
        """Fallback truncation works without embeddings."""
        from src.core.semantic.citation_excerpt import _truncate_to_sentence

        long_text = "First sentence. Second sentence. Third sentence. " * 10
        result = _truncate_to_sentence(long_text, 100)
        assert len(result) <= 110  # some tolerance for "..."
        assert result.endswith("...")

    def test_long_context_with_no_embeddings(self):
        """Long context without embeddings uses sentence truncation."""
        from src.core.semantic.citation_excerpt import extract_citation_excerpt

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
        from src.core.semantic.token_budget import count_tokens

        count = count_tokens("Hello world")
        assert count >= 2

    def test_count_tokens_empty(self):
        """Empty string returns 0 tokens."""
        from src.core.semantic.token_budget import count_tokens

        assert count_tokens("") == 0

    def test_compute_context_budget(self):
        """Budget = context_window - template - question - output - margin."""
        from src.core.semantic.token_budget import compute_context_budget

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
        from src.core.semantic.token_budget import compute_context_budget

        budget = compute_context_budget(
            context_window=100,
            prompt_template_tokens=50,
            question_tokens=50,
            max_output_tokens=50,
        )
        assert budget == 64

    def test_ensure_fits_short_text(self):
        """Short text passes through unchanged."""
        from src.core.semantic.token_budget import _ensure_fits

        text = "Short text"
        assert _ensure_fits(text, 1000) == text

    def test_ensure_fits_truncates(self):
        """Long text is truncated to fit budget."""
        from src.core.semantic.token_budget import _ensure_fits, count_tokens

        long_text = "word " * 1000
        result = _ensure_fits(long_text, 50)
        assert count_tokens(result) <= 50

    def test_build_windows(self):
        """Builds correct number of overlapping windows."""
        from src.core.semantic.token_budget import _build_windows

        text = "x" * 1000
        windows = _build_windows(text, 0.6, 3)
        assert len(windows) == 3
        # Each window ~600 chars
        for w in windows:
            assert abs(len(w) - 600) < 50


# ============================================================================
# C. Semantic Constants
# ============================================================================


class TestQAConstants:
    """Tests for Q&A prompt templates and constants."""

    def test_prompts_have_placeholders(self):
        """Both prompts contain {context} and {question} placeholders."""
        from src.core.semantic.semantic_constants import (
            COMPACT_SEMANTIC_PROMPT,
            FULL_SEMANTIC_PROMPT,
        )

        for prompt in [COMPACT_SEMANTIC_PROMPT, FULL_SEMANTIC_PROMPT]:
            assert "{context}" in prompt
            assert "{question}" in prompt

    def test_compact_prompt_shorter(self):
        """Compact prompt is shorter than full prompt."""
        from src.core.semantic.semantic_constants import (
            COMPACT_SEMANTIC_PROMPT,
            FULL_SEMANTIC_PROMPT,
        )

        assert len(COMPACT_SEMANTIC_PROMPT) < len(FULL_SEMANTIC_PROMPT)

    def test_unanswered_text_defined(self):
        """UNANSWERED_TEXT is a non-empty string."""
        from src.core.semantic.semantic_constants import UNANSWERED_TEXT

        assert isinstance(UNANSWERED_TEXT, str)
        assert len(UNANSWERED_TEXT) > 0

    def test_compact_prompt_threshold(self):
        """Threshold is a reasonable integer."""
        from src.core.semantic.semantic_constants import COMPACT_PROMPT_THRESHOLD

        assert isinstance(COMPACT_PROMPT_THRESHOLD, int)
        assert COMPACT_PROMPT_THRESHOLD >= 2048
