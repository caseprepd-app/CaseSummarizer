"""
Tests for semantic search core modules.

Covers:
- Citation excerpt extraction (citation_excerpt.py) — window building, truncation
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
# C. Semantic Constants (LLM prompt constants removed Mar 2026)
# ============================================================================
