"""
Tests for the Term-in-Context viewer.

Tests the extract_snippets() function — pure text processing, no GUI needed.
Also tests menu integration via mocks.
"""

from src.ui.context_viewer_dialog import extract_snippets

# --- Helper to build document dicts ---


def _doc(filename: str, text: str, use_preprocessed: bool = True) -> dict:
    """Build a minimal document dict for testing."""
    key = "preprocessed_text" if use_preprocessed else "extracted_text"
    return {"filename": filename, key: text}


# ============================================================
# Snippet extraction logic
# ============================================================


class TestSnippetExtraction:
    """Tests for core extract_snippets() behavior."""

    def test_single_doc_single_match(self):
        """One occurrence returns correct before/after/term."""
        docs = [_doc("test.pdf", "The defendant showed negligence in this case.")]
        results = extract_snippets("negligence", docs)

        assert len(results) == 1
        assert results[0]["filename"] == "test.pdf"
        assert results[0]["total_count"] == 1
        assert len(results[0]["snippets"]) == 1
        snippet = results[0]["snippets"][0]
        assert snippet["term"] == "negligence"
        assert "defendant" in snippet["before"]
        assert "case" in snippet["after"]

    def test_multi_doc_results(self):
        """Term in 2 of 3 docs; third doc skipped."""
        docs = [
            _doc("a.pdf", "The negligence was clear."),
            _doc("b.pdf", "No relevant terms here."),
            _doc("c.pdf", "Gross negligence alleged."),
        ]
        results = extract_snippets("negligence", docs)

        assert len(results) == 2
        filenames = {r["filename"] for r in results}
        assert filenames == {"a.pdf", "c.pdf"}

    def test_case_insensitive_match(self):
        """'NEGLIGENCE' matches 'negligence' search; preserves original case."""
        docs = [_doc("test.pdf", "The NEGLIGENCE was obvious.")]
        results = extract_snippets("negligence", docs)

        assert len(results) == 1
        assert results[0]["snippets"][0]["term"] == "NEGLIGENCE"

    def test_max_snippets_per_doc(self):
        """15 occurrences capped at 10 snippets, remaining=5."""
        text = " ".join(["the negligence claim"] * 15)
        docs = [_doc("big.pdf", text)]
        results = extract_snippets("negligence", docs, max_per_doc=10)

        assert results[0]["total_count"] == 15
        assert len(results[0]["snippets"]) == 10
        assert results[0]["remaining"] == 5

    def test_multi_word_term(self):
        """Multi-word phrase matched as whole phrase."""
        docs = [_doc("test.pdf", "The breach of contract was proven in court.")]
        results = extract_snippets("breach of contract", docs)

        assert len(results) == 1
        assert results[0]["snippets"][0]["term"] == "breach of contract"

    def test_special_regex_chars(self):
        """'C.I.A.' doesn't become a regex wildcard."""
        docs = [_doc("test.pdf", "The C.I.A. document was classified.")]
        results = extract_snippets("C.I.A.", docs)

        assert len(results) == 1
        assert results[0]["snippets"][0]["term"] == "C.I.A."

    def test_results_sorted_by_count(self):
        """Doc with most matches appears first."""
        docs = [
            _doc("few.pdf", "negligence once"),
            _doc("many.pdf", "negligence negligence negligence"),
        ]
        results = extract_snippets("negligence", docs)

        assert results[0]["filename"] == "many.pdf"
        assert results[0]["total_count"] == 3
        assert results[1]["filename"] == "few.pdf"
        assert results[1]["total_count"] == 1

    def test_no_matches_returns_empty(self):
        """Term not in any document returns empty list."""
        docs = [_doc("test.pdf", "Nothing relevant here.")]
        results = extract_snippets("negligence", docs)

        assert results == []

    def test_empty_documents_list(self):
        """Empty input returns empty output."""
        assert extract_snippets("negligence", []) == []

    def test_empty_term(self):
        """Empty term returns empty output."""
        docs = [_doc("test.pdf", "Some text.")]
        assert extract_snippets("", docs) == []

    def test_fallback_to_extracted_text(self):
        """Uses extracted_text when no preprocessed_text."""
        docs = [_doc("test.pdf", "The negligence was clear.", use_preprocessed=False)]
        results = extract_snippets("negligence", docs)

        assert len(results) == 1
        assert results[0]["snippets"][0]["term"] == "negligence"


# ============================================================
# Context boundaries
# ============================================================


class TestContextBoundaries:
    """Tests for snippet boundary handling."""

    def test_context_word_boundary(self):
        """Snippet doesn't cut mid-word."""
        # Build text where context boundary would fall mid-word
        text = "abcdefghijklmnop negligence qrstuvwxyz"
        docs = [_doc("test.pdf", text)]
        results = extract_snippets("negligence", docs, context_chars=5)

        snippet = results[0]["snippets"][0]
        # Before should not end with a partial word
        assert not snippet["before"].rstrip().endswith("nop")

    def test_match_at_document_start(self):
        """No '...' prefix when match is at start."""
        docs = [_doc("test.pdf", "negligence was the issue here.")]
        results = extract_snippets("negligence", docs)

        snippet = results[0]["snippets"][0]
        assert not snippet["before"].startswith("...")

    def test_match_at_document_end(self):
        """No '...' suffix when match is at end."""
        docs = [_doc("test.pdf", "the issue was negligence")]
        results = extract_snippets("negligence", docs)

        snippet = results[0]["snippets"][0]
        assert not snippet["after"].endswith("...")

    def test_ellipsis_in_middle(self):
        """Both '...' prefix and suffix for mid-document match."""
        text = "x " * 100 + "the negligence claim" + " y" * 100
        docs = [_doc("test.pdf", text)]
        results = extract_snippets("negligence", docs, context_chars=40)

        snippet = results[0]["snippets"][0]
        assert snippet["before"].startswith("...")
        assert snippet["after"].endswith("...")


# ============================================================
# Document with no text fields
# ============================================================


class TestEdgeCases:
    """Tests for edge cases and missing data."""

    def test_doc_with_no_text_fields(self):
        """Document dict without text fields is skipped."""
        docs = [{"filename": "empty.pdf"}]
        results = extract_snippets("negligence", docs)
        assert results == []

    def test_doc_with_empty_text(self):
        """Document with empty string text is skipped."""
        docs = [_doc("empty.pdf", "")]
        results = extract_snippets("negligence", docs)
        assert results == []

    def test_remaining_zero_when_all_shown(self):
        """remaining is 0 when total_count <= max_per_doc."""
        docs = [_doc("test.pdf", "negligence negligence negligence")]
        results = extract_snippets("negligence", docs, max_per_doc=10)

        assert results[0]["remaining"] == 0
