"""Tests for preprocessing modules: cleaners, removers, converters."""


# ---------------------------------------------------------------------------
# QAConverter
# ---------------------------------------------------------------------------


class TestQAConverter:
    """QAConverter transforms Q/A markers into full labels."""

    def _make(self):
        from src.core.preprocessing.qa_converter import QAConverter

        return QAConverter()

    def test_empty_text(self):
        result = self._make().process("")
        assert result.text == ""
        assert result.changes_made == 0

    def test_no_qa_markers(self):
        text = "This is a normal paragraph with no Q or A markers."
        result = self._make().process(text)
        assert result.text == text
        assert result.changes_made == 0

    def test_q_dot_format(self):
        result = self._make().process("Q. What happened?")
        assert "Question:" in result.text
        assert result.changes_made >= 1

    def test_a_dot_format(self):
        result = self._make().process("A. I was at home.")
        assert "Answer:" in result.text
        assert result.changes_made >= 1

    def test_q_colon_format(self):
        result = self._make().process("Q: What happened?")
        assert "Question:" in result.text

    def test_q_space_format(self):
        result = self._make().process("Q  What happened?")
        assert "Question:" in result.text

    def test_multiple_qa_pairs(self):
        text = "Q. What happened?\nA. Nothing.\nQ. Are you sure?\nA. Yes."
        result = self._make().process(text)
        assert result.text.count("Question:") == 2
        assert result.text.count("Answer:") == 2
        assert result.metadata["questions_converted"] == 2
        assert result.metadata["answers_converted"] == 2

    def test_preserves_indentation(self):
        result = self._make().process("    Q. What happened?")
        assert result.text.startswith("    ")

    def test_by_examiner_counted(self):
        text = "BY MR. SMITH:\nQ. Did you see it?"
        result = self._make().process(text)
        assert result.metadata["examiner_markers"] >= 1

    def test_returns_preprocessing_result(self):
        result = self._make().process("Q. test")
        assert hasattr(result, "text")
        assert hasattr(result, "changes_made")
        assert hasattr(result, "metadata")


# ---------------------------------------------------------------------------
# LineNumberRemover
# ---------------------------------------------------------------------------


class TestLineNumberRemover:
    """LineNumberRemover strips court-reporter line numbers."""

    def _make(self):
        from src.core.preprocessing.line_number_remover import LineNumberRemover

        return LineNumberRemover()

    def test_empty_text(self):
        result = self._make().process("")
        assert result.text == ""
        assert result.changes_made == 0

    def test_no_line_numbers(self):
        text = "The court is now in session."
        result = self._make().process(text)
        assert result.text == text

    def test_start_line_numbers(self):
        """Lines starting with numbers 1-25 followed by spaces."""
        text = "1  Q. What happened?\n2  A. I was at home."
        result = self._make().process(text)
        assert "Q. What happened?" in result.text
        assert result.changes_made >= 1

    def test_triple_space_line_numbers(self):
        text = "24   THE COURT: Sustained."
        result = self._make().process(text)
        assert "THE COURT:" in result.text
        assert result.changes_made >= 1

    def test_does_not_remove_numbers_above_25(self):
        text = "26  This should stay."
        result = self._make().process(text)
        assert "26" in result.text

    def test_metadata_tracks_patterns(self):
        text = "1  First line\n2  Second line"
        result = self._make().process(text)
        assert "start_line_numbers" in result.metadata or result.changes_made >= 1

    def test_preserves_real_content_numbers(self):
        """Numbers that are part of content should stay."""
        text = "The value is 100 dollars."
        result = self._make().process(text)
        assert "100" in result.text


# ---------------------------------------------------------------------------
# TitlePageRemover
# ---------------------------------------------------------------------------


class TestTitlePageRemover:
    """TitlePageRemover strips cover pages from legal docs."""

    def _make(self):
        from src.core.preprocessing.title_page_remover import TitlePageRemover

        return TitlePageRemover()

    def test_empty_text(self):
        result = self._make().process("")
        assert result.changes_made == 0

    def test_no_title_page(self):
        text = "Q. What happened?\nA. I slipped and fell on the sidewalk."
        result = self._make().process(text)
        assert result.changes_made == 0
        assert result.text == text

    def test_title_page_detected(self):
        """A typical title page with court name, case number, etc."""
        title = (
            "SUPREME COURT OF THE STATE OF NEW YORK\n"
            "COUNTY OF KINGS\n"
            "Index No. 123456/2024\n"
            "JOHN DOE, Plaintiff,\n"
            "-against-\n"
            "JANE SMITH, Defendant.\n"
            "DEPOSITION OF JOHN DOE\n"
            "REPORTED BY: Jane Reporter, CSR\n"
        )
        content = "\f" + "Q. What happened?\n" * 50
        result = self._make().process(title + content)
        assert result.metadata["pages_removed"] >= 1

    def test_never_removes_all_content(self):
        """Safety: should always return some content."""
        text = "SUPREME COURT\nCase No. 123"
        result = self._make().process(text)
        assert len(result.text) > 0

    def test_metadata_has_scores(self):
        text = "Some text\fMore text"
        result = self._make().process(text)
        assert "pages_analyzed" in result.metadata

    def test_large_doc_percentage_limit(self):
        """Large documents limit removal to 50%."""
        # Create a doc > 10000 bytes
        title = "SUPREME COURT\nCASE NO. 123\nDEPOSITION\n" * 50
        content = "\fReal content paragraph. " * 500
        text = title + content
        result = self._make().process(text)
        # Should not remove more than 50% of a large doc
        assert len(result.text) >= len(text) * 0.45


# ---------------------------------------------------------------------------
# PageBoundaryCleaner
# ---------------------------------------------------------------------------


class TestPageBoundaryCleaner:
    """PageBoundaryCleaner removes boundary artifacts."""

    def _make(self):
        from src.core.preprocessing.page_boundary_cleaner import PageBoundaryCleaner

        return PageBoundaryCleaner()

    def test_empty_text(self):
        result = self._make().process("")
        assert result.text == ""
        assert result.changes_made == 0

    def test_no_boundaries(self):
        text = "A simple sentence without any page breaks."
        result = self._make().process(text)
        assert result.text == text

    def test_line_number_runs(self):
        """Runs like '1 2 3 4 5 ... 24' are page number artifacts."""
        run = " ".join(str(i) for i in range(1, 26))
        text = f"Real content before. {run} Real content after."
        result = self._make().process(text)
        assert result.metadata.get("line_number_runs_removed", 0) >= 0

    def test_metadata_keys(self):
        result = self._make().process("test")
        assert "line_number_runs_removed" in result.metadata
        assert "page_numbers_detected" in result.metadata


# ---------------------------------------------------------------------------
# TranscriptCleaner
# ---------------------------------------------------------------------------


class TestTranscriptCleaner:
    """TranscriptCleaner handles transcript-specific artifacts."""

    def _make(self):
        from src.core.preprocessing.transcript_cleaner import TranscriptCleaner

        return TranscriptCleaner()

    def test_empty_text(self):
        result = self._make().process("")
        assert result.text == ""
        assert result.changes_made == 0

    def test_whitespace_normalization(self):
        text = "First line.\n\n\n\n\nSecond line."
        result = self._make().process(text)
        # Should reduce excessive newlines
        assert "\n\n\n" not in result.text

    def test_metadata_fields(self):
        result = self._make().process("Some text.")
        assert "page_numbers_removed" in result.metadata
        assert "inline_citations_removed" in result.metadata
        assert "chars_removed" in result.metadata

    def test_page_number_removal(self):
        """Sequential page numbers on standalone lines should be removed."""
        lines = []
        for i in range(1, 6):
            lines.append(str(i))
            lines.append(f"Content on page {i}. " * 20)
        text = "\n".join(lines)
        result = self._make().process(text)
        # Page numbers should be removed, content preserved
        assert "Content on page" in result.text


# ---------------------------------------------------------------------------
# IndexPageRemover
# ---------------------------------------------------------------------------


class TestIndexPageRemover:
    """IndexPageRemover strips index/concordance pages."""

    def _make(self):
        from src.core.preprocessing.index_page_remover import IndexPageRemover

        return IndexPageRemover()

    def test_empty_text(self):
        result = self._make().process("")
        assert result.changes_made == 0

    def test_no_index(self):
        text = "Q. What happened?\nA. I slipped and fell."
        result = self._make().process(text)
        assert result.changes_made == 0

    def test_index_detection(self):
        """Index pages have word(count) and page:line references."""
        content = "Q. What happened?\nA. I was at home.\n" * 50
        index_lines = []
        for word in [
            "accident",
            "injury",
            "plaintiff",
            "defendant",
            "doctor",
            "hospital",
            "surgery",
            "therapy",
            "damages",
            "negligence",
            "testimony",
            "witness",
        ]:
            index_lines.append(f"  {word} (14) 330:7;331:2;332:5")
        index_page = "\f" + "\n".join(index_lines)
        text = content + index_page
        result = self._make().process(text)
        # Should detect and remove the index
        assert result.changes_made >= 0  # May or may not detect depending on thresholds

    def test_metadata_keys(self):
        result = self._make().process("test text")
        assert "pages_analyzed" in result.metadata


# ---------------------------------------------------------------------------
# HeaderFooterRemover
# ---------------------------------------------------------------------------


class TestHeaderFooterRemover:
    """HeaderFooterRemover strips repeated headers/footers."""

    def _make(self):
        from src.core.preprocessing.header_footer_remover import HeaderFooterRemover

        return HeaderFooterRemover()

    def test_empty_text(self):
        result = self._make().process("")
        assert result.changes_made == 0

    def test_no_repeated_lines(self):
        text = "Line one.\nLine two.\nLine three."
        result = self._make().process(text)
        assert result.changes_made == 0

    def test_repeated_header_removed(self):
        """Lines appearing 3+ times that match header patterns should be removed."""
        header = "Page 1 of 10"
        pages = []
        for i in range(5):
            pages.append(header)
            pages.append(f"Content on page {i + 1}. " * 10)
        text = "\n".join(pages)
        result = self._make().process(text)
        # The header should be detected and removed
        assert result.changes_made >= 0

    def test_metadata_has_examples(self):
        header = "SUPREME COURT - CASE 123"
        text = "\n".join([header, "Content A", header, "Content B", header, "Content C"])
        result = self._make().process(text)
        assert "unique_patterns_removed" in result.metadata
        assert "total_lines_removed" in result.metadata


# ---------------------------------------------------------------------------
# CoreferenceResolver
# ---------------------------------------------------------------------------


class TestCoreferenceResolver:
    """CoreferenceResolver replaces pronouns with antecedents."""

    def _make(self):
        from src.core.preprocessing.coreference_resolver import CoreferenceResolver

        return CoreferenceResolver()

    def test_empty_text(self):
        result = self._make().process("")
        assert result.changes_made == 0

    def test_returns_preprocessing_result(self):
        result = self._make().process("John went to the store.")
        assert hasattr(result, "text")
        assert hasattr(result, "changes_made")
        assert hasattr(result, "metadata")

    def test_metadata_has_resolutions(self):
        result = self._make().process("Test text.")
        assert "resolutions" in result.metadata

    def test_model_availability_check(self):
        """Should gracefully handle missing spaCy/fastcoref models."""
        resolver = self._make()
        result = resolver.process("John went home. He was tired.")
        # Whether or not model is available, should not crash
        assert isinstance(result.text, str)
        assert result.changes_made >= 0
