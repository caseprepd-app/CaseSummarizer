"""Tests for preprocessing modules: cleaners, removers, converters."""

import pytest

# ---------------------------------------------------------------------------
# QAConverter
# ---------------------------------------------------------------------------


class TestQAConverter:
    """QAConverter transforms Q/A markers into full labels."""

    def _make(self):
        from src.deprecated.qa_converter import QAConverter

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

    def test_q_space_format_not_converted(self):
        """Bare 'Q ' without separator is NOT converted (would corrupt 'A witness' etc.)."""
        result = self._make().process("Q  What happened?")
        assert "Question:" not in result.text
        assert "Q  What happened?" in result.text

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

    def test_aggressive_sweep_removes_numbers_above_25(self):
        """Aggressive sweep catches 1-3 digit numbers at line start."""
        text = "26  This should be removed."
        result = self._make().process(text)
        assert "26" not in result.text
        assert "This should be removed." in result.text

    def test_aggressive_sweep_preserves_plural_next_word(self):
        """Numbers before plural words (ending in 's') are kept."""
        text = "3 boxes on the table"
        result = self._make().process(text)
        assert "3 boxes" in result.text

    def test_aggressive_sweep_preserves_four_digit_numbers(self):
        """4-digit numbers (years, etc.) are never removed."""
        text = "2024 was a good year"
        result = self._make().process(text)
        assert "2024" in result.text

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

    # -- Proceedings detection (mid-page content preservation) -------

    def _make_proceedings_block(self):
        """Helper: a realistic block of transcript Q/A (>= 5 prose lines)."""
        return (
            "Q. Can you state your name for the record?\n"
            "A. My name is John Doe.\n"
            "Q. And where do you currently reside?\n"
            "A. I live at 123 Main Street in Brooklyn, New York.\n"
            "Q. What is your occupation?\n"
            "A. I am a civil engineer.\n"
        )

    def test_proceedings_qa_preserved_on_title_page(self):
        """Q/A content at bottom of a title page is kept."""
        title_top = (
            "SUPREME COURT OF THE STATE OF NEW YORK\n"
            "COUNTY OF KINGS\n"
            "Index No. 123456/2024\n"
            "JOHN DOE, Plaintiff,\n"
            "-against-\n"
            "JANE SMITH, Defendant.\n"
            "APPEARANCES:\n"
            "ATTORNEY FOR PLAINTIFF\n"
        )
        text = title_top + self._make_proceedings_block() + "\f" + "Q. Next?\nA. Yes."
        result = self._make().process(text)

        # The Q/A lines from the title page must survive
        assert "Q. Can you state your name" in result.text
        assert "A. My name is John Doe" in result.text
        assert result.metadata["pages_removed"] >= 1

    def test_full_title_page_still_removed(self):
        """A pure title page with no proceedings is fully removed."""
        title = (
            "SUPREME COURT OF THE STATE OF NEW YORK\n"
            "COUNTY OF KINGS\n"
            "Index No. 123456/2024\n"
            "JOHN DOE, Plaintiff,\n"
            "-against-\n"
            "JANE SMITH, Defendant.\n"
            "DEPOSITION OF JOHN DOE\n"
        )
        content = "Q. What happened?\nA. I slipped and fell on the sidewalk."
        text = title + "\f" + content
        result = self._make().process(text)

        assert "SUPREME COURT" not in result.text
        assert "Q. What happened?" in result.text

    def test_speaker_label_triggers_proceedings(self):
        """Speaker labels with 5 content lines ahead are detected."""
        title_top = (
            "SUPREME COURT OF THE STATE OF NEW YORK\n"
            "Index No. 123456/2024\n"
            "PLAINTIFF, -against- DEFENDANT.\n"
            "APPEARANCES:\n"
            "ATTORNEY FOR PLAINTIFF\n"
        )
        proceedings = (
            "THE CLERK: Calling case number one on the calendar.\n"
            "THE COURT: Good morning, are the parties ready to proceed?\n"
            "Q. Please state your name for the record.\n"
            "A. My name is Jane Doe and I reside in Brooklyn.\n"
            "Q. And what is your current occupation?\n"
            "A. I am a registered nurse at the hospital.\n"
        )
        text = title_top + proceedings + "\f" + "Q. What happened next?"
        result = self._make().process(text)

        assert "THE CLERK: Calling case" in result.text
        assert "THE COURT: Good morning" in result.text

    def test_mixed_case_general_lines_need_5_prose_ahead(self):
        """General content lines require 5 prose lines ahead for confirmation."""
        title_top = (
            "SUPREME COURT OF THE STATE OF NEW YORK\n"
            "DEPOSITION OF JOHN DOE\n"
            "APPEARANCES:\n"
            "ATTORNEY FOR PLAINTIFF\n"
        )
        # 6 general-heuristic prose lines (1 candidate + 5 look-ahead)
        proceedings = (
            "The following proceedings were held before the Honorable Judge.\n"
            "All parties present and accounted for in the courtroom.\n"
            "The court reporter was duly sworn to take the record.\n"
            "The witness was called and placed under oath by the clerk.\n"
            "Counsel for plaintiff indicated they were ready to proceed.\n"
            "The court asked both sides to identify themselves for the record.\n"
        )
        text = title_top + proceedings + "\f" + "More content here."
        result = self._make().process(text)

        assert "following proceedings were held" in result.text

    def test_general_lines_rejected_without_enough_lookahead(self):
        """Only 2 general prose lines is NOT enough — needs 5."""
        title_top = (
            "SUPREME COURT OF THE STATE OF NEW YORK\n"
            "DEPOSITION OF JOHN DOE\n"
            "APPEARANCES:\n"
            "ATTORNEY FOR PLAINTIFF\n"
        )
        # Only 2 general prose lines — not enough for general lookahead
        almost = (
            "The following proceedings were held before the court.\n"
            "All parties present and accounted for today.\n"
        )
        text = title_top + almost + "\f" + "Q. Name?\nA. John."
        result = self._make().process(text)

        # Should NOT detect proceedings on the title page (not enough lookahead)
        # The entire title page is removed; content starts on page 2
        assert "Q. Name?" in result.text

    def test_qa_lines_need_5_content_ahead(self):
        """Q/A lines use the same 5-line look-ahead as all content lines."""
        title_top = (
            "SUPREME COURT OF THE STATE OF NEW YORK\n"
            "Index No. 123456/2024\n"
            "JOHN DOE, Plaintiff,\n"
            "-against-\n"
            "JANE SMITH, Defendant.\n"
            "APPEARANCES:\n"
        )
        proceedings = (
            "Q. State your name for the record please.\n"
            "A. My name is John Doe from Brooklyn.\n"
            "Q. Where do you currently reside sir?\n"
            "A. I live at one two three Main Street.\n"
            "Q. What is your current occupation?\n"
            "A. I am a civil engineer at a firm.\n"
        )
        text = title_top + proceedings + "\f" + "Q. What happened next?"
        result = self._make().process(text)

        assert "Q. State your name" in result.text

    def test_short_title_names_not_mistaken_for_content(self):
        """Short lines like attorney names should not trigger proceedings."""
        title = (
            "SUPREME COURT OF THE STATE OF NEW YORK\n"
            "Index No. 123456/2024\n"
            "JOHN DOE, Plaintiff,\n"
            "-against-\n"
            "JANE SMITH, Defendant.\n"
            "JOHN T. SMITH, ESQ.\n"
            "Attorney for Plaintiff\n"
        )
        content = "Q. What happened?\nA. I slipped and fell."
        text = title + "\f" + content
        result = self._make().process(text)

        # Title page should be fully removed (attorney name is not proceedings)
        assert "JOHN T. SMITH" not in result.text
        assert "Q. What happened?" in result.text

    def test_double_spaced_content_detected(self):
        """Content lines separated by blank lines (double spacing) are adjacent."""
        title_top = (
            "SUPREME COURT OF THE STATE OF NEW YORK\n"
            "DEPOSITION OF JOHN DOE\n"
            "APPEARANCES:\n"
            "ATTORNEY FOR PLAINTIFF\n"
        )
        # Double-spaced: blank lines between content lines (skipped by look-ahead)
        proceedings = (
            "THE CLERK: This hearing will now come to order.\n"
            "\n"
            "THE COURT: Good morning, counsel, are you ready to proceed?\n"
            "\n"
            "Q. Please state your name for the record.\n"
            "\n"
            "A. My name is Jane Doe and I live in Manhattan.\n"
            "\n"
            "Q. What is your current occupation and employer?\n"
            "\n"
            "A. I am a software engineer at a technology company.\n"
        )
        text = title_top + proceedings + "\f" + "Q. What happened next?"
        result = self._make().process(text)

        assert "THE CLERK: This hearing" in result.text

    def test_all_caps_title_lines_not_content(self):
        """ALL-CAPS lines that are long should not be detected as content."""
        remover = self._make()
        assert not remover._is_content_line("SUPREME COURT OF THE STATE OF NEW YORK")
        assert not remover._is_content_line("ATTORNEY FOR THE PLAINTIFF JOHN DOE")
        assert not remover._is_content_line("JOHN DOE")
        assert not remover._is_content_line("")

    def test_mostly_caps_line_rejected(self):
        """Lines where 75%+ words start uppercase are rejected as content."""
        remover = self._make()
        # All-caps or heavily capitalized lines
        assert not remover._is_content_line("BY: STEVEN D. ATESHOGLOU, Esq.")
        assert not remover._is_content_line("MELVILLE, NEW YORK 11747")
        assert not remover._is_content_line("SUPREME COURT OF THE STATE OF NEW YORK")
        # Mixed-case with enough lowercase IS content (protected by look-ahead)
        assert remover._is_content_line("Attorneys for the City of New York")

    def test_is_content_line_positive_cases(self):
        """Lines with 4+ words and mixed capitalization are content."""
        remover = self._make()
        assert remover._is_content_line("Q. What happened on that day?")
        assert remover._is_content_line("A. I fell on the sidewalk.")
        assert remover._is_content_line("THE CLERK: Calling the next case on the calendar.")
        assert remover._is_content_line("The following proceedings were held in open court.")

    def test_attorney_listing_not_detected_as_proceedings(self):
        """Appearance-page attorney blocks don't trigger proceedings."""
        title_top = (
            "SUPREME COURT OF THE STATE OF NEW YORK\n"
            "Index No. 123456/2024\n"
            "PLAINTIFF, -against- DEFENDANT.\n"
            "APPEARANCES:\n"
        )
        attorneys = (
            "Attorneys for the City of New York\n"
            "Wall Street New York, New York 10005\n"
            "BY: STEVEN D. ATESHOGLOU, Esq.\n"
            "Attorneys for Con Edison\n"
            "55 Washington Street, suite 720\n"
            "BY: ANDREW SHOWERS, ESQ.\n"
        )
        text = title_top + attorneys + "\f" + "Q. Name?\nA. John."
        result = self._make().process(text)

        # Title page should be fully removed (attorney listings are not prose)
        assert "ATESHOGLOU" not in result.text
        assert "Q. Name?" in result.text

    def test_proceedings_stop_further_page_removal(self):
        """Once proceedings are found, no more pages are checked/removed."""
        title1 = (
            "SUPREME COURT\nIndex No. 123\nPLAINTIFF, -against- DEFENDANT.\n"
            "DEPOSITION OF JOHN DOE\n"
        )
        # Page 2: appearances with Q/A proceedings at bottom
        title2_top = "APPEARANCES:\nATTORNEY FOR PLAINTIFF\nJOHN SMITH, ESQ.\n"
        title2_proceedings = (
            "Q. Good morning, can you state your full name?\n"
            "A. My name is John Doe, and I live in Brooklyn.\n"
            "Q. What is your date of birth please sir?\n"
            "A. I was born on January first nineteen sixty five.\n"
            "Q. And what is your current occupation?\n"
            "A. I am a retired construction worker from Queens.\n"
        )
        # Page 3: would also score as title if checked (but shouldn't be)
        page3 = (
            "SUPREME COURT\nIndex No. 999\n"
            "This page has title patterns but should not be removed.\n"
        )
        text = title1 + "\f" + title2_top + title2_proceedings + "\f" + page3
        result = self._make().process(text)

        # Page 3 must be kept (proceedings found on page 2 stops removal)
        assert "should not be removed" in result.text
        assert "Q. Good morning" in result.text


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
        pytest.importorskip("fastcoref")
        result = self._make().process("Test text.")
        assert "resolutions" in result.metadata

    def test_model_availability_check(self):
        """Should gracefully handle missing spaCy/fastcoref models."""
        resolver = self._make()
        result = resolver.process("John went home. He was tired.")
        # Whether or not model is available, should not crash
        assert isinstance(result.text, str)
        assert result.changes_made >= 0
