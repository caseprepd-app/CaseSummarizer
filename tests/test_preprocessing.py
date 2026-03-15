"""
Tests for the Smart Preprocessing Pipeline

Tests each preprocessor in isolation and the pipeline as a whole.
"""

from src.core.preprocessing import (
    HeaderFooterRemover,
    LineNumberRemover,
    PreprocessingPipeline,
    TitlePageRemover,
    create_default_pipeline,
)


class TestLineNumberRemover:
    """Tests for LineNumberRemover preprocessor."""

    def test_removes_line_numbers_at_start(self):
        """Should remove transcript-style line numbers (1-25) at line start."""
        remover = LineNumberRemover()
        text = "1  Q.  Good morning.\n2  A.  Good morning.\n3  Q.  State your name."
        result = remover.process(text)

        assert "1  Q." not in result.text
        assert "Q.  Good morning." in result.text
        assert "A.  Good morning." in result.text
        assert result.changes_made == 3

    def test_preserves_numbers_in_content(self):
        """Should not remove numbers that are part of content."""
        remover = LineNumberRemover()
        text = "The accident occurred on January 15, 2024."
        result = remover.process(text)

        assert result.text == text
        assert result.changes_made == 0

    def test_handles_pipe_format(self):
        """Should remove pipe-prefixed line numbers."""
        remover = LineNumberRemover()
        text = "|1 First line\n|2 Second line"
        result = remover.process(text)

        assert result.text == "First line\nSecond line"
        assert result.changes_made == 2


class TestHeaderFooterRemover:
    """Tests for HeaderFooterRemover preprocessor."""

    def test_removes_repetitive_page_numbers(self):
        """Should remove lines that appear on multiple pages."""
        remover = HeaderFooterRemover()
        # Simulate a header appearing 4 times
        text = (
            "Page 1\nContent line 1\n\n"
            "Page 2\nContent line 2\n\n"
            "Page 3\nContent line 3\n\n"
            "Page 4\nContent line 4"
        )
        result = remover.process(text)

        # "Page X" lines should be removed
        assert "Content line" in result.text
        assert result.changes_made >= 4

    def test_preserves_unique_content(self):
        """Should preserve lines that appear only once."""
        remover = HeaderFooterRemover()
        text = "This is unique content.\nAnother unique line."
        result = remover.process(text)

        assert result.text == text
        assert result.changes_made == 0

    def test_removes_section_headers(self):
        """Should remove transcript section headers that repeat."""
        remover = HeaderFooterRemover()
        # Section headers appearing 4 times (on 4 pages)
        text = (
            "DIRECT EXAMINATION\nQ. First question?\nA. First answer.\n\n"
            "DIRECT EXAMINATION\nQ. Second question?\nA. Second answer.\n\n"
            "DIRECT EXAMINATION\nQ. Third question?\nA. Third answer.\n\n"
            "DIRECT EXAMINATION\nQ. Fourth question?\nA. Fourth answer."
        )
        result = remover.process(text)

        assert "DIRECT EXAMINATION" not in result.text
        assert "First question" in result.text
        assert result.changes_made == 4

    def test_removes_opening_statements_header(self):
        """Should remove OPENING STATEMENTS headers."""
        remover = HeaderFooterRemover()
        text = (
            "OPENING STATEMENTS - PLAINTIFF / MR. KAUFER\nContent 1\n\n"
            "OPENING STATEMENTS - PLAINTIFF / MR. KAUFER\nContent 2\n\n"
            "OPENING STATEMENTS - PLAINTIFF / MR. KAUFER\nContent 3\n\n"
            "OPENING STATEMENTS - PLAINTIFF / MR. KAUFER\nContent 4"
        )
        result = remover.process(text)

        assert "OPENING STATEMENTS" not in result.text
        assert "Content 1" in result.text
        assert result.changes_made == 4

    def test_removes_short_plaintiff_header(self):
        """Should remove short lines with PLAINTIFF when they repeat."""
        remover = HeaderFooterRemover()
        text = (
            "PLAINTIFF / MR. SMITH\nSome testimony here.\n\n"
            "PLAINTIFF / MR. SMITH\nMore testimony here.\n\n"
            "PLAINTIFF / MR. SMITH\nEven more testimony.\n\n"
            "PLAINTIFF / MR. SMITH\nFinal testimony."
        )
        result = remover.process(text)

        assert "PLAINTIFF / MR. SMITH" not in result.text
        assert "Some testimony here" in result.text
        assert result.changes_made == 4

    def test_preserves_plaintiff_in_prose(self):
        """Should NOT remove 'plaintiff' when it's part of prose content."""
        remover = HeaderFooterRemover()
        # Even though "plaintiff" appears, these are content sentences
        text = (
            "The plaintiff was injured in the accident.\n"
            "The plaintiff testified about the incident.\n"
            "The plaintiff arrived at 3pm that day.\n"
            "The plaintiff met with his doctor."
        )
        result = remover.process(text)

        # Prose should be preserved (sentences end with periods, have many words)
        assert "plaintiff was injured" in result.text
        assert result.changes_made == 0

    def test_removes_cross_examination_header(self):
        """Should remove CROSS EXAMINATION headers."""
        remover = HeaderFooterRemover()
        text = (
            "CROSS EXAMINATION\nQ. First cross?\n\n"
            "CROSS EXAMINATION\nQ. Second cross?\n\n"
            "CROSS EXAMINATION\nQ. Third cross?\n\n"
            "CROSS EXAMINATION\nQ. Fourth cross?"
        )
        result = remover.process(text)

        assert "CROSS EXAMINATION" not in result.text
        assert "First cross" in result.text

    def test_removes_proceedings_header(self):
        """Should remove PROCEEDINGS as a standalone header."""
        remover = HeaderFooterRemover()
        text = (
            "PROCEEDINGS\nThe court convened at 9am.\n\n"
            "PROCEEDINGS\nThe witness was sworn.\n\n"
            "PROCEEDINGS\nCounsel made objections.\n\n"
            "PROCEEDINGS\nThe court recessed."
        )
        result = remover.process(text)

        assert "PROCEEDINGS" not in result.text
        assert "court convened" in result.text

    def test_removes_closing_arguments_header(self):
        """Should remove CLOSING ARGUMENTS headers."""
        remover = HeaderFooterRemover()
        text = (
            "CLOSING ARGUMENTS\nContent 1\n\n"
            "CLOSING ARGUMENTS\nContent 2\n\n"
            "CLOSING ARGUMENTS\nContent 3\n\n"
            "CLOSING ARGUMENTS\nContent 4"
        )
        result = remover.process(text)

        assert "CLOSING ARGUMENTS" not in result.text

    def test_candidate_check_respects_line_length(self):
        """Short-line keywords should only match short lines."""
        remover = HeaderFooterRemover()

        # Short line with PLAINTIFF - should be candidate
        assert remover._is_header_footer_candidate("PLAINTIFF / MR. SMITH")

        # Long prose line with plaintiff - should NOT be candidate
        long_line = "The plaintiff was severely injured in the automobile accident on January 15."
        assert not remover._is_header_footer_candidate(long_line)

    def test_candidate_check_excludes_sentences(self):
        """Lines ending with sentence punctuation should not match short-line keywords."""
        remover = HeaderFooterRemover()

        # Header-like (no period) - should match
        assert remover._is_header_footer_candidate("DEFENDANT - CROSS")

        # Sentence (ends with period) - should not match short-line keywords
        # Even though it's short and has "defendant"
        assert not remover._is_header_footer_candidate("The defendant testified.")

    def test_custom_patterns_from_settings(self):
        """Should use custom patterns from user preferences."""
        from unittest.mock import patch

        # Mock user preferences to include a custom pattern
        mock_prefs = {
            "custom_header_footer_patterns": "SMITH & JONES LLP\nJANE DOE CSR",
            "header_footer_short_line_detection": True,
            "header_footer_min_occurrences": 3,
        }

        with patch("src.user_preferences.get_user_preferences") as mock_get_prefs:
            mock_get_prefs.return_value.get = lambda k, default=None: mock_prefs.get(k, default)
            remover = HeaderFooterRemover()

            # Custom patterns should match
            assert remover._is_header_footer_candidate("Smith & Jones LLP")
            assert remover._is_header_footer_candidate("Jane Doe CSR")

    def test_short_line_detection_can_be_disabled(self):
        """Should respect short_line_detection setting."""
        from unittest.mock import patch

        # Disable short-line detection
        mock_prefs = {
            "custom_header_footer_patterns": "",
            "header_footer_short_line_detection": False,
            "header_footer_min_occurrences": 3,
        }

        with patch("src.user_preferences.get_user_preferences") as mock_get_prefs:
            mock_get_prefs.return_value.get = lambda k, default=None: mock_prefs.get(k, default)
            remover = HeaderFooterRemover()

            # Short-line keywords should NOT match when detection is disabled
            # (unless they match standard patterns like "direct examination")
            assert not remover._is_header_footer_candidate("PLAINTIFF / MR. SMITH")

            # But standard patterns should still work
            assert remover._is_header_footer_candidate("DIRECT EXAMINATION")

    def test_configurable_min_occurrences(self):
        """Should use configurable min_occurrences threshold."""
        from unittest.mock import patch

        # Set higher threshold
        mock_prefs = {
            "custom_header_footer_patterns": "",
            "header_footer_short_line_detection": True,
            "header_footer_min_occurrences": 5,
        }

        with patch("src.user_preferences.get_user_preferences") as mock_get_prefs:
            mock_get_prefs.return_value.get = lambda k, default=None: mock_prefs.get(k, default)
            remover = HeaderFooterRemover()

            # Header appearing only 4 times should NOT be removed (threshold is 5)
            text = (
                "Page 1\nContent line 1\n\n"
                "Page 2\nContent line 2\n\n"
                "Page 3\nContent line 3\n\n"
                "Page 4\nContent line 4"
            )
            result = remover.process(text)

            # Pages should NOT be removed (only 4 occurrences, threshold is 5)
            assert "Page 1" in result.text
            assert result.changes_made == 0


class TestTitlePageRemover:
    """Tests for TitlePageRemover preprocessor."""

    def test_removes_obvious_title_page(self):
        """Should remove pages with case captions and court headers."""
        remover = TitlePageRemover()
        title_page = """
SUPREME COURT OF THE STATE OF NEW YORK
COUNTY OF QUEENS

JOHN DOE,
                         Plaintiff,
    -against-

JANE SMITH,
                         Defendant.

Index No. 123456/2024

DEPOSITION OF JOHN DOE
"""
        content_page = """
Q.  Good morning, Mr. Doe.
A.  Good morning.
Q.  Please state your name for the record.
A.  John Doe.
"""
        text = title_page + "\f" + content_page  # Form feed separates pages
        result = remover.process(text)

        # Content should be preserved
        assert "Good morning" in result.text
        # Title page elements should be reduced or removed
        assert result.changes_made >= 1

    def test_preserves_content_only_document(self):
        """Should not remove content if no clear title page."""
        remover = TitlePageRemover()
        text = "Q.  What is your name?\nA.  John Smith."
        result = remover.process(text)

        assert result.text == text
        assert result.changes_made == 0


class TestPreprocessingPipeline:
    """Tests for the PreprocessingPipeline orchestrator."""

    def test_pipeline_creation(self):
        """Should create pipeline with multiple preprocessors."""
        pipeline = create_default_pipeline()

        # 6 preprocessors: TitlePage, IndexPage, HeaderFooter, LineNumber, PageBoundary, Transcript
        assert len(pipeline.preprocessors) == 6
        assert any(p.name == "Index Page Remover" for p in pipeline.preprocessors)
        assert any(p.name == "Line Number Remover" for p in pipeline.preprocessors)
        assert any(p.name == "Page Boundary Cleaner" for p in pipeline.preprocessors)
        assert any(p.name == "Transcript Cleaner" for p in pipeline.preprocessors)

    def test_default_settings_none_disables_title_page_remover(self):
        """settings=None (default call) disables TitlePageRemover — vocab_only is the default."""
        pipeline = create_default_pipeline()
        remover = next(p for p in pipeline.preprocessors if p.name == "Title Page Remover")
        assert remover.enabled is False

    def test_settings_missing_title_page_handling_disables_remover(self):
        """Empty settings dict with no title_page_handling key defaults to disabled."""
        pipeline = create_default_pipeline({})
        remover = next(p for p in pipeline.preprocessors if p.name == "Title Page Remover")
        assert remover.enabled is False

    def test_title_page_handling_exclude_all_enables_remover(self):
        """exclude_all: TitlePageRemover should be enabled in the pipeline."""
        pipeline = create_default_pipeline({"title_page_handling": "exclude_all"})
        remover = next(p for p in pipeline.preprocessors if p.name == "Title Page Remover")
        assert remover.enabled is True

    def test_title_page_handling_vocab_only_disables_remover(self):
        """vocab_only: TitlePageRemover disabled in pipeline (workers.py handles search)."""
        pipeline = create_default_pipeline({"title_page_handling": "vocab_only"})
        remover = next(p for p in pipeline.preprocessors if p.name == "Title Page Remover")
        assert remover.enabled is False

    def test_title_page_handling_include_all_disables_remover(self):
        """include_all: TitlePageRemover should be disabled everywhere."""
        pipeline = create_default_pipeline({"title_page_handling": "include_all"})
        remover = next(p for p in pipeline.preprocessors if p.name == "Title Page Remover")
        assert remover.enabled is False

    def test_pipeline_processes_in_order(self):
        """Should process text through all preprocessors in order."""
        pipeline = create_default_pipeline()
        text = "1  Q.  Good morning.\n2  A.  Good morning."
        result = pipeline.process(text)

        # Line numbers removed, Q/A notation preserved (QAConverter removed)
        assert "Q." in result or "Good morning." in result
        assert "1  Q." not in result

    def test_pipeline_tracks_total_changes(self):
        """Should track cumulative changes across all preprocessors."""
        pipeline = create_default_pipeline()
        text = "1  Q.  What happened?\n2  A.  I saw it."
        pipeline.process(text)

        # Should have changes from LineNumberRemover (QAConverter removed)
        assert pipeline.total_changes >= 2  # 2 line numbers

    def test_disabled_preprocessors_skipped(self):
        """Should skip disabled preprocessors."""
        # Pipeline with LineNumberRemover active and HeaderFooterRemover disabled
        pipeline = PreprocessingPipeline(
            [
                LineNumberRemover(),
                HeaderFooterRemover(),
            ]
        )
        pipeline.preprocessors[1].enabled = False  # Disable header/footer remover

        # "DIRECT EXAMINATION" appears 4 times — HeaderFooterRemover would remove it,
        # but since it's disabled the header text must survive.
        # Line numbers are removed by the still-active LineNumberRemover.
        text = (
            "1  DIRECT EXAMINATION\n"
            "2  DIRECT EXAMINATION\n"
            "3  DIRECT EXAMINATION\n"
            "4  DIRECT EXAMINATION"
        )
        result = pipeline.process(text)

        # Line numbers removed (LineNumberRemover active)
        assert "1  DIRECT EXAMINATION" not in result
        # Header text preserved (HeaderFooterRemover disabled)
        assert "DIRECT EXAMINATION" in result

    def test_empty_text_handled(self):
        """Should handle empty text gracefully."""
        pipeline = create_default_pipeline()
        result = pipeline.process("")

        assert result == ""
        assert pipeline.total_changes == 0

    def test_get_stats_returns_info(self):
        """Should return stats from last run."""
        pipeline = create_default_pipeline()
        text = "Q.  Test question.\nA.  Test answer."
        pipeline.process(text)

        stats = pipeline.get_stats()
        assert isinstance(stats, dict)
        assert any("Transcript Cleaner" in name for name in stats)
