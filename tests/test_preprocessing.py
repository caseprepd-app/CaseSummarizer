"""
Tests for the Smart Preprocessing Pipeline

Tests each preprocessor in isolation and the pipeline as a whole.
"""

from src.core.preprocessing import (
    HeaderFooterRemover,
    LineNumberRemover,
    PreprocessingPipeline,
    QAConverter,
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


class TestQAConverter:
    """Tests for Q/A Converter preprocessor."""

    def test_converts_q_dot_format(self):
        """Should convert Q. to Question:."""
        converter = QAConverter()
        text = "Q.  Where were you on January 5th?"
        result = converter.process(text)

        assert "Question: Where were you" in result.text
        assert "Q." not in result.text

    def test_converts_a_dot_format(self):
        """Should convert A. to Answer:."""
        converter = QAConverter()
        text = "A.  I was at home."
        result = converter.process(text)

        assert "Answer: I was at home." in result.text
        assert "A." not in result.text

    def test_handles_full_qa_exchange(self):
        """Should convert full Q&A exchanges."""
        converter = QAConverter()
        text = "Q.  What happened?\nA.  I saw the accident."
        result = converter.process(text)

        assert "Question: What happened?" in result.text
        assert "Answer: I saw the accident." in result.text
        assert result.changes_made == 2


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

        assert len(pipeline.preprocessors) == 5
        assert any(p.name == "Line Number Remover" for p in pipeline.preprocessors)
        assert any(p.name == "Transcript Cleaner" for p in pipeline.preprocessors)
        assert any(p.name == "Q/A Converter" for p in pipeline.preprocessors)

    def test_pipeline_processes_in_order(self):
        """Should process text through all preprocessors in order."""
        pipeline = create_default_pipeline()
        text = "1  Q.  Good morning.\n2  A.  Good morning."
        result = pipeline.process(text)

        # Line numbers removed AND Q/A converted
        assert "Question:" in result
        assert "Answer:" in result
        assert "1  Q." not in result

    def test_pipeline_tracks_total_changes(self):
        """Should track cumulative changes across all preprocessors."""
        pipeline = create_default_pipeline()
        text = "1  Q.  What happened?\n2  A.  I saw it."
        pipeline.process(text)

        # Should have changes from both LineNumberRemover and QAConverter
        assert pipeline.total_changes >= 4  # 2 line numbers + 2 Q/A conversions

    def test_disabled_preprocessors_skipped(self):
        """Should skip disabled preprocessors."""
        pipeline = PreprocessingPipeline([
            LineNumberRemover(),
            QAConverter(),
        ])
        pipeline.preprocessors[1].enabled = False  # Disable Q/A converter

        text = "Q.  What happened?\nA.  I saw it."
        result = pipeline.process(text)

        # Q/A NOT converted (disabled)
        assert "Q." in result
        assert "A." in result
        # Question:/Answer: should not appear
        assert "Question:" not in result
        assert "Answer:" not in result

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
        assert any("Q/A Converter" in name for name in stats.keys())
