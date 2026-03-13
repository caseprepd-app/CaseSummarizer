"""
Tests for Word and PDF document builders and combined export.

Covers:
- WordDocumentBuilder (word_builder.py)
- PdfDocumentBuilder (pdf_builder.py)
- Combined export orchestration (combined_exporter.py)
- Vocabulary exporter via builder interface (vocab_exporter.py)
- Q&A exporter via builder interface (qa_exporter.py)
- Base classes and verification colors (base.py)
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

# ============================================================================
# A. Base Classes and Verification Colors
# ============================================================================


class TestTextSpan:
    """Tests for the TextSpan dataclass."""

    def test_default_values(self):
        """TextSpan defaults: no color, no bold/italic/strikethrough."""
        from src.core.export.base import TextSpan

        span = TextSpan(text="hello")
        assert span.text == "hello"
        assert span.color is None
        assert span.bold is False
        assert span.italic is False
        assert span.strikethrough is False

    def test_with_all_attributes(self):
        """TextSpan stores all styling attributes."""
        from src.core.export.base import TextSpan

        span = TextSpan(text="red", color=(255, 0, 0), bold=True, strikethrough=True)
        assert span.color == (255, 0, 0)
        assert span.bold is True
        assert span.strikethrough is True


class TestVerificationColors:
    """Tests for get_verification_color thresholds."""

    def test_verified(self):
        """Low hallucination prob -> verified (green)."""
        from src.core.export.base import get_verification_color

        color, strike, category = get_verification_color(0.10)
        assert category == "verified"
        assert strike is False

    def test_uncertain(self):
        """Moderate prob -> uncertain."""
        from src.core.export.base import get_verification_color

        _, _, category = get_verification_color(0.35)
        assert category == "uncertain"

    def test_suspicious(self):
        """Higher prob -> suspicious."""
        from src.core.export.base import get_verification_color

        _, _, category = get_verification_color(0.55)
        assert category == "suspicious"

    def test_unreliable(self):
        """High prob -> unreliable."""
        from src.core.export.base import get_verification_color

        _, _, category = get_verification_color(0.75)
        assert category == "unreliable"

    def test_hallucinated(self):
        """Very high prob -> hallucinated with strikethrough."""
        from src.core.export.base import get_verification_color

        color, strike, category = get_verification_color(0.90)
        assert category == "hallucinated"
        assert strike is True

    def test_boundary_verified_uncertain(self):
        """Exactly 0.30 -> uncertain (not verified)."""
        from src.core.export.base import get_verification_color

        _, _, category = get_verification_color(0.30)
        assert category == "uncertain"

    def test_zero_prob(self):
        """Zero prob -> verified."""
        from src.core.export.base import get_verification_color

        _, _, category = get_verification_color(0.0)
        assert category == "verified"

    def test_one_prob(self):
        """Max prob -> hallucinated."""
        from src.core.export.base import get_verification_color

        _, _, category = get_verification_color(1.0)
        assert category == "hallucinated"


# ============================================================================
# B. Word Document Builder
# ============================================================================


class TestWordDocumentBuilder:
    """Tests for WordDocumentBuilder."""

    def test_creates_valid_docx(self, tmp_path):
        """Builder creates a valid .docx file."""
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder(title="Test Doc")
        builder.add_heading("Title", level=1)
        builder.add_paragraph("Hello world")
        path = tmp_path / "test.docx"
        builder.save(str(path))
        assert path.exists()
        assert path.stat().st_size > 0

    def test_heading_levels(self):
        """Builder accepts heading levels 1-3."""
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        builder.add_heading("H1", level=1)
        builder.add_heading("H2", level=2)
        builder.add_heading("H3", level=3)
        # No exception means success

    def test_paragraph_bold_italic(self):
        """Builder creates bold and italic paragraphs."""
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        builder.add_paragraph("Bold", bold=True)
        builder.add_paragraph("Italic", italic=True)
        builder.add_paragraph("Both", bold=True, italic=True)

    def test_styled_paragraph_with_colors(self):
        """Builder renders spans with RGB colors."""
        from src.core.export.base import TextSpan
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        spans = [
            TextSpan("green text", color=(0, 255, 0)),
            TextSpan("red text", color=(255, 0, 0), bold=True),
            TextSpan("struck", strikethrough=True),
        ]
        builder.add_styled_paragraph(spans)

    def test_table_with_data(self, tmp_path):
        """Builder creates table with headers and rows."""
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        headers = ["Term", "Score", "Person"]
        rows = [
            ["plaintiff", "0.95", "No"],
            ["John Smith", "0.88", "Yes"],
        ]
        builder.add_table(headers, rows)
        path = tmp_path / "table.docx"
        builder.save(str(path))
        assert path.exists()

    def test_empty_table_no_crash(self):
        """Builder handles empty headers gracefully."""
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        builder.add_table([], [])  # Should not crash

    def test_separator(self):
        """Builder adds separator without error."""
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        builder.add_separator()

    def test_save_to_bad_path_raises(self):
        """Saving to non-existent directory raises OSError."""
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        builder.add_paragraph("test")
        with pytest.raises(OSError):
            builder.save("/nonexistent/dir/test.docx")

    def test_special_characters(self, tmp_path):
        """Builder handles special characters (em dash, curly quotes)."""
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        builder.add_paragraph("Em dash \u2014 and curly quotes \u201c \u201d")
        builder.add_heading("Section \u00a7 2.1")
        path = tmp_path / "special.docx"
        builder.save(str(path))
        assert path.exists()


# ============================================================================
# C. PDF Document Builder
# ============================================================================


class TestPdfDocumentBuilder:
    """Tests for PdfDocumentBuilder."""

    def test_creates_valid_pdf(self, tmp_path):
        """Builder creates a valid PDF file."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder(title="Test PDF")
        builder.add_heading("Title", level=1)
        builder.add_paragraph("Hello world")
        path = tmp_path / "test.pdf"
        builder.save(str(path))
        assert path.exists()
        assert path.stat().st_size > 0

    def test_heading_levels(self):
        """Builder accepts heading levels 1-3 with different sizes."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder()
        builder.add_heading("H1", level=1)
        builder.add_heading("H2", level=2)
        builder.add_heading("H3", level=3)

    def test_paragraph_bold_italic(self):
        """Builder creates bold and italic paragraphs."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder()
        builder.add_paragraph("Bold", bold=True)
        builder.add_paragraph("Italic", italic=True)

    def test_styled_paragraph_with_colors(self):
        """Builder renders spans with RGB colors."""
        from src.core.export.base import TextSpan
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder()
        spans = [
            TextSpan("green text", color=(0, 128, 0)),
            TextSpan("red bold", color=(255, 0, 0), bold=True),
            TextSpan("struck", strikethrough=True, color=(128, 128, 128)),
        ]
        builder.add_styled_paragraph(spans)

    def test_table_with_data(self, tmp_path):
        """Builder creates table with headers and rows."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder()
        headers = ["Term", "Score"]
        rows = [["plaintiff", "0.95"], ["defendant", "0.88"]]
        builder.add_table(headers, rows)
        path = tmp_path / "table.pdf"
        builder.save(str(path))
        assert path.exists()

    def test_empty_table_no_crash(self):
        """Builder handles empty headers gracefully."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder()
        builder.add_table([], [])

    def test_separator(self):
        """Builder adds separator without error."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder()
        builder.add_separator()

    def test_save_to_bad_path_raises(self):
        """Saving to non-existent directory raises OSError."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder()
        builder.add_paragraph("test")
        with pytest.raises(OSError):
            builder.save("/nonexistent/dir/test.pdf")

    def test_latin1_sanitization(self):
        """Non-Latin-1 characters are replaced, not crashing."""
        from src.core.export.pdf_builder import _sanitize_for_latin1

        # Chinese characters should be replaced with '?'
        result = _sanitize_for_latin1("Hello \u4e16\u754c")
        assert "?" in result
        assert "Hello" in result

    def test_latin1_preserves_accented(self):
        """Accented Latin characters are preserved or normalized."""
        from src.core.export.pdf_builder import _sanitize_for_latin1

        result = _sanitize_for_latin1("caf\u00e9")
        # Should preserve the 'e' at minimum (NFKD decomposes then encodes)
        assert "caf" in result

    def test_unicode_in_heading(self, tmp_path):
        """Builder handles Unicode in headings via sanitization."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder()
        builder.add_heading("Section \u00a7 2 \u2014 Overview")
        builder.add_paragraph("Em dash: \u2014, bullet: \u2022")
        path = tmp_path / "unicode.pdf"
        builder.save(str(path))
        assert path.exists()

    def test_long_table_triggers_page_break(self, tmp_path):
        """Large table triggers automatic page break."""
        from src.core.export.pdf_builder import PdfDocumentBuilder

        builder = PdfDocumentBuilder()
        headers = ["Col1", "Col2"]
        rows = [[f"Row {i}", f"Value {i}"] for i in range(100)]
        builder.add_table(headers, rows)
        path = tmp_path / "big_table.pdf"
        builder.save(str(path))
        assert path.exists()


# ============================================================================
# D. Vocabulary Export via Builder
# ============================================================================


class TestVocabularyExportViaBuilder:
    """Tests for export_vocabulary() with Word and PDF builders."""

    SAMPLE_VOCAB = [
        {
            "Term": "plaintiff",
            "Quality Score": 0.95,
            "Is Person": "No",
            "Found By": "NER",
            "Occurrences": 5,
        },
        {
            "Term": "John Smith",
            "Quality Score": 0.88,
            "Is Person": "Yes",
            "Found By": "NER",
            "Occurrences": 3,
        },
    ]

    def test_word_vocab_export(self, tmp_path):
        """Vocabulary export to Word creates valid file."""
        from src.core.export.vocab_exporter import export_vocabulary
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        export_vocabulary(self.SAMPLE_VOCAB, builder)
        path = tmp_path / "vocab.docx"
        builder.save(str(path))
        assert path.exists()

    def test_pdf_vocab_export(self, tmp_path):
        """Vocabulary export to PDF creates valid file."""
        from src.core.export.pdf_builder import PdfDocumentBuilder
        from src.core.export.vocab_exporter import export_vocabulary

        builder = PdfDocumentBuilder()
        export_vocabulary(self.SAMPLE_VOCAB, builder)
        path = tmp_path / "vocab.pdf"
        builder.save(str(path))
        assert path.exists()

    def test_empty_vocab(self):
        """Empty vocab data produces 'no data' message."""
        from src.core.export.vocab_exporter import export_vocabulary
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        export_vocabulary([], builder)
        # Should not crash

    def test_include_details_adds_columns(self):
        """include_details=True adds NER/RAKE/BM25 columns."""
        from src.core.export.vocab_exporter import export_vocabulary

        builder = MagicMock()
        vocab = [
            {
                "Term": "test",
                "Quality Score": 0.5,
                "Is Person": "No",
                "Found By": "NER",
                "Occurrences": 1,
                "NER": 0.8,
                "RAKE": 0.2,
                "BM25": 0.1,
            }
        ]
        export_vocabulary(vocab, builder, include_details=True)
        # Check add_table was called with extended headers
        call_args = builder.add_table.call_args[0]
        headers = call_args[0]
        assert "NER" in headers
        assert "RAKE" in headers
        assert "BM25" in headers


# ============================================================================
# E. Q&A Export via Builder
# ============================================================================


@dataclass
class MockVerifiedSpan:
    """Mock span for verification testing."""

    text: str
    hallucination_prob: float


@dataclass
class MockVerification:
    """Mock verification result."""

    overall_reliability: float
    answer_rejected: bool
    spans: list


@dataclass
class MockSemanticResult:
    """Mock SemanticResult for export testing."""

    question: str
    quick_answer: str
    citation: str
    source_summary: str = ""
    verification: object = None


class TestQAExportViaBuilder:
    """Tests for export_semantic_results() with Word and PDF builders."""

    def test_word_qa_export(self, tmp_path):
        """Q&A export to Word creates valid file."""
        from src.core.export.semantic_exporter import export_semantic_results
        from src.core.export.word_builder import WordDocumentBuilder

        results = [MockSemanticResult("Who?", "John Smith", "John Smith filed...")]
        builder = WordDocumentBuilder()
        export_semantic_results(results, builder)
        path = tmp_path / "qa.docx"
        builder.save(str(path))
        assert path.exists()

    def test_pdf_qa_export(self, tmp_path):
        """Q&A export to PDF creates valid file."""
        from src.core.export.pdf_builder import PdfDocumentBuilder
        from src.core.export.semantic_exporter import export_semantic_results

        results = [MockSemanticResult("What?", "Personal injury", "This case involves...")]
        builder = PdfDocumentBuilder()
        export_semantic_results(results, builder)
        path = tmp_path / "qa.pdf"
        builder.save(str(path))
        assert path.exists()

    def test_empty_results(self):
        """Empty results produces 'no data' message."""
        from src.core.export.semantic_exporter import export_semantic_results
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        export_semantic_results([], builder)

    def test_verified_answer_with_spans(self, tmp_path):
        """Verified answer renders colored spans."""
        from src.core.export.semantic_exporter import export_semantic_results
        from src.core.export.word_builder import WordDocumentBuilder

        verification = MockVerification(
            overall_reliability=0.85,
            answer_rejected=False,
            spans=[
                MockVerifiedSpan("John Smith filed", 0.1),
                MockVerifiedSpan("a lawsuit", 0.8),
            ],
        )
        results = [
            MockSemanticResult(
                "Who?", "John Smith filed a lawsuit", "citation...", verification=verification
            )
        ]
        builder = WordDocumentBuilder()
        export_semantic_results(results, builder, include_verification_colors=True)
        path = tmp_path / "verified.docx"
        builder.save(str(path))
        assert path.exists()

    def test_rejected_answer(self):
        """Rejected answer shows rejection message."""
        from src.core.export.semantic_exporter import export_semantic_results

        verification = MockVerification(
            overall_reliability=0.3,
            answer_rejected=True,
            spans=[],
        )
        results = [MockSemanticResult("Who?", "bad answer", "citation...", verification=verification)]
        builder = MagicMock()
        export_semantic_results(results, builder)
        # Should not crash; rejected message is added as paragraph


# ============================================================================
# F. Combined Export
# ============================================================================


class TestCombinedExport:
    """Tests for combined vocabulary + Q&A export."""

    def test_combined_word_export(self, tmp_path):
        """Combined export to Word creates valid file."""
        from src.core.export.combined_exporter import export_combined
        from src.core.export.word_builder import WordDocumentBuilder

        vocab = [
            {
                "Term": "plaintiff",
                "Quality Score": 0.95,
                "Is Person": "No",
                "Found By": "NER",
                "Occurrences": 5,
            }
        ]
        qa = [MockSemanticResult("Who?", "John Smith", "citation")]
        builder = WordDocumentBuilder()
        export_combined(vocab, qa, builder)
        path = tmp_path / "combined.docx"
        builder.save(str(path))
        assert path.exists()

    def test_combined_pdf_export(self, tmp_path):
        """Combined export to PDF creates valid file."""
        from src.core.export.combined_exporter import export_combined
        from src.core.export.pdf_builder import PdfDocumentBuilder

        vocab = [
            {
                "Term": "defendant",
                "Quality Score": 0.9,
                "Is Person": "No",
                "Found By": "RAKE",
                "Occurrences": 2,
            }
        ]
        qa = [MockSemanticResult("What?", "Personal injury", "The case...")]
        builder = PdfDocumentBuilder()
        export_combined(vocab, qa, builder)
        path = tmp_path / "combined.pdf"
        builder.save(str(path))
        assert path.exists()

    def test_combined_empty_vocab_and_qa(self):
        """Combined export handles empty data."""
        from src.core.export.combined_exporter import export_combined
        from src.core.export.word_builder import WordDocumentBuilder

        builder = WordDocumentBuilder()
        export_combined([], [], builder)

    def test_combined_vocab_only(self, tmp_path):
        """Combined export with vocab but no Q&A."""
        from src.core.export.combined_exporter import export_combined
        from src.core.export.word_builder import WordDocumentBuilder

        vocab = [
            {
                "Term": "test",
                "Quality Score": 0.5,
                "Is Person": "No",
                "Found By": "BM25",
                "Occurrences": 1,
            }
        ]
        builder = WordDocumentBuilder()
        export_combined(vocab, [], builder)
        path = tmp_path / "vocab_only.docx"
        builder.save(str(path))
        assert path.exists()

    def test_combined_qa_only(self, tmp_path):
        """Combined export with Q&A but no vocab."""
        from src.core.export.combined_exporter import export_combined
        from src.core.export.word_builder import WordDocumentBuilder

        qa = [MockSemanticResult("Who?", "John", "The plaintiff")]
        builder = WordDocumentBuilder()
        export_combined([], qa, builder)
        path = tmp_path / "qa_only.docx"
        builder.save(str(path))
        assert path.exists()
