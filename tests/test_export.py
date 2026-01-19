"""
Unit tests for export functionality.

Tests vocabulary and Q&A export to various formats (HTML, TXT).
"""

from dataclasses import dataclass


class TestVocabularyHtmlExport:
    """Test vocabulary HTML export."""

    def test_export_vocabulary_html_creates_file(self, tmp_path):
        """HTML export should create a file at the specified path."""
        from src.core.export.html_builder import export_vocabulary_html

        vocab_data = [
            {"Term": "plaintiff", "Quality Score": 0.95, "Is Person": "No", "Found By": "NER"},
            {"Term": "John Smith", "Quality Score": 0.88, "Is Person": "Yes", "Found By": "NER"},
        ]
        output_file = tmp_path / "vocab.html"

        result = export_vocabulary_html(vocab_data, str(output_file))

        assert result is True
        assert output_file.exists()

    def test_export_vocabulary_html_contains_terms(self, tmp_path):
        """HTML export should contain all vocabulary terms."""
        from src.core.export.html_builder import export_vocabulary_html

        vocab_data = [
            {"Term": "hypertension", "Quality Score": 0.9, "Is Person": "No", "Found By": "RAKE"},
            {"Term": "Dr. Jane Doe", "Quality Score": 0.85, "Is Person": "Yes", "Found By": "NER"},
        ]
        output_file = tmp_path / "vocab.html"

        export_vocabulary_html(vocab_data, str(output_file))
        content = output_file.read_text(encoding="utf-8")

        assert "hypertension" in content
        assert "Dr. Jane Doe" in content

    def test_export_vocabulary_html_escapes_special_chars(self, tmp_path):
        """HTML export should escape HTML special characters in data."""
        from src.core.export.html_builder import export_vocabulary_html

        vocab_data = [
            {
                "Term": "<script>alert('xss')</script>",
                "Quality Score": 0.5,
                "Is Person": "No",
                "Found By": "BM25",
            },
        ]
        output_file = tmp_path / "vocab.html"

        export_vocabulary_html(vocab_data, str(output_file))
        content = output_file.read_text(encoding="utf-8")

        # The XSS attempt should be escaped in table data (not raw)
        # Note: HTML template itself has <script> tags for JavaScript functionality
        assert "&lt;script&gt;alert" in content
        assert (
            "alert(&#x27;xss&#x27;)" in content or "alert('xss')" not in content.split("<tbody>")[1]
        )

    def test_export_vocabulary_html_empty_data(self, tmp_path):
        """HTML export should handle empty vocabulary data."""
        from src.core.export.html_builder import export_vocabulary_html

        vocab_data = []
        output_file = tmp_path / "vocab.html"

        result = export_vocabulary_html(vocab_data, str(output_file))

        assert result is True
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "0 entries" in content

    def test_export_vocabulary_html_with_visible_columns(self, tmp_path):
        """HTML export should respect visible_columns parameter."""
        from src.core.export.html_builder import export_vocabulary_html

        vocab_data = [
            {"Term": "test term", "Quality Score": 0.8, "Is Person": "No", "Found By": "NER"},
        ]
        output_file = tmp_path / "vocab.html"

        result = export_vocabulary_html(
            vocab_data,
            str(output_file),
            visible_columns=["Term", "Score"],
        )

        assert result is True

    def test_export_vocabulary_html_invalid_path_returns_false(self):
        """HTML export should return False for invalid file paths."""
        from src.core.export.html_builder import export_vocabulary_html

        vocab_data = [{"Term": "test", "Quality Score": 0.5, "Is Person": "No", "Found By": "NER"}]
        # Invalid path (directory that doesn't exist)
        invalid_path = "/nonexistent/directory/file.html"

        result = export_vocabulary_html(vocab_data, invalid_path)

        assert result is False


class TestQAHtmlExport:
    """Test Q&A HTML export."""

    @dataclass
    class MockQAResult:
        """Mock QAResult for testing."""

        question: str
        quick_answer: str
        citation: str
        source_summary: str = ""
        verification: object = None

    def test_export_qa_html_creates_file(self, tmp_path):
        """Q&A HTML export should create a file at the specified path."""
        from src.core.export.html_builder import export_qa_html

        results = [
            self.MockQAResult(
                question="Who is the plaintiff?",
                quick_answer="John Smith",
                citation="The plaintiff, John Smith, filed...",
            ),
        ]
        output_file = tmp_path / "qa.html"

        result = export_qa_html(results, str(output_file))

        assert result is True
        assert output_file.exists()

    def test_export_qa_html_contains_questions_and_answers(self, tmp_path):
        """Q&A HTML export should contain questions and answers."""
        from src.core.export.html_builder import export_qa_html

        results = [
            self.MockQAResult(
                question="What is the case about?",
                quick_answer="Personal injury claim",
                citation="This case involves a personal injury...",
            ),
        ]
        output_file = tmp_path / "qa.html"

        export_qa_html(results, str(output_file))
        content = output_file.read_text(encoding="utf-8")

        assert "What is the case about?" in content
        assert "Personal injury claim" in content

    def test_export_qa_html_escapes_special_chars(self, tmp_path):
        """Q&A HTML export should escape HTML special characters in data."""
        from src.core.export.html_builder import export_qa_html

        results = [
            self.MockQAResult(
                question="What about <script> tags?",
                quick_answer="They should be escaped",
                citation="Test & verify",
            ),
        ]
        output_file = tmp_path / "qa.html"

        export_qa_html(results, str(output_file))
        content = output_file.read_text(encoding="utf-8")

        # The XSS attempt in the question should be escaped in the data
        # Note: HTML template itself has <script> tags for JavaScript functionality
        assert "&lt;script&gt;" in content
        assert "&amp;" in content

    def test_export_qa_html_empty_results(self, tmp_path):
        """Q&A HTML export should handle empty results."""
        from src.core.export.html_builder import export_qa_html

        results = []
        output_file = tmp_path / "qa.html"

        result = export_qa_html(results, str(output_file))

        assert result is True
        assert output_file.exists()


class TestVocabularyTxtExport:
    """Test vocabulary TXT export."""

    def test_export_vocabulary_txt_creates_file(self, tmp_path):
        """TXT export should create a file at the specified path."""
        from src.core.export.vocab_exporter import export_vocabulary_txt

        vocab_data = [
            {"Term": "plaintiff"},
            {"Term": "defendant"},
        ]
        output_file = tmp_path / "vocab.txt"

        result = export_vocabulary_txt(vocab_data, str(output_file))

        assert result is True
        assert output_file.exists()

    def test_export_vocabulary_txt_contains_terms(self, tmp_path):
        """TXT export should contain all terms, one per line."""
        from src.core.export.vocab_exporter import export_vocabulary_txt

        vocab_data = [
            {"Term": "hypertension"},
            {"Term": "cervical strain"},
            {"Term": "plaintiff"},
        ]
        output_file = tmp_path / "vocab.txt"

        export_vocabulary_txt(vocab_data, str(output_file))
        content = output_file.read_text(encoding="utf-8")
        lines = content.strip().split("\n")

        assert len(lines) == 3
        assert "hypertension" in lines
        assert "cervical strain" in lines
        assert "plaintiff" in lines

    def test_export_vocabulary_txt_skips_empty_terms(self, tmp_path):
        """TXT export should skip entries with empty Term."""
        from src.core.export.vocab_exporter import export_vocabulary_txt

        vocab_data = [
            {"Term": "valid term"},
            {"Term": ""},
            {"Term": None},
            {"OtherKey": "no term key"},
        ]
        output_file = tmp_path / "vocab.txt"

        export_vocabulary_txt(vocab_data, str(output_file))
        content = output_file.read_text(encoding="utf-8")
        lines = [l for l in content.strip().split("\n") if l]

        assert len(lines) == 1
        assert "valid term" in lines

    def test_export_vocabulary_txt_empty_data(self, tmp_path):
        """TXT export should handle empty vocabulary data."""
        from src.core.export.vocab_exporter import export_vocabulary_txt

        vocab_data = []
        output_file = tmp_path / "vocab.txt"

        result = export_vocabulary_txt(vocab_data, str(output_file))

        assert result is True
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert content == ""

    def test_export_vocabulary_txt_invalid_path_returns_false(self):
        """TXT export should return False for invalid file paths."""
        from src.core.export.vocab_exporter import export_vocabulary_txt

        vocab_data = [{"Term": "test"}]
        invalid_path = "/nonexistent/directory/file.txt"

        result = export_vocabulary_txt(vocab_data, invalid_path)

        assert result is False
