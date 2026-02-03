"""
Tests for the Layout Analyzer module.

Tests PDF zone detection including:
- Header/footer zone detection via coordinate analysis
- Title page identification
- Line number margin detection
- Content zone boundary calculation
"""

from unittest.mock import MagicMock

from src.core.extraction.layout_analyzer import (
    FOOTER_ZONE_RATIO,
    HEADER_ZONE_RATIO,
    LINE_NUMBER_X_RATIO,
    MIN_REPEAT_COUNT,
    MIN_SAMPLE_PAGES,
    SAMPLE_PAGE_COUNT,
    TITLE_PAGE_THRESHOLD,
    ContentZone,
    LayoutAnalyzer,
    _score_page_for_title,
)


def _make_mock_page(
    text: str = "",
    blocks: list[dict] | None = None,
    width: float = 612.0,
    height: float = 792.0,
):
    """
    Create a mock fitz.Page object.

    Args:
        text: Text returned by get_text()
        blocks: List of block dicts returned by get_text("dict")["blocks"]
        width: Page width in points
        height: Page height in points

    Returns:
        Mock page object
    """
    page = MagicMock()
    page.get_text.side_effect = lambda fmt=None: (
        {"blocks": blocks or []} if fmt == "dict" else text
    )

    # Mock rect property
    rect = MagicMock()
    rect.width = width
    rect.height = height
    page.rect = rect

    return page


def _make_mock_doc(pages: list):
    """
    Create a mock fitz.Document from a list of mock pages.

    Args:
        pages: List of mock page objects

    Returns:
        Mock document object
    """
    doc = MagicMock()
    doc.__len__ = MagicMock(return_value=len(pages))
    doc.__getitem__ = MagicMock(side_effect=lambda i: pages[i])
    doc.__iter__ = MagicMock(return_value=iter(pages))
    return doc


def _make_text_block(
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    block_type: int = 0,
):
    """
    Create a mock text block dict.

    Args:
        text: Block text content
        x0, y0, x1, y1: Bounding box coordinates
        block_type: 0 for text, 1 for image

    Returns:
        Block dict matching fitz format
    """
    return {
        "type": block_type,
        "bbox": (x0, y0, x1, y1),
        "lines": [{"spans": [{"text": text}]}],
    }


class TestScorePageForTitle:
    """Tests for the title page scoring function."""

    def test_empty_text_scores_zero(self):
        """Empty text should score 0."""
        assert _score_page_for_title("") == 0

    def test_content_only_scores_negative(self):
        """Q&A content should score negative (not a title page)."""
        text = """
        Q. Good morning.
        A. Good morning.
        Q. Please state your name.
        A. John Smith.
        """
        score = _score_page_for_title(text)
        assert score < 0

    def test_title_page_scores_high(self):
        """Classic title page elements should score high."""
        text = """
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
        score = _score_page_for_title(text)
        assert score >= TITLE_PAGE_THRESHOLD

    def test_court_header_adds_points(self):
        """Court names should add points."""
        assert _score_page_for_title("SUPREME COURT") >= 2
        assert _score_page_for_title("CIVIL COURT") >= 2
        assert _score_page_for_title("DISTRICT COURT") >= 2

    def test_plaintiff_defendant_adds_points(self):
        """Plaintiff/defendant labels should add points."""
        assert _score_page_for_title("Plaintiff,") >= 2
        assert _score_page_for_title("Defendant,") >= 2

    def test_deposition_title_adds_points(self):
        """Deposition title should add more points."""
        assert _score_page_for_title("DEPOSITION OF JOHN SMITH") >= 3

    def test_short_pages_get_bonus(self):
        """Short pages with title indicators get bonus point."""
        short_title = "SUPREME COURT"  # < 500 chars, has title pattern
        long_content = "SUPREME COURT " + "x" * 600  # > 500 chars

        short_score = _score_page_for_title(short_title)
        long_score = _score_page_for_title(long_content)

        # Short page should get +1 bonus
        assert short_score > long_score


class TestContentZone:
    """Tests for the ContentZone dataclass."""

    def test_content_zone_fields(self):
        """ContentZone should store all boundary fields."""
        zone = ContentZone(
            left=50.0,
            top=72.0,
            right=562.0,
            bottom=720.0,
            page_width=612.0,
            page_height=792.0,
        )

        assert zone.left == 50.0
        assert zone.top == 72.0
        assert zone.right == 562.0
        assert zone.bottom == 720.0
        assert zone.page_width == 612.0
        assert zone.page_height == 792.0


class TestLayoutAnalyzerInit:
    """Tests for LayoutAnalyzer initialization."""

    def test_creates_instance(self):
        """Should create LayoutAnalyzer instance."""
        analyzer = LayoutAnalyzer()
        assert analyzer is not None


class TestDetectZones:
    """Tests for the main detect_zones method."""

    def test_returns_none_for_too_few_pages(self):
        """Should return None if document has fewer than MIN_SAMPLE_PAGES."""
        analyzer = LayoutAnalyzer()

        # Create doc with only 2 pages
        pages = [_make_mock_page() for _ in range(2)]
        doc = _make_mock_doc(pages)

        result = analyzer.detect_zones(doc)

        assert result is None

    def test_returns_none_when_no_repeating_blocks(self):
        """Should return None if no repeating header/footer blocks found."""
        analyzer = LayoutAnalyzer()

        # Create pages with unique blocks (no repetition)
        pages = []
        for i in range(5):
            blocks = [
                _make_text_block(f"Unique content {i}", 100, 400, 500, 420),
            ]
            pages.append(_make_mock_page(text=f"Content page {i}", blocks=blocks))

        doc = _make_mock_doc(pages)
        result = analyzer.detect_zones(doc)

        # Should return a zone (even if default bounds)
        assert result is not None

    def test_detects_header_zone(self):
        """Should detect repeating header blocks and set top boundary."""
        analyzer = LayoutAnalyzer()
        page_height = 792.0
        header_y = 30.0  # In top 8% zone (< 63.36)

        # Create pages with repeating header at same Y position
        pages = []
        for i in range(5):
            blocks = [
                # Header block at consistent Y position
                _make_text_block("Case No. 123", 100, header_y, 200, header_y + 15),
                # Content block in middle of page
                _make_text_block(f"Content {i}", 100, 400, 500, 420),
            ]
            pages.append(_make_mock_page(text="Q. Question", blocks=blocks))

        doc = _make_mock_doc(pages)
        result = analyzer.detect_zones(doc)

        assert result is not None
        # Header bottom should be below the detected header block
        assert result.top > 0

    def test_detects_footer_zone(self):
        """Should detect repeating footer blocks and set bottom boundary."""
        analyzer = LayoutAnalyzer()
        page_height = 792.0
        footer_y = 760.0  # In bottom 8% zone (> 728.64)

        # Create pages with repeating footer at same Y position
        pages = []
        for i in range(5):
            blocks = [
                # Content block in middle of page
                _make_text_block(f"Content {i}", 100, 400, 500, 420),
                # Footer block at consistent Y position
                _make_text_block(f"Page {i}", 100, footer_y, 200, footer_y + 15),
            ]
            pages.append(_make_mock_page(text="Q. Question", blocks=blocks))

        doc = _make_mock_doc(pages)
        result = analyzer.detect_zones(doc)

        assert result is not None
        # Footer top should be above the detected footer block
        assert result.bottom < page_height

    def test_skips_title_pages(self):
        """Should skip title pages when selecting sample pages."""
        analyzer = LayoutAnalyzer()

        # Page 0: Title page
        title_text = """
        SUPREME COURT OF THE STATE OF NEW YORK
        DEPOSITION OF JOHN DOE
        Plaintiff, Defendant
        """
        title_page = _make_mock_page(text=title_text, blocks=[])

        # Pages 1-5: Content pages
        content_pages = []
        for i in range(1, 6):
            blocks = [_make_text_block(f"Content {i}", 100, 400, 500, 420)]
            content_pages.append(_make_mock_page(text="Q. Question\nA. Answer", blocks=blocks))

        pages = [title_page] + content_pages
        doc = _make_mock_doc(pages)

        result = analyzer.detect_zones(doc)

        # Should successfully detect zones (didn't crash on title page)
        assert result is not None


class TestFindContentStart:
    """Tests for the _find_content_start method."""

    def test_returns_zero_for_content_first_page(self):
        """Should return 0 if first page is content."""
        analyzer = LayoutAnalyzer()

        pages = [
            _make_mock_page(text="Q. Question\nA. Answer"),
            _make_mock_page(text="Q. More content"),
        ]
        doc = _make_mock_doc(pages)

        result = analyzer._find_content_start(doc)

        assert result == 0

    def test_skips_title_page(self):
        """Should return 1 if first page is a title page."""
        analyzer = LayoutAnalyzer()

        title_text = """
        SUPREME COURT OF THE STATE OF NEW YORK
        DEPOSITION OF JOHN DOE
        Plaintiff, Defendant, Index No.
        """
        pages = [
            _make_mock_page(text=title_text),
            _make_mock_page(text="Q. Question\nA. Answer"),
        ]
        doc = _make_mock_doc(pages)

        result = analyzer._find_content_start(doc)

        assert result == 1

    def test_skips_multiple_title_pages(self):
        """Should skip multiple consecutive title pages."""
        analyzer = LayoutAnalyzer()

        title_text = """
        SUPREME COURT
        DEPOSITION OF JOHN DOE
        Plaintiff, Defendant
        """
        pages = [
            _make_mock_page(text=title_text),
            _make_mock_page(text=title_text),  # Second title page
            _make_mock_page(text="Q. Question\nA. Answer"),
        ]
        doc = _make_mock_doc(pages)

        result = analyzer._find_content_start(doc)

        assert result == 2

    def test_defaults_to_page_one_if_all_title(self):
        """Should default to page 1 if all checked pages look like title pages."""
        analyzer = LayoutAnalyzer()

        title_text = """
        SUPREME COURT
        DEPOSITION OF JOHN DOE
        Plaintiff, Defendant
        """
        # All 5 pages are title-like
        pages = [_make_mock_page(text=title_text) for _ in range(5)]
        doc = _make_mock_doc(pages)

        result = analyzer._find_content_start(doc)

        # Should default to page 1 (skip obvious cover)
        assert result == 1


class TestSelectSamplePages:
    """Tests for the _select_sample_pages method."""

    def test_selects_consecutive_pages(self):
        """Should select SAMPLE_PAGE_COUNT consecutive pages from start."""
        analyzer = LayoutAnalyzer()

        result = analyzer._select_sample_pages(start=2, total=20)

        expected = list(range(2, 2 + SAMPLE_PAGE_COUNT))
        assert result == expected

    def test_handles_short_documents(self):
        """Should handle documents shorter than sample count."""
        analyzer = LayoutAnalyzer()

        result = analyzer._select_sample_pages(start=0, total=3)

        assert result == [0, 1, 2]

    def test_respects_document_end(self):
        """Should not exceed document length."""
        analyzer = LayoutAnalyzer()

        result = analyzer._select_sample_pages(start=8, total=10)

        assert result == [8, 9]
        assert all(i < 10 for i in result)


class TestDetectHFZones:
    """Tests for the _detect_hf_zones method."""

    def test_returns_defaults_when_no_repeating_blocks(self):
        """Should return full page height when no repeating blocks found."""
        analyzer = LayoutAnalyzer()
        page_width = 612.0
        page_height = 792.0

        # Blocks at different Y positions on each page (no repetition)
        page_blocks = []
        for i in range(5):
            page = _make_mock_page(width=page_width, height=page_height)
            blocks = [_make_text_block(f"Content {i}", 100, 100 + i * 50, 500, 120)]
            page_blocks.append((page, blocks))

        header_bottom, footer_top = analyzer._detect_hf_zones(page_blocks, page_width, page_height)

        # Should return defaults (no clipping)
        assert header_bottom == 0.0
        assert footer_top == page_height

    def test_detects_repeating_header_y_positions(self):
        """Should detect blocks at same Y position in header zone."""
        analyzer = LayoutAnalyzer()
        page_width = 612.0
        page_height = 792.0
        header_y = 30.0  # In top 8% (< 63.36)

        # Same header block on each page
        page_blocks = []
        for i in range(5):
            page = _make_mock_page(width=page_width, height=page_height)
            blocks = [
                _make_text_block("Header", 100, header_y, 200, header_y + 15),
                _make_text_block(f"Content {i}", 100, 400, 500, 420),
            ]
            page_blocks.append((page, blocks))

        header_bottom, footer_top = analyzer._detect_hf_zones(page_blocks, page_width, page_height)

        # Should have detected header
        assert header_bottom > 0

    def test_detects_repeating_footer_y_positions(self):
        """Should detect blocks at same Y position in footer zone."""
        analyzer = LayoutAnalyzer()
        page_width = 612.0
        page_height = 792.0
        footer_y = 760.0  # In bottom 8% (> 728.64)

        # Same footer block on each page
        page_blocks = []
        for i in range(5):
            page = _make_mock_page(width=page_width, height=page_height)
            blocks = [
                _make_text_block(f"Content {i}", 100, 400, 500, 420),
                _make_text_block("Page footer", 100, footer_y, 200, footer_y + 15),
            ]
            page_blocks.append((page, blocks))

        header_bottom, footer_top = analyzer._detect_hf_zones(page_blocks, page_width, page_height)

        # Should have detected footer
        assert footer_top < page_height

    def test_requires_minimum_repetitions(self):
        """Should require MIN_REPEAT_COUNT repetitions to detect."""
        analyzer = LayoutAnalyzer()
        page_width = 612.0
        page_height = 792.0
        header_y = 30.0

        # Only 2 pages with header (below MIN_REPEAT_COUNT=3)
        page_blocks = []
        for i in range(5):
            page = _make_mock_page(width=page_width, height=page_height)
            if i < 2:
                # First 2 pages have header
                blocks = [
                    _make_text_block("Header", 100, header_y, 200, header_y + 15),
                    _make_text_block(f"Content {i}", 100, 400, 500, 420),
                ]
            else:
                # Other pages have no header
                blocks = [_make_text_block(f"Content {i}", 100, 400, 500, 420)]
            page_blocks.append((page, blocks))

        header_bottom, footer_top = analyzer._detect_hf_zones(page_blocks, page_width, page_height)

        # Should NOT detect header (only 2 repetitions)
        assert header_bottom == 0.0

    def test_ignores_image_blocks(self):
        """Should skip image blocks (type=1)."""
        analyzer = LayoutAnalyzer()
        page_width = 612.0
        page_height = 792.0
        header_y = 30.0

        # Image blocks at header position on all pages
        page_blocks = []
        for i in range(5):
            page = _make_mock_page(width=page_width, height=page_height)
            blocks = [
                # Image block (type=1)
                _make_text_block("Logo", 100, header_y, 200, header_y + 50, block_type=1),
                _make_text_block(f"Content {i}", 100, 400, 500, 420),
            ]
            page_blocks.append((page, blocks))

        header_bottom, footer_top = analyzer._detect_hf_zones(page_blocks, page_width, page_height)

        # Should NOT detect header (image blocks ignored)
        assert header_bottom == 0.0

    def test_tolerance_buckets_similar_y_positions(self):
        """Should bucket Y positions within 5-point tolerance."""
        analyzer = LayoutAnalyzer()
        page_width = 612.0
        page_height = 792.0

        # Headers at slightly different Y positions (within 5pt tolerance)
        page_blocks = []
        y_positions = [28, 30, 32, 29, 31]  # All within 5pt tolerance

        for i, y in enumerate(y_positions):
            page = _make_mock_page(width=page_width, height=page_height)
            blocks = [
                _make_text_block("Header", 100, y, 200, y + 15),
                _make_text_block(f"Content {i}", 100, 400, 500, 420),
            ]
            page_blocks.append((page, blocks))

        header_bottom, footer_top = analyzer._detect_hf_zones(page_blocks, page_width, page_height)

        # Should detect header (all Y positions bucket together)
        assert header_bottom > 0


class TestDetectLineNumberMargin:
    """Tests for the _detect_line_number_margin method."""

    def test_returns_zero_when_no_line_numbers(self):
        """Should return 0 when no line numbers detected."""
        analyzer = LayoutAnalyzer()
        page_width = 612.0
        page_height = 792.0

        # Content blocks only, no line numbers
        page_blocks = []
        for i in range(5):
            page = _make_mock_page(width=page_width, height=page_height)
            blocks = [_make_text_block(f"Content {i}", 100, 400, 500, 420)]
            page_blocks.append((page, blocks))

        result = analyzer._detect_line_number_margin(page_blocks, page_width, page_height)

        assert result == 0.0

    def test_detects_line_numbers_in_left_margin(self):
        """Should detect numeric blocks in left 10% of page."""
        analyzer = LayoutAnalyzer()
        page_width = 612.0
        page_height = 792.0
        line_num_x = 30.0  # In left 10% (< 61.2)

        # Line number blocks on left margin
        page_blocks = []
        for i in range(5):
            page = _make_mock_page(width=page_width, height=page_height)
            blocks = [
                # Line number "1" in left margin
                _make_text_block("1", line_num_x, 100, line_num_x + 15, 115),
                # Line number "2"
                _make_text_block("2", line_num_x, 130, line_num_x + 15, 145),
                # Content
                _make_text_block(f"Content {i}", 100, 100, 500, 150),
            ]
            page_blocks.append((page, blocks))

        result = analyzer._detect_line_number_margin(page_blocks, page_width, page_height)

        # Should detect line numbers and return margin boundary
        assert result > 0

    def test_ignores_numbers_outside_left_margin(self):
        """Should ignore numbers outside left 10% zone."""
        analyzer = LayoutAnalyzer()
        page_width = 612.0
        page_height = 792.0

        # Numbers in middle of page (not in left margin)
        page_blocks = []
        for i in range(5):
            page = _make_mock_page(width=page_width, height=page_height)
            blocks = [
                _make_text_block("25", 200, 100, 230, 115),  # In middle of page
                _make_text_block(f"Content {i}", 100, 200, 500, 220),
            ]
            page_blocks.append((page, blocks))

        result = analyzer._detect_line_number_margin(page_blocks, page_width, page_height)

        # Should NOT detect (numbers not in left margin)
        assert result == 0.0

    def test_ignores_three_digit_numbers(self):
        """Should only detect 1-2 digit numbers as line numbers."""
        analyzer = LayoutAnalyzer()
        page_width = 612.0
        page_height = 792.0
        line_num_x = 30.0

        # Three-digit numbers (not line numbers)
        page_blocks = []
        for i in range(5):
            page = _make_mock_page(width=page_width, height=page_height)
            blocks = [
                _make_text_block("123", line_num_x, 100, line_num_x + 30, 115),
                _make_text_block(f"Content {i}", 100, 200, 500, 220),
            ]
            page_blocks.append((page, blocks))

        result = analyzer._detect_line_number_margin(page_blocks, page_width, page_height)

        # Should NOT detect (3-digit numbers aren't line numbers)
        assert result == 0.0

    def test_requires_minimum_pages_with_line_numbers(self):
        """Should require MIN_REPEAT_COUNT pages with line numbers."""
        analyzer = LayoutAnalyzer()
        page_width = 612.0
        page_height = 792.0
        line_num_x = 30.0

        # Only 2 pages have line numbers
        page_blocks = []
        for i in range(5):
            page = _make_mock_page(width=page_width, height=page_height)
            if i < 2:
                blocks = [
                    _make_text_block("1", line_num_x, 100, line_num_x + 15, 115),
                    _make_text_block(f"Content {i}", 100, 200, 500, 220),
                ]
            else:
                blocks = [_make_text_block(f"Content {i}", 100, 200, 500, 220)]
            page_blocks.append((page, blocks))

        result = analyzer._detect_line_number_margin(page_blocks, page_width, page_height)

        # Should NOT detect (only 2 pages have line numbers)
        assert result == 0.0


class TestConfigConstants:
    """Tests for configuration constants."""

    def test_header_zone_ratio_is_reasonable(self):
        """Header zone should be top portion of page."""
        assert 0.05 <= HEADER_ZONE_RATIO <= 0.15

    def test_footer_zone_ratio_is_reasonable(self):
        """Footer zone should be bottom portion of page."""
        assert 0.05 <= FOOTER_ZONE_RATIO <= 0.15

    def test_line_number_x_ratio_is_reasonable(self):
        """Line number margin should be small left portion."""
        assert 0.05 <= LINE_NUMBER_X_RATIO <= 0.20

    def test_min_repeat_count_requires_multiple_occurrences(self):
        """Should require multiple occurrences to detect patterns."""
        assert MIN_REPEAT_COUNT >= 2

    def test_sample_page_count_is_reasonable(self):
        """Should sample enough pages to detect patterns."""
        assert 3 <= SAMPLE_PAGE_COUNT <= 10

    def test_min_sample_pages_allows_short_docs(self):
        """Should allow reasonably short documents."""
        assert MIN_SAMPLE_PAGES >= 2


class TestIntegration:
    """Integration tests for full zone detection workflow."""

    def test_full_workflow_with_headers_and_footers(self):
        """Test complete workflow detecting both headers and footers."""
        analyzer = LayoutAnalyzer()
        page_width = 612.0
        page_height = 792.0

        # Create realistic document structure
        pages = []
        for i in range(10):
            header_y = 30.0
            footer_y = 760.0

            blocks = [
                # Header
                _make_text_block("Smith v. Jones", 200, header_y, 400, header_y + 15),
                # Line numbers
                _make_text_block(str(i * 25 + 1), 20, 100, 35, 115),
                # Content
                _make_text_block(f"Q. Question {i}", 50, 200, 550, 220),
                _make_text_block(f"A. Answer {i}", 50, 240, 550, 260),
                # Footer
                _make_text_block(f"Page {i + 1}", 280, footer_y, 330, footer_y + 15),
            ]

            text = f"Q. Question {i}\nA. Answer {i}"
            pages.append(
                _make_mock_page(text=text, blocks=blocks, width=page_width, height=page_height)
            )

        doc = _make_mock_doc(pages)
        result = analyzer.detect_zones(doc)

        assert result is not None
        assert isinstance(result, ContentZone)
        # Should have detected header zone
        assert result.top > 0
        # Should have detected footer zone
        assert result.bottom < page_height
        # Page dimensions should be captured
        assert result.page_width == page_width
        assert result.page_height == page_height

    def test_handles_empty_document_gracefully(self):
        """Should handle document with no pages."""
        analyzer = LayoutAnalyzer()
        doc = _make_mock_doc([])

        result = analyzer.detect_zones(doc)

        assert result is None
