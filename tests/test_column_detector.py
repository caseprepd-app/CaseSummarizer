"""Tests for multi-column PDF page detection and extraction."""

from unittest.mock import MagicMock

from src.core.extraction.column_detector import (
    MIN_BLOCKS_PER_COLUMN,
    MIN_COLUMN_GAP_RATIO,
    _cluster_x_positions,
    _is_multi_column,
    extract_page_text,
)


class TestClusterXPositions:
    """Tests for column clustering by x-position."""

    def test_empty_blocks(self):
        assert _cluster_x_positions([], page_width=612) == []

    def test_single_column(self):
        """Blocks close together should form one column."""
        # All blocks centered near x=200 on a 612pt-wide page
        blocks = [
            (180, 100, 220, 120, "line 1", 0, 0),
            (180, 130, 220, 150, "line 2", 1, 0),
            (180, 160, 220, 180, "line 3", 2, 0),
        ]
        columns = _cluster_x_positions(blocks, page_width=612)
        assert len(columns) == 1
        assert len(columns[0]) == 3

    def test_two_columns(self):
        """Blocks in two distinct x-regions should form two columns."""
        page_width = 612
        gap = page_width * MIN_COLUMN_GAP_RATIO + 10  # Exceed threshold

        left_blocks = [
            (50, 100, 100, 120, "left 1", 0, 0),
            (50, 130, 100, 150, "left 2", 1, 0),
            (50, 160, 100, 180, "left 3", 2, 0),
        ]
        right_blocks = [
            (50 + gap, 100, 100 + gap, 120, "right 1", 3, 0),
            (50 + gap, 130, 100 + gap, 150, "right 2", 4, 0),
            (50 + gap, 160, 100 + gap, 180, "right 3", 5, 0),
        ]
        columns = _cluster_x_positions(left_blocks + right_blocks, page_width)
        assert len(columns) == 2
        assert len(columns[0]) == 3  # left column
        assert len(columns[1]) == 3  # right column

    def test_columns_ordered_left_to_right(self):
        """Column groups should be in left-to-right order."""
        page_width = 612
        gap = page_width * MIN_COLUMN_GAP_RATIO + 10

        blocks = [
            (300 + gap, 100, 350 + gap, 120, "right first", 0, 0),
            (50, 100, 100, 120, "left first", 1, 0),
        ]
        columns = _cluster_x_positions(blocks, page_width)
        assert len(columns) == 2
        # Left column should be first
        assert "left first" in columns[0][0][4]


class TestIsMultiColumn:
    """Tests for multi-column detection threshold."""

    def test_single_column_not_multi(self):
        assert _is_multi_column([[(0,) * 7] * 10]) is False

    def test_two_significant_non_overlapping_columns(self):
        """Two significant columns with non-overlapping x-ranges → multi-column."""
        # Left column blocks: x0=50, x1=280
        col1 = [
            (50, 100 + i * 30, 280, 120 + i * 30, f"L{i}", i, 0)
            for i in range(MIN_BLOCKS_PER_COLUMN)
        ]
        # Right column blocks: x0=320, x1=560 (no overlap with left)
        col2 = [
            (320, 100 + i * 30, 560, 120 + i * 30, f"R{i}", i + 10, 0)
            for i in range(MIN_BLOCKS_PER_COLUMN)
        ]
        assert _is_multi_column([col1, col2]) is True

    def test_one_column_too_small(self):
        """A column with fewer than MIN_BLOCKS_PER_COLUMN isn't significant."""
        col1 = [(0,) * 7] * MIN_BLOCKS_PER_COLUMN
        col2 = [(0,) * 7] * (MIN_BLOCKS_PER_COLUMN - 1)
        assert _is_multi_column([col1, col2]) is False


class TestExtractPageText:
    """Tests for the main extract_page_text function."""

    def _make_page(self, blocks, width=612, height=792):
        """Create a mock fitz.Page with given text blocks."""
        page = MagicMock()
        page.number = 0
        page.rect = MagicMock()
        page.rect.width = width
        page.rect.height = height

        # get_text("blocks") returns the block tuples
        # get_text(sort=True) returns sorted text
        def mock_get_text(opt=None, clip=None, sort=False, flags=0):
            if opt == "blocks":
                if clip:
                    return [b for b in blocks if b[6] == 0]
                return blocks
            # For sort=True text extraction
            text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
            text_blocks.sort(key=lambda b: (b[1], b[0]))
            return "\n".join(b[4].strip() for b in text_blocks)

        page.get_text = mock_get_text
        return page

    def test_empty_page(self):
        page = self._make_page([])
        assert extract_page_text(page) == ""

    def test_single_column_uses_sort(self):
        """Single-column pages should use sort=True extraction."""
        blocks = [
            (50, 200, 200, 220, "second line\n", 1, 0),
            (50, 100, 200, 120, "first line\n", 0, 0),
            (50, 300, 200, 320, "third line\n", 2, 0),
        ]
        page = self._make_page(blocks)
        result = extract_page_text(page)
        # sort=True should order by y position
        assert result.index("first") < result.index("second")
        assert result.index("second") < result.index("third")

    def test_multi_column_reads_left_then_right(self):
        """Multi-column should read left column fully before right column."""
        page_width = 612
        gap = page_width * MIN_COLUMN_GAP_RATIO + 50

        blocks = []
        # Left column: 4 blocks (exceeds MIN_BLOCKS_PER_COLUMN)
        for i in range(4):
            y = 100 + i * 30
            blocks.append((50, y, 150, y + 20, f"L{i + 1}", i, 0))
        # Right column: 4 blocks
        for i in range(4):
            y = 100 + i * 30
            blocks.append((50 + gap, y, 150 + gap, y + 20, f"R{i + 1}", i + 4, 0))

        page = self._make_page(blocks, width=page_width)
        result = extract_page_text(page)

        # Left column should appear before right column
        assert result.index("L1") < result.index("R1")
        assert result.index("L4") < result.index("R1")
        # Within each column, order should be top-to-bottom
        assert result.index("L1") < result.index("L2")
        assert result.index("R1") < result.index("R2")

    def test_image_blocks_filtered(self):
        """Image blocks (type=1) should be excluded."""
        blocks = [
            (50, 100, 200, 120, "text line\n", 0, 0),
            (50, 200, 200, 220, "image data", 1, 1),  # type 1 = image
        ]
        page = self._make_page(blocks)
        result = extract_page_text(page)
        assert "text line" in result
        assert "image data" not in result

    def test_clip_rect_passed_through(self):
        """Clip rectangle should be passed to get_text calls."""
        import fitz

        blocks = [
            (50, 100, 200, 120, "clipped text\n", 0, 0),
        ]
        page = self._make_page(blocks)
        clip = fitz.Rect(40, 90, 300, 400)
        result = extract_page_text(page, clip=clip)
        assert "clipped text" in result

    def test_transcript_indented_text_is_single_column(self):
        """Legal transcript with speaker labels + indented body → single column.

        Speaker labels at x=72-200, body text at x=200-550. The body text
        x-range overlaps with the label x-range, so this should NOT be
        detected as multi-column.
        """
        page_width = 612
        blocks = []
        # Speaker labels (narrow, left-aligned) — centers near x=136
        for i in range(4):
            y = 100 + i * 60
            blocks.append((72, y, 200, y + 15, f"SPEAKER {i}:", i, 0))
        # Body text (wide, indented but overlapping label x-range)
        # In real transcripts, body blocks often start inside the label region
        for i in range(4):
            y = 115 + i * 60
            blocks.append((180, y, 550, y + 15, f"Body text line {i}", i + 10, 0))

        page = self._make_page(blocks, width=page_width)
        result = extract_page_text(page)
        # Should use sort=True (single-column), so reading order is by y
        assert result.index("SPEAKER 0") < result.index("Body text line 0")
        assert result.index("Body text line 0") < result.index("SPEAKER 1")

    def test_true_multi_column_still_detected(self):
        """True multi-column (non-overlapping x-ranges) still works."""
        page_width = 612
        blocks = []
        # Left column: x=50-280
        for i in range(4):
            y = 100 + i * 30
            blocks.append((50, y, 280, y + 20, f"L{i + 1}", i, 0))
        # Right column: x=320-560 (clear gap, no overlap)
        for i in range(4):
            y = 100 + i * 30
            blocks.append((320, y, 560, y + 20, f"R{i + 1}", i + 4, 0))

        page = self._make_page(blocks, width=page_width)
        result = extract_page_text(page)
        # Multi-column: all left before any right
        assert result.index("L4") < result.index("R1")


class TestColumnsOverlapX:
    """Tests for the _columns_overlap_x helper."""

    def test_non_overlapping_columns(self):
        """Columns with clear gap between x-ranges → no overlap."""
        from src.core.extraction.column_detector import _columns_overlap_x

        col1 = [(50, 100, 280, 120, "a", 0, 0), (50, 130, 280, 150, "b", 1, 0)]
        col2 = [(320, 100, 560, 120, "c", 2, 0), (320, 130, 560, 150, "d", 3, 0)]
        assert _columns_overlap_x([col1, col2]) is False

    def test_overlapping_columns(self):
        """Body text x1 extends past next column's x0 → overlap."""
        from src.core.extraction.column_detector import _columns_overlap_x

        col1 = [(72, 100, 200, 120, "label", 0, 0), (72, 130, 200, 150, "label2", 1, 0)]
        col2 = [(200, 100, 550, 120, "body", 2, 0), (200, 130, 550, 150, "body2", 3, 0)]
        # col1 max x1 = 200, col2 min x0 = 200 → NOT overlapping (equal, not greater)
        assert _columns_overlap_x([col1, col2]) is False

    def test_overlapping_columns_body_spans_past(self):
        """Left column block extends into right column's x-range."""
        from src.core.extraction.column_detector import _columns_overlap_x

        col1 = [(72, 100, 350, 120, "wide block", 0, 0)]
        col2 = [(300, 100, 550, 120, "right block", 1, 0)]
        # col1 max x1=350 > col2 min x0=300 → overlap
        assert _columns_overlap_x([col1, col2]) is True

    def test_empty_columns(self):
        """Single column list → no adjacent pairs → no overlap."""
        from src.core.extraction.column_detector import _columns_overlap_x

        col1 = [(50, 100, 280, 120, "a", 0, 0)]
        assert _columns_overlap_x([col1]) is False
