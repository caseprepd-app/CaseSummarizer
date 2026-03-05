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

    def test_two_significant_columns(self):
        col1 = [(0,) * 7] * MIN_BLOCKS_PER_COLUMN
        col2 = [(0,) * 7] * MIN_BLOCKS_PER_COLUMN
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
