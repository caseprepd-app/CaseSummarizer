"""
Layout-Aware PDF Zone Detection.

Analyzes PDF page structure using PyMuPDF block coordinates to identify
header/footer zones and content boundaries. This enables extraction-time
filtering of boilerplate (headers, footers, page numbers, line numbers)
rather than post-extraction regex cleanup.

Example usage:
    >>> import fitz
    >>> analyzer = LayoutAnalyzer()
    >>> doc = fitz.open("transcript.pdf")
    >>> zones = analyzer.detect_zones(doc)
    >>> print(f"Content zone: y={zones.header_bottom:.0f} to y={zones.footer_top:.0f}")
"""

import logging
import re
from dataclasses import dataclass

import fitz

logger = logging.getLogger(__name__)

# Title page scoring patterns (duplicated from TitlePageRemover to avoid
# backward dependency — extraction must not import from preprocessing)
_TITLE_PATTERNS = [
    (re.compile(r"SUPREME\s+COURT", re.IGNORECASE), 2),
    (re.compile(r"CIVIL\s+COURT", re.IGNORECASE), 2),
    (re.compile(r"DISTRICT\s+COURT", re.IGNORECASE), 2),
    (re.compile(r"COURT\s+OF\s+(?:THE\s+)?STATE", re.IGNORECASE), 2),
    (re.compile(r"PLAINTIFF[,\s]", re.IGNORECASE), 2),
    (re.compile(r"DEFENDANT[,\s]", re.IGNORECASE), 2),
    (re.compile(r"DEPOSITION\s+OF\s+[A-Z]", re.IGNORECASE), 3),
    (re.compile(r"EXAMINATION\s+BEFORE\s+TRIAL", re.IGNORECASE), 3),
    (re.compile(r"INDEX\s*(?:NO\.?|NUMBER)", re.IGNORECASE), 2),
    (re.compile(r"CASE\s*(?:NO\.?|NUMBER)", re.IGNORECASE), 2),
    (re.compile(r"ATTORNEY[S]?\s+FOR", re.IGNORECASE), 2),
    (re.compile(r"APPEARANCES?:", re.IGNORECASE), 2),
    (re.compile(r"COURT\s+REPORTER", re.IGNORECASE), 2),
]

_CONTENT_PATTERNS = [
    (re.compile(r"^\s*Q[\.:]", re.MULTILINE), -3),
    (re.compile(r"^\s*A[\.:]", re.MULTILINE), -3),
    (re.compile(r"THE\s+WITNESS:", re.IGNORECASE), -2),
    (re.compile(r"BY\s+(?:MR\.|MS\.|MRS\.)", re.IGNORECASE), -1),
]

TITLE_PAGE_THRESHOLD = 4
MIN_SAMPLE_PAGES = 3
SAMPLE_PAGE_COUNT = 5
HEADER_ZONE_RATIO = 0.08
FOOTER_ZONE_RATIO = 0.08
LINE_NUMBER_X_RATIO = 0.10
MIN_REPEAT_COUNT = 3


@dataclass
class ContentZone:
    """
    Defines the rectangular content area on a page.

    Coordinates are in PDF points (72 points = 1 inch).

    Attributes:
        left: Left boundary X coordinate
        top: Top boundary Y coordinate (below header zone)
        right: Right boundary X coordinate
        bottom: Bottom boundary Y coordinate (above footer zone)
        page_width: Full page width for reference
        page_height: Full page height for reference
    """

    left: float
    top: float
    right: float
    bottom: float
    page_width: float
    page_height: float


def _score_page_for_title(page_text: str) -> int:
    """
    Score a page for title page characteristics.

    Uses the same patterns as TitlePageRemover but without importing it.

    Args:
        page_text: Text content of a single page

    Returns:
        Integer score; >= TITLE_PAGE_THRESHOLD means title page
    """
    score = 0
    for pattern, points in _TITLE_PATTERNS:
        if pattern.search(page_text):
            score += points
    for pattern, points in _CONTENT_PATTERNS:
        if pattern.search(page_text):
            score += points
    if len(page_text.strip()) < 500 and score > 0:
        score += 1
    return score


class LayoutAnalyzer:
    """
    Detects header/footer zones and content boundaries in PDF documents.

    Samples representative content pages (skipping title pages) and uses
    block coordinates to find repeating elements at consistent Y positions,
    which indicate headers, footers, and margin annotations.
    """

    def detect_zones(self, doc: fitz.Document) -> ContentZone | None:
        """
        Analyze a PDF document and return the content zone rectangle.

        Pipeline:
            1. Find first content page (skip title pages)
            2. Sample consecutive pages from that point
            3. Extract block coordinates from sample pages
            4. Identify repeating header/footer blocks by Y position
            5. Detect left-margin line numbers by X position
            6. Return content zone excluding detected boilerplate

        Args:
            doc: An open fitz.Document

        Returns:
            ContentZone with content boundaries, or None if detection fails
            (e.g., too few pages, no repeating blocks found)
        """
        page_count = len(doc)
        if page_count < MIN_SAMPLE_PAGES:
            logger.debug("Layout: Too few pages (%d) for zone detection", page_count)
            return None

        # Step 1: Find first content page
        content_start = self._find_content_start(doc)
        logger.debug("Layout: First content page index: %d", content_start)

        # Step 2: Select sample pages
        sample_indices = self._select_sample_pages(content_start, page_count)
        if len(sample_indices) < MIN_SAMPLE_PAGES:
            logger.debug("Layout: Not enough sample pages (%d)", len(sample_indices))
            return None

        logger.debug("Layout: Sampling pages %s", sample_indices)

        # Step 3: Extract blocks from sample pages
        page_blocks = []
        for idx in sample_indices:
            page = doc[idx]
            blocks = page.get_text("dict")["blocks"]
            page_blocks.append((page, blocks))

        if not page_blocks:
            return None

        # Get page dimensions from first sample page
        first_page = page_blocks[0][0]
        page_width = first_page.rect.width
        page_height = first_page.rect.height

        # Step 4: Detect header/footer zones
        header_bottom, footer_top = self._detect_hf_zones(page_blocks, page_width, page_height)

        # Step 5: Detect left-margin line numbers
        left_margin = self._detect_line_number_margin(page_blocks, page_width, page_height)

        zone = ContentZone(
            left=left_margin,
            top=header_bottom,
            right=page_width,
            bottom=footer_top,
            page_width=page_width,
            page_height=page_height,
        )

        logger.info(
            "Layout: Content zone detected — "
            "left=%.0f, top=%.0f, right=%.0f, bottom=%.0f "
            "(page: %.0f x %.0f)",
            zone.left,
            zone.top,
            zone.right,
            zone.bottom,
            page_width,
            page_height,
        )

        return zone

    def _find_content_start(self, doc: fitz.Document) -> int:
        """
        Find the index of the first non-title page.

        Scores each page starting from page 0. Returns the index of the
        first page that scores below TITLE_PAGE_THRESHOLD.

        Args:
            doc: An open fitz.Document

        Returns:
            Page index (0-based) of first content page
        """
        max_check = min(len(doc), 5)
        for i in range(max_check):
            page_text = doc[i].get_text()
            score = _score_page_for_title(page_text)
            logger.debug("Layout: Page %d title score: %d", i, score)
            if score < TITLE_PAGE_THRESHOLD:
                return i

        # All checked pages look like title pages; start from page 1
        # (page 0 is almost certainly a cover)
        return min(1, len(doc) - 1)

    def _select_sample_pages(self, start: int, total: int) -> list[int]:
        """
        Select consecutive page indices for sampling.

        Args:
            start: First content page index
            total: Total page count

        Returns:
            List of page indices to sample
        """
        end = min(start + SAMPLE_PAGE_COUNT, total)
        return list(range(start, end))

    def _detect_hf_zones(
        self,
        page_blocks: list[tuple],
        page_width: float,
        page_height: float,
    ) -> tuple[float, float]:
        """
        Detect header and footer zones by finding repeating Y-position blocks.

        Blocks that appear at the same Y position (within 5pt tolerance) on
        3+ of the sample pages, and sit in the top/bottom 8% of the page,
        are classified as headers/footers.

        Args:
            page_blocks: List of (page, blocks_dict_list) tuples
            page_width: Page width in points
            page_height: Page height in points

        Returns:
            Tuple of (header_bottom, footer_top) Y coordinates
        """
        header_zone_limit = page_height * HEADER_ZONE_RATIO
        footer_zone_limit = page_height * (1 - FOOTER_ZONE_RATIO)

        # Collect Y-positions of text blocks in header/footer zones
        # Key: rounded Y position, Value: count of pages with a block there
        header_y_counts: dict[int, int] = {}
        footer_y_counts: dict[int, int] = {}

        tolerance = 5  # points

        for _page, blocks in page_blocks:
            # Track Y positions seen on this page to avoid double-counting
            seen_header_y = set()
            seen_footer_y = set()

            for block in blocks:
                # Skip image blocks
                if block.get("type", 0) == 1:
                    continue

                bbox = block.get("bbox", (0, 0, 0, 0))
                y0, y1 = bbox[1], bbox[3]
                block_mid_y = (y0 + y1) / 2

                # Round to tolerance bucket
                bucket = round(block_mid_y / tolerance) * tolerance

                if block_mid_y < header_zone_limit:
                    if bucket not in seen_header_y:
                        seen_header_y.add(bucket)
                        header_y_counts[bucket] = header_y_counts.get(bucket, 0) + 1

                elif block_mid_y > footer_zone_limit:
                    if bucket not in seen_footer_y:
                        seen_footer_y.add(bucket)
                        footer_y_counts[bucket] = footer_y_counts.get(bucket, 0) + 1

        # Find the lowest repeating header block
        header_bottom = 0.0
        for y_bucket, count in header_y_counts.items():
            if count >= MIN_REPEAT_COUNT:
                # Find actual bottom of blocks at this Y
                candidate = y_bucket + tolerance
                if candidate > header_bottom:
                    header_bottom = candidate
                    logger.debug(
                        "Layout: Header block at y≈%d repeated on %d pages",
                        y_bucket,
                        count,
                    )

        # Find the highest repeating footer block
        footer_top = page_height
        for y_bucket, count in footer_y_counts.items():
            if count >= MIN_REPEAT_COUNT:
                # Use the top of blocks at this Y
                candidate = y_bucket - tolerance
                if candidate < footer_top:
                    footer_top = candidate
                    logger.debug(
                        "Layout: Footer block at y≈%d repeated on %d pages",
                        y_bucket,
                        count,
                    )

        # If no repeating blocks found, use defaults (no clipping)
        if header_bottom == 0.0 and footer_top == page_height:
            logger.debug("Layout: No repeating header/footer blocks found")

        return header_bottom, footer_top

    def _detect_line_number_margin(
        self,
        page_blocks: list[tuple],
        page_width: float,
        page_height: float,
    ) -> float:
        """
        Detect left-margin line numbers by looking for narrow numeric blocks.

        Line numbers appear as small blocks of 1-2 digit numbers positioned
        at X < 10% of page width, repeated across multiple pages.

        Args:
            page_blocks: List of (page, blocks_dict_list) tuples
            page_width: Page width in points
            page_height: Page height in points

        Returns:
            Left margin X coordinate (0.0 if no line numbers detected)
        """
        line_num_x_limit = page_width * LINE_NUMBER_X_RATIO
        line_number_pattern = re.compile(r"^\s*\d{1,2}\s*$")

        # Count pages that have numeric blocks in left margin
        pages_with_margin_numbers = 0
        max_x1 = 0.0

        for _page, blocks in page_blocks:
            found_on_page = False
            for block in blocks:
                if block.get("type", 0) == 1:
                    continue

                bbox = block.get("bbox", (0, 0, 0, 0))
                x0, x1 = bbox[0], bbox[2]

                # Block must be in left margin zone
                if x1 > line_num_x_limit:
                    continue

                # Check if block text is just a number
                lines = block.get("lines", [])
                for line in lines:
                    text = "".join(span.get("text", "") for span in line.get("spans", []))
                    if line_number_pattern.match(text):
                        found_on_page = True
                        if x1 > max_x1:
                            max_x1 = x1

            if found_on_page:
                pages_with_margin_numbers += 1

        if pages_with_margin_numbers >= MIN_REPEAT_COUNT:
            # Add small padding beyond the rightmost line number
            left_margin = max_x1 + 5
            logger.debug(
                "Layout: Line numbers detected in left margin (x < %.0f) on %d pages",
                left_margin,
                pages_with_margin_numbers,
            )
            return left_margin

        return 0.0
