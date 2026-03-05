"""
Multi-Column PDF Page Detection and Extraction.

Detects whether a PDF page has a multi-column layout (e.g., Min-U-Script
condensed transcripts) and extracts text in proper reading order:
left column top-to-bottom, then right column top-to-bottom.

Single-column pages are extracted with sort=True for correct spatial order.

Example usage:
    >>> import fitz
    >>> doc = fitz.open("transcript.pdf")
    >>> text = extract_page_text(doc[0])
"""

import logging

import fitz

logger = logging.getLogger(__name__)

# Extraction flags: expand ligatures for better dictionary matching,
# skip image data, enable intra-block dehyphenation, clip to mediabox.
# NOT setting TEXT_PRESERVE_LIGATURES — we want "ffi"→"ffi" expansion.
# NOT setting TEXT_PRESERVE_IMAGES — we only need text blocks.
EXTRACT_FLAGS = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP | fitz.TEXT_DEHYPHENATE

# Minimum gap between column centers to consider them separate columns.
# Expressed as fraction of page width (15% = columns must be >15% apart).
MIN_COLUMN_GAP_RATIO = 0.15

# Minimum number of blocks in a cluster to count as a column.
MIN_BLOCKS_PER_COLUMN = 3


def _cluster_x_positions(blocks: list[tuple], page_width: float) -> list[list[tuple]]:
    """
    Cluster text blocks into columns by their x-center position.

    Uses a simple sweep: sorts blocks by x-center, then splits into
    groups when the gap between consecutive x-centers exceeds the
    minimum column gap threshold.

    Args:
        blocks: List of PyMuPDF text block tuples
                (x0, y0, x1, y1, text, block_no, block_type)
        page_width: Width of the page in points

    Returns:
        List of column groups, each a list of block tuples,
        ordered left-to-right
    """
    if not blocks:
        return []

    min_gap = page_width * MIN_COLUMN_GAP_RATIO

    # Calculate x-center for each block and sort by it
    blocks_with_center = [(b, (b[0] + b[2]) / 2) for b in blocks]
    blocks_with_center.sort(key=lambda item: item[1])

    # Split into groups when gap between consecutive centers is large
    columns: list[list[tuple]] = [[blocks_with_center[0][0]]]
    prev_center = blocks_with_center[0][1]

    for block, center in blocks_with_center[1:]:
        if center - prev_center > min_gap:
            columns.append([block])
        else:
            columns[-1].append(block)
        prev_center = center

    return columns


def _is_multi_column(columns: list[list[tuple]]) -> bool:
    """
    Determine if the detected columns represent a true multi-column layout.

    A page is multi-column only if there are 2+ columns, each with enough
    text blocks to be meaningful (not just a stray margin annotation).

    Args:
        columns: Column groups from _cluster_x_positions

    Returns:
        True if the page has a genuine multi-column layout
    """
    significant_columns = [c for c in columns if len(c) >= MIN_BLOCKS_PER_COLUMN]
    return len(significant_columns) >= 2


def extract_page_text(
    page: fitz.Page,
    clip: fitz.Rect | None = None,
) -> str:
    """
    Extract text from a page in correct reading order.

    For single-column pages, uses sort=True for spatial ordering.
    For multi-column pages, detects columns and reads each column
    top-to-bottom before moving to the next column.

    Args:
        page: A PyMuPDF Page object
        clip: Optional clip rectangle (from layout zone detection)

    Returns:
        Extracted text in reading order
    """
    # Get text blocks with positions and optimized flags
    # Each block: (x0, y0, x1, y1, "text", block_no, block_type)
    blocks = page.get_text("blocks", clip=clip, flags=EXTRACT_FLAGS)

    # Filter to text blocks only (type 0)
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

    if not text_blocks:
        return ""

    page_width = clip.width if clip else page.rect.width

    # Detect columns
    columns = _cluster_x_positions(text_blocks, page_width)

    if _is_multi_column(columns):
        # Multi-column: read each column top-to-bottom, left-to-right
        logger.debug(
            "Multi-column detected on page %d: %d columns",
            page.number + 1,
            len(columns),
        )
        parts = []
        for column_blocks in columns:
            # Sort blocks within column by y-position (top to bottom)
            column_blocks.sort(key=lambda b: b[1])
            for block in column_blocks:
                parts.append(block[4].strip())
        return "\n".join(parts)

    # Single-column: use sort=True for correct spatial order
    return page.get_text(clip=clip, sort=True, flags=EXTRACT_FLAGS)
