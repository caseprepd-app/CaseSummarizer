"""
Centralized Treeview Style Configuration

All ttk.Treeview styles are initialized here at app startup to avoid
the expensive theme_use("default") call during view switching.

This prevents the ~30 second GUI freeze that occurs when style
configuration happens lazily on first view switch.

DPI-aware rowheight strategy:
    ttk.Treeview rowheight is specified in pixels, but on high-DPI
    displays (e.g. 4K at 200% Windows scaling), point-size fonts
    render larger automatically via SetProcessDpiAwareness(2). A 10pt
    font that fits in 20px at 100% DPI may need ~26px at 200% DPI.
    Hardcoding rowheight=25 clips the text on those displays.

    Fix: tkfont.Font.metrics('linespace') returns the actual pixel
    height the font renders at under the current DPI, so we derive
    rowheight = linespace + padding. This works at any DPI without
    needing to know the scaling factor, because Tk's font subsystem
    already accounts for DPI when reporting metrics.

    The font_offset parameter keeps treeview font sizes in sync with
    the rest of the UI (theme.scale_fonts applies the same offset to
    CTk widgets, but ttk widgets are styled separately here).
"""

import logging
import tkinter.font as tkfont
from tkinter import ttk

logger = logging.getLogger(__name__)

_styles_initialized = False
_vocab_font_spec = ("Segoe UI", 10)
_vocab_heading_font_spec = ("Segoe UI", 10, "bold")

# Cache init params for re-initialization on theme change
_last_scale_factor = 1.0
_last_font_offset = 0


def _get_rowheight(font_spec: tuple, padding: int = 8) -> int:
    """
    Calculate Treeview rowheight from actual rendered font metrics.

    Uses tkfont.Font.metrics('linespace') which returns the real pixel
    height at the current DPI -- no manual DPI math needed. On a 4K
    display at 200% scaling, a 10pt font reports ~26px linespace vs
    ~13px at 100%. Adding padding ensures the text is not clipped.

    Args:
        font_spec: Tk font tuple, e.g. ("Segoe UI", 10).
        padding: Extra vertical pixels above/below text.

    Returns:
        int: Row height in pixels that fits the font at current DPI.
    """
    f = tkfont.Font(font=font_spec)
    linespace = f.metrics("linespace")
    return linespace + padding


def initialize_all_styles(scale_factor: float = 1.0, font_offset: int = 0) -> None:
    """
    Initialize all Treeview styles at app startup.

    Args:
        scale_factor: UI scale multiplier (1.0 = 100%). Scales padding
                      for ttk widgets (not affected by CTk scaling).
        font_offset: Point offset applied to treeview font sizes (default 0).
                     Keeps treeview text consistent with the rest of the UI.

    Must be called once before any Treeview widgets are created.
    Calling multiple times is safe (no-op after first call).
    """
    global _styles_initialized, _last_scale_factor, _last_font_offset
    if _styles_initialized:
        return

    _last_scale_factor = scale_factor
    _last_font_offset = font_offset

    logger.debug(
        "Initializing all Treeview styles (scale=%.2f, font_offset=%d)",
        scale_factor,
        font_offset,
    )

    style = ttk.Style()
    # This is the expensive call - do it exactly once at startup
    style.theme_use("default")

    _apply_all_styles(style, scale_factor, font_offset)

    _styles_initialized = True
    logger.debug("All Treeview styles initialized")


def reinitialize_styles() -> None:
    """
    Re-apply all Treeview styles after a theme mode change.

    Uses the same scale_factor and font_offset from the initial call.
    Skips the expensive theme_use("default") since it was already done.
    """
    if not _styles_initialized:
        return

    style = ttk.Style()
    _apply_all_styles(style, _last_scale_factor, _last_font_offset)
    logger.debug("Treeview styles re-applied for theme change")


def _apply_all_styles(style: ttk.Style, scale_factor: float, font_offset: int) -> None:
    """Apply all treeview style configurations using current theme colors."""
    _configure_vocab_treeview_style(style, scale_factor, font_offset)
    _configure_qa_table_style(style, scale_factor, font_offset)
    _configure_file_review_style(style, scale_factor, font_offset)
    _configure_question_list_style(style, scale_factor, font_offset)


def get_vocab_font_specs() -> tuple[tuple, tuple]:
    """
    Return (content_font_spec, heading_font_spec) for the vocab treeview.

    These are the same font tuples used by _configure_vocab_treeview_style,
    stored at initialization time so column_config can compute DPI-aware widths.

    Returns:
        Tuple of (content_font, heading_font) e.g.
        (("Segoe UI", 10), ("Segoe UI", 10, "bold"))
    """
    return _vocab_font_spec, _vocab_heading_font_spec


def _configure_vocab_treeview_style(style: ttk.Style, sf: float, font_offset: int) -> None:
    """Configure style for vocabulary/NER grid (DynamicOutputWidget)."""
    from src.ui.theme import get_color

    global _vocab_font_spec, _vocab_heading_font_spec
    # 8pt floor matches theme.scale_fonts() -- prevents unreadable text
    font_size = max(8, int(10 * sf) + font_offset)
    font_spec = ("Segoe UI", font_size)
    _vocab_font_spec = font_spec
    _vocab_heading_font_spec = ("Segoe UI", font_size, "bold")
    style.configure(
        "Vocab.Treeview",
        background=get_color("tree_bg"),
        foreground=get_color("tree_fg"),
        fieldbackground=get_color("tree_field_bg"),
        borderwidth=0,
        rowheight=_get_rowheight(font_spec),
        font=font_spec,
    )
    style.map("Vocab.Treeview", background=[("selected", get_color("row_selected"))])

    style.configure(
        "Vocab.Treeview.Heading",
        background=get_color("tree_heading_bg"),
        foreground=get_color("tree_heading_fg"),
        relief="flat",
        font=("Segoe UI", font_size, "bold"),
        padding=(int(8 * sf), int(4 * sf)),
    )
    style.map("Vocab.Treeview.Heading", background=[("active", get_color("tree_heading_hover"))])

    style.configure(
        "Vocab.Vertical.TScrollbar",
        background=get_color("tree_scroll_bg"),
        troughcolor=get_color("tree_scroll_trough"),
        borderwidth=0,
        arrowcolor=get_color("tree_arrow"),
    )
    style.configure(
        "Vocab.Horizontal.TScrollbar",
        background=get_color("tree_scroll_bg"),
        troughcolor=get_color("tree_scroll_trough"),
        borderwidth=0,
        arrowcolor=get_color("tree_arrow"),
    )


def _configure_qa_table_style(style: ttk.Style, sf: float, font_offset: int) -> None:
    """Configure style for search results table (Search Panel)."""
    from src.ui.theme import get_color

    font_size = max(8, int(10 * sf) + font_offset)
    font_spec = ("Segoe UI", font_size)
    style.configure(
        "QATable.Treeview",
        background=get_color("tree_bg"),
        foreground=get_color("tree_fg"),
        fieldbackground=get_color("tree_field_bg"),
        borderwidth=0,
        rowheight=_get_rowheight(font_spec),
        font=font_spec,
    )
    style.map("QATable.Treeview", background=[("selected", get_color("row_selected"))])

    style.configure(
        "QATable.Treeview.Heading",
        background=get_color("tree_heading_bg"),
        foreground=get_color("tree_heading_fg"),
        relief="flat",
        font=("Segoe UI", font_size, "bold"),
        padding=(int(8 * sf), int(4 * sf)),
    )
    style.map("QATable.Treeview.Heading", background=[("active", get_color("tree_heading_hover"))])

    style.configure(
        "QATable.Vertical.TScrollbar",
        background=get_color("tree_scroll_bg"),
        troughcolor=get_color("tree_scroll_trough"),
        borderwidth=0,
        arrowcolor=get_color("tree_arrow"),
    )
    style.configure(
        "QATable.Horizontal.TScrollbar",
        background=get_color("tree_scroll_bg"),
        troughcolor=get_color("tree_scroll_trough"),
        borderwidth=0,
        arrowcolor=get_color("tree_arrow"),
    )


def _configure_file_review_style(style: ttk.Style, sf: float, font_offset: int) -> None:
    """Configure style for file review table (FileReviewTable/widgets.py)."""
    from src.ui.theme import get_color

    font_size = max(8, int(10 * sf) + font_offset)
    font_spec = ("Segoe UI", font_size)
    style.configure(
        "Treeview",
        background=get_color("tree_bg"),
        foreground=get_color("tree_fg"),
        fieldbackground=get_color("tree_field_bg"),
        borderwidth=0,
        rowheight=_get_rowheight(font_spec),
        font=font_spec,
    )
    style.map("Treeview", background=[("selected", get_color("row_selected"))])

    style.configure(
        "Treeview.Heading",
        background=get_color("tree_file_heading_bg"),
        foreground=get_color("tree_heading_fg"),
        relief="flat",
    )
    style.map("Treeview.Heading", background=[("active", get_color("tree_file_heading_hover"))])


def _configure_question_list_style(style: ttk.Style, sf: float, font_offset: int) -> None:
    """Configure style for question editor list (QAQuestionEditor)."""
    from src.ui.theme import get_color

    font_size = max(8, int(10 * sf) + font_offset)
    font_spec = ("Segoe UI", font_size)
    style.configure(
        "QuestionList.Treeview",
        background=get_color("tree_bg"),
        foreground=get_color("tree_fg"),
        fieldbackground=get_color("tree_field_bg"),
        borderwidth=0,
        rowheight=_get_rowheight(font_spec),
        font=font_spec,
    )
    style.map("QuestionList.Treeview", background=[("selected", get_color("row_selected"))])

    style.configure(
        "QuestionList.Treeview.Heading",
        background=get_color("tree_heading_bg"),
        foreground=get_color("tree_heading_fg"),
        relief="flat",
        font=("Segoe UI", font_size, "bold"),
    )
    style.map(
        "QuestionList.Treeview.Heading",
        background=[("active", get_color("tree_heading_hover"))],
    )
