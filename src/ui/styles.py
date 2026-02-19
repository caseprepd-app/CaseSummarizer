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
    global _styles_initialized
    if _styles_initialized:
        return

    logger.debug(
        "Initializing all Treeview styles (scale=%.2f, font_offset=%d)",
        scale_factor,
        font_offset,
    )

    style = ttk.Style()
    # This is the expensive call - do it exactly once at startup
    style.theme_use("default")

    _configure_vocab_treeview_style(style, scale_factor, font_offset)
    _configure_qa_table_style(style, scale_factor, font_offset)
    _configure_file_review_style(style, scale_factor, font_offset)
    _configure_question_list_style(style, scale_factor, font_offset)

    _styles_initialized = True
    logger.debug("All Treeview styles initialized")


def _configure_vocab_treeview_style(style: ttk.Style, sf: float, font_offset: int) -> None:
    """Configure style for vocabulary/NER grid (DynamicOutputWidget)."""
    # 8pt floor matches theme.scale_fonts() -- prevents unreadable text
    font_size = max(8, int(10 * sf) + font_offset)
    font_spec = ("Segoe UI", font_size)
    style.configure(
        "Vocab.Treeview",
        background="#2b2b2b",
        foreground="white",
        fieldbackground="#2b2b2b",
        borderwidth=0,
        rowheight=_get_rowheight(font_spec),
        font=font_spec,
    )
    style.map("Vocab.Treeview", background=[("selected", "#3470b6")])

    style.configure(
        "Vocab.Treeview.Heading",
        background="#404040",
        foreground="white",
        relief="flat",
        font=("Segoe UI", font_size, "bold"),
        padding=(int(8 * sf), int(4 * sf)),
    )
    style.map("Vocab.Treeview.Heading", background=[("active", "#505050")])

    style.configure(
        "Vocab.Vertical.TScrollbar",
        background="#404040",
        troughcolor="#2b2b2b",
        borderwidth=0,
        arrowcolor="white",
    )
    style.configure(
        "Vocab.Horizontal.TScrollbar",
        background="#404040",
        troughcolor="#2b2b2b",
        borderwidth=0,
        arrowcolor="white",
    )


def _configure_qa_table_style(style: ttk.Style, sf: float, font_offset: int) -> None:
    """Configure style for Q&A results table (QAPanel)."""
    font_size = max(8, int(10 * sf) + font_offset)
    font_spec = ("Segoe UI", font_size)
    style.configure(
        "QATable.Treeview",
        background="#2b2b2b",
        foreground="white",
        fieldbackground="#2b2b2b",
        borderwidth=0,
        rowheight=_get_rowheight(font_spec),
        font=font_spec,
    )
    style.map("QATable.Treeview", background=[("selected", "#3470b6")])

    style.configure(
        "QATable.Treeview.Heading",
        background="#404040",
        foreground="white",
        relief="flat",
        font=("Segoe UI", font_size, "bold"),
        padding=(int(8 * sf), int(4 * sf)),
    )
    style.map("QATable.Treeview.Heading", background=[("active", "#505050")])

    style.configure(
        "QATable.Vertical.TScrollbar",
        background="#404040",
        troughcolor="#2b2b2b",
        borderwidth=0,
        arrowcolor="white",
    )
    style.configure(
        "QATable.Horizontal.TScrollbar",
        background="#404040",
        troughcolor="#2b2b2b",
        borderwidth=0,
        arrowcolor="white",
    )


def _configure_file_review_style(style: ttk.Style, sf: float, font_offset: int) -> None:
    """Configure style for file review table (FileReviewTable/widgets.py)."""
    font_size = max(8, int(10 * sf) + font_offset)
    font_spec = ("Segoe UI", font_size)
    style.configure(
        "Treeview",
        background="#2b2b2b",
        foreground="white",
        fieldbackground="#2b2b2b",
        borderwidth=0,
        rowheight=_get_rowheight(font_spec),
        font=font_spec,
    )
    style.map("Treeview", background=[("selected", "#3470b6")])

    style.configure("Treeview.Heading", background="#565b5e", foreground="white", relief="flat")
    style.map("Treeview.Heading", background=[("active", "#6c757d")])


def _configure_question_list_style(style: ttk.Style, sf: float, font_offset: int) -> None:
    """Configure style for question editor list (QAQuestionEditor)."""
    font_size = max(8, int(10 * sf) + font_offset)
    font_spec = ("Segoe UI", font_size)
    style.configure(
        "QuestionList.Treeview",
        background="#2b2b2b",
        foreground="white",
        fieldbackground="#2b2b2b",
        borderwidth=0,
        rowheight=_get_rowheight(font_spec),
        font=font_spec,
    )
    style.map("QuestionList.Treeview", background=[("selected", "#3470b6")])

    style.configure(
        "QuestionList.Treeview.Heading",
        background="#404040",
        foreground="white",
        relief="flat",
        font=("Segoe UI", font_size, "bold"),
    )
    style.map("QuestionList.Treeview.Heading", background=[("active", "#505050")])
