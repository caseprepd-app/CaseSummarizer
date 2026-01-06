"""
Centralized Treeview Style Configuration

All ttk.Treeview styles are initialized here at app startup to avoid
the expensive theme_use("default") call during view switching.

This prevents the ~30 second GUI freeze that occurs when style
configuration happens lazily on first view switch.
"""

from tkinter import ttk

from src.logging_config import debug_log

_styles_initialized = False


def initialize_all_styles() -> None:
    """
    Initialize all Treeview styles at app startup.

    Must be called once before any Treeview widgets are created.
    Calling multiple times is safe (no-op after first call).
    """
    global _styles_initialized
    if _styles_initialized:
        return

    debug_log("[Styles] Initializing all Treeview styles...")

    style = ttk.Style()
    # This is the expensive call - do it exactly once at startup
    style.theme_use("default")

    _configure_vocab_treeview_style(style)
    _configure_qa_table_style(style)
    _configure_file_review_style(style)
    _configure_question_list_style(style)

    _styles_initialized = True
    debug_log("[Styles] All Treeview styles initialized")


def _configure_vocab_treeview_style(style: ttk.Style) -> None:
    """Configure style for vocabulary/NER grid (DynamicOutputWidget)."""
    style.configure(
        "Vocab.Treeview",
        background="#2b2b2b",
        foreground="white",
        fieldbackground="#2b2b2b",
        borderwidth=0,
        rowheight=25,
        font=("Segoe UI", 10),
    )
    style.map("Vocab.Treeview", background=[("selected", "#3470b6")])

    style.configure(
        "Vocab.Treeview.Heading",
        background="#404040",
        foreground="white",
        relief="flat",
        font=("Segoe UI", 10, "bold"),
        padding=(8, 4),
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


def _configure_qa_table_style(style: ttk.Style) -> None:
    """Configure style for Q&A results table (QAPanel)."""
    style.configure(
        "QATable.Treeview",
        background="#2b2b2b",
        foreground="white",
        fieldbackground="#2b2b2b",
        borderwidth=0,
        rowheight=28,
        font=("Segoe UI", 10),
    )
    style.map("QATable.Treeview", background=[("selected", "#3470b6")])

    style.configure(
        "QATable.Treeview.Heading",
        background="#404040",
        foreground="white",
        relief="flat",
        font=("Segoe UI", 10, "bold"),
        padding=(8, 4),
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


def _configure_file_review_style(style: ttk.Style) -> None:
    """Configure style for file review table (FileReviewTable/widgets.py)."""
    style.configure(
        "Treeview",
        background="#2b2b2b",
        foreground="white",
        fieldbackground="#2b2b2b",
        borderwidth=0,
    )
    style.map("Treeview", background=[("selected", "#3470b6")])

    style.configure("Treeview.Heading", background="#565b5e", foreground="white", relief="flat")
    style.map("Treeview.Heading", background=[("active", "#6c757d")])


def _configure_question_list_style(style: ttk.Style) -> None:
    """Configure style for question editor list (QAQuestionEditor)."""
    style.configure(
        "QuestionList.Treeview",
        background="#2b2b2b",
        foreground="white",
        fieldbackground="#2b2b2b",
        borderwidth=0,
        rowheight=28,
        font=("Segoe UI", 10),
    )
    style.map("QuestionList.Treeview", background=[("selected", "#3470b6")])

    style.configure(
        "QuestionList.Treeview.Heading",
        background="#404040",
        foreground="white",
        relief="flat",
        font=("Segoe UI", 10, "bold"),
    )
    style.map("QuestionList.Treeview.Heading", background=[("active", "#505050")])
