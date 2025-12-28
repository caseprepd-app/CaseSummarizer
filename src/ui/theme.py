"""
Centralized Theme Configuration for LocalScribe

All fonts, colors, and common widget configurations are defined here.
This prevents scattered magic values and ensures consistency.

Usage:
    from src.ui.theme import FONTS, COLORS, BUTTON_STYLES

    label = ctk.CTkLabel(parent, font=FONTS["heading"], text_color=COLORS["text_muted"])
    button = ctk.CTkButton(parent, **BUTTON_STYLES["primary"])

IMPORTANT: Use tuple fonts, NOT CTkFont objects, to avoid scaling conflicts
with CTkTextbox.tag_config(). See RESEARCH_LOG.md "Q&A Follow-up Font Scaling Error".
"""

# =============================================================================
# FONTS
# =============================================================================
# All fonts are tuples: (family, size) or (family, size, weight)
# This format is compatible with both CTk widgets and tkinter tag_config

FONTS = {
    # Headings
    "heading_xl": ("Segoe UI", 20, "bold"),      # Main window title
    "heading_lg": ("Segoe UI", 17, "bold"),      # Section titles (quadrant builder)
    "heading": ("Segoe UI", 14, "bold"),         # Panel headers
    "heading_sm": ("Segoe UI", 13, "bold"),      # Sub-headers

    # Body text
    "body": ("Segoe UI", 12),                    # Default body text
    "body_bold": ("Segoe UI", 12, "bold"),       # Emphasized body text
    "body_italic": ("Segoe UI", 12, "italic"),   # Italic body text

    # Small text
    "small": ("Segoe UI", 11),                   # Secondary text, labels
    "small_bold": ("Segoe UI", 11, "bold"),      # Small headers
    "tiny": ("Segoe UI", 10),                    # Status bar, system monitor

    # Q&A Panel specific (used in tag_config with cnf={})
    "qa_question": ("Segoe UI", 13, "bold"),           # Question text
    "qa_question_default": ("Segoe UI", 13, "bold italic"),  # Default questions
    "qa_label": ("Segoe UI", 11, "bold"),              # "Answer:", "Sources:" labels
    "qa_source": ("Segoe UI", 12, "italic"),           # Source citations

    # Monospace (for code/technical content)
    "mono": ("Consolas", 11),
    "mono_sm": ("Consolas", 10),
}


# =============================================================================
# COLORS
# =============================================================================
# Organized by semantic meaning, not visual appearance
# Format: (light_mode, dark_mode) or just string for dark-mode-only

COLORS = {
    # Backgrounds
    "bg_dark": "#2b2b2b",                        # Main dark background (panels, frames)
    "bg_darker": "#1e1e1e",                      # Deeper background (textboxes)
    "bg_card": "#333333",                        # Card/section background
    "bg_input": "#404040",                       # Input field backgrounds
    "bg_hover": "#505050",                       # Hover state for backgrounds

    # Buttons - Primary (action buttons)
    "btn_primary": "#2d5a87",                    # Primary button background
    "btn_primary_hover": "#3d6a97",              # Primary button hover

    # Buttons - Secondary (neutral buttons)
    "btn_secondary": "#555555",                  # Secondary button background
    "btn_secondary_hover": "#666666",            # Secondary button hover

    # Buttons - Tertiary (less prominent)
    "btn_tertiary": "#444444",                   # Tertiary button background
    "btn_tertiary_hover": "#555555",             # Tertiary button hover

    # Buttons - Danger
    "btn_danger": "#994444",                     # Danger/delete button
    "btn_danger_hover": "#bb5555",               # Danger button hover

    # Buttons - Disabled
    "btn_disabled": "#6c757d",                   # Disabled button
    "btn_disabled_hover": "#5a6268",             # Disabled button hover (shouldn't happen)

    # Text
    "text_primary": "#e0e0e0",                   # Primary text (bright)
    "text_secondary": "#aaaaaa",                 # Secondary/muted text
    "text_disabled": "#666666",                  # Disabled text
    "text_white": "#ffffff",                     # Pure white text

    # Semantic colors
    "success": "#28a745",                        # Success/approved/keep (green)
    "success_light": "#90EE90",                  # Light green (system monitor)
    "warning": "#ffc107",                        # Warning (yellow)
    "danger": "#dc3545",                         # Error/rejected/skip (red)
    "info": "#17a2b8",                           # Info/LLM (blue)

    # Q&A Panel text tags
    "qa_question": "#5dade2",                    # Question text (light blue)
    "qa_question_default": "#7dacd6",            # Default question (muted blue)
    "qa_answer": "#aed6f1",                      # Answer text (pale blue)
    "qa_citation": "#d7dbdd",                    # Citation text (light gray)
    "qa_source": "#52be80",                      # Source info (green)
    "qa_label": "#85929e",                       # Labels (gray)
    "qa_separator": "#566573",                   # Section separators

    # Algorithm detection colors (vocabulary table)
    "algo_multi": "#28a745",                     # Multiple algorithms (green)
    "algo_ner": "#2c3e50",                       # NER only (dark slate)
    "algo_rake": "#8e44ad",                      # RAKE only (purple)
    "algo_bm25": "#e67e22",                      # BM25 only (orange)
    "algo_llm": "#17a2b8",                       # LLM only (blue)

    # Table rows
    "row_odd": "#f8f9fa",                        # Odd row (light mode)
    "row_even": "#ffffff",                       # Even row (light mode)
    "row_selected": "#3470b6",                   # Selected row

    # Output pane (Session 45)
    "output_pane": ("#e8e8e8", "#1a1a2e"),       # Light/dark mode pair

    # Progress/status
    "progress_partial": ("orange", "#ffaa00"),   # NER only
    "progress_complete": ("green", "#00cc66"),   # NER + LLM

    # System monitor
    "monitor_bg": "#1a3a1a",                     # System monitor frame bg
}


# =============================================================================
# BUTTON STYLE PRESETS
# =============================================================================
# Common button configurations to reduce repetition

BUTTON_STYLES = {
    "primary": {
        "fg_color": COLORS["btn_primary"],
        "hover_color": COLORS["btn_primary_hover"],
    },
    "secondary": {
        "fg_color": COLORS["btn_secondary"],
        "hover_color": COLORS["btn_secondary_hover"],
    },
    "tertiary": {
        "fg_color": COLORS["btn_tertiary"],
        "hover_color": COLORS["btn_tertiary_hover"],
    },
    "danger": {
        "fg_color": COLORS["btn_danger"],
        "hover_color": COLORS["btn_danger_hover"],
    },
    "disabled": {
        "fg_color": COLORS["btn_disabled"],
        "hover_color": COLORS["btn_disabled_hover"],
    },
}


# =============================================================================
# FRAME STYLE PRESETS
# =============================================================================

FRAME_STYLES = {
    "card": {
        "fg_color": COLORS["bg_dark"],
        "corner_radius": 6,
    },
    "card_dark": {
        "fg_color": COLORS["bg_darker"],
        "corner_radius": 6,
    },
    "input_area": {
        "fg_color": COLORS["bg_card"],
        "corner_radius": 6,
    },
    "transparent": {
        "fg_color": "transparent",
    },
}


# =============================================================================
# TEXTBOX STYLE PRESETS
# =============================================================================

TEXTBOX_STYLES = {
    "default": {
        "fg_color": COLORS["bg_darker"],
        "text_color": COLORS["text_primary"],
        "font": FONTS["body"],
    },
    "code": {
        "fg_color": COLORS["bg_darker"],
        "text_color": COLORS["text_primary"],
        "font": FONTS["mono"],
    },
}


# =============================================================================
# LABEL STYLE PRESETS
# =============================================================================

LABEL_STYLES = {
    "heading": {
        "font": FONTS["heading"],
    },
    "heading_xl": {
        "font": FONTS["heading_xl"],
    },
    "muted": {
        "font": FONTS["small"],
        "text_color": COLORS["text_secondary"],
    },
    "tiny": {
        "font": FONTS["tiny"],
        "text_color": COLORS["text_secondary"],
    },
}


# =============================================================================
# Q&A TEXT TAGS
# =============================================================================
# Used with CTkTextbox.tag_config(name, cnf={...})
# IMPORTANT: Must use cnf={} parameter, not keyword arguments

QA_TEXT_TAGS = {
    "question": {"foreground": COLORS["qa_question"], "font": FONTS["qa_question"]},
    "question_default": {"foreground": COLORS["qa_question_default"], "font": FONTS["qa_question_default"]},
    "label": {"foreground": COLORS["qa_label"], "font": FONTS["qa_label"]},
    "answer": {"foreground": COLORS["qa_answer"]},
    "citation": {"foreground": COLORS["qa_citation"]},
    "source": {"foreground": COLORS["qa_source"], "font": FONTS["qa_source"]},
    "separator": {"foreground": COLORS["qa_separator"]},
}


# =============================================================================
# VOCABULARY TABLE TAGS
# =============================================================================
# Used with ttk.Treeview.tag_configure()

VOCAB_TABLE_TAGS = {
    "rated_up": {"foreground": COLORS["success"]},
    "rated_down": {"foreground": COLORS["danger"]},
    "found_multi": {"foreground": COLORS["algo_multi"]},
    "found_ner": {"foreground": COLORS["algo_ner"]},
    "found_rake": {"foreground": COLORS["algo_rake"]},
    "found_bm25": {"foreground": COLORS["algo_bm25"]},
    "found_llm": {"foreground": COLORS["algo_llm"]},
    "oddrow": {"background": COLORS["row_odd"]},
    "evenrow": {"background": COLORS["row_even"]},
}


# =============================================================================
# FILE STATUS TAGS
# =============================================================================
# Used with FileReviewTable (widgets.py)

FILE_STATUS_TAGS = {
    "green": {"foreground": COLORS["success"]},
    "yellow": {"foreground": COLORS["warning"]},
    "red": {"foreground": COLORS["danger"]},
    "pending": {"foreground": "gray"},
}
