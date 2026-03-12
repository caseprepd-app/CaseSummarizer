"""
Centralized Theme Configuration for CasePrepd

All fonts, colors, and common widget configurations are defined here.
This prevents scattered magic values and ensures consistency.

Usage:
    from src.ui.theme import FONTS, COLORS, BUTTON_STYLES, get_color

    label = ctk.CTkLabel(parent, font=FONTS["heading"], text_color=get_color("text_secondary"))
    button = ctk.CTkButton(parent, **BUTTON_STYLES["primary"])

IMPORTANT: Use tuple fonts, NOT CTkFont objects, to avoid scaling conflicts
with CTkTextbox.tag_config() — CTkTextbox ignores font size in insert().
"""

import customtkinter as ctk

# =============================================================================
# FONTS
# =============================================================================
# All fonts are tuples: (family, size) or (family, size, weight)
# This format is compatible with both CTk widgets and tkinter tag_config

FONTS = {
    # Headings
    "heading_xl": ("Segoe UI", 20, "bold"),  # Main window title
    "heading_lg": ("Segoe UI", 17, "bold"),  # Section titles (quadrant builder)
    "heading": ("Segoe UI", 14, "bold"),  # Panel headers
    "heading_sm": ("Segoe UI", 13, "bold"),  # Sub-headers
    # Body text
    "body": ("Segoe UI", 12),  # Default body text
    "body_bold": ("Segoe UI", 12, "bold"),  # Emphasized body text
    "body_italic": ("Segoe UI", 12, "italic"),  # Italic body text
    # Small text
    "small": ("Segoe UI", 11),  # Secondary text, labels
    "small_bold": ("Segoe UI", 11, "bold"),  # Small headers
    "tiny": ("Segoe UI", 10),  # Status bar, system monitor
    # Search Panel specific (used in tag_config with cnf={})
    "qa_question": ("Segoe UI", 13, "bold"),  # Question text
    "qa_question_default": ("Segoe UI", 13, "bold italic"),  # Default questions
    "qa_label": ("Segoe UI", 11, "bold"),  # "Answer:", "Sources:" labels
    "qa_source": ("Segoe UI", 12, "italic"),  # Source citations
    # Monospace (for code/technical content)
    "mono": ("Consolas", 11),
    "mono_sm": ("Consolas", 10),
}

# Font size offset labels: Small=-2, Medium=0, Large=+2
FONT_SIZE_OPTIONS = [
    ("Small", "small"),
    ("Medium (Default)", "medium"),
    ("Large", "large"),
]

FONT_SIZE_OFFSETS = {"small": -2, "medium": 0, "large": 2}

# Store base sizes for rebuilding
_BASE_FONTS = {key: val for key, val in FONTS.items()}


def scale_fonts(offset: int = 0) -> None:
    """
    Rebuild FONTS dict in-place with scaled sizes.

    Args:
        offset: Integer point offset to apply (e.g. -2, 0, +4).
                Negative shrinks, positive enlarges. Floor of 8pt enforced.
    """
    for key, base in _BASE_FONTS.items():
        family = base[0]
        size = max(8, base[1] + offset)
        if len(base) == 3:
            FONTS[key] = (family, size, base[2])
        else:
            FONTS[key] = (family, size)


# =============================================================================
# THEME MODE HELPERS
# =============================================================================


def get_mode() -> str:
    """
    Get current appearance mode as 'light' or 'dark'.

    Returns:
        'light' or 'dark' (resolves 'system' to actual mode).
    """
    mode = ctk.get_appearance_mode()
    return mode.lower() if mode else "dark"


def get_color(key: str) -> str:
    """
    Get the color value for the current appearance mode.

    For (light, dark) tuples, returns the appropriate value.
    For plain strings, returns as-is (backwards compatible).

    Args:
        key: Color key from COLORS dict.

    Returns:
        Hex color string for the current mode.
    """
    value = COLORS[key]
    if isinstance(value, tuple):
        return value[0] if get_mode() == "light" else value[1]
    return value


def color_pair(key: str) -> tuple[str, str]:
    """
    Get the (light, dark) color tuple for a key.

    For plain strings, returns (value, value).

    Args:
        key: Color key from COLORS dict.

    Returns:
        (light_value, dark_value) tuple.
    """
    value = COLORS[key]
    if isinstance(value, tuple):
        return value
    return (value, value)


def resolve_tags(tag_dict: dict) -> dict:
    """
    Resolve a tag config dict for the current appearance mode.

    Tkinter tag_config() requires single color strings, not (light, dark) tuples.
    This resolves all tuple values in the tag dict to the current mode's color.

    Args:
        tag_dict: Dict of {tag_name: {property: color_or_tuple, ...}}.

    Returns:
        New dict with all color tuples resolved to single strings.
    """
    mode = get_mode()
    idx = 0 if mode == "light" else 1
    resolved = {}
    for tag_name, config in tag_dict.items():
        resolved_config = {}
        for prop, value in config.items():
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and all(isinstance(v, str) for v in value)
            ):
                resolved_config[prop] = value[idx]
            else:
                resolved_config[prop] = value
        resolved[tag_name] = resolved_config
    return resolved


# =============================================================================
# COLORS
# =============================================================================
# All values are (light_mode, dark_mode) tuples.
# Use get_color("key") for current mode, or color_pair("key") for CTk widgets.
#
# Light mode palette:
# - Backgrounds: warm off-whites (#f0f0f0 range), not pure white
# - Text: dark grays (#1a1a1a to #666666)
# - Attention colors (red/orange/yellow): slightly deeper for light bg contrast

COLORS = {
    # Backgrounds
    "bg_dark": ("#e8e8e8", "#2b2b2b"),  # Main panel/frame background
    "bg_darker": ("#dcdcdc", "#1e1e1e"),  # Deeper background (textboxes)
    "bg_card": ("#d8d8d8", "#333333"),  # Card/section background
    "bg_input": ("#d0d0d0", "#404040"),  # Input field backgrounds
    "bg_hover": ("#c8c8c8", "#505050"),  # Hover state
    # Buttons - Primary
    "btn_primary": ("#2d5a87", "#2d5a87"),
    "btn_primary_hover": ("#3d6a97", "#3d6a97"),
    # Buttons - Secondary
    "btn_secondary": ("#9e9e9e", "#555555"),
    "btn_secondary_hover": ("#b0b0b0", "#666666"),
    # Buttons - Tertiary
    "btn_tertiary": ("#bdbdbd", "#444444"),
    "btn_tertiary_hover": ("#9e9e9e", "#555555"),
    # Buttons - Danger
    "btn_danger": ("#c62828", "#994444"),
    "btn_danger_hover": ("#d32f2f", "#bb5555"),
    # Buttons - Disabled
    "btn_disabled": ("#b0b0b0", "#6c757d"),
    "btn_disabled_hover": ("#9e9e9e", "#5a6268"),
    # Text
    "text_primary": ("#1a1a1a", "#e0e0e0"),  # Primary text
    "text_secondary": ("#666666", "#aaaaaa"),  # Muted text
    "text_disabled": ("#999999", "#666666"),  # Disabled text
    "text_white": ("#ffffff", "#ffffff"),  # Pure white (buttons, headings)
    # Semantic colors — attention-grabbing in both modes
    "success": ("#1b7a2f", "#28a745"),  # Keep/approved (green)
    "success_light": ("#4caf50", "#90EE90"),  # Light green (loaded feedback)
    "warning": ("#e6a800", "#ffc107"),  # Warning (yellow) — darker on light bg
    "danger": ("#c62828", "#dc3545"),  # Error/skip (red) — deeper on light bg
    "danger_light": ("#e57373", "#f5a9b0"),  # Light red (loaded feedback)
    "info": ("#0d7a8a", "#17a2b8"),  # Info (blue)
    # Search Panel text tags
    "qa_question": ("#1565c0", "#5dade2"),  # Question text (blue)
    "qa_question_default": ("#5b86a7", "#7dacd6"),  # Default question (muted blue)
    "qa_answer": ("#1a3a5c", "#aed6f1"),  # Answer text
    "qa_citation": ("#4a4a4a", "#d7dbdd"),  # Citation text
    "qa_source": ("#1b7a2f", "#52be80"),  # Source info (green)
    "qa_label": ("#5a6b7a", "#85929e"),  # Labels (gray)
    "qa_separator": ("#b0bec5", "#566573"),  # Section separators
    # Algorithm detection colors (vocabulary table)
    "algo_multi": ("#1b8a3a", "#5dde77"),  # Multiple algorithms (green)
    "algo_ner": ("#1565c0", "#7ec8e3"),  # NER only (blue)
    "algo_rake": ("#7b1fa2", "#c792ea"),  # RAKE only (purple)
    "algo_bm25": ("#e67e22", "#ffb347"),  # BM25 only (orange)
    "algo_llm": ("#00838f", "#4dd0e1"),  # Reserved (cyan)
    # Table rows
    "row_odd": ("#f0f0f0", "#2b2b2b"),  # Odd row
    "row_even": ("#e4e4e4", "#353535"),  # Even row
    "row_selected": ("#3470b6", "#3470b6"),  # Selected row (same both modes)
    "row_text": ("#1a1a1a", "#e0e0e0"),  # Row text
    # Output pane
    "output_pane": ("#e8e8e8", "#1a1a2e"),
    # Progress/status
    "progress_partial": ("#cc8800", "#ffaa00"),  # Partial progress (orange)
    "progress_complete": ("#1b7a2f", "#00cc66"),  # Complete (green)
    # Status bar
    "status_error": ("#c75000", "#f0932b"),  # Orange — visible but not alarming
    "status_bar_bg": ("#d8d8e8", "#1a1a2e"),  # Matches output pane
    # Progress bar
    "progress_bar": ("#3d8bfd", "#3d8bfd"),
    # Drag-and-drop zone
    "drop_zone_border": ("#3d8bfd", "#3d8bfd"),
    "drop_zone_bg": ("#d0e0f0", "#1a3050"),
    "drop_zone_idle_border": ("#b0b0b0", "#555555"),
    "drop_zone_idle_bg": ("#e8e8e8", "#2b2b2b"),
    # System monitor
    "monitor_bg": ("#d8e8d8", "#1a3a1a"),
    # Hallucination verification colors
    "verify_verified": ("#1b7a2f", "#28a745"),  # Green
    "verify_uncertain": ("#e6a800", "#ffc107"),  # Yellow
    "verify_suspicious": ("#c75000", "#fd7e14"),  # Orange
    "verify_unreliable": ("#c62828", "#dc3545"),  # Red
    "verify_hallucinated": ("#999999", "#888888"),  # Gray + strikethrough
    # Reliability header colors
    "reliability_high": ("#1b7a2f", "#28a745"),  # Green
    "reliability_medium": ("#e6a800", "#ffc107"),  # Yellow
    "reliability_low": ("#c62828", "#dc3545"),  # Red
    # ---------------------------------------------------------------
    # UI element colors (used directly by specific widgets)
    # ---------------------------------------------------------------
    # Menus (tkinter Menu widgets don't support CTk tuples)
    "menu_bg": ("#e8e8e8", "#212121"),
    "menu_fg": ("#1a1a1a", "#ffffff"),
    "menu_active_bg": ("#d0d0d0", "#333333"),
    "menu_active_fg": ("#1a1a1a", "#ffffff"),
    "menu_disabled_fg": ("#999999", "#666666"),
    # Treeview (ttk widgets, need single values via get_color)
    "tree_bg": ("#f0f0f0", "#2b2b2b"),
    "tree_fg": ("#1a1a1a", "#ffffff"),
    "tree_field_bg": ("#f0f0f0", "#2b2b2b"),
    "tree_heading_bg": ("#d0d0d0", "#404040"),
    "tree_heading_fg": ("#1a1a1a", "#ffffff"),
    "tree_heading_hover": ("#c0c0c0", "#505050"),
    "tree_scroll_bg": ("#d0d0d0", "#404040"),
    "tree_scroll_trough": ("#f0f0f0", "#2b2b2b"),
    "tree_arrow": ("#1a1a1a", "#ffffff"),
    # File review treeview heading (slightly different shade)
    "tree_file_heading_bg": ("#c0c5c8", "#565b5e"),
    "tree_file_heading_hover": ("#b0b5b8", "#6c757d"),
    # Placeholder text (attention-grabbing)
    "placeholder_golden": ("#b07800", "#E8A838"),  # Disabled search entry
    "placeholder_red": ("#c62828", "#E05555"),  # Search failure/unavailable
    # Corpus dropdown error
    "corpus_error_text": ("#c62828", "#e07070"),  # Red — no corpora available
    # Separator
    "separator": ("#c8c8c8", "#2b2b2b"),
    # Tooltips
    "tooltip_bg": ("#FFF9C4", "#424242"),
    "tooltip_fg": ("#333333", "#FFFFFF"),
    # Find bar highlights
    "find_highlight_bg": ("#FFEB3B", "#FFEB3B"),  # Yellow (same both modes)
    "find_highlight_fg": ("#000000", "#000000"),
    "find_current_bg": ("#FF9800", "#FF9800"),  # Orange current match
    "find_current_fg": ("#000000", "#000000"),
    # Term highlight in context viewer
    "term_highlight_bg": ("#FFEB3B", "#FFEB3B"),
    "term_highlight_fg": ("#000000", "#000000"),
    # Document header in context viewer
    "doc_header_fg": ("#1565c0", "#4A9EFF"),
    # "More..." note text
    "more_note_fg": ("#666666", "#888888"),
    # Dialog colors
    "dialog_link": ("#1565c0", "#5dade2"),  # Clickable links
    "dialog_subtitle": ("#666666", "#aaaaaa"),  # Subtitle/description text
    "dialog_muted": ("#777777", "#888888"),  # Muted text in dialogs
    "dialog_separator": ("#c0c0c0", "#555555"),  # Separator lines
    "dialog_chosen": ("#1565c0", "#4a9eff"),  # Chosen/selected item
    "dialog_rejected": ("#c62828", "#cc6666"),  # Rejected/error items
    # Warning banner (settings corpus warning)
    "warning_banner_bg": ("#FFF3CD", "#FFF3CD"),  # Yellow-cream
    "warning_banner_fg": ("#856404", "#856404"),  # Dark brown text
    # System monitor status tiers
    "sysmon_good_bg": ("#d8e8d8", "#1a3a1a"),
    "sysmon_good_fg": ("#1b7a2f", "#90EE90"),
    "sysmon_warn_bg": ("#e8e8d0", "#3a3a1a"),
    "sysmon_warn_fg": ("#b08800", "#FFEB3B"),
    "sysmon_caution_bg": ("#e8ddd0", "#3a2a1a"),
    "sysmon_caution_fg": ("#c75000", "#FFA500"),
    "sysmon_critical_bg": ("#e8d0d0", "#3a1a1a"),
    "sysmon_critical_fg": ("#c62828", "#FF4444"),
    # Extracting status (purple)
    "extracting": ("#7b1fa2", "#9b59b6"),
    # Corpus dialog buttons
    "corpus_add_btn": ("#217346", "#217346"),
    "corpus_add_hover": ("#1a5c38", "#1a5c38"),
    "corpus_delete_hover": ("#c42b1c", "#c42b1c"),
    # Settings dialog tab buttons
    "settings_tab_selected": ("#3B8ED0", "#1F6AA5"),
    "settings_tab_hover": ("#36719F", "#144870"),
    # Remove icon (file review)
    "remove_icon": ("#c62828", "#e67e22"),
    # Question editor list bg
    "question_list_bg": ("#e8e8e8", "#2b2b2b"),
    # Splash screen
    "splash_bg": ("#e0e0e8", "#1a1a2e"),
    "splash_fg": ("#1a1a1a", "#ffffff"),
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
    "success": {
        "fg_color": COLORS["success"],
        "hover_color": COLORS["success"],
    },
    "caution": {
        "fg_color": ("#6a3a9e", "#7744aa"),
        "hover_color": ("#8650b8", "#9955cc"),
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
# SEARCH TEXT TAGS
# =============================================================================
# Used with CTkTextbox.tag_config(name, cnf={...})
# IMPORTANT: Must use cnf={} parameter, not keyword arguments

QA_TEXT_TAGS = {
    "question": {"foreground": COLORS["qa_question"], "font": FONTS["qa_question"]},
    "question_default": {
        "foreground": COLORS["qa_question_default"],
        "font": FONTS["qa_question_default"],
    },
    "label": {"foreground": COLORS["qa_label"], "font": FONTS["qa_label"]},
    "answer": {"foreground": COLORS["qa_answer"]},
    "citation": {"foreground": COLORS["qa_citation"]},
    "source": {"foreground": COLORS["qa_source"], "font": FONTS["qa_source"]},
    "separator": {"foreground": COLORS["qa_separator"]},
    # Hallucination verification span tags
    "verify_verified": {"foreground": COLORS["verify_verified"]},
    "verify_uncertain": {"foreground": COLORS["verify_uncertain"]},
    "verify_suspicious": {"foreground": COLORS["verify_suspicious"]},
    "verify_unreliable": {"foreground": COLORS["verify_unreliable"]},
    "verify_hallucinated": {"foreground": COLORS["verify_hallucinated"], "overstrike": True},
    # Reliability header tags (bold + colored)
    "reliability_high": {"foreground": COLORS["reliability_high"], "font": FONTS["body_bold"]},
    "reliability_medium": {"foreground": COLORS["reliability_medium"], "font": FONTS["body_bold"]},
    "reliability_low": {"foreground": COLORS["reliability_low"], "font": FONTS["body_bold"]},
    # Legend label
    "legend_label": {"foreground": COLORS["text_secondary"], "font": FONTS["small"]},
    # Score detail (retrieval confidence, shown alongside verification)
    "score_detail": {"foreground": COLORS["text_secondary"], "font": FONTS["body_bold"]},
}


# =============================================================================
# VOCABULARY TABLE TAGS
# =============================================================================
# Used with ttk.Treeview.tag_configure()

VOCAB_TABLE_TAGS = {
    # Distinguish session (user clicked) vs loaded (from dataset) feedback
    "rated_up_session": {"foreground": COLORS["success"]},  # User clicked Keep
    "rated_up_loaded": {"foreground": COLORS["success_light"]},  # From dataset
    "rated_down_session": {"foreground": COLORS["danger"]},  # User clicked Skip
    "rated_down_loaded": {"foreground": COLORS["danger_light"]},  # From dataset
    "found_multi": {"foreground": COLORS["algo_multi"]},
    "found_ner": {"foreground": COLORS["algo_ner"]},
    "found_rake": {"foreground": COLORS["algo_rake"]},
    "found_bm25": {"foreground": COLORS["algo_bm25"]},
    "found_llm": {"foreground": COLORS["algo_llm"]},
    "oddrow": {"background": COLORS["row_odd"], "foreground": COLORS["row_text"]},
    "evenrow": {"background": COLORS["row_even"], "foreground": COLORS["row_text"]},
    # Muted rows for filtered/lesser terms section
    "filtered_oddrow": {"background": COLORS["row_odd"], "foreground": "#888888"},
    "filtered_evenrow": {"background": COLORS["row_even"], "foreground": "#888888"},
    "filtered_rated_up_session": {"foreground": COLORS["success"]},
    "filtered_rated_up_loaded": {"foreground": COLORS["success_light"]},
    "filtered_rated_down_session": {"foreground": COLORS["danger"]},
    "filtered_rated_down_loaded": {"foreground": COLORS["danger_light"]},
}


# =============================================================================
# FILE STATUS TAGS
# =============================================================================
# Used with FileReviewTable (widgets.py)

FILE_STATUS_TAGS = {
    "green": {"foreground": COLORS["success"]},
    "yellow": {"foreground": COLORS["warning"]},
    "red": {"foreground": COLORS["danger"]},
    "pending": {"foreground": COLORS["text_secondary"]},
    "extracting": {"foreground": COLORS["extracting"]},
}
