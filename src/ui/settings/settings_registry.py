"""
Settings Registry for CasePrepd.

Provides a declarative way to define application settings with metadata
for automatic UI generation. Adding a new setting requires only a single
SettingsRegistry.register() call - no UI code changes needed.

Architecture:
- SettingDefinition: Dataclass with all metadata for one setting
- SettingsRegistry: Class-level registry that organizes settings by category
- _register_all_settings(): Called on import to register all app settings

Example - Adding a new setting:
    SettingsRegistry.register(SettingDefinition(
        key="my_new_setting",
        label="My New Feature",
        category="General",  # Creates tab if needed
        setting_type=SettingType.CHECKBOX,
        tooltip="Description of what this does.",
        default=False,
        getter=lambda: prefs.get("my_new_setting", False),
        setter=lambda v: prefs.set("my_new_setting", v),
    ))
"""

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar


class SettingType(Enum):
    """
    Types of settings with corresponding UI widgets.

    Each type maps to a widget class in settings_widgets.py:
    - SLIDER: SliderSetting (numeric range)
    - CHECKBOX: CheckboxSetting (boolean)
    - DROPDOWN: DropdownSetting (selection)
    - SPINBOX: SpinboxSetting (integer +/-)
    - PATH: Reserved for future file/folder picker
    - BUTTON: ActionButton (executes action on click)
    - CUSTOM: CustomWidgetSetting (custom widget factory)
    """

    SLIDER = "slider"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"
    PATH = "path"
    SPINBOX = "spinbox"
    BUTTON = "button"
    CUSTOM = "custom"  # For complex widgets like question list


@dataclass
class SettingDefinition:
    """
    Complete metadata for a single application setting.

    The SettingsDialog reads these definitions and auto-generates
    appropriate widgets, tooltips, and save/load behavior.

    Attributes:
        key: Unique identifier (used for storage in preferences)
        label: Display name shown in the UI
        category: Tab name in settings dialog (groups related settings)
        setting_type: Widget type to render
        tooltip: Explanation shown on hover. Can be a string or callable
            returning a string (for dynamic content evaluated at display time).
        default: Default value when no preference is saved
        min_value: Minimum value (for SLIDER, SPINBOX)
        max_value: Maximum value (for SLIDER, SPINBOX)
        step: Increment between values (for SLIDER)
        options: List of (display_text, value) tuples (for DROPDOWN).
            Can also be a callable returning such a list (for dynamic options
            evaluated when the dialog opens, not at registration time).
        getter: Function that returns the current value
        setter: Function that applies a new value
        action: Function to execute on click (for BUTTON)
        widget_factory: Function(parent) -> widget (for CUSTOM type)
    """

    key: str
    label: str
    category: str
    setting_type: SettingType
    tooltip: str | Callable[[], str]
    default: Any
    min_value: float = None
    max_value: float = None
    step: float = 1
    options: list | Callable[[], list] = field(default_factory=list)
    getter: Callable[[], Any] = None
    setter: Callable[[Any], None] = None
    action: Callable[[], None] = None
    widget_factory: Callable = None
    section: str | None = None  # Sub-group within a category (for collapsible sections)


class SettingsRegistry:
    """
    Global registry of all application settings.

    Settings are organized by category for tabbed display. The dialog
    reads from this registry to generate its UI dynamically.

    Usage:
        # Register a setting
        SettingsRegistry.register(my_setting_definition)

        # Get all categories (for tabs)
        categories = SettingsRegistry.get_categories()

        # Get settings in a category
        settings = SettingsRegistry.get_settings_for_category("Performance")
    """

    _settings: ClassVar[dict[str, SettingDefinition]] = {}
    _categories: ClassVar[dict[str, list[str]]] = {}  # category -> [setting_keys]
    _category_order: ClassVar[list[str]] = []  # Preserve registration order

    @classmethod
    def register(cls, setting: SettingDefinition) -> None:
        """
        Register a setting definition.

        Args:
            setting: SettingDefinition with all metadata.
        """
        cls._settings[setting.key] = setting

        if setting.category not in cls._categories:
            cls._categories[setting.category] = []
            cls._category_order.append(setting.category)

        if setting.key not in cls._categories[setting.category]:
            cls._categories[setting.category].append(setting.key)

    @classmethod
    def get_categories(cls) -> list[str]:
        """Get all category names in registration order."""
        return cls._category_order.copy()

    @classmethod
    def get_settings_for_category(cls, category: str) -> list[SettingDefinition]:
        """Get all settings in a category."""
        keys = cls._categories.get(category, [])
        return [cls._settings[k] for k in keys]

    @classmethod
    def get_all_settings(cls) -> list[SettingDefinition]:
        """Get all registered settings."""
        return list(cls._settings.values())

    @classmethod
    def get_setting(cls, key: str) -> SettingDefinition | None:
        """Get a specific setting by key."""
        return cls._settings.get(key)

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (for testing)."""
        cls._settings.clear()
        cls._categories.clear()
        cls._category_order.clear()


def _register_all_settings():
    """
    Register all CasePrepd settings.

    This function is called on module import. To add a new setting,
    add a SettingsRegistry.register() call here.

    Settings are grouped by category (tab):
    - Performance: Parallel processing, CPU allocation
    - Summarization: AI summary settings
    - Vocabulary: Term extraction settings
    """
    # Import lazily to avoid circular imports
    import os

    from src.config import (
        BM25_ENABLED,
        CORPUS_DIR,
        VOCABULARY_SORT_METHOD,
    )
    from src.user_preferences import get_user_preferences

    prefs = get_user_preferences()

    # ===================================================================
    # APPEARANCE TAB
    # ===================================================================

    SettingsRegistry.register(
        SettingDefinition(
            key="appearance_mode",
            label="Theme",
            category="Appearance",
            setting_type=SettingType.DROPDOWN,
            tooltip=(
                "Choose the application color theme.\n\n"
                "• Dark: Dark backgrounds with light text (default)\n"
                "• Light: Light backgrounds with dark text\n"
                "• System: Follow your Windows theme setting\n\n"
                "Takes effect immediately."
            ),
            default="Dark",
            options=[
                ("Dark", "Dark"),
                ("Light", "Light"),
                ("System", "System"),
            ],
            getter=lambda: prefs.get("appearance_mode", "Dark"),
            setter=lambda v: _apply_appearance_mode(v),
        )
    )

    def _apply_appearance_mode(mode: str):
        """Apply appearance mode change and refresh styles."""
        import customtkinter as ctk

        prefs.set("appearance_mode", mode)
        ctk.set_appearance_mode(mode)

        from src.ui.styles import reinitialize_styles

        reinitialize_styles()

    SettingsRegistry.register(
        SettingDefinition(
            key="font_size_offset",
            label="Font Size Adjustment",
            category="Appearance",
            setting_type=SettingType.SPINBOX,
            tooltip=(
                "Adjust the font size used throughout the application,\n"
                "including table rows (vocabulary, Q&A, documents).\n\n"
                "Enter a point offset (positive = larger, negative = smaller).\n"
                "Examples: +4 for high-DPI screens, -2 for compact layout.\n\n"
                "Requires restart to take effect."
            ),
            default=0,
            min_value=-4,
            max_value=10,
            getter=lambda: prefs.get("font_size_offset", 0),
            setter=lambda v: prefs.set("font_size_offset", v),
        )
    )

    SettingsRegistry.register(
        SettingDefinition(
            key="ui_scale_pct",
            label="UI Scale (%)",
            category="Appearance",
            setting_type=SettingType.SLIDER,
            tooltip=(
                "Scale all widget dimensions (buttons, tables, dialogs).\n\n"
                "Does NOT affect font sizes (use Font Size Adjustment for that).\n"
                "Useful for 4K monitors where widgets appear cramped.\n\n"
                "• 75%: Compact layout\n"
                "• 100%: Default\n"
                "• 125-150%: Recommended for 4K monitors\n"
                "• 200%: Maximum\n\n"
                "Requires restart to take effect."
            ),
            default=100,
            min_value=75,
            max_value=200,
            step=25,
            getter=lambda: prefs.get("ui_scale_pct", 100),
            setter=lambda v: prefs.set("ui_scale_pct", int(v)),
        )
    )

    # ===================================================================
    # VOCABULARY TAB
    # ===================================================================

    def _create_corpus_warning_widget(parent):
        """Factory for corpus status warning banner with dynamic refresh."""
        import customtkinter as ctk

        from src.services import VocabularyService
        from src.ui.theme import COLORS

        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame._warning_frame = None
        frame._warning_label = None

        def update_warning():
            """Update the warning banner based on current corpus status."""
            corpus_manager = VocabularyService().get_corpus_manager()
            doc_count = corpus_manager.get_document_count()

            if doc_count < 5:
                warning_text = (
                    f"Corpus not ready ({doc_count}/5 documents). "
                    "ML predictions are less accurate without a corpus of past transcripts. "
                    "Add documents in Settings > Corpus."
                )

                if frame._warning_frame is None:
                    frame._warning_frame = ctk.CTkFrame(
                        frame, fg_color=COLORS["warning_banner_bg"], corner_radius=6
                    )
                    frame._warning_frame.pack(fill="x", pady=(0, 10), padx=5)
                    frame._warning_label = ctk.CTkLabel(
                        frame._warning_frame,
                        text=warning_text,
                        text_color=COLORS["warning_banner_fg"],
                        wraplength=400,
                        justify="left",
                        anchor="w",
                    )
                    frame._warning_label.pack(anchor="w", padx=10, pady=8)
                else:
                    frame._warning_label.configure(text=warning_text)
                    if not frame._warning_frame.winfo_ismapped():
                        frame._warning_frame.pack(fill="x", pady=(0, 10), padx=5)
            else:
                if frame._warning_frame is not None and frame._warning_frame.winfo_ismapped():
                    frame._warning_frame.pack_forget()

        update_warning()
        frame.bind("<Map>", lambda e: update_warning())

        return frame

    SettingsRegistry.register(
        SettingDefinition(
            key="corpus_status_warning",
            label="",  # No label for banner
            category="Vocabulary",
            setting_type=SettingType.CUSTOM,
            tooltip="",
            default=None,
            widget_factory=_create_corpus_warning_widget,
        )
    )

    SettingsRegistry.register(
        SettingDefinition(
            key="vocab_sort_method",
            label="Sort vocabulary by",
            category="Vocabulary",
            setting_type=SettingType.DROPDOWN,
            tooltip=(
                "Controls how vocabulary terms are sorted in the results table.\n\n"
                "• Quality Score: Terms the ML model predicts you'll approve appear first. "
                "This improves as you rate more terms.\n\n"
                "• Rarity: Unusual/rare words appear first (based on Google Books corpus)."
            ),
            options=[
                ("Quality Score", "quality_score"),
                ("Rarity", "rarity"),
            ],
            default="quality_score" if VOCABULARY_SORT_METHOD == "quality_score" else "rarity",
            getter=lambda: prefs.get("vocab_sort_method", VOCABULARY_SORT_METHOD),
            setter=lambda v: prefs.set("vocab_sort_method", v),
        )
    )

    # CSV export column format setting
    SettingsRegistry.register(
        SettingDefinition(
            key="vocab_export_format",
            label="CSV export columns",
            category="Vocabulary",
            setting_type=SettingType.DROPDOWN,
            tooltip=(
                "Controls which columns are included when saving vocabulary to CSV. "
                "'All columns' includes Quality Score, Frequency, and Rank for "
                "Excel filtering. 'Basic' exports Term, Type, and Role. "
                "'Terms only' exports just the vocabulary terms."
            ),
            default="basic",
            options=[
                ("All columns (with quality metrics)", "all"),
                ("Basic (Term, Type, Role)", "basic"),
                ("Terms only", "terms_only"),
            ],
            getter=lambda: prefs.get("vocab_export_format", "basic"),
            setter=lambda v: prefs.set("vocab_export_format", v),
        )
    )

    # Column visibility configuration
    def _create_column_visibility_widget(parent):
        """Factory function to create the ColumnVisibilityWidget."""
        from src.ui.settings.columns_widget import ColumnVisibilityWidget

        return ColumnVisibilityWidget(parent)

    def _save_column_visibility(visibility: dict) -> None:
        """Persist column visibility dict to preferences."""
        if visibility is not None:
            prefs.set("vocab_column_visibility", visibility)

    SettingsRegistry.register(
        SettingDefinition(
            key="vocab_column_visibility",
            label="",  # Widget has its own header
            category="Vocabulary",
            setting_type=SettingType.CUSTOM,
            tooltip="",  # Widget has its own tooltip
            default=None,
            getter=lambda: prefs.get("vocab_column_visibility", {}),
            setter=_save_column_visibility,
            widget_factory=_create_column_visibility_widget,
        )
    )

    # BM25 Corpus-based term extraction
    SettingsRegistry.register(
        SettingDefinition(
            key="bm25_enabled",
            label="Enable Corpus Analysis (BM25)",
            category="Vocabulary",
            setting_type=SettingType.CHECKBOX,
            tooltip=(
                "Compare your current document against your library of past "
                "transcripts to identify case-specific terminology. Terms that "
                "are frequent in this document but rare in your corpus are likely "
                "important. Requires 5+ documents in your corpus folder.\n\n"
                "🔒 Privacy: All analysis happens locally on your computer - "
                "no documents or data are ever sent to external servers."
            ),
            default=BM25_ENABLED,
            getter=lambda: prefs.get("bm25_enabled", BM25_ENABLED),
            setter=lambda v: prefs.set("bm25_enabled", v),
        )
    )

    def _open_corpus_folder():
        """Open the corpus folder in the system file explorer."""
        # UI-003: Verify CORPUS_DIR exists before trying to open
        if not CORPUS_DIR.exists():
            CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            # Windows
            os.startfile(str(CORPUS_DIR))
        except AttributeError:
            # macOS/Linux fallback
            import subprocess
            import sys

            if sys.platform == "darwin":
                subprocess.run(["open", str(CORPUS_DIR)])
            else:
                subprocess.run(["xdg-open", str(CORPUS_DIR)])

    SettingsRegistry.register(
        SettingDefinition(
            key="open_corpus_folder",
            label="Open Corpus Folder",
            category="Vocabulary",
            setting_type=SettingType.BUTTON,
            tooltip=(
                "Add your past transcripts (PDF, TXT, RTF) to this folder to "
                "build your personal vocabulary baseline. The more documents "
                "you add, the better the system identifies unusual terms "
                "specific to each new case.\n\n"
                "📁 Location: " + str(CORPUS_DIR)
            ),
            default=None,
            action=_open_corpus_folder,
        )
    )

    # ML Model Reset buttons
    def _reset_vocab_model():
        """Reset vocabulary ML model to default (keep feedback history)."""
        from tkinter import messagebox

        result = messagebox.askyesno(
            "Reset Vocabulary Model",
            "Reset the vocabulary ranking model to default settings?\n\n"
            "This will undo any personalization from your thumbs up/down "
            "feedback, but your feedback history will be preserved.\n\n"
            "You can retrain the model later using your existing feedback.",
            icon="warning",
        )

        if result:
            from src.services import VocabularyService

            learner = VocabularyService().get_meta_learner()
            if learner.reset_to_default():
                messagebox.showinfo(
                    "Reset Complete",
                    "Vocabulary model has been reset to default.\n\n"
                    "Your feedback history is preserved. The model will "
                    "retrain when you provide more feedback.",
                )
            else:
                messagebox.showerror(
                    "Reset Failed",
                    "Failed to reset vocabulary model. Check the log file for details.",
                )

    SettingsRegistry.register(
        SettingDefinition(
            key="reset_vocab_model",
            label="Reset Vocabulary Model",
            category="Vocabulary",
            setting_type=SettingType.BUTTON,
            tooltip=(
                "Reset the vocabulary ranking model to its default (shipped) state. "
                "This undoes any personalization from your thumbs up/down feedback, "
                "but keeps your feedback history so you can retrain later.\n\n"
                "Use this if the model seems to be ranking terms poorly after "
                "you've given it feedback."
            ),
            default=None,
            action=_reset_vocab_model,
        )
    )

    def _reset_vocab_model_and_history():
        """Reset vocabulary ML model AND clear all feedback history."""
        from tkinter import messagebox

        result = messagebox.askyesno(
            "Reset Model and Clear History",
            "⚠️ CAUTION: This will:\n\n"
            "• Reset the vocabulary ranking model to default\n"
            "• DELETE all your thumbs up/down feedback history\n\n"
            "This action cannot be undone. Are you sure?",
            icon="warning",
        )

        if result:
            # Double-check for destructive action
            confirm = messagebox.askyesno(
                "Confirm Complete Reset",
                "Are you absolutely sure?\n\n"
                "All feedback you've given will be permanently deleted.",
                icon="warning",
            )

            if confirm:
                from src.services import VocabularyService

                vocab_svc = VocabularyService()
                learner = vocab_svc.get_meta_learner()
                feedback_manager = vocab_svc.get_feedback_manager()

                model_ok = learner.reset_to_default()
                feedback_ok = feedback_manager.clear_all_feedback()

                if model_ok and feedback_ok:
                    messagebox.showinfo(
                        "Reset Complete",
                        "Vocabulary model and feedback history have been reset.\n\n"
                        "The system is now using default settings.",
                    )
                else:
                    messagebox.showerror(
                        "Reset Partially Failed",
                        f"Model reset: {'OK' if model_ok else 'FAILED'}\n"
                        f"Feedback clear: {'OK' if feedback_ok else 'FAILED'}\n\n"
                        "Check the log file for details.",
                    )

    SettingsRegistry.register(
        SettingDefinition(
            key="reset_vocab_model_and_history",
            label="Reset Model and Clear History",
            category="Vocabulary",
            setting_type=SettingType.BUTTON,
            tooltip=(
                "⚠️ COMPLETE RESET: Resets the vocabulary model AND deletes all "
                "your thumbs up/down feedback history. This cannot be undone.\n\n"
                "Use this for a complete fresh start if you want to begin "
                "personalizing from scratch."
            ),
            default=None,
            action=_reset_vocab_model_and_history,
        )
    )

    # Model & Feedback Export/Import buttons
    def _export_vocab_model():
        """Export user vocabulary model from settings."""
        from tkinter import filedialog, messagebox

        from src.services.model_io_service import export_user_model

        dest = filedialog.asksaveasfilename(
            title="Export Vocabulary Model",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl")],
            initialfile="vocab_model.pkl",
        )
        if not dest:
            return
        from pathlib import Path

        ok, msg = export_user_model(Path(dest))
        if ok:
            messagebox.showinfo("Export Complete", msg)
        else:
            messagebox.showerror("Export Failed", msg)

    def _import_vocab_model():
        """Import vocabulary model from settings."""
        from tkinter import filedialog, messagebox

        messagebox.showwarning(
            "Security Warning",
            "Only load model files from sources you trust.\n"
            "Model files can contain executable code.\n\n"
            "Press OK to continue.",
        )

        src = filedialog.askopenfilename(
            title="Import Vocabulary Model",
            filetypes=[("Pickle files", "*.pkl")],
        )
        if not src:
            return
        from pathlib import Path

        from src.services.model_io_service import import_user_model

        ok, msg = import_user_model(Path(src))
        if ok:
            messagebox.showinfo("Import Complete", msg)
        else:
            messagebox.showerror("Import Failed", msg)

    def _export_vocab_feedback():
        """Export user feedback history from settings."""
        from tkinter import filedialog, messagebox

        from src.services import VocabularyService
        from src.services.model_io_service import export_user_feedback

        feedback_mgr = VocabularyService().get_feedback_manager()
        dest = filedialog.asksaveasfilename(
            title="Export Feedback History",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="feedback_history.csv",
        )
        if not dest:
            return
        from pathlib import Path

        ok, msg = export_user_feedback(Path(dest), feedback_mgr)
        if ok:
            messagebox.showinfo("Export Complete", msg)
        else:
            messagebox.showerror("Export Failed", msg)

    def _import_vocab_feedback():
        """Import feedback history from settings."""
        from tkinter import filedialog, messagebox

        src = filedialog.askopenfilename(
            title="Import Feedback History",
            filetypes=[("CSV files", "*.csv")],
        )
        if not src:
            return

        result = messagebox.askyesnocancel(
            "Import Mode",
            "How should the imported feedback be combined?\n\n"
            "Yes = Replace (old data backed up)\n"
            "No = Append to existing\n"
            "Cancel = Abort",
        )
        if result is None:
            return
        mode = "replace" if result else "append"

        from pathlib import Path

        from src.services import VocabularyService
        from src.services.model_io_service import import_user_feedback

        feedback_mgr = VocabularyService().get_feedback_manager()
        ok, msg, count = import_user_feedback(Path(src), mode, feedback_mgr)

        if not ok:
            messagebox.showerror("Import Failed", msg)
            return

        retrain = messagebox.askyesno(
            "Retrain Model?",
            f"{msg}\n\nRetrain the vocabulary model with the new feedback?",
        )
        if retrain:
            learner = VocabularyService().get_meta_learner()
            learner.train()
            messagebox.showinfo("Import Complete", f"Imported {count} records. Model retrained.")
        else:
            messagebox.showinfo("Import Complete", msg)

    SettingsRegistry.register(
        SettingDefinition(
            key="export_vocab_model",
            label="Export Model",
            category="Vocabulary",
            setting_type=SettingType.BUTTON,
            tooltip="Export your personalized vocabulary model to a file for backup or transfer.",
            default=None,
            action=_export_vocab_model,
        )
    )

    SettingsRegistry.register(
        SettingDefinition(
            key="import_vocab_model",
            label="Import Model",
            category="Vocabulary",
            setting_type=SettingType.BUTTON,
            tooltip="Import a vocabulary model from a file. The current model will be backed up.",
            default=None,
            action=_import_vocab_model,
        )
    )

    SettingsRegistry.register(
        SettingDefinition(
            key="export_vocab_feedback",
            label="Export Feedback History",
            category="Vocabulary",
            setting_type=SettingType.BUTTON,
            tooltip="Export your feedback history (thumbs up/down ratings) to a CSV file.",
            default=None,
            action=_export_vocab_feedback,
        )
    )

    SettingsRegistry.register(
        SettingDefinition(
            key="import_vocab_feedback",
            label="Import Feedback History",
            category="Vocabulary",
            setting_type=SettingType.BUTTON,
            tooltip="Import feedback history from a CSV file. Choose to replace or append.",
            default=None,
            action=_import_vocab_feedback,
        )
    )

    # LLM vocabulary extraction setting (moved from Performance to Vocabulary)
    SettingsRegistry.register(
        SettingDefinition(
            key="vocab_use_llm",
            label="LLM vocabulary extraction",
            category="Vocabulary",
            setting_type=SettingType.DROPDOWN,
            tooltip=lambda: (
                "Controls whether LLM (Ollama) is used for vocabulary extraction "
                "in addition to NER (spaCy).\n\n"
                "• Auto: Enables LLM if a dedicated GPU is detected. Without a "
                "GPU, LLM extraction is very slow and often not worth the wait.\n"
                "• Always enable: Force LLM extraction on every run.\n"
                "• NER only (default): Only NER-based extraction will run. "
                "Fastest option, finds person names and organizations.\n\n"
                "Adding LLM extraction also finds medical terms, legal "
                "terminology, and domain-specific vocabulary.\n\n"
                f"Current status: {_get_gpu_status_for_tooltip()}"
            ),
            default="no",
            options=[
                ("NER only (no LLM)", "no"),
                ("Auto (enable if GPU detected)", "auto"),
                ("Always enable", "yes"),
            ],
            getter=lambda: prefs.get_vocab_llm_mode(),
            setter=lambda v: prefs.set_vocab_llm_mode(v),
        )
    )

    # Vocabulary filtering thresholds (user-configurable)
    SettingsRegistry.register(
        SettingDefinition(
            key="single_word_rarity_threshold",
            label="Single-word filtering threshold",
            category="Vocabulary",
            setting_type=SettingType.SLIDER,
            tooltip=(
                "Filter single words in the top X% of English vocabulary. "
                "Lower = more aggressive filtering, Higher = keep more words.\n\n"
                "Example: 0.50 filters the most common 50% of English words."
            ),
            default=0.50,
            min_value=0.10,
            max_value=0.90,
            step=0.05,
            getter=lambda: prefs.get("single_word_rarity_threshold", 0.50),
            setter=lambda v: prefs.set("single_word_rarity_threshold", float(v)),
        )
    )

    SettingsRegistry.register(
        SettingDefinition(
            key="phrase_rarity_threshold",
            label="Phrase filtering threshold",
            category="Vocabulary",
            setting_type=SettingType.SLIDER,
            tooltip=(
                "Filter multi-word phrases where all words are in the top X% of English. "
                "Lower = more aggressive filtering, Higher = keep more phrases.\n\n"
                "Example: 0.50 filters phrases where every word is in the top 50%."
            ),
            default=0.50,
            min_value=0.10,
            max_value=0.90,
            step=0.05,
            getter=lambda: prefs.get("phrase_rarity_threshold", 0.50),
            setter=lambda v: prefs.set("phrase_rarity_threshold", float(v)),
        )
    )

    # Additional vocabulary filtering controls
    SettingsRegistry.register(
        SettingDefinition(
            key="vocab_min_occurrences",
            label="Minimum term occurrences",
            category="Vocabulary",
            setting_type=SettingType.SPINBOX,
            tooltip=(
                "Filter terms appearing fewer than N times in your documents.\n\n"
                "Higher values filter more aggressively, removing OCR errors and "
                "one-off terms. Value of 1 keeps all terms.\n\n"
                "Note: Person names are exempt from this filter."
            ),
            default=2,
            min_value=1,
            max_value=5,
            getter=lambda: prefs.get("vocab_min_occurrences", 2),
            setter=lambda v: prefs.set("vocab_min_occurrences", int(v)),
        )
    )

    SettingsRegistry.register(
        SettingDefinition(
            key="vocab_score_floor",
            label="Minimum quality score",
            category="Vocabulary",
            setting_type=SettingType.SLIDER,
            tooltip=(
                "Filter terms with quality scores below this threshold.\n\n"
                "Higher values show only high-confidence results. "
                "Lower values include more terms but may include noise.\n\n"
                "The quality score is based on ML predictions of term usefulness."
            ),
            default=55,
            min_value=45,
            max_value=85,
            step=5,
            getter=lambda: prefs.get("vocab_score_floor", 55),
            setter=lambda v: prefs.set("vocab_score_floor", int(v)),
        )
    )

    SettingsRegistry.register(
        SettingDefinition(
            key="phrase_mean_rarity_threshold",
            label="Phrase mean commonality",
            category="Vocabulary",
            setting_type=SettingType.SLIDER,
            tooltip=(
                "Filter phrases where the AVERAGE word commonality exceeds this "
                "threshold. Lower values = more aggressive filtering.\n\n"
                "Example: 0.40 filters phrases where the average word is in the "
                "top 40% of common English words.\n\n"
                "Works alongside 'Phrase filtering threshold' which checks the "
                "RAREST word in the phrase."
            ),
            default=0.40,
            min_value=0.10,
            max_value=0.90,
            step=0.05,
            getter=lambda: prefs.get("phrase_mean_rarity_threshold", 0.40),
            setter=lambda v: prefs.set("phrase_mean_rarity_threshold", float(v)),
        )
    )

    # Non-NER Rarity Passthrough Thresholds
    # These settings control when RAKE/BM25-found terms pass through rarity filtering
    SettingsRegistry.register(
        SettingDefinition(
            key="non_ner_single_passthrough_threshold",
            label="RAKE/BM25 single-word passthrough",
            category="Vocabulary",
            setting_type=SettingType.SLIDER,
            tooltip=(
                "Allow single words found by RAKE/BM25 to pass through rarity "
                "filtering if their rarity score meets this threshold. "
                "Higher = stricter (fewer passthroughs).\n\n"
                "Words not in the dictionary are treated as rare (score = "
                "'Rarity score for unknown words' setting below)."
            ),
            default=0.80,
            min_value=0.50,
            max_value=0.95,
            step=0.05,
            getter=lambda: prefs.get("non_ner_single_passthrough_threshold", 0.80),
            setter=lambda v: prefs.set("non_ner_single_passthrough_threshold", float(v)),
        )
    )

    SettingsRegistry.register(
        SettingDefinition(
            key="non_ner_phrase_max_passthrough_threshold",
            label="RAKE/BM25 phrase passthrough (rarest word)",
            category="Vocabulary",
            setting_type=SettingType.SLIDER,
            tooltip=(
                "Allow multi-word RAKE/BM25 phrases to pass through rarity "
                "filtering if the rarest word's score meets this threshold. "
                "Both this AND the average threshold below must be met.\n\n"
                "Higher = stricter (fewer phrase passthroughs)."
            ),
            default=0.85,
            min_value=0.50,
            max_value=0.95,
            step=0.05,
            getter=lambda: prefs.get("non_ner_phrase_max_passthrough_threshold", 0.85),
            setter=lambda v: prefs.set("non_ner_phrase_max_passthrough_threshold", float(v)),
        )
    )

    SettingsRegistry.register(
        SettingDefinition(
            key="non_ner_phrase_mean_passthrough_threshold",
            label="RAKE/BM25 phrase passthrough (adjusted mean)",
            category="Vocabulary",
            setting_type=SettingType.SLIDER,
            tooltip=(
                "Allow multi-word RAKE/BM25 phrases to pass through rarity "
                "filtering if the adjusted mean word rarity meets this threshold. "
                "Both this AND the rarest-word threshold above must be met.\n\n"
                "The adjusted mean excludes common filler words (controlled by "
                "the 'common word floor' setting below).\n\n"
                "Higher = stricter (fewer phrase passthroughs)."
            ),
            default=0.65,
            min_value=0.30,
            max_value=0.80,
            step=0.05,
            getter=lambda: prefs.get("non_ner_phrase_mean_passthrough_threshold", 0.65),
            setter=lambda v: prefs.set("non_ner_phrase_mean_passthrough_threshold", float(v)),
        )
    )

    SettingsRegistry.register(
        SettingDefinition(
            key="non_ner_phrase_common_word_floor",
            label="Adjusted mean: common word floor",
            category="Vocabulary",
            setting_type=SettingType.SLIDER,
            tooltip=(
                "Words with rarity score below this floor are excluded from "
                "the adjusted mean rarity calculation. This prevents common "
                "filler words (like 'of', 'the', 'and') from dragging down "
                "the mean rarity of phrases that contain rare words.\n\n"
                "Example: 0.10 excludes the top 10% most common English words."
            ),
            default=0.10,
            min_value=0.05,
            max_value=0.30,
            step=0.05,
            getter=lambda: prefs.get("non_ner_phrase_common_word_floor", 0.10),
            setter=lambda v: prefs.set("non_ner_phrase_common_word_floor", float(v)),
        )
    )

    SettingsRegistry.register(
        SettingDefinition(
            key="non_ner_unknown_word_rarity",
            label="Rarity score for unknown words",
            category="Vocabulary",
            setting_type=SettingType.SLIDER,
            tooltip=(
                "Rarity score assigned to words not found in the Google frequency "
                "dataset. Higher values treat unknown words as rarer, making them "
                "more likely to be rescued.\n\n"
                "Unknown words are often proper nouns or specialized terms."
            ),
            default=0.85,
            min_value=0.50,
            max_value=1.00,
            step=0.05,
            getter=lambda: prefs.get("non_ner_unknown_word_rarity", 0.85),
            setter=lambda v: prefs.set("non_ner_unknown_word_rarity", float(v)),
        )
    )

    # Session: User-defined indicator patterns for vocabulary ML
    def _get_indicator_patterns(p):
        """Get current indicator pattern config with shipped defaults."""
        from src.config import (
            DEFAULT_NEGATIVE_INDICATORS,
            DEFAULT_NEGATIVE_REGEX_OVERRIDE,
            DEFAULT_POSITIVE_INDICATORS,
            DEFAULT_POSITIVE_REGEX_OVERRIDE,
        )

        return {
            "positive_strings": p.get("vocab_positive_indicators", DEFAULT_POSITIVE_INDICATORS),
            "negative_strings": p.get("vocab_negative_indicators", DEFAULT_NEGATIVE_INDICATORS),
            "positive_override": p.get(
                "vocab_positive_regex_override", DEFAULT_POSITIVE_REGEX_OVERRIDE
            ),
            "negative_override": p.get(
                "vocab_negative_regex_override", DEFAULT_NEGATIVE_REGEX_OVERRIDE
            ),
        }

    def _create_indicator_pattern_widget(parent):
        """Factory function to create the IndicatorPatternWidget."""
        from src.ui.settings.indicator_pattern_widget import IndicatorPatternWidget

        return IndicatorPatternWidget(parent)

    def _save_indicator_patterns(value: dict) -> None:
        """Save indicator patterns to preferences and trigger retrain."""
        if not isinstance(value, dict):
            return
        prefs.set("vocab_positive_indicators", value.get("positive_strings", []))
        prefs.set("vocab_negative_indicators", value.get("negative_strings", []))
        prefs.set("vocab_positive_regex_override", value.get("positive_override", ""))
        prefs.set("vocab_negative_regex_override", value.get("negative_override", ""))

        # Trigger preference learner retrain with new features
        # (indicator_patterns cache auto-invalidates when preferences change)
        try:
            from src.services import VocabularyService

            learner = VocabularyService().get_meta_learner()
            if learner.train():
                logger.info("Vocabulary model retrained with updated indicator patterns")
            else:
                logger.debug("Vocabulary model retrain skipped (insufficient data)")
        except Exception as e:
            logger.warning("Could not retrain vocabulary model: %s", e)

    SettingsRegistry.register(
        SettingDefinition(
            key="vocab_indicator_patterns",
            label="Indicator Patterns",
            category="Vocabulary",
            setting_type=SettingType.CUSTOM,
            tooltip=(
                "Define strings that indicate good or bad vocabulary terms.\n\n"
                "Positive indicators: Strings found in terms you tend to keep "
                "(e.g., 'dr.', 'plaintiff').\n\n"
                "Negative indicators: Strings found in terms you tend to skip "
                "(e.g., 'direct', 'redirect', 'cross').\n\n"
                "These become ML features — the model learns from your votes "
                "whether these patterns correlate with terms you keep or skip."
            ),
            default=None,
            getter=lambda: _get_indicator_patterns(prefs),
            setter=_save_indicator_patterns,
            widget_factory=_create_indicator_pattern_widget,
        )
    )

    # ===================================================================
    # TEXT PREPROCESSING TAB
    # ===================================================================

    _preprocess_toggles = [
        (
            "preprocess_title_pages",
            "Remove title pages",
            "Remove cover/title pages from the start of legal transcripts.\n\n"
            "These pages typically contain case caption, court info, and "
            "reporter details that aren't part of the testimony.",
        ),
        (
            "preprocess_index_pages",
            "Remove index pages",
            "Remove index/concordance pages from the end of transcripts.\n\n"
            "These are alphabetical reference pages that list where topics "
            "appear — useful in print but noise for AI summarization.",
        ),
        (
            "preprocess_headers_footers",
            "Remove headers/footers",
            "Remove repetitive headers and footers that appear on every page.\n\n"
            "Detected by frequency analysis — text that repeats on most pages "
            "is identified as a header or footer and removed.",
        ),
        (
            "preprocess_line_numbers",
            "Remove line numbers",
            "Remove margin line numbers (1-25) common in court transcripts.\n\n"
            "These numbers are used for reference during depositions but add "
            "noise when processing text for AI analysis.",
        ),
        (
            "preprocess_page_boundaries",
            "Clean page boundary artifacts",
            "Clean artifacts caused by collapsed page boundaries in PDF extraction.\n\n"
            "When PDF text extraction doesn't preserve page breaks, line numbers, "
            "page numbers, reporter initials, and headers can merge into body text. "
            "This cleaner detects and removes those artifacts.\n\n"
            "Example: '...this 1 2 3 ... 24 sn Proceedings 29 1 Court.' becomes "
            "'...this Court.'",
        ),
        (
            "preprocess_transcript_artifacts",
            "Clean transcript artifacts",
            "Remove transcript-specific artifacts like standalone page numbers "
            "and inline concordance citations.\n\n"
            "Handles patterns like embedded page:line references that appear "
            "in some transcript formats.",
        ),
        (
            "preprocess_qa_notation",
            "Convert Q/A notation",
            "Convert shorthand Q./A. notation to readable 'Question:'/'Answer:' "
            "format.\n\n"
            "Makes transcript dialogue easier to read in summaries.",
        ),
        (
            "preprocess_coreference",
            "Resolve pronoun references",
            "Replace pronouns (he, she, they) with the names they refer to.\n\n"
            "This improves search accuracy — chunks that only contain pronouns "
            "become findable by name. Uses AI-based coreference resolution which "
            "is correct approximately 81% of the time. Better to use than not, "
            "but be aware it slightly modifies your input text.",
        ),
    ]

    for _key, _label, _tooltip in _preprocess_toggles:
        SettingsRegistry.register(
            SettingDefinition(
                key=_key,
                label=_label,
                category="Text Preprocessing",
                setting_type=SettingType.CHECKBOX,
                tooltip=_tooltip + "\n\nChanges apply on next document load.",
                default=True,
                getter=(lambda k=_key: prefs.get(k, True)),
                setter=(lambda v, k=_key: prefs.set(k, v)),
            )
        )

    # -------------------------------------------------------------------
    # Header/Footer Removal Configuration
    # -------------------------------------------------------------------

    # Custom patterns widget
    def _create_custom_patterns_widget(parent):
        """Factory function to create the CustomPatternsWidget."""
        from src.ui.settings.patterns_widget import CustomPatternsWidget

        return CustomPatternsWidget(
            parent,
            tooltip_text=(
                "Add custom text patterns to remove from documents.\n\n"
                "These are matched case-insensitively and removed if they "
                "appear 3 or more times (indicating they're repeated "
                "headers or footers).\n\n"
                "Examples:\n"
                "• Firm names: 'SMITH & JONES LLP'\n"
                "• Reporter info: 'JANE DOE, CSR'\n"
                "• Custom headers: 'CONFIDENTIAL - DO NOT DISTRIBUTE'"
            ),
        )

    def _save_custom_patterns(value: str) -> None:
        """Save custom patterns to preferences."""
        if value is not None:
            prefs.set("custom_header_footer_patterns", value)

    SettingsRegistry.register(
        SettingDefinition(
            key="custom_header_footer_patterns",
            label="Custom Header/Footer Patterns",
            category="Text Preprocessing",
            setting_type=SettingType.CUSTOM,
            tooltip="Add custom patterns to remove from document headers and footers.",
            default="",
            setter=_save_custom_patterns,
            widget_factory=_create_custom_patterns_widget,
        )
    )

    SettingsRegistry.register(
        SettingDefinition(
            key="header_footer_short_line_detection",
            label="Aggressive short-line detection",
            category="Text Preprocessing",
            setting_type=SettingType.CHECKBOX,
            tooltip=(
                "When enabled, short lines (under 70 characters) containing words "
                "like 'plaintiff', 'defendant', 'direct', or 'cross' are treated "
                "as headers if they repeat 3+ times.\n\n"
                "This catches section headers like 'PLAINTIFF / MR. SMITH' that "
                "might otherwise be missed.\n\n"
                "Disable if legitimate short lines are being incorrectly removed."
            ),
            default=True,
            getter=lambda: prefs.get("header_footer_short_line_detection", True),
            setter=lambda v: prefs.set("header_footer_short_line_detection", v),
        )
    )

    SettingsRegistry.register(
        SettingDefinition(
            key="header_footer_min_occurrences",
            label="Minimum occurrences for removal",
            category="Text Preprocessing",
            setting_type=SettingType.SPINBOX,
            tooltip=(
                "How many times a line must appear to be considered a header/footer.\n\n"
                "Lower values (2-3) are more aggressive and catch headers that "
                "don't appear on every page.\n\n"
                "Higher values (4-5) are more conservative and only remove "
                "content that appears very frequently.\n\n"
                "Default: 3"
            ),
            default=3,
            min_value=2,
            max_value=10,
            getter=lambda: prefs.get("header_footer_min_occurrences", 3),
            setter=lambda v: prefs.set("header_footer_min_occurrences", int(v)),
        )
    )

    # ===================================================================
    # CORPUS TAB
    # ===================================================================

    # Custom widget for corpus management
    def _create_corpus_settings_widget(parent):
        """Factory function to create the CorpusSettingsWidget."""
        from src.ui.settings.corpus_widget import CorpusSettingsWidget

        return CorpusSettingsWidget(parent)

    SettingsRegistry.register(
        SettingDefinition(
            key="corpus_management",
            label="Corpus Management",
            category="Corpus",
            setting_type=SettingType.CUSTOM,
            tooltip=(
                "Manage your corpus of past transcripts for BM25 vocabulary extraction. "
                "The corpus helps identify case-specific terminology by comparing against "
                "your typical work."
            ),
            default=None,
            widget_factory=_create_corpus_settings_widget,
        )
    )

    # ===================================================================
    # Q&A TAB
    # ===================================================================

    # Ollama Model Selector
    def _get_ollama_model_options() -> list[tuple[str, str]]:
        """Fetch available models from Ollama for dropdown options."""
        try:
            from src.services import AIService

            manager = AIService().get_ollama_manager()
            if not manager.is_connected:
                return [("(Ollama not running - start Ollama first)", "")]
            models = manager.get_available_models()
            if models:
                # Sort by name and return (display, value) tuples
                options = []
                for name in sorted(models.keys()):
                    # Show model name with size if available
                    info = models[name]
                    size_gb = info.get("size", 0) / (1024**3) if info.get("size") else 0
                    display = f"{name} ({size_gb:.1f} GB)" if size_gb > 0 else name
                    options.append((display, name))
                return options
            return [("(No models installed - run 'ollama pull gemma3:1b')", "")]
        except Exception:
            return [("(Error connecting to Ollama)", "")]

    def _set_ollama_model(model_name: str) -> None:
        """Save and activate selected Ollama model."""
        if not model_name:
            return
        prefs.set("ollama_model", model_name)
        # Also update the model manager to use this model
        try:
            from src.services import AIService

            manager = AIService().get_ollama_manager()
            manager.load_model(model_name)
        except Exception as e:
            # Log exception instead of silent pass
            logger.debug("Model load deferred: %s", e)

    SettingsRegistry.register(
        SettingDefinition(
            key="ollama_model",
            label="AI Model",
            category="AI Model",
            setting_type=SettingType.DROPDOWN,
            tooltip=(
                "Select which Ollama model to use for AI features.\n\n"
                "This program was tested with Gemma 3. More parameters generally "
                "produce better results, but a dedicated GPU is recommended for "
                "larger models (7B+).\n\n"
                "Pick the largest Gemma model suitable for your hardware."
            ),
            default="gemma3:1b",
            options=_get_ollama_model_options,
            getter=lambda: prefs.get("ollama_model", "gemma3:1b"),
            setter=_set_ollama_model,
        )
    )

    # "Learn more about LLMs" educational link
    def _show_ollama_education_dialog():
        """Show educational popup about Ollama and local LLMs."""
        import webbrowser

        import customtkinter as ctk

        from src.ui.scaling import scale_value
        from src.ui.theme import COLORS, FONTS

        _w, _h = scale_value(550), scale_value(580)

        # Create modal dialog - larger to fit new content
        dialog = ctk.CTkToplevel()
        dialog.title("Understanding Local LLM Models")
        dialog.geometry(f"{_w}x{_h}")
        dialog.resizable(False, False)
        dialog.grab_set()  # Modal behavior

        # Center on screen
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - _w) // 2
        y = (dialog.winfo_screenheight() - _h) // 2
        dialog.geometry(f"+{x}+{y}")

        # Scrollable content frame
        content = ctk.CTkScrollableFrame(dialog, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        title = ctk.CTkLabel(
            content,
            text="Understanding Local LLM Models",
            font=FONTS["heading"],
        )
        title.pack(anchor="w", pady=(0, 10))

        # Intro text
        intro = ctk.CTkLabel(
            content,
            text=(
                "CasePrepd uses Ollama to run AI language models locally on your\n"
                "computer for summaries and Q&A. Your documents never leave your machine."
            ),
            font=FONTS["body"],
            text_color=COLORS["text_secondary"],
            justify="left",
            anchor="w",
        )
        intro.pack(anchor="w", pady=(0, 10))

        # YouTube button - understand how LLMs work (in Understanding LLMs section)
        youtube_btn = ctk.CTkButton(
            content,
            text="Watch: How LLMs Work (YouTube)",
            command=lambda: webbrowser.open("https://www.youtube.com/watch?v=LPZh9BOjkQs"),
            width=200,
        )
        youtube_btn.pack(anchor="w", pady=(0, 10))

        # Sections with updated content
        sections = [
            (
                "HOW PARAMETERS AFFECT QUALITY",
                "The 'parameters' in a model name (e.g., 27B = 27 billion) indicate\n"
                "the model's size and capability. More parameters generally means:\n"
                "• Better understanding of complex questions\n"
                "• More accurate and coherent summaries\n"
                "• Fewer hallucinations (made-up facts)",
            ),
            (
                "HARDWARE CONSIDERATIONS",
                "• Dedicated GPU (NVIDIA/AMD): Can run 12B-27B models efficiently\n"
                "• No dedicated GPU (CPU only): Limited to smaller models, slower\n"
                "  processing, and higher hallucination rates",
            ),
            (
                "OUR RECOMMENDATIONS",
                "• With dedicated GPU: gemma3:27b (best quality)\n"
                "• Without dedicated GPU: gemma3:12b (good balance)\n"
                "• Models 4B or smaller: Not recommended - we've observed poor\n"
                "  performance and frequent hallucinations with these models",
            ),
            (
                "GETTING STARTED",
                "For step-by-step installation instructions, click the\n"
                "'Ollama Setup Guide' button below or visit:\n"
                "Help menu > Ollama Setup Guide",
            ),
        ]

        for section_title, section_text in sections:
            section_label = ctk.CTkLabel(
                content,
                text=section_title,
                font=FONTS["heading_sm"],
                anchor="w",
            )
            section_label.pack(anchor="w", pady=(10, 2))

            section_content = ctk.CTkLabel(
                content,
                text=section_text,
                font=FONTS["body"],
                text_color=COLORS["text_secondary"],
                justify="left",
                anchor="w",
            )
            section_content.pack(anchor="w", pady=(0, 5))

        # Button frame
        button_frame = ctk.CTkFrame(content, fg_color="transparent")
        button_frame.pack(fill="x", pady=(15, 0))

        # Ollama Setup Guide button
        visit_btn = ctk.CTkButton(
            button_frame,
            text="Ollama Setup Guide",
            command=lambda: webbrowser.open(
                "https://sites.google.com/view/caseprepd/ollama-instructions"
            ),
            width=140,
        )
        visit_btn.pack(side="left")

        # Close button
        close_btn = ctk.CTkButton(
            button_frame,
            text="Close",
            command=dialog.destroy,
            width=80,
            fg_color="gray40",
            hover_color="gray30",
        )
        close_btn.pack(side="right")

    def _create_ollama_learn_more_link(parent):
        """Create a 'Learn more' link widget for Ollama education."""
        import customtkinter as ctk

        from src.ui.theme import COLORS, FONTS

        frame = ctk.CTkFrame(parent, fg_color="transparent")

        link = ctk.CTkButton(
            frame,
            text="Learn more about local LLM models ↗",
            font=FONTS["small"],
            fg_color="transparent",
            text_color=COLORS["dialog_link"],
            hover_color=("gray90", "gray25"),
            anchor="w",
            width=250,
            height=24,
            command=_show_ollama_education_dialog,
        )
        link.pack(anchor="w")

        tip = ctk.CTkLabel(
            frame,
            text="Tip: Models with 8B+ parameters are recommended for reliable answers.",
            font=FONTS["small"],
            text_color=COLORS["dialog_muted"],
            anchor="w",
        )
        tip.pack(anchor="w", pady=(2, 0))

        return frame

    SettingsRegistry.register(
        SettingDefinition(
            key="ollama_learn_more",
            label="",  # No label - link speaks for itself
            category="AI Model",
            setting_type=SettingType.CUSTOM,
            tooltip="",
            default=None,
            widget_factory=_create_ollama_learn_more_link,
        )
    )

    # Context window size based on VRAM
    def _get_context_size_options() -> list[tuple[str, str]]:
        """Generate context size options with auto-detected recommendation."""
        from src.services import AIService

        ai_svc = AIService()
        vram = ai_svc.get_vram_gb()
        optimal = ai_svc.get_optimal_context_size()

        if vram > 0:
            auto_label = f"Auto ({optimal // 1000}K - detected {vram:.1f}GB VRAM)"
        else:
            auto_label = f"Auto ({optimal // 1000}K - no dedicated GPU)"

        return [
            (auto_label, "auto"),
            ("2K (CPU only - fastest)", "2048"),
            ("4K (low VRAM / CPU)", "4000"),
            ("8K (6-8GB VRAM)", "8000"),
            ("16K (8-12GB VRAM)", "16000"),
            ("32K (12-16GB VRAM)", "32000"),
            ("48K (16-24GB VRAM)", "48000"),
            ("64K (24GB+ VRAM)", "64000"),
        ]

    def _get_context_size() -> str:
        """Get current context size mode as string for dropdown."""
        mode = prefs.get("llm_context_size", "auto")
        return str(mode)

    def _set_context_size(value: str) -> None:
        """Set context size from dropdown value."""
        if value == "auto":
            prefs.set("llm_context_size", "auto")
        else:
            prefs.set("llm_context_size", int(value))

    SettingsRegistry.register(
        SettingDefinition(
            key="llm_context_size",
            label="Context window size",
            category="AI Model",
            setting_type=SettingType.DROPDOWN,
            tooltip=(
                "How much text the AI can process at once (in tokens).\n\n"
                "Larger context = better comprehension of long documents, but "
                "requires more GPU memory (VRAM).\n\n"
                "• Auto: Automatically selects optimal size based on your GPU\n"
                "• Manual: Override if you experience issues or want to experiment\n\n"
                "If Ollama becomes slow or unresponsive, try a smaller context size."
            ),
            default="auto",
            options=_get_context_size_options,
            getter=_get_context_size,
            setter=_set_context_size,
        )
    )

    # ===================================================================
    # Q&A RETRIEVAL WEIGHTS
    # ===================================================================

    SettingsRegistry.register(
        SettingDefinition(
            key="retrieval_weight_faiss",
            label="Semantic search weight (FAISS)",
            category="Q&A",
            setting_type=SettingType.SLIDER,
            tooltip=(
                "Weight for semantic (FAISS) search when retrieving document context "
                "for Q&A answers.\n\n"
                "Semantic search understands meaning and concepts. Phrasing is "
                "forgiving — asking 'Who are the parties?' can find passages about "
                "'plaintiff and defendant' even without those exact words.\n\n"
                "Higher values give semantic results more influence when both "
                "algorithms find the same passage."
            ),
            default=1.0,
            min_value=0.0,
            max_value=2.0,
            step=0.1,
            getter=lambda: prefs.get("retrieval_weight_faiss", 1.0),
            setter=lambda v: prefs.set("retrieval_weight_faiss", float(v)),
        )
    )

    SettingsRegistry.register(
        SettingDefinition(
            key="retrieval_weight_bm25",
            label="Exact match weight (BM25+)",
            category="Q&A",
            setting_type=SettingType.SLIDER,
            tooltip=(
                "Weight for exact-match (BM25+) search when retrieving document "
                "context for Q&A answers.\n\n"
                "BM25+ favors exact text matches — it finds passages containing "
                "the precise words in your question. Best for specific legal terms, "
                "names, and dates.\n\n"
                "Higher values give exact-match results more influence when both "
                "algorithms find the same passage."
            ),
            default=0.8,
            min_value=0.0,
            max_value=2.0,
            step=0.1,
            getter=lambda: prefs.get("retrieval_weight_bm25", 0.8),
            setter=lambda v: prefs.set("retrieval_weight_bm25", float(v)),
        )
    )

    SettingsRegistry.register(
        SettingDefinition(
            key="qa_answer_mode",
            label="Answer generation mode",
            category="Q&A",
            setting_type=SettingType.DROPDOWN,
            tooltip=(
                "How to generate answers from retrieved document context.\n\n"
                "• Extraction: Pulls matching sentences directly from the document "
                "text. Fast and deterministic, but can produce garbled results on "
                "dialogue-format documents like trial transcripts.\n\n"
                "• Ollama AI (Recommended): Uses the local LLM to synthesize a "
                "natural, coherent answer from relevant passages. Slower but "
                "produces more readable responses. Requires Ollama to be running."
            ),
            default="ollama",
            options=[
                ("Ollama AI (recommended, synthesized)", "ollama"),
                ("Extraction (fast, from document)", "extraction"),
            ],
            getter=lambda: prefs.get("qa_answer_mode", "ollama"),
            setter=lambda v: prefs.set("qa_answer_mode", v),
        )
    )

    # Custom widget for default questions management
    # (Replaces the old "Edit Default Questions" YAML editor button)
    def _create_default_questions_widget(parent):
        """Factory function to create the DefaultQuestionsWidget."""
        from src.ui.settings.questions_widget import DefaultQuestionsWidget

        return DefaultQuestionsWidget(parent)

    def _save_default_questions(questions_data: list[dict]) -> None:
        """Persist buffered questions list to the manager (single disk write)."""
        if questions_data is None:
            return
        from src.services import QAService

        manager = QAService().get_default_questions_manager()
        manager.replace_all(questions_data)

    SettingsRegistry.register(
        SettingDefinition(
            key="qa_default_questions",
            label="Default Questions",
            category="Q&A",
            setting_type=SettingType.CUSTOM,
            tooltip=(
                "Manage the questions that are automatically asked after document "
                "processing. Enable/disable individual questions using checkboxes. "
                "Add new questions or edit existing ones."
            ),
            default=None,
            setter=_save_default_questions,
            widget_factory=_create_default_questions_widget,
        )
    )

    # ===================================================================
    # EXPORT TAB
    # ===================================================================

    SettingsRegistry.register(
        SettingDefinition(
            key="auto_open_exports",
            label="Auto-open exported files",
            category="Export",
            setting_type=SettingType.CHECKBOX,
            tooltip=(
                "When enabled, exported files (CSV, Word, PDF, HTML) are "
                "automatically opened in their default application after export.\n\n"
                "Disable this if you export many files at once or prefer to "
                "manually open files."
            ),
            default=True,
            getter=lambda: prefs.get("auto_open_exports", True),
            setter=lambda v: prefs.set("auto_open_exports", v),
        )
    )

    # ===================================================================
    # LOGGING TAB
    # ===================================================================

    from src.config import LOGS_DIR

    SettingsRegistry.register(
        SettingDefinition(
            key="logging_level",
            label="Log detail level",
            category="Logging",
            setting_type=SettingType.DROPDOWN,
            tooltip=(
                "Controls how much detail is written to the log file.\n\n"
                "• Off: No logging (saves disk space)\n"
                "• Brief: Key milestones only - document processing, results, "
                "errors. Recommended for normal use.\n"
                "• Comprehensive: Everything - timing details, algorithm internals, "
                "chunk details. Use for debugging issues.\n\n"
                "Errors and warnings are always logged regardless of this setting."
            ),
            default="brief",
            options=[
                ("Off (no logging)", "off"),
                ("Brief (recommended)", "brief"),
                ("Comprehensive (debugging)", "comprehensive"),
                ("Custom (pick categories)", "custom"),
            ],
            getter=lambda: prefs.get_logging_level(),
            setter=lambda v: prefs.set_logging_level(v),
        )
    )

    def _open_logging_dialog():
        """Open the custom log categories dialog."""
        from src.ui.logging_dialog import LoggingDialog

        LoggingDialog(parent=None)

    SettingsRegistry.register(
        SettingDefinition(
            key="customize_logging",
            label="Customize Categories...",
            category="Logging",
            setting_type=SettingType.BUTTON,
            tooltip=("Choose which log categories are included when using Custom logging mode."),
            default=None,
            action=_open_logging_dialog,
        )
    )

    def _open_log_folder():
        """Open the logs folder in the system file explorer."""
        if not LOGS_DIR.exists():
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            # Windows
            os.startfile(str(LOGS_DIR))
        except AttributeError:
            # macOS/Linux fallback
            import subprocess
            import sys

            if sys.platform == "darwin":
                subprocess.run(["open", str(LOGS_DIR)])
            else:
                subprocess.run(["xdg-open", str(LOGS_DIR)])

    SettingsRegistry.register(
        SettingDefinition(
            key="open_log_folder",
            label="Open Log Folder",
            category="Logging",
            setting_type=SettingType.BUTTON,
            tooltip=(
                "Open the folder containing log files in your system file explorer.\n\n"
                "Log files:\n"
                "• caseprepd.log - Application log\n\n"
                f"Location: {LOGS_DIR}"
            ),
            default=None,
            action=_open_log_folder,
        )
    )

    def _clear_log_file():
        """Clear the log file with confirmation."""
        from tkinter import messagebox

        from src.logging_config import clear_log_file, get_log_file_size_mb

        size_mb = get_log_file_size_mb()
        result = messagebox.askyesno(
            "Clear Log File",
            f"Clear the log file?\n\n"
            f"Current size: {size_mb:.2f} MB\n\n"
            "This will erase all logged information. "
            "A new session header will be written.",
            icon="question",
        )

        if result:
            if clear_log_file():
                messagebox.showinfo(
                    "Log Cleared",
                    "Log file has been cleared and reinitialized.",
                )
            else:
                messagebox.showerror(
                    "Clear Failed",
                    "Failed to clear the log file. The file may be in use.",
                )

    SettingsRegistry.register(
        SettingDefinition(
            key="clear_log_file",
            label="Clear Log File",
            category="Logging",
            setting_type=SettingType.BUTTON,
            tooltip=(
                "Clear the caseprepd.log file to free disk space.\n\n"
                "The file will be emptied. Use this if the log file "
                "has grown too large.\n\n"
                "Note: This cannot be undone."
            ),
            default=None,
            action=_clear_log_file,
        )
    )

    SettingsRegistry.register(
        SettingDefinition(
            key="log_retention_days",
            label="Auto-delete old logs",
            category="Logging",
            setting_type=SettingType.DROPDOWN,
            tooltip=(
                "Automatically delete old main_log_*.txt debug files "
                "when the application starts.\n\n"
                "These files capture stdout/stderr output for crash "
                "debugging. They accumulate over time in:\n"
                "  %APPDATA%/CasePrepd/logs/\n\n"
                "This does NOT affect the structured caseprepd.log file."
            ),
            default="90",
            options=[
                ("Keep forever", "0"),
                ("7 days", "7"),
                ("30 days", "30"),
                ("90 days (Recommended)", "90"),
            ],
            getter=lambda: str(prefs.get("log_retention_days", "90")),
            setter=lambda v: prefs.set("log_retention_days", v),
        )
    )

    # ===================================================================
    # ADVANCED TAB — Processing section (relocated from Performance)
    # ===================================================================

    SettingsRegistry.register(
        SettingDefinition(
            key="resource_usage_pct",
            label="Parallel processing limit",
            category="Advanced",
            section="Processing",
            setting_type=SettingType.SLIDER,
            tooltip=(
                "How many parallel workers to allow during processing, expressed "
                "as a percentage of your CPU cores.\n\n"
                "Example: On an 8-core machine at 75%, up to 6 workers can run "
                "simultaneously. RAM is also checked — each LLM worker needs "
                "~2GB, so low-RAM systems will be capped regardless.\n\n"
                "Affects:\n"
                "• Vocabulary extraction: Up to 4 algorithms run in parallel\n"
                "• LLM extraction: Up to 3 document chunks processed at once\n"
                "• Q&A: Up to 4 questions answered simultaneously\n\n"
                "Lower values keep your computer responsive during processing. "
                "Higher values finish faster but may cause slowdowns."
            ),
            default=75,
            min_value=25,
            max_value=100,
            step=5,
            getter=lambda: prefs.get("resource_usage_pct", 75),
            setter=lambda v: prefs.set("resource_usage_pct", int(v)),
        )
    )

    # ===================================================================
    # AI MODEL TAB — Summary settings (relocated from Performance)
    # ===================================================================

    # GPU status helper for tooltips
    def _get_gpu_status_for_tooltip() -> str:
        """Get GPU status text for tooltip display."""
        try:
            from src.services import AIService

            return AIService().get_gpu_status_text()
        except Exception:
            return "GPU detection unavailable"

    SettingsRegistry.register(
        SettingDefinition(
            key="summary_enhanced_mode",
            label="Enhanced summary mode (two-pass)",
            category="AI Model",
            setting_type=SettingType.DROPDOWN,
            tooltip=lambda: (
                "When enabled, summarization uses a two-pass approach:\n"
                "1. Pass 1: Extract key claims, facts, and relief from each chunk\n"
                "2. Pass 2: Summarize with extracted facts as context\n\n"
                "This prevents important details from fading during progressive "
                "summarization, but doubles the number of LLM calls.\n\n"
                "• Auto: Enable if a dedicated GPU is detected\n"
                "• Always enable: Always use two-pass (2× processing time)\n"
                "• Standard only: Single-pass summarization (faster)\n\n"
                f"Current status: {_get_gpu_status_for_tooltip()}"
            ),
            default="auto",
            options=[
                ("Auto (enable if GPU detected)", "auto"),
                ("Always enable (2× processing time)", "yes"),
                ("Standard only (faster)", "no"),
            ],
            getter=lambda: prefs.get_summary_enhanced_mode(),
            setter=lambda v: prefs.set_summary_enhanced_mode(v),
        )
    )

    SettingsRegistry.register(
        SettingDefinition(
            key="summary_gpu_override",
            label="Summary generation",
            category="AI Model",
            setting_type=SettingType.DROPDOWN,
            tooltip=lambda: (
                "Summary generation is a long-running task that processes all "
                "document chunks through the LLM. Without a dedicated GPU, this "
                "can take several hours.\n\n"
                "• Require GPU: Only enable the Summary checkbox if a dedicated "
                "GPU is detected. Without a GPU, the checkbox is grayed out "
                "with an explanation.\n"
                "• Allow without GPU: Enable the Summary checkbox regardless of "
                "GPU availability. Use this if you're willing to wait several "
                "hours for the summary to complete.\n\n"
                f"Current status: {_get_gpu_status_for_tooltip()}"
            ),
            default="auto",
            options=[
                ("Require GPU (recommended)", "auto"),
                ("Allow without GPU", "yes"),
            ],
            getter=lambda: prefs.get_summary_gpu_override_mode(),
            setter=lambda v: prefs.set_summary_gpu_override_mode(v),
        )
    )

    # ===================================================================
    # Q&A EXPORT TAB
    # ===================================================================

    from src.config import QA_EXPORT_CONFIDENCE_FLOOR, QA_EXPORT_VERIFICATION_FLOOR

    SettingsRegistry.register(
        SettingDefinition(
            key="qa_export_confidence_floor",
            label="Retrieval confidence floor",
            category="Q&A Export",
            setting_type=SettingType.SLIDER,
            tooltip=(
                "Minimum FAISS retrieval confidence (0-1) for a Q&A answer\n"
                "to be included in exports. This measures how relevant the\n"
                "retrieved document chunks were to the question.\n\n"
                "Both this AND the verification floor must be met.\n\n"
                "Default: 0.40 (40%)"
            ),
            default=QA_EXPORT_CONFIDENCE_FLOOR,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            getter=lambda: prefs.get("qa_export_confidence_floor", QA_EXPORT_CONFIDENCE_FLOOR),
            setter=lambda v: prefs.set("qa_export_confidence_floor", float(v)),
        )
    )

    SettingsRegistry.register(
        SettingDefinition(
            key="qa_export_verification_floor",
            label="Verification reliability floor",
            category="Q&A Export",
            setting_type=SettingType.SLIDER,
            tooltip=(
                "Minimum hallucination verification reliability (0-1) for\n"
                "a Q&A answer to be included in exports. This measures how\n"
                "well the answer is supported by the source text.\n\n"
                "Both this AND the retrieval floor must be met.\n\n"
                "Default: 0.80 (80%)"
            ),
            default=QA_EXPORT_VERIFICATION_FLOOR,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            getter=lambda: prefs.get("qa_export_verification_floor", QA_EXPORT_VERIFICATION_FLOOR),
            setter=lambda v: prefs.set("qa_export_verification_floor", float(v)),
        )
    )


# Register all settings when this module is imported
_register_all_settings()

# Register Advanced tab settings (must be after base registration)
from .advanced_registry import _register_advanced_settings

_register_advanced_settings()
