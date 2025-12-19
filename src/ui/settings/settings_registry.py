"""
Settings Registry for LocalScribe.

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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


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
    """
    SLIDER = "slider"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"
    PATH = "path"
    SPINBOX = "spinbox"
    BUTTON = "button"


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
        tooltip: Explanation shown on hover (helps users understand the setting)
        default: Default value when no preference is saved
        min_value: Minimum value (for SLIDER, SPINBOX)
        max_value: Maximum value (for SLIDER, SPINBOX)
        step: Increment between values (for SLIDER)
        options: List of (display_text, value) tuples (for DROPDOWN)
        getter: Function that returns the current value
        setter: Function that applies a new value
        action: Function to execute on click (for BUTTON)
    """
    key: str
    label: str
    category: str
    setting_type: SettingType
    tooltip: str
    default: Any
    min_value: float = None
    max_value: float = None
    step: float = 1
    options: list = field(default_factory=list)
    getter: Callable[[], Any] = None
    setter: Callable[[Any], None] = None
    action: Callable[[], None] = None


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

    _settings: dict[str, SettingDefinition] = {}
    _categories: dict[str, list[str]] = {}  # category -> [setting_keys]
    _category_order: list[str] = []  # Preserve registration order

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
    Register all LocalScribe settings.

    This function is called on module import. To add a new setting,
    add a SettingsRegistry.register() call here.

    Settings are grouped by category (tab):
    - Performance: Parallel processing, CPU allocation
    - Summarization: AI summary settings
    - Vocabulary: Term extraction settings
    """
    # Import lazily to avoid circular imports
    import os
    from src.user_preferences import get_user_preferences
    from src.config import (
        USER_DEFINED_MAX_WORKER_COUNT,
        VOCABULARY_DISPLAY_LIMIT,
        VOCABULARY_DISPLAY_MAX,
        VOCABULARY_SORT_METHOD,
        DEFAULT_SUMMARY_WORDS,
        MIN_SUMMARY_WORDS,
        MAX_SUMMARY_WORDS,
        CORPUS_DIR,
        BM25_ENABLED,
    )

    prefs = get_user_preferences()

    # ===================================================================
    # PERFORMANCE TAB
    # ===================================================================

    SettingsRegistry.register(SettingDefinition(
        key="parallel_workers_auto",
        label="Auto-detect CPU cores",
        category="Performance",
        setting_type=SettingType.CHECKBOX,
        tooltip=(
            "When enabled, LocalScribe automatically detects the optimal "
            "number of parallel workers based on your CPU. Disable this "
            "to manually set the worker count below."
        ),
        default=True,
        getter=lambda: not prefs.get("user_picks_max_workers", False),
        setter=lambda v: prefs.set("user_picks_max_workers", not v),
    ))

    SettingsRegistry.register(SettingDefinition(
        key="parallel_workers_count",
        label="Manual worker count",
        category="Performance",
        setting_type=SettingType.SPINBOX,
        tooltip=(
            "Number of parallel workers when auto-detect is disabled. "
            "Higher values process documents faster but use more RAM. "
            "Range: 1-8. Recommended: 2 for most systems, 4 for 16GB+ RAM."
        ),
        default=USER_DEFINED_MAX_WORKER_COUNT,
        min_value=1,
        max_value=8,
        getter=lambda: prefs.get("user_defined_max_workers", 2),
        setter=lambda v: prefs.set("user_defined_max_workers", v),
    ))

    SettingsRegistry.register(SettingDefinition(
        key="resource_usage_pct",
        label="System resource usage",
        category="Performance",
        setting_type=SettingType.SLIDER,
        tooltip=(
            "Percentage of system resources (CPU and RAM) to use for processing. "
            "Higher values process faster but may slow your computer during processing.\n\n"
            "• 25%: Minimal impact - computer stays responsive\n"
            "• 50%: Moderate - some slowdown during processing\n"
            "• 75%: Recommended - good balance of speed and responsiveness\n"
            "• 100%: Maximum speed - computer may be slow during processing"
        ),
        default=75,
        min_value=25,
        max_value=100,
        step=5,
        getter=lambda: prefs.get("resource_usage_pct", 75),
        setter=lambda v: prefs.set("resource_usage_pct", int(v)),
    ))

    # ===================================================================
    # SUMMARIZATION TAB
    # ===================================================================

    SettingsRegistry.register(SettingDefinition(
        key="default_summary_words",
        label="Default summary length (words)",
        category="Summarization",
        setting_type=SettingType.SLIDER,
        tooltip=(
            "Target word count for AI-generated summaries. The actual "
            "length may vary slightly. Longer summaries capture more "
            "detail but take more time to generate."
        ),
        default=DEFAULT_SUMMARY_WORDS,
        min_value=MIN_SUMMARY_WORDS,
        max_value=MAX_SUMMARY_WORDS,
        step=50,
        getter=lambda: prefs.get("summary_words", DEFAULT_SUMMARY_WORDS),
        setter=lambda v: prefs.set("summary_words", int(v)),
    ))

    # ===================================================================
    # VOCABULARY TAB
    # ===================================================================

    SettingsRegistry.register(SettingDefinition(
        key="vocab_display_limit",
        label="Vocabulary display limit",
        category="Vocabulary",
        setting_type=SettingType.SLIDER,
        tooltip=(
            f"Maximum terms shown in the vocabulary table. Higher values "
            f"may slow the GUI on large documents. Range: 10-{VOCABULARY_DISPLAY_MAX}. "
            f"The full vocabulary is always saved to CSV."
        ),
        default=VOCABULARY_DISPLAY_LIMIT,
        min_value=10,
        max_value=VOCABULARY_DISPLAY_MAX,
        step=10,
        getter=lambda: prefs.get("vocab_display_limit", VOCABULARY_DISPLAY_LIMIT),
        setter=lambda v: prefs.set("vocab_display_limit", int(v)),
    ))

    SettingsRegistry.register(SettingDefinition(
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
        options=["Quality Score", "Rarity"],
        default="Quality Score" if VOCABULARY_SORT_METHOD == "quality_score" else "Rarity",
        getter=lambda: "Quality Score" if prefs.get("vocab_sort_method", VOCABULARY_SORT_METHOD) == "quality_score" else "Rarity",
        setter=lambda v: prefs.set("vocab_sort_method", "quality_score" if v == "Quality Score" else "rarity"),
    ))

    # Session 23: CSV export column format setting
    SettingsRegistry.register(SettingDefinition(
        key="vocab_export_format",
        label="CSV export columns",
        category="Vocabulary",
        setting_type=SettingType.DROPDOWN,
        tooltip=(
            "Controls which columns are included when saving vocabulary to CSV. "
            "'All columns' includes Quality Score, Frequency, and Rank for "
            "Excel filtering. 'Basic' exports Term, Type, Role, and Definition. "
            "'Terms only' exports just the vocabulary terms."
        ),
        default="basic",
        options=[
            ("All columns (with quality metrics)", "all"),
            ("Basic (Term, Type, Role, Definition)", "basic"),
            ("Terms only", "terms_only"),
        ],
        getter=lambda: prefs.get("vocab_export_format", "basic"),
        setter=lambda v: prefs.set("vocab_export_format", v),
    ))

    # Session 26: BM25 Corpus-based term extraction
    SettingsRegistry.register(SettingDefinition(
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
    ))

    def _open_corpus_folder():
        """Open the corpus folder in the system file explorer."""
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

    SettingsRegistry.register(SettingDefinition(
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
    ))

    # Session 47: ML Model Reset buttons
    def _reset_vocab_model():
        """Reset vocabulary ML model to default (keep feedback history)."""
        from tkinter import messagebox
        from src.vocabulary.meta_learner import get_meta_learner

        result = messagebox.askyesno(
            "Reset Vocabulary Model",
            "Reset the vocabulary ranking model to default settings?\n\n"
            "This will undo any personalization from your thumbs up/down "
            "feedback, but your feedback history will be preserved.\n\n"
            "You can retrain the model later using your existing feedback.",
            icon="warning"
        )

        if result:
            learner = get_meta_learner()
            if learner.reset_to_default():
                messagebox.showinfo(
                    "Reset Complete",
                    "Vocabulary model has been reset to default.\n\n"
                    "Your feedback history is preserved. The model will "
                    "retrain when you provide more feedback."
                )
            else:
                messagebox.showerror(
                    "Reset Failed",
                    "Failed to reset vocabulary model. Check the debug log for details."
                )

    SettingsRegistry.register(SettingDefinition(
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
    ))

    def _reset_vocab_model_and_history():
        """Reset vocabulary ML model AND clear all feedback history."""
        from tkinter import messagebox
        from src.vocabulary.meta_learner import get_meta_learner
        from src.vocabulary.feedback_manager import get_feedback_manager

        result = messagebox.askyesno(
            "Reset Model and Clear History",
            "⚠️ CAUTION: This will:\n\n"
            "• Reset the vocabulary ranking model to default\n"
            "• DELETE all your thumbs up/down feedback history\n\n"
            "This action cannot be undone. Are you sure?",
            icon="warning"
        )

        if result:
            # Double-check for destructive action
            confirm = messagebox.askyesno(
                "Confirm Complete Reset",
                "Are you absolutely sure?\n\n"
                "All feedback you've given will be permanently deleted.",
                icon="warning"
            )

            if confirm:
                learner = get_meta_learner()
                feedback_manager = get_feedback_manager()

                model_ok = learner.reset_to_default()
                feedback_ok = feedback_manager.clear_all_feedback()

                if model_ok and feedback_ok:
                    messagebox.showinfo(
                        "Reset Complete",
                        "Vocabulary model and feedback history have been reset.\n\n"
                        "The system is now using default settings."
                    )
                else:
                    messagebox.showerror(
                        "Reset Partially Failed",
                        f"Model reset: {'OK' if model_ok else 'FAILED'}\n"
                        f"Feedback clear: {'OK' if feedback_ok else 'FAILED'}\n\n"
                        "Check the debug log for details."
                    )

    SettingsRegistry.register(SettingDefinition(
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
    ))


    # ===================================================================
    # QUESTIONS TAB
    # ===================================================================

    SettingsRegistry.register(SettingDefinition(
        key="qa_answer_mode",
        label="Answer generation mode",
        category="Questions",
        setting_type=SettingType.DROPDOWN,
        tooltip=(
            "How to generate answers from retrieved document context.\n\n"
            "• Extraction: Finds the most relevant sentences directly from "
            "your document. Fast and deterministic - same question always "
            "gives the same answer. Best for quick lookups.\n\n"
            "• Ollama: Uses AI to synthesize a natural-language answer from "
            "relevant passages. Slower but produces more readable, comprehensive "
            "responses. Requires Ollama to be running."
        ),
        default="extraction",
        options=[
            ("Extraction (fast, from document)", "extraction"),
            ("Ollama AI (slower, synthesized)", "ollama"),
        ],
        getter=lambda: prefs.get("qa_answer_mode", "extraction"),
        setter=lambda v: prefs.set("qa_answer_mode", v),
    ))

    SettingsRegistry.register(SettingDefinition(
        key="qa_auto_run",
        label="Auto-run default questions",
        category="Questions",
        setting_type=SettingType.CHECKBOX,
        tooltip=(
            "Automatically run the default questions after document processing "
            "completes. Disable this if you prefer to manually trigger questions "
            "or if processing large documents where questions add overhead."
        ),
        default=True,
        getter=lambda: prefs.get("qa_auto_run", True),
        setter=lambda v: prefs.set("qa_auto_run", v),
    ))

    def _open_question_editor():
        """Open the Q&A question editor dialog."""
        from src.ui.qa_question_editor import QAQuestionEditor
        # Get root window - traverse up the widget tree
        import tkinter as tk
        for widget in tk._default_root.winfo_children():
            if widget.winfo_class() == 'CTkToplevel':
                # Find the settings dialog
                editor = QAQuestionEditor(widget)
                editor.wait_window()
                return
        # Fallback to root
        if tk._default_root:
            editor = QAQuestionEditor(tk._default_root)
            editor.wait_window()

    SettingsRegistry.register(SettingDefinition(
        key="qa_edit_questions",
        label="Edit Default Questions",
        category="Questions",
        setting_type=SettingType.BUTTON,
        tooltip=(
            "Customize the questions that are automatically asked for every "
            "document. You can add, edit, delete, or reorder questions. "
            "Changes are saved to config/qa_questions.yaml."
        ),
        default=None,
        action=_open_question_editor,
    ))

    def _open_default_questions_editor(parent_widget=None):
        """Open text editor for default questions file."""
        from pathlib import Path
        import subprocess
        import platform

        questions_file = Path(__file__).parent.parent.parent.parent / "config" / "qa_default_questions.txt"

        # Ensure file exists
        if not questions_file.exists():
            questions_file.parent.mkdir(parents=True, exist_ok=True)
            questions_file.write_text(
                "# Default Q&A Questions\n"
                "# One question per line. Lines starting with # are comments.\n\n"
                "What is this case about?\n"
                "What are the main allegations?\n",
                encoding='utf-8'
            )

        # Open in system default text editor
        try:
            if platform.system() == 'Windows':
                subprocess.run(['notepad.exe', str(questions_file)])
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', '-t', str(questions_file)])
            else:  # Linux
                subprocess.run(['xdg-open', str(questions_file)])

            # Refresh the checkbox label in main window after editing
            # Find main window and refresh label
            widget = parent_widget
            while widget:
                if hasattr(widget, 'refresh_default_questions_label'):
                    widget.refresh_default_questions_label()
                    break
                widget = widget.master if hasattr(widget, 'master') else None

        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Could not open file: {e}")

    SettingsRegistry.register(SettingDefinition(
        key="qa_edit_auto_questions",
        label="Edit Auto Questions (TXT)",
        category="Questions",
        setting_type=SettingType.BUTTON,
        tooltip=(
            "Edit the simple text file of questions that are automatically asked "
            "after document processing completes. One question per line. "
            "Changes are saved to config/qa_default_questions.txt."
        ),
        default=None,
        action=_open_default_questions_editor,
    ))

    # ===================================================================
    # EXPERIMENTAL TAB (Session 43)
    # ===================================================================

    SettingsRegistry.register(SettingDefinition(
        key="vocab_use_llm",
        label="Use LLM for vocabulary extraction",
        category="Experimental",
        setting_type=SettingType.CHECKBOX,
        tooltip=(
            "When enabled, vocabulary extraction uses both NER (spaCy) and "
            "LLM (Ollama) to find terms. The results are reconciled and "
            "terms found by both methods are ranked higher.\n\n"
            "This may take longer but typically provides more accurate results. "
            "Disable to use NER-only extraction (faster but may miss some terms)."
        ),
        default=True,
        getter=lambda: prefs.is_vocab_llm_enabled(),
        setter=lambda v: prefs.set_vocab_llm_enabled(v),
    ))

    SettingsRegistry.register(SettingDefinition(
        key="experimental_briefing_enabled",
        label="Enable Case Briefing (experimental)",
        category="Experimental",
        setting_type=SettingType.CHECKBOX,
        tooltip=(
            "⚠️ EXPERIMENTAL: Case Briefing extracts parties, allegations, "
            "and case facts from legal documents. This feature is still "
            "being refined and may produce incomplete or inaccurate results.\n\n"
            "When enabled, a 'Case Briefing (experimental)' checkbox will "
            "appear in the output options. Requires application restart."
        ),
        default=False,
        getter=lambda: prefs.is_experimental_briefing_enabled(),
        setter=lambda v: prefs.set_experimental_briefing_enabled(v),
    ))


# Register all settings when this module is imported
_register_all_settings()
