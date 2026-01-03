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
    - CUSTOM: CustomWidgetSetting (custom widget factory)
    """
    SLIDER = "slider"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"
    PATH = "path"
    SPINBOX = "spinbox"
    BUTTON = "button"
    CUSTOM = "custom"  # Session 63c: For complex widgets like question list


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
        widget_factory: Function(parent) -> widget (for CUSTOM type)
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
    widget_factory: Callable = None  # Session 63c: For CUSTOM type widgets


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

    # Session 62b: Summary setting moved to Performance tab (consolidated)

    SettingsRegistry.register(SettingDefinition(
        key="default_summary_words",
        label="Default summary length (words)",
        category="Performance",
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

    # Session 68: Corpus status warning banner (dynamic refresh)
    # Shows warning if corpus has < 5 documents, updates when tab is shown
    def _create_corpus_warning_widget(parent):
        """Factory for corpus status warning banner with dynamic refresh."""
        import tkinter as tk
        from src.core.vocabulary.corpus_manager import get_corpus_manager

        frame = tk.Frame(parent)
        frame._warning_frame = None  # Store reference for updates
        frame._warning_label = None

        def update_warning():
            """Update the warning banner based on current corpus status."""
            corpus_manager = get_corpus_manager()
            doc_count = corpus_manager.get_document_count()

            if doc_count < 5:
                # Show or update warning banner
                warning_text = (
                    f"Corpus not ready ({doc_count}/5 documents). "
                    "ML predictions are less accurate without a corpus of past transcripts. "
                    "Add documents in Settings > Corpus."
                )

                if frame._warning_frame is None:
                    # Create warning frame
                    frame._warning_frame = tk.Frame(frame, bg="#FFF3CD", padx=10, pady=8)
                    frame._warning_frame.pack(fill="x", pady=(0, 10))
                    frame._warning_label = tk.Label(
                        frame._warning_frame,
                        text=warning_text,
                        bg="#FFF3CD",
                        fg="#856404",
                        wraplength=400,
                        justify="left",
                    )
                    frame._warning_label.pack(anchor="w")
                else:
                    # Update existing label text
                    frame._warning_label.configure(text=warning_text)
                    if not frame._warning_frame.winfo_ismapped():
                        frame._warning_frame.pack(fill="x", pady=(0, 10))
            else:
                # Corpus is ready - hide warning if it exists
                if frame._warning_frame is not None and frame._warning_frame.winfo_ismapped():
                    frame._warning_frame.pack_forget()

        # Initial update
        update_warning()

        # Refresh when widget becomes visible (tab switch)
        frame.bind("<Map>", lambda e: update_warning())

        return frame

    SettingsRegistry.register(SettingDefinition(
        key="corpus_status_warning",
        label="",  # No label for banner
        category="Vocabulary",
        setting_type=SettingType.CUSTOM,
        tooltip="",
        default=None,
        widget_factory=_create_corpus_warning_widget,
    ))

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
        options=[
            ("Quality Score", "quality_score"),
            ("Rarity", "rarity"),
        ],
        default="quality_score" if VOCABULARY_SORT_METHOD == "quality_score" else "rarity",
        getter=lambda: prefs.get("vocab_sort_method", VOCABULARY_SORT_METHOD),
        setter=lambda v: prefs.set("vocab_sort_method", v),
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
        from src.core.vocabulary.meta_learner import get_meta_learner

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
        from src.core.vocabulary.meta_learner import get_meta_learner
        from src.core.vocabulary.feedback_manager import get_feedback_manager

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

    # Session 59: Vocabulary filtering thresholds (user-configurable)
    # Session 62b: Moved from Advanced to Vocabulary tab (consolidated)
    SettingsRegistry.register(SettingDefinition(
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
    ))

    SettingsRegistry.register(SettingDefinition(
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
    ))

    # Session 62: Additional vocabulary filtering controls
    SettingsRegistry.register(SettingDefinition(
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
    ))

    SettingsRegistry.register(SettingDefinition(
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
    ))

    # Session 68: Corpus Familiarity Filtering
    SettingsRegistry.register(SettingDefinition(
        key="corpus_familiarity_threshold",
        label="Corpus familiarity threshold",
        category="Vocabulary",
        setting_type=SettingType.SLIDER,
        tooltip=(
            "Filter terms appearing in this percentage or more of your corpus "
            "documents. Terms above this threshold are removed (you likely "
            "already know them).\n\n"
            "Example: 0.75 removes terms appearing in 75%+ of your past "
            "transcripts.\n\n"
            "Lower values = more aggressive filtering.\n"
            "Higher values = keep more terms.\n"
            "Set to 1.0 to disable percentage-based filtering."
        ),
        default=0.75,
        min_value=0.25,
        max_value=1.0,
        step=0.05,
        getter=lambda: prefs.get("corpus_familiarity_threshold", 0.75),
        setter=lambda v: prefs.set("corpus_familiarity_threshold", float(v)),
    ))

    SettingsRegistry.register(SettingDefinition(
        key="corpus_familiarity_min_docs",
        label="Corpus familiarity min docs",
        category="Vocabulary",
        setting_type=SettingType.SPINBOX,
        tooltip=(
            "Alternative threshold: Filter terms appearing in N or more "
            "documents. This provides a hard floor regardless of corpus size.\n\n"
            "Example: 10 removes terms appearing in 10+ documents.\n\n"
            "Set to 0 to use only percentage-based filtering."
        ),
        default=10,
        min_value=0,
        max_value=50,
        getter=lambda: prefs.get("corpus_familiarity_min_docs", 10),
        setter=lambda v: prefs.set("corpus_familiarity_min_docs", int(v)),
    ))

    SettingsRegistry.register(SettingDefinition(
        key="corpus_familiarity_exempt_persons",
        label="Exempt person names from corpus filter",
        category="Vocabulary",
        setting_type=SettingType.CHECKBOX,
        tooltip=(
            "When enabled, person names are never filtered by corpus familiarity. "
            "Recommended: Names in legal documents are always case-specific, "
            "even common names like 'John Smith'.\n\n"
            "Disable if you want to filter frequently-appearing party names."
        ),
        default=True,
        getter=lambda: prefs.get("corpus_familiarity_exempt_persons", True),
        setter=lambda v: prefs.set("corpus_familiarity_exempt_persons", v),
    ))

    # ===================================================================
    # CORPUS TAB (Session 64)
    # ===================================================================

    # Session 64: Custom widget for corpus management
    def _create_corpus_settings_widget(parent):
        """Factory function to create the CorpusSettingsWidget."""
        from src.ui.settings.settings_widgets import CorpusSettingsWidget
        return CorpusSettingsWidget(parent)

    SettingsRegistry.register(SettingDefinition(
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
    ))

    # ===================================================================
    # Q&A TAB
    # ===================================================================

    # Session 62: Ollama Model Selector
    def _get_ollama_model_options() -> list[tuple[str, str]]:
        """Fetch available models from Ollama for dropdown options."""
        try:
            from src.core.ai import OllamaModelManager
            manager = OllamaModelManager()
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
                    if size_gb > 0:
                        display = f"{name} ({size_gb:.1f} GB)"
                    else:
                        display = name
                    options.append((display, name))
                return options
            return [("(No models installed - run 'ollama pull gemma3:1b')", "")]
        except Exception as e:
            return [(f"(Error connecting to Ollama)", "")]

    def _set_ollama_model(model_name: str) -> None:
        """Save and activate selected Ollama model."""
        if not model_name:
            return
        prefs.set("ollama_model", model_name)
        # Also update the model manager to use this model
        try:
            from src.core.ai import OllamaModelManager
            manager = OllamaModelManager()
            manager.load_model(model_name)
        except Exception as e:
            # LOG-017: Log exception instead of silent pass
            from src.logging_config import debug_log
            debug_log(f"[SETTINGS] Model load deferred: {e}")

    SettingsRegistry.register(SettingDefinition(
        key="ollama_model",
        label="AI Model",
        category="Q&A",
        setting_type=SettingType.DROPDOWN,
        tooltip=(
            "Select which Ollama model to use for AI features.\n\n"
            "This program was tested with Gemma 3. More parameters generally "
            "produce better results, but a dedicated GPU is recommended for "
            "larger models (7B+).\n\n"
            "Pick the largest Gemma model suitable for your hardware."
        ),
        default="gemma3:1b",
        options=_get_ollama_model_options(),
        getter=lambda: prefs.get("ollama_model", "gemma3:1b"),
        setter=_set_ollama_model,
    ))

    # Session 64: Context window size based on VRAM
    def _get_context_size_options() -> list[tuple[str, str]]:
        """Generate context size options with auto-detected recommendation."""
        from src.core.utils.gpu_detector import get_optimal_context_size, get_vram_gb

        vram = get_vram_gb()
        optimal = get_optimal_context_size()

        if vram > 0:
            auto_label = f"Auto ({optimal // 1000}K - detected {vram:.1f}GB VRAM)"
        else:
            auto_label = f"Auto ({optimal // 1000}K - no dedicated GPU)"

        return [
            (auto_label, "auto"),
            ("4K (low VRAM / CPU only)", "4000"),
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

    SettingsRegistry.register(SettingDefinition(
        key="llm_context_size",
        label="Context window size",
        category="Q&A",
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
        options=_get_context_size_options(),
        getter=_get_context_size,
        setter=_set_context_size,
    ))

    SettingsRegistry.register(SettingDefinition(
        key="qa_answer_mode",
        label="Answer generation mode",
        category="Q&A",
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
        category="Q&A",
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

    # Session 63c: Custom widget for default questions management
    # (Replaces the old "Edit Default Questions" YAML editor button)
    def _create_default_questions_widget(parent):
        """Factory function to create the DefaultQuestionsWidget."""
        from src.ui.settings.settings_widgets import DefaultQuestionsWidget
        return DefaultQuestionsWidget(parent)

    SettingsRegistry.register(SettingDefinition(
        key="qa_default_questions",
        label="Default Questions",
        category="Q&A",
        setting_type=SettingType.CUSTOM,
        tooltip=(
            "Manage the questions that are automatically asked after document "
            "processing. Enable/disable individual questions using checkboxes. "
            "Add new questions or edit existing ones. Changes are saved automatically."
        ),
        default=None,
        widget_factory=_create_default_questions_widget,
    ))

    # ===================================================================
    # EXPORT TAB (Session 73)
    # ===================================================================

    SettingsRegistry.register(SettingDefinition(
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
    ))

    # ===================================================================
    # EXPERIMENTAL TAB (Session 43)
    # Session 62b: LLM setting moved to Performance; only briefing remains here
    # ===================================================================

    # Session 62b: GPU-based auto-detection for LLM extraction (moved to Performance)
    def _get_gpu_status_for_tooltip() -> str:
        """Get GPU status text for tooltip display."""
        try:
            from src.core.utils.gpu_detector import get_gpu_status_text
            return get_gpu_status_text()
        except Exception:
            return "GPU detection unavailable"

    SettingsRegistry.register(SettingDefinition(
        key="vocab_use_llm",
        label="LLM vocabulary extraction",
        category="Performance",
        setting_type=SettingType.DROPDOWN,
        tooltip=(
            "Whether to use LLM (Ollama) for enhanced vocabulary extraction.\n\n"
            "• Auto: Use LLM if dedicated GPU detected, skip otherwise\n"
            "• Yes: Always use LLM (slower without GPU)\n"
            "• No: Never use LLM (fast NER-only extraction)\n\n"
            f"Current status: {_get_gpu_status_for_tooltip()}"
        ),
        default="auto",
        options=[
            ("Auto (based on GPU)", "auto"),
            ("Yes (always use LLM)", "yes"),
            ("No (NER only)", "no"),
        ],
        getter=lambda: prefs.get_vocab_llm_mode(),
        setter=lambda v: prefs.set_vocab_llm_mode(v),
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
