"""
Window Layout Mixin for MainWindow

Provides UI layout creation methods for the MainWindow class.
Extracted from main_window.py in Session 33 to improve code organization.

This module contains all widget creation and layout code, separating
visual structure from business logic in the main window.

Usage:
    class MainWindow(WindowLayoutMixin, ctk.CTk):
        def __init__(self):
            super().__init__()
            self._create_header()
            self._create_warning_banner()
            self._create_main_panels()
            self._create_status_bar()
"""

import customtkinter as ctk

from src.ui.theme import BUTTON_STYLES, COLORS, FONTS, FRAME_STYLES


class WindowLayoutMixin:
    """
    Mixin providing layout creation methods for MainWindow.

    This mixin expects the following attributes to be defined:
    - self (ctk.CTk window instance)
    - self._open_settings (callback method)
    - self._open_model_settings (callback method)
    - self._on_corpus_changed (callback method)
    - self._open_corpus_dialog (callback method)
    - self._select_files (callback method)
    - self._clear_files (callback method)
    - self._update_generate_button_state (callback method)
    - self._on_summary_checked (callback method)
    - self._perform_tasks (callback method)
    - self._ask_followup (callback method)

    And expects to create these widget references:
    - self.header_frame, self.title_label, self.settings_btn
    - self.model_display_frame, self.model_name_label, self.model_configure_btn
    - self.corpus_frame, self.corpus_dropdown, self.manage_corpus_btn
    - self.banner_frame, self.banner_label, self.setup_corpus_btn
    - self.main_frame, self.left_panel, self.right_panel
    - self.file_table, self.add_files_btn, self.clear_files_btn
    - self.qa_check, self.vocab_check, self.summary_check, self.generate_btn
    - self.output_display, self.followup_frame, self.followup_entry, self.followup_btn
    - self.status_frame, self.status_label, self.timer_label, self.corpus_info_label
    - self.ollama_status_frame, self.ollama_status_dot, self.ollama_status_label
    """

    def _create_header(self):
        """Create header row with corpus dropdown and settings button."""
        self.header_frame = ctk.CTkFrame(self, height=50, corner_radius=0)
        self.header_frame.pack(fill="x", padx=0, pady=0)
        self.header_frame.pack_propagate(False)

        # App title (left)
        self.title_label = ctk.CTkLabel(
            self.header_frame, text="CasePrepd", font=FONTS["heading_xl"]
        )
        self.title_label.pack(side="left", padx=15, pady=10)

        # Model display frame (after title, normal prominence)
        self.model_display_frame = ctk.CTkFrame(self.header_frame, **FRAME_STYLES["transparent"])
        self.model_display_frame.pack(side="left", padx=(20, 10), pady=10)

        model_icon_label = ctk.CTkLabel(self.model_display_frame, text="🤖", font=FONTS["body"])
        model_icon_label.pack(side="left", padx=(0, 5))

        self.model_name_label = ctk.CTkLabel(
            self.model_display_frame, text="Loading...", font=FONTS["body"]
        )
        self.model_name_label.pack(side="left", padx=(0, 8))

        self.model_configure_btn = ctk.CTkButton(
            self.model_display_frame,
            text="Configure",
            width=75,
            height=28,
            font=FONTS["small"],
            fg_color=("gray70", "gray30"),
            command=self._open_model_settings,
        )
        self.model_configure_btn.pack(side="left")

        # Settings button (right)
        self.settings_btn = ctk.CTkButton(
            self.header_frame, text="Settings", width=100, command=self._open_settings
        )
        self.settings_btn.pack(side="right", padx=15, pady=10)

        # Corpus dropdown (right of title)
        self.corpus_frame = ctk.CTkFrame(self.header_frame, **FRAME_STYLES["transparent"])
        self.corpus_frame.pack(side="right", padx=10, pady=10)

        corpus_label = ctk.CTkLabel(self.corpus_frame, text="Corpus:", font=FONTS["body"])
        corpus_label.pack(side="left", padx=(0, 5))

        self.corpus_dropdown = ctk.CTkComboBox(
            self.corpus_frame, values=["Loading..."], width=150, command=self._on_corpus_changed
        )
        self.corpus_dropdown.pack(side="left")

        # Corpus document count badge (Session 67)
        self.corpus_doc_count_label = ctk.CTkLabel(
            self.corpus_frame, text="", font=FONTS["small"], text_color=COLORS["text_secondary"]
        )
        self.corpus_doc_count_label.pack(side="left", padx=(8, 0))

        # Manage button
        self.manage_corpus_btn = ctk.CTkButton(
            self.corpus_frame,
            text="Manage",
            width=70,
            fg_color=("gray70", "gray30"),
            command=self._open_corpus_dialog,
        )
        self.manage_corpus_btn.pack(side="left", padx=(5, 0))

    def _create_main_panels(self):
        """Create the two-panel main content area."""
        self.main_frame = ctk.CTkFrame(self, **FRAME_STYLES["transparent"])
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Configure grid for two panels
        self.main_frame.grid_columnconfigure(0, weight=2)  # Left panel
        self.main_frame.grid_columnconfigure(1, weight=3)  # Right panel (larger)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Left panel: Session Documents + Tasks
        self._create_left_panel()

        # Right panel: Results
        self._create_right_panel()

    def _create_left_panel(self):
        """Create the left panel with session documents and task options."""
        from src.ui.widgets import FileReviewTable

        self.left_panel = ctk.CTkFrame(self.main_frame)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)

        self.left_panel.grid_columnconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(1, weight=1)  # File table expands

        # Section header
        docs_header = ctk.CTkLabel(self.left_panel, text="SESSION DOCUMENTS", font=FONTS["heading"])
        docs_header.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        # File Review Table
        self.file_table = FileReviewTable(self.left_panel)
        self.file_table.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # File buttons
        file_btn_frame = ctk.CTkFrame(self.left_panel, **FRAME_STYLES["transparent"])
        file_btn_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

        self.add_files_btn = ctk.CTkButton(
            file_btn_frame, text="+ Add Files", width=100, command=self._select_files
        )
        self.add_files_btn.pack(side="left", padx=(0, 5))

        self.clear_files_btn = ctk.CTkButton(
            file_btn_frame,
            text="Clear All",
            width=80,
            fg_color=("gray70", "gray30"),
            command=self._clear_files,
        )
        self.clear_files_btn.pack(side="left")

        # Session Stats section (Session 73)
        self.stats_label = ctk.CTkLabel(
            self.left_panel,
            text="",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"],
            justify="left",
            anchor="w",
        )
        self.stats_label.grid(row=3, column=0, sticky="w", padx=10, pady=(5, 0))

        # Task checkboxes section
        task_header = ctk.CTkLabel(self.left_panel, text="TASKS", font=FONTS["body_bold"])
        task_header.grid(row=4, column=0, sticky="w", padx=10, pady=(10, 5))

        task_frame = ctk.CTkFrame(self.left_panel, **FRAME_STYLES["transparent"])
        task_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=0)

        # Vocabulary checkbox (default ON) - moved to first position
        self.vocab_check = ctk.CTkCheckBox(
            task_frame, text="Extract Vocabulary", command=self._on_vocab_check_changed
        )
        self.vocab_check.pack(anchor="w", pady=2)
        self.vocab_check.select()  # ON by default

        # LLM Enhancement sub-checkbox (indented under Vocabulary)
        self.vocab_llm_check = ctk.CTkCheckBox(
            task_frame, text="Use LLM Enhancement", command=self._on_vocab_llm_check_changed
        )
        self.vocab_llm_check.pack(anchor="w", pady=(0, 2), padx=(20, 0))
        # Initial state set by _update_vocab_llm_checkbox_state() in MainWindow.__init__

        # Q&A checkbox (default ON)
        self.qa_check = ctk.CTkCheckBox(
            task_frame, text="Ask Questions", command=self._on_qa_check_changed
        )
        self.qa_check.pack(anchor="w", pady=2)
        self.qa_check.select()  # ON by default

        # Default questions sub-checkbox (indented under Q&A)
        self.ask_default_questions_check = ctk.CTkCheckBox(
            task_frame,
            text="Ask 0 default questions",
            command=self._on_default_questions_toggled,
        )
        self.ask_default_questions_check.pack(anchor="w", pady=(0, 2), padx=(20, 0))
        self.ask_default_questions_check.select()  # ON by default

        # Summary checkbox (default OFF, with warning)
        self.summary_check = ctk.CTkCheckBox(
            task_frame, text="Generate Summary (slow)", command=self._on_summary_checked
        )
        self.summary_check.pack(anchor="w", pady=2)
        # OFF by default - no select()

        # "Perform N Tasks" button
        self.generate_btn = ctk.CTkButton(
            self.left_panel,
            text="Perform 2 Tasks",
            font=FONTS["heading"],
            height=40,
            command=self._perform_tasks,
        )
        self.generate_btn.grid(row=6, column=0, sticky="ew", padx=10, pady=(15, 5))

        # Task preview label (Session 69) - shows what will run
        self.task_preview_label = ctk.CTkLabel(
            self.left_panel,
            text="",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"],
            wraplength=280,  # Fixed width suitable for left panel (approx 300px)
            justify="left",
        )
        self.task_preview_label.grid(row=7, column=0, sticky="w", padx=10, pady=(0, 10))

    def _create_right_panel(self):
        """Create the right panel with results display."""
        from src.ui.dynamic_output import DynamicOutputWidget

        self.right_panel = ctk.CTkFrame(self.main_frame)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)

        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(1, weight=1)  # Results area expands

        # Header with results dropdown
        results_header = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        results_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        results_label = ctk.CTkLabel(results_header, text="📋 RESULTS", font=FONTS["heading"])
        results_label.pack(side="left")

        # Dynamic Output Widget (contains the results selector and display)
        self.output_display = DynamicOutputWidget(self.right_panel)
        self.output_display.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # Follow-up question input (for Q&A mode)
        # Session 78: Store as class attribute so it can be shown/hidden based on active tab
        self.followup_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.followup_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.followup_frame.grid_columnconfigure(0, weight=1)
        # Start hidden - only shown when Q&A tab is active
        self.followup_frame.grid_remove()

        # Tip label above question input
        tip_label = ctk.CTkLabel(
            self.followup_frame,
            text="Tip: Ask one specific question at a time (dates, injuries, warnings) - not broad or compound questions.",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"],
            anchor="w",
        )
        tip_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        self.followup_entry = ctk.CTkEntry(
            self.followup_frame,
            placeholder_text="Q&A not ready - run tasks first",
            height=35,
            # state="disabled",  # TEST: Starting enabled to diagnose typing issue
        )
        self.followup_entry.grid(row=1, column=0, sticky="ew", padx=(0, 5))
        self.followup_entry.bind("<Return>", lambda e: self._ask_followup())

        self.followup_btn = ctk.CTkButton(
            self.followup_frame,
            text="Ask",
            width=60,
            command=self._ask_followup,
            state="disabled",  # Enabled after Q&A vector store is built
        )
        self.followup_btn.grid(row=1, column=1)

    def _create_status_bar(self):
        """Create status bar at bottom of window."""
        self.status_frame = ctk.CTkFrame(self, height=30, corner_radius=0)
        self.status_frame.pack(fill="x", side="bottom")
        self.status_frame.pack_propagate(False)

        # Status text
        self.status_label = ctk.CTkLabel(self.status_frame, text="Ready", font=FONTS["small"])
        self.status_label.pack(side="left", padx=10, pady=5)

        # Ollama status indicator (small, less prominent)
        self.ollama_status_frame = ctk.CTkFrame(self.status_frame, **FRAME_STYLES["transparent"])
        self.ollama_status_frame.pack(side="left", padx=(15, 0), pady=5)

        self.ollama_status_dot = ctk.CTkLabel(
            self.ollama_status_frame,
            text="●",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"],  # Default gray until checked
        )
        self.ollama_status_dot.pack(side="left", padx=(0, 3))

        self.ollama_status_label = ctk.CTkLabel(
            self.ollama_status_frame,
            text="Ollama",
            font=FONTS["tiny"],
            text_color=COLORS["text_secondary"],
        )
        self.ollama_status_label.pack(side="left")

        # Activity indicator (animated progress bar) - shows during processing
        self.activity_indicator = ctk.CTkProgressBar(
            self.status_frame,
            width=60,
            height=8,
            mode="indeterminate",
            indeterminate_speed=0.7,
        )
        # Hidden initially - shown during processing
        self._activity_indicator_visible = False

        # Timer (right side)
        self.timer_label = ctk.CTkLabel(self.status_frame, text="⏱ 0:00", font=FONTS["small"])
        self.timer_label.pack(side="right", padx=10, pady=5)

        # Export All button (right side, hidden until processing completes)
        self.export_all_btn = ctk.CTkButton(
            self.status_frame,
            text="Export All",
            command=self._export_all,
            width=90,
            height=24,
            font=FONTS["small"],
            **BUTTON_STYLES["secondary"],
        )
        # Hidden initially - shown after processing completes
        self._export_all_visible = False

        # Combined Report button (Session 73: single document with vocab + Q&A)
        self.combined_report_btn = ctk.CTkButton(
            self.status_frame,
            text="Combined Report",
            command=self._export_combined_report,
            width=110,
            height=24,
            font=FONTS["small"],
            **BUTTON_STYLES["secondary"],
        )
        # Hidden initially - shown after processing completes
        self._combined_report_visible = False

        # Corpus info (middle)
        self.corpus_info_label = ctk.CTkLabel(
            self.status_frame, text="", font=FONTS["small"], text_color=COLORS["text_secondary"]
        )
        self.corpus_info_label.pack(side="right", padx=20, pady=5)
