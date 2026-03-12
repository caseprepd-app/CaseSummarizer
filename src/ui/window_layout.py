"""
Window Layout Mixin for MainWindow

Provides UI layout creation methods for the MainWindow class.
Extracted from main_window.py to improve code organization.

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
    - self._on_corpus_changed (callback method)
    - self._open_corpus_dialog (callback method)
    - self._select_files (callback method)
    - self._clear_files (callback method)
    - self._update_generate_button_state (callback method)
    - self._perform_tasks (callback method)
    - self._on_stop_clicked (callback method)
    - self._ask_followup (callback method)

    And expects to create these widget references:
    - self.header_frame, self.title_label, self.settings_btn
    - self.doc_count_frame, self.doc_count_label
    - self.corpus_frame, self.corpus_dropdown, self.manage_corpus_btn
    - self.banner_frame, self.banner_label, self.setup_corpus_btn
    - self.main_frame, self.left_panel, self.right_panel
    - self.file_table, self.add_files_btn, self.clear_files_btn
    - self.ask_default_questions_check, self.generate_btn, self.stop_btn
    - self.output_display, self.followup_frame, self.followup_entry, self.followup_btn
    - self.status_frame, self.status_label, self.timer_label
    """

    def _create_header(self):
        """Create header row with corpus dropdown and settings button."""
        from src.ui.scaling import scale_value

        self.header_frame = ctk.CTkFrame(self, height=scale_value(50), corner_radius=0)
        self.header_frame.pack(fill="x", padx=0, pady=0)
        self.header_frame.pack_propagate(False)

        # App title (left)
        from src.config import DEBUG_MODE

        title_text = "CasePrepd [DEBUG]" if DEBUG_MODE else "CasePrepd"
        self.title_label = ctk.CTkLabel(
            self.header_frame, text=title_text, font=FONTS["heading_xl"]
        )
        self.title_label.pack(side="left", padx=15, pady=10)

        # Document count badge (after title)
        self.doc_count_frame = ctk.CTkFrame(self.header_frame, **FRAME_STYLES["transparent"])
        self.doc_count_frame.pack(side="left", padx=(20, 10), pady=10)

        self.doc_count_label = ctk.CTkLabel(
            self.doc_count_frame,
            text="",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"],
        )
        self.doc_count_label.pack(side="left")

        # Settings button (right)
        self.settings_btn = ctk.CTkButton(
            self.header_frame,
            text="Settings",
            width=scale_value(100),
            corner_radius=6,
            command=self._open_settings,
        )
        self.settings_btn.pack(side="right", padx=15, pady=10)

        # Corpus dropdown (right of title)
        self.corpus_frame = ctk.CTkFrame(self.header_frame, **FRAME_STYLES["transparent"])
        self.corpus_frame.pack(side="right", padx=10, pady=10)

        corpus_label = ctk.CTkLabel(self.corpus_frame, text="Corpus:", font=FONTS["body"])
        corpus_label.pack(side="left", padx=(0, 5))

        self.corpus_dropdown = ctk.CTkComboBox(
            self.corpus_frame,
            values=["Loading..."],
            width=scale_value(150),
            command=self._on_corpus_changed,
        )
        self.corpus_dropdown.pack(side="left")

        # Corpus document count badge
        self.corpus_doc_count_label = ctk.CTkLabel(
            self.corpus_frame, text="", font=FONTS["small"], text_color=COLORS["text_secondary"]
        )
        self.corpus_doc_count_label.pack(side="left", padx=(8, 0))

        # Corpus dialog button
        self.manage_corpus_btn = ctk.CTkButton(
            self.corpus_frame,
            text="Corpus...",
            width=scale_value(70),
            fg_color=("gray70", "gray30"),
            command=self._open_corpus_dialog,
        )
        self.manage_corpus_btn.pack(side="left", padx=(5, 0))

    def _create_pipeline_indicator(self):
        """Create the pipeline step indicator between header and main panels."""
        from src.ui.pipeline_indicator import PipelineIndicator

        self.pipeline_indicator = PipelineIndicator(self)
        # Hidden initially - shown when processing starts
        self._pipeline_indicator_visible = False

    def _create_main_panels(self):
        """Create the two-panel main content area."""
        self.main_frame = ctk.CTkFrame(self, **FRAME_STYLES["transparent"])
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Configure grid for two panels
        self.main_frame.grid_columnconfigure(0, weight=1, uniform="panel")  # Left panel (1/3)
        self.main_frame.grid_columnconfigure(1, weight=2, uniform="panel")  # Right panel (2/3)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Left panel: Session Documents + Tasks
        self._create_left_panel()

        # Right panel: Results
        self._create_right_panel()

    def _create_left_panel(self):
        """Create the left panel with session documents and task options."""
        from src.ui.scaling import scale_value
        from src.ui.widgets import FileReviewTable

        self.left_panel = ctk.CTkFrame(self.main_frame, border_width=0)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)

        self.left_panel.grid_columnconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(1, weight=1)  # File table expands

        # Section header
        docs_header = ctk.CTkLabel(self.left_panel, text="SESSION DOCUMENTS", font=FONTS["heading"])
        docs_header.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        # File Review Table
        self.file_table = FileReviewTable(
            self.left_panel, on_remove=self._remove_file, on_select=self._on_file_selected
        )
        self.file_table.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # File buttons
        file_btn_frame = ctk.CTkFrame(self.left_panel, **FRAME_STYLES["transparent"])
        file_btn_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

        self.add_files_btn = ctk.CTkButton(
            file_btn_frame,
            text="+ Add Files",
            width=scale_value(100),
            corner_radius=6,
            command=self._select_files,
        )
        self.add_files_btn.pack(side="left", padx=(0, 5))

        self.clear_files_btn = ctk.CTkButton(
            file_btn_frame,
            text="Clear All",
            width=scale_value(80),
            corner_radius=6,
            **BUTTON_STYLES["caution"],
            command=self._clear_files,
        )
        self.clear_files_btn.pack(side="left")

        # Session Stats section
        self.stats_label = ctk.CTkLabel(
            self.left_panel,
            text="",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"],
            justify="left",
            anchor="w",
        )
        self.stats_label.grid(row=3, column=0, sticky="w", padx=10, pady=(5, 0))

        # Separator between documents and tasks
        separator = ctk.CTkFrame(self.left_panel, height=1, fg_color=COLORS["text_disabled"])
        separator.grid(row=4, column=0, sticky="ew", padx=10, pady=(8, 2))

        # Options section
        options_frame = ctk.CTkFrame(self.left_panel, **FRAME_STYLES["transparent"])
        options_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=(8, 0))

        # Default searches checkbox (vocab + search + key excerpts always run)
        self.ask_default_questions_check = ctk.CTkCheckBox(
            options_frame,
            text="Run default searches",
            command=self._on_default_questions_toggled,
        )
        self.ask_default_questions_check.pack(anchor="w", pady=2)
        self.ask_default_questions_check.select()  # ON by default

        # "Process Documents" button
        self.generate_btn = ctk.CTkButton(
            self.left_panel,
            text="Process Documents",
            font=FONTS["heading"],
            height=scale_value(40),
            corner_radius=6,
            command=self._perform_tasks,
        )
        self.generate_btn.grid(row=6, column=0, sticky="ew", padx=10, pady=(15, 5))

        # Stop button (hidden by default, shown during processing)
        self.stop_btn = ctk.CTkButton(
            self.left_panel,
            text="STOP",
            font=FONTS["heading"],
            height=scale_value(40),
            corner_radius=6,
            **BUTTON_STYLES["danger"],
            command=self._on_stop_clicked,
        )
        # Same grid position as generate_btn — only one visible at a time

    def _create_right_panel(self):
        """Create the right panel with results display."""
        from src.ui.dynamic_output import DynamicOutputWidget
        from src.ui.scaling import scale_value

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

        # Follow-up search input (for Search tab)
        # Store as class attribute so it can be shown/hidden based on active tab
        self.followup_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.followup_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.followup_frame.grid_columnconfigure(0, weight=1)
        # Start hidden - only shown when Search tab is active
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
            placeholder_text="Search your documents after processing completes...",
            placeholder_text_color=COLORS["placeholder_golden"],
            height=scale_value(35),
            state="disabled",
        )
        self.followup_entry.grid(row=1, column=0, sticky="ew", padx=(0, 5))
        self.followup_entry.bind("<Return>", lambda e: self._ask_followup())

        self.followup_btn = ctk.CTkButton(
            self.followup_frame,
            text="Ask",
            width=scale_value(60),
            corner_radius=6,
            command=self._ask_followup,
            state="disabled",  # Enabled after search vector store is built
        )
        self.followup_btn.grid(row=1, column=1)

    def _create_status_bar(self):
        """Create status bar at bottom of window with accent border."""
        from src.ui.scaling import scale_value as _sv

        # Top border accent (thin blue line separating status bar from content)
        self.status_border = ctk.CTkFrame(
            self, height=2, corner_radius=0, fg_color=COLORS["btn_primary"]
        )
        self.status_border.pack(fill="x", side="bottom")

        self.status_frame = ctk.CTkFrame(
            self, height=_sv(30), corner_radius=0, fg_color=COLORS["status_bar_bg"]
        )
        self.status_frame.pack(fill="x", side="bottom")
        self.status_frame.pack_propagate(False)

        # Status text (bright for active messages; idle "Ready" uses text_secondary)
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready",
            font=FONTS["body_bold"],
            text_color=COLORS["text_secondary"],
        )
        self.status_label.pack(side="left", padx=10, pady=5)

        # Determinate progress bar - shows actual percentage during processing
        self.progress_bar = ctk.CTkProgressBar(
            self.status_frame,
            width=120,
            height=8,
            mode="determinate",
            progress_color=COLORS["progress_bar"],
        )
        self.progress_bar.set(0)
        # Hidden initially - shown during processing
        self._progress_bar_visible = False

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
            width=_sv(90),
            height=_sv(24),
            font=FONTS["small"],
            **BUTTON_STYLES["secondary"],
        )
        # Hidden initially - shown after processing completes
        self._export_all_visible = False
