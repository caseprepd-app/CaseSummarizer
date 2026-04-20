"""
CasePrepd - Main Window (CustomTkinter)

Main application window with:
- Header: Corpus dropdown + Settings button
- No-corpus warning banner
- Two-panel layout: Left (Documents + Tasks), Right (Results)
- Status bar with processing timer

Architecture:
    MainWindow inherits from:
    - WindowLayoutMixin: UI creation methods (_create_header, _create_main_panels, etc.)
    - ctk.CTk: CustomTkinter main window base class

    Layout code is in: src/ui/window_layout.py
    Business logic is in: This file (main_window.py)
"""

import logging
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from src.ui.styles import initialize_all_styles
from src.ui.window_layout import WindowLayoutMixin

logger = logging.getLogger(__name__)

# Try to import tkinterdnd2 for drag-and-drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    HAS_DND = True
except ImportError:
    HAS_DND = False

# Placeholder text shown in search panel while a follow-up search is pending
PENDING_ANSWER_TEXT = "Answer pending..."


def _pick_no_valid_files_msg(selected_files: list, processing_results: list) -> tuple[str, str]:
    """
    Choose a (title, message) pair when _perform_tasks has no usable results.

    Distinguishes three cases so the user knows what to do:
    - No files selected at all.
    - Files selected but preprocessing has not produced any results yet.
    - Files selected, preprocessing produced results, but every file failed.

    Args:
        selected_files: List of selected file paths (may be empty).
        processing_results: List of per-file result dicts from preprocessing.
            Each dict is expected to carry a "status" key whose value is
            "success" on a good preprocess.

    Returns:
        Tuple of (title, message) suitable for messagebox display.
    """
    if not selected_files:
        return ("No Files", "Please add files before processing.")
    if not processing_results:
        return (
            "Still Preparing Files",
            "Files are still being prepared. Please wait a moment and try again.",
        )
    failed = sum(1 for r in processing_results if r.get("status") != "success")
    total = len(processing_results)
    return (
        "All Files Failed",
        f"All {failed} of {total} files failed preprocessing. "
        "Check the file list for errors, then remove or replace failed files.",
    )


class MainWindow(WindowLayoutMixin, ctk.CTk):
    """
    Main application window for CasePrepd.

    Layout:
    - Header row: App title, corpus dropdown, settings button
    - Warning banner: Shown when no corpus configured
    - Left panel: Session documents + options + "Process Documents" button
    - Right panel: Results display with output type selector
    - Status bar: Status text + corpus info + processing timer

    Layout methods (from WindowLayoutMixin):
    - _create_header, _create_warning_banner, _create_main_panels
    - _create_left_panel, _create_right_panel, _create_status_bar
    """

    _FOLLOWUP_TIMEOUT_POLLS = 3_000  # 5 minutes at 100 ms per poll

    def __init__(self, worker_manager=None):
        super().__init__()
        self._worker_manager = worker_manager

        from src.config import APP_NAME, BUNDLED_BASE_DIR, DEBUG_MODE

        self.title(f"{APP_NAME} [DEBUG]" if DEBUG_MODE else APP_NAME)
        from src.ui.scaling import scale_value

        self.geometry(f"{scale_value(1200)}x{scale_value(750)}")
        icon_path = BUNDLED_BASE_DIR / "assets" / "icon.ico"
        if icon_path.exists():
            self.iconbitmap(str(icon_path))
        self.minsize(scale_value(900), scale_value(600))

        # State
        self.selected_files: list[str] = []
        self.processing_results: list[dict] = []
        self._processing_start_time: float | None = None
        self._timer_after_id: str | None = None
        self._processing_active: bool = False
        self._preprocessing_active: bool = False

        # Resize debounce -- tracks whether the window is mid-resize
        self._resize_in_progress: bool = False
        self._resize_debounce_id: str | None = None
        self.bind("<Configure>", self._on_configure)

        # Managers (via service layer for pipeline architecture)
        from src.services import VocabularyService

        self.corpus_registry = VocabularyService().get_corpus_registry()

        # Shutdown guard — prevents after() callbacks on destroyed widgets
        self._destroying = False

        # Worker subprocess manager (replaces individual worker threads)
        # When worker_manager is provided, all heavy work runs in a subprocess
        self._queue_poll_id: str | None = None

        # Search infrastructure (vector_store_path still tracked here for UI checks)
        self._vector_store_path = None  # Path to current session's vector store
        self._semantic_results: list = []  # Store SemanticResult objects
        self._semantic_results_lock = threading.Lock()  # Thread-safe access
        self._semantic_ready = False  # Search becomes available after indexing
        self._semantic_answering_active = False  # True while default searches are being answered
        self._semantic_failed = (
            False  # True when search indexing fails (embedding model error, etc.)
        )
        self._key_sentences_pending = False  # True after semantic_ready until key_sentences_result
        self._worker_ready_retries = 0  # Auto-retry counter for worker startup

        # Task tracking defaults (normally set in _perform_tasks, but init here
        # so _all_tasks_complete / _finalize_tasks never hit AttributeError)
        self._pending_tasks: dict = {}
        self._completed_tasks: set = set()
        self._failed_tasks: set = set()
        self._degraded_algorithms: list[str] = []  # Algorithms skipped/failed in vocab
        self._exporting_all = False  # Re-entrancy guard for export-all
        self._deferred_status_id: str | None = None  # Pending set_status() during error hold

        # Follow-up polling timeout counter (BUG 1 fix)
        self._followup_poll_count: int = 0
        self._followup_pending: bool = False
        self._pending_followup_index: int | None = None

        self._tab_followup_result = None  # Stash when _poll_queue() steals a tab followup result
        self._status_clear_id = None  # Tk after-ID for auto-clearing status bar
        self._status_error_hold_until = None  # Timestamp until error message is held
        # (_deferred_status_id already initialized above with type annotation)

        # Worker-ready retry after-IDs (tracked so rapid re-invocation can cancel
        # orphaned callbacks; otherwise stacked .after() schedules leak — fixes
        # "Add Files clicked twice stacks retries" class of bugs).
        self._preprocessing_retry_id: str | None = None
        self._extraction_retry_id: str | None = None

        # Initialize ttk styles with UI scale factor and font offset.
        # Must happen AFTER super().__init__() creates the Tk root (ttk.Style needs it).
        from src.ui.scaling import get_effective_font_offset, get_effective_ui_scale

        initialize_all_styles(get_effective_ui_scale(), get_effective_font_offset())

        # Build UI
        self._create_header()
        self._create_pipeline_indicator()
        self._create_main_panels()
        self._create_status_bar()

        # Create menu bar (File, Help)
        from src.ui.menu_handler import create_menus

        create_menus(
            window=self,
            select_files_callback=self._select_files,
            show_settings_callback=self._open_settings,
            quit_callback=self.quit,
        )

        # Initialize state
        self._refresh_corpus_dropdown()
        self._update_generate_button_state()
        self._update_default_questions_label()  # Set initial question count + grey out
        self._restore_task_checkbox_states()  # Restore user's checkbox preferences

        # Initialize drag-and-drop support
        self._setup_drag_drop()

        # Startup checks and status updates
        self._check_corpus_limit()  # Check if corpus exceeds doc limit

        dnd_status = "enabled" if HAS_DND else "disabled (tkinterdnd2 not installed)"
        logger.debug("Initialized with two-panel layout, drag-drop %s", dnd_status)

    # =========================================================================
    # Corpus Management
    # =========================================================================

    def _refresh_corpus_dropdown(self):
        """
        Refresh the corpus dropdown with available corpora.

        Badge states:
          0 corpora  → dropdown "None" (reddish), badge blank
          0 docs     → corpus name, badge "(empty)"
          1-4 docs   → corpus name, badge "(N/5+ docs required for corpus functionality)"
          5+ docs    → corpus name, badge "(N docs · BM25 active)"
        """
        from src.ui.theme import COLORS

        try:
            corpora = self.corpus_registry.list_corpora()
            active = self.corpus_registry.get_active_corpus()

            if active is None:
                # No corpora exist — show placeholder
                self.corpus_dropdown.configure(values=["None"])
                self.corpus_dropdown.set("None")
                self.corpus_dropdown.configure(text_color=COLORS["corpus_error_text"])
                self.corpus_doc_count_label.configure(text="")
                return

            # Reset text color to normal
            self.corpus_dropdown.configure(text_color=COLORS["text_primary"])

            names = [c.name for c in corpora]
            self.corpus_dropdown.configure(values=names)
            self.corpus_dropdown.set(active)

            # Update badge based on document count
            active_info = next((c for c in corpora if c.name == active), None)
            doc_count = active_info.doc_count if active_info else 0

            if doc_count == 0:
                self.corpus_doc_count_label.configure(
                    text="(empty)", text_color=COLORS["text_secondary"]
                )
            elif doc_count < 5:
                self.corpus_doc_count_label.configure(
                    text=f"({doc_count}/5+ docs required for corpus functionality)",
                    text_color=COLORS["warning"],
                )
            else:
                self.corpus_doc_count_label.configure(
                    text=f"({doc_count} docs · BM25 active)",
                    text_color=COLORS["text_secondary"],
                )

        except Exception as e:
            logger.warning("Error refreshing corpus dropdown: %s", e)
            self.corpus_dropdown.configure(values=["Error"])
            self.corpus_dropdown.set("Error")
            self.corpus_doc_count_label.configure(text="")
            self.set_status_error("Could not load saved summaries")

    def _on_corpus_changed(self, corpus_name: str):
        """Handle corpus selection change."""
        if corpus_name == "None":
            return

        try:
            self.corpus_registry.set_active_corpus(corpus_name)
            self._refresh_corpus_dropdown()
            self.set_status(f"Active corpus: {corpus_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to switch corpus: {e}")

    def _open_modal_dialog(self, button, dialog_factory, *, after=None):
        """
        Open a blocking modal dialog with button-disable and re-entrancy guards.

        Args:
            button: The CTkButton that triggered this — disabled while open.
            dialog_factory: Callable returning a BaseModalDialog (called inside try).
            after: Optional callback to run after the dialog closes.
        """
        if getattr(self, "_dialog_open", False):
            return
        self._dialog_open = True
        button.configure(state="disabled")

        dialog = None
        try:
            dialog = dialog_factory()
            dialog.wait_window()
        except Exception as e:
            logger.warning("Dialog failed: %s", e)
            self.set_status_error("Dialog failed to open. Try again.")
            if dialog is not None:
                try:
                    dialog.destroy()
                except Exception:
                    pass
        finally:
            self._dialog_open = False
            button.configure(state="normal")

        if after:
            after()

    def _open_corpus_dialog(self):
        """Open Settings dialog to the Corpus tab."""
        from src.ui.settings import SettingsDialog

        self._open_modal_dialog(
            self.manage_corpus_btn,
            lambda: SettingsDialog(parent=self, initial_tab="Corpus"),
            after=self._refresh_corpus_dropdown,
        )

    # =========================================================================
    # File Management
    # =========================================================================

    def _setup_drag_drop(self):
        """
        Initialize drag-and-drop file support.

        Registers the file table area as a drop target for files.
        Requires tkinterdnd2 library to be installed.
        """
        if not HAS_DND:
            logger.debug("Drag-drop disabled: tkinterdnd2 not installed")
            return

        try:
            # Initialize TkinterDnD on the underlying Tk instance
            # This approach works with CustomTkinter
            TkinterDnD._require(self)

            # Register the left panel (file table area) as a drop target
            self.left_panel.drop_target_register(DND_FILES)
            self.left_panel.dnd_bind("<<Drop>>", self._on_file_drop)
            self.left_panel.dnd_bind("<<DragEnter>>", self._on_drag_enter)
            self.left_panel.dnd_bind("<<DragLeave>>", self._on_drag_leave)

            # Store original border state for restoration
            self._original_border_color = self.left_panel.cget("border_color")
            self._original_border_width = self.left_panel.cget("border_width")

            logger.debug("Drag-drop enabled on file table area")

        except Exception as e:
            logger.debug("Failed to initialize drag-drop: %s", e)

    def _on_file_drop(self, event):
        """
        Handle files dropped onto the file table area.

        Args:
            event: Drop event containing file paths
        """
        if self._processing_active or self._preprocessing_active:
            self.set_status("Cannot add files during active processing")
            return

        # Parse the dropped file paths
        # tkinterdnd2 provides paths as a space-separated string or Tcl list
        raw_data = event.data

        # Handle Tcl list format (paths with spaces are enclosed in braces)
        if "{" in raw_data:
            # Parse as Tcl list - paths with spaces are in braces
            paths = []
            i = 0
            while i < len(raw_data):
                if raw_data[i] == "{":
                    # Find closing brace
                    try:
                        end = raw_data.index("}", i)
                    except ValueError:
                        break
                    paths.append(raw_data[i + 1 : end])
                    i = end + 1
                elif raw_data[i] == " ":
                    i += 1
                else:
                    # Find next space or end
                    end = raw_data.find(" ", i)
                    if end == -1:
                        end = len(raw_data)
                    paths.append(raw_data[i:end])
                    i = end
        else:
            # Simple space-separated paths (no spaces in filenames)
            paths = raw_data.split()

        # Filter to supported file types
        supported_extensions = {".pdf", ".txt", ".rtf", ".docx", ".png", ".jpg", ".jpeg"}
        valid_files = []
        for path in paths:
            ext = Path(path).suffix.lower()
            if ext in supported_extensions and Path(path).is_file():
                valid_files.append(path)

        if not valid_files:
            self.set_status("No supported files dropped (PDF, TXT, RTF, DOCX, PNG, JPG)")
            return

        logger.debug("Files dropped: %s valid files", len(valid_files))

        # Hide Export All button when new files are dropped
        if self._export_all_visible:
            self.export_all_btn.pack_forget()
            self._export_all_visible = False

        # Reset drop zone highlight
        self._reset_drop_zone_border()

        # Deduplicate: only add files not already in the session
        existing = set(self.selected_files)
        new_files = [f for f in valid_files if f not in existing]
        if not new_files:
            self.set_status("All dropped files are already in the session")
            return

        self.selected_files.extend(new_files)
        self.set_status(
            f"Processing {len(new_files)} dropped {'file' if len(new_files) == 1 else 'files'}..."
        )
        self._start_preprocessing(new_files)

    def _on_drag_enter(self, _event):
        """Highlight the file table area when files are dragged over it."""
        from src.ui.theme import COLORS

        self.left_panel.configure(border_color=COLORS["drop_zone_border"], border_width=2)

    def _on_drag_leave(self, _event):
        """Remove highlight when files leave the drop zone."""
        self._reset_drop_zone_border()

    def _reset_drop_zone_border(self):
        """Restore the left panel's original border state."""
        original_color = getattr(self, "_original_border_color", None)
        original_width = getattr(self, "_original_border_width", 0)
        self.left_panel.configure(
            border_color=original_color or "transparent",
            border_width=original_width or 0,
        )

    def _select_files(self):
        """Open file dialog to select documents for this session."""
        if self._processing_active or self._preprocessing_active:
            logger.warning("_select_files called during active processing — ignored")
            return

        files = filedialog.askopenfilenames(
            title="Select Documents for This Session",
            filetypes=[
                ("Documents", "*.pdf *.txt *.rtf *.docx"),
                ("PDF files", "*.pdf"),
                ("Word documents", "*.docx"),
                ("Text files", "*.txt"),
                ("RTF files", "*.rtf"),
                ("Images (OCR)", "*.png *.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )

        if not files:
            return

        # Hide Export All button when new files are selected
        if self._export_all_visible:
            self.export_all_btn.pack_forget()
            self._export_all_visible = False

        # Deduplicate: only add files not already in the session
        existing = set(self.selected_files)
        new_files = [f for f in files if f not in existing]
        if not new_files:
            self.set_status("All selected files are already in the session")
            return

        self.selected_files.extend(new_files)
        self.set_status(
            f"Processing {len(new_files)} new {'file' if len(new_files) == 1 else 'files'}..."
        )
        self._start_preprocessing(new_files)

    def _clear_files(self):
        """Clear all files from the session."""
        from src.ui.theme import COLORS

        if (
            self._processing_active
            or self._preprocessing_active
            or self._semantic_answering_active
            or self._key_sentences_pending
            or self._followup_pending
        ):
            logger.warning("Cannot clear files during active processing")
            self.set_status("Cannot clear files during processing")
            return

        self.selected_files.clear()
        self.processing_results.clear()
        self.file_table.clear()
        self.output_display.clear_document_preview()
        # Reset search state so old results don't persist
        self._semantic_ready = False
        self._semantic_answering_active = False
        self._semantic_failed = False
        self._key_sentences_pending = False
        with self._semantic_results_lock:
            self._semantic_results.clear()
        self._vector_store_path = None
        self.followup_btn.configure(state="disabled")
        self.followup_entry.configure(
            state="disabled",
            placeholder_text="Search your documents after processing completes...",
            placeholder_text_color=COLORS["placeholder_golden"],
        )
        self._update_generate_button_state()
        self._update_session_stats()  # Clear stats display
        self._update_doc_count_badge()
        self.set_status("Files cleared")

    def _on_file_selected(self, filename):
        """
        Handle file row click — show document preview in the Document tab.

        Args:
            filename: The display filename that was clicked.
        """
        # Find the result dict for this filename
        result = next(
            (r for r in self.processing_results if r.get("filename") == filename),
            None,
        )
        if result:
            self.output_display.show_document_preview(result)
        else:
            # File added but not yet extracted
            logger.debug("No extraction result for %s yet", filename)

    def _remove_file(self, filename):
        """
        Remove a single file from the session by filename.

        Blocked during active processing to prevent state inconsistency
        with the worker subprocess.

        Args:
            filename: The display filename to remove.
        """
        if (
            self._processing_active
            or self._preprocessing_active
            or self._semantic_answering_active
            or self._key_sentences_pending
            or self._followup_pending
        ):
            self.set_status("Cannot remove files during processing")
            return

        # Remove from selected_files (match by basename)
        self.selected_files = [f for f in self.selected_files if Path(f).name != filename]

        # Remove from processing_results
        self.processing_results = [
            r for r in self.processing_results if r.get("filename") != filename
        ]

        # Remove from file table widget
        self.file_table.remove_result(filename)

        # Clear document preview if the removed file is currently previewed
        if self.output_display.document_preview_filename == filename:
            self.output_display.clear_document_preview()

        # Update UI state
        self._update_generate_button_state()
        self._update_session_stats()
        self._update_doc_count_badge()
        self.set_status(f"Removed {filename}")
        logger.debug("Removed file: %s (%d files remain)", filename, len(self.selected_files))

    def _check_ocr_availability(self) -> bool:
        """
        Check if OCR is available and prompt user if not.

        Returns:
            True if OCR should be allowed, False to skip OCR.
        """
        import time

        from src.services.ocr_availability import OCRStatus, check_ocr_availability
        from src.user_preferences import get_user_preferences

        status = check_ocr_availability()
        if status in (OCRStatus.AVAILABLE, OCRStatus.POPPLER_MISSING):
            # Poppler missing only affects PDF-to-image conversion;
            # Tesseract can still OCR image files directly.
            return True

        # Check 90-day snooze
        prefs = get_user_preferences()
        dismiss_until = prefs.get("ocr_dismiss_until", 0)
        if dismiss_until > time.time():
            return False

        # Show dialog on the UI thread
        from src.ui.ocr_dialog import OCRDialog

        dialog = OCRDialog(self)
        result = dialog.result

        if result == "snooze":
            prefs.set("ocr_dismiss_until", time.time() + 90 * 86400)
        # "download" and "skip" both skip OCR for this session
        return False

    def _start_preprocessing(self, file_paths=None):
        """
        Start preprocessing files.

        Args:
            file_paths: List of new file paths to process. If None, uses self.selected_files.
        """
        if self._preprocessing_active:
            logger.warning("_start_preprocessing called while already preprocessing — ignored")
            return

        paths_to_process = file_paths or self.selected_files
        if not paths_to_process:
            return

        # Check OCR availability once per batch (not on retries)
        if not hasattr(self, "_ocr_allowed_cache"):
            self._ocr_allowed_cache = None
        if self._worker_ready_retries == 0:
            self._ocr_allowed_cache = self._check_ocr_availability()
        ocr_allowed = self._ocr_allowed_cache

        # Disable controls during preprocessing
        self.add_files_btn.configure(state="disabled")
        self.generate_btn.configure(state="disabled")
        self.clear_files_btn.configure(state="disabled")

        # Ensure worker subprocess is ready before sending work
        if not self._worker_manager.is_ready():
            self._worker_ready_retries += 1
            if self._worker_ready_retries > 20:  # ~60s of retries
                self.set_status_error("Processing engine failed to start. Please restart the app.")
                self.add_files_btn.configure(state="normal")
                self.generate_btn.configure(state="normal")
                self.clear_files_btn.configure(state="normal")
                self._worker_ready_retries = 0
                return
            retries = self._worker_ready_retries
            self.set_status(f"Processing engine starting up, please wait... ({retries}/20)")
            self._cancel_and_reschedule(
                "_preprocessing_retry_id",
                3000,
                lambda: self._start_preprocessing(paths_to_process),
            )
            return

        self._worker_ready_retries = 0  # Reset retry counter on success
        self._preprocessing_retry_id = None  # Retry completed / not needed

        # Start timer
        self._start_timer()

        # Show placeholder rows immediately so user sees files were accepted
        for fpath in paths_to_process:
            fname = Path(fpath).name
            self.file_table.add_pending_file(fname, fpath)

        # Send only the new batch to the worker subprocess
        logger.debug("Sending process_files: %d file(s)", len(paths_to_process))
        self._worker_manager.send_command(
            "process_files",
            {
                "file_paths": paths_to_process,
                "ocr_allowed": ocr_allowed,
            },
        )
        self._preprocessing_active = True

        # Start polling the queue
        logger.debug("Polling started (preprocessing)")
        self._poll_queue()

    def _on_configure(self, event):
        """Debounce resize events -- tracks whether window is mid-resize."""
        if event.widget is not self:
            return
        self._resize_in_progress = True
        if self._resize_debounce_id is not None:
            self.after_cancel(self._resize_debounce_id)
        self._resize_debounce_id = self.after(150, self._on_resize_complete)

    def _on_resize_complete(self):
        """Called 150ms after the last resize event -- resize is done."""
        self._resize_in_progress = False
        self._resize_debounce_id = None

    def _poll_queue(self):
        """Poll the worker subprocess result queue for messages."""
        if self._destroying:
            return

        # Drain messages from worker subprocess (non-blocking)
        messages = self._worker_manager.check_for_messages()
        for msg in messages:
            try:
                msg_type, data = msg
                self._handle_queue_message(msg_type, data)
            except (TypeError, ValueError):
                logger.warning("Invalid message from worker subprocess: %s", msg)
            except Exception:
                logger.error("Unhandled error processing message: %s", msg, exc_info=True)

        # Detect dead subprocess while we think work is in progress
        any_active = (
            self._processing_active or self._preprocessing_active or self._semantic_answering_active
        )
        if any_active and not messages and not self._worker_manager.is_alive():
            logger.error("Worker subprocess died while tasks were active — recovering")
            self.set_status_error("Worker process crashed. Results may be incomplete.")
            self._preprocessing_active = False
            self._semantic_answering_active = False
            self._key_sentences_pending = False
            # Reset search state so stale session data doesn't persist
            self._semantic_ready = False
            self._semantic_failed = False
            self._vector_store_path = None
            # Disable followup controls — session is dead
            self.followup_entry.configure(
                state="disabled",
                placeholder_text="Session ended — reprocess files",
            )
            self.followup_btn.configure(state="disabled")
            if self._processing_active:
                self._processing_active = False
                self.output_display.set_extraction_in_progress(False)
                self._on_tasks_complete(False, "Worker process crashed")
            any_active = False

        # Continue polling while processing is active, key sentences pending, or we got messages
        if any_active or self._key_sentences_pending or messages:
            self._queue_poll_id = self.after(33, self._poll_queue)  # ~30fps
        else:
            self._queue_poll_id = None
            logger.debug("Polling stopped (all flags inactive, no pending messages)")

    def _handle_queue_message(self, msg_type: str, data):
        """Handle a message from the worker queue."""
        from src.ui.theme import COLORS

        if msg_type == "progress":
            _percentage, message = data
            # Don't overwrite completion status with stale progress messages
            if not self._processing_active:
                return
            # Append search status note if index is ready but answers haven't appeared yet
            if self._semantic_ready and "search" not in message.lower():
                message = f"{message} (searching documents...)"
            self.set_status(message)

        elif msg_type == "file_processed":
            self.processing_results.append(data)
            self.file_table.add_result(data)

        elif msg_type == "processing_finished":
            self._on_preprocessing_complete(data)

        elif msg_type == "error":
            logger.error("Worker error: %s", data)
            self.set_status_error(f"Error: {data}")
            messagebox.showerror("Processing Error", str(data))
            # Snapshot flags before clearing
            was_processing = self._processing_active
            was_preprocessing = self._preprocessing_active
            # Reset ALL processing flags to prevent stuck UI
            self._preprocessing_active = False
            # Reset search state so stale session data doesn't persist
            self._semantic_ready = False
            self._semantic_failed = False
            self._vector_store_path = None
            if was_processing:
                self._processing_active = False
                self._semantic_answering_active = False
                self.output_display.set_extraction_in_progress(False)
                self._on_tasks_complete(False, str(data))
            elif was_preprocessing:
                self._on_preprocessing_complete([])

        # Progressive Extraction handlers
        elif msg_type == "extraction_started":
            # Dim feedback buttons while extraction is in progress
            logger.debug("Extraction started - dimming feedback buttons")
            self.output_display.set_extraction_in_progress(True)

        elif msg_type == "extraction_complete":
            # Re-enable feedback buttons after extraction completes
            logger.debug("Extraction complete - enabling feedback buttons")
            self.output_display.set_extraction_in_progress(False)

        elif msg_type == "partial_vocab_complete":
            # Show BM25 + RAKE results before NER completes
            term_count = len(data) if data else 0
            logger.debug("Partial results: %s terms from BM25+RAKE", term_count)
            self.output_display.update_outputs(vocab_csv_data=data)
            self.output_display.set_extraction_source("partial")
            from src.config import DEBUG_MODE

            if DEBUG_MODE:
                self.set_status(f"Found {term_count} terms (BM25+RAKE). Running NER...")
            else:
                self.set_status(f"Found {term_count} key terms. Now scanning for names...")

        elif msg_type == "ner_progress":
            # Update status bar with NER chunk progress
            chunk_num = data.get("chunk_num", 0)
            total_chunks = data.get("total_chunks", 1)
            pct = int((chunk_num / max(total_chunks, 1)) * 100)
            from src.config import DEBUG_MODE

            if DEBUG_MODE:
                self.set_status(f"NER: {pct}% complete (chunk {chunk_num}/{total_chunks})...")
            else:
                self.set_status(f"Extracting names... {pct}% complete")
            # Note: We don't update the vocab table with each chunk because raw NER
            # candidates need post-processing. The final merged results come with ner_complete.

        elif msg_type == "ner_complete":
            vocab_data = data.get("vocab", []) if isinstance(data, dict) else data
            filtered_data = data.get("filtered", []) if isinstance(data, dict) else []
            skipped = data.get("skipped_algorithms", []) if isinstance(data, dict) else []
            if skipped:
                self._degraded_algorithms = skipped
            term_count = len(vocab_data)
            logger.debug(
                "NER complete: %s terms, %s filtered, %s skipped - displaying",
                term_count,
                len(filtered_data),
                skipped or "none",
            )
            self.output_display.update_outputs(
                vocab_csv_data=vocab_data, filtered_vocab_data=filtered_data
            )
            self.output_display.set_extraction_source("ner")

            # Vocab results are now visible — transition Search tab from
            # "vocab in progress" to "building search index" so users
            # aren't confused by the vocab tab being done while the Search
            # tab still says vocab is running.
            if not self._semantic_ready:
                from src.ui.workflow_status import WorkflowPhase

                self.output_display.set_workflow_phase(WorkflowPhase.SEMANTIC_INDEXING)

            if self._pending_tasks.get("semantic"):
                self.set_status(f"Found {term_count} terms. Building search index...")
            else:
                self.set_status(f"Vocabulary extraction complete: {term_count} terms")

            # Vocabulary extraction is complete — mark task done
            self._completed_tasks.add("vocab")
            if self._all_tasks_complete():
                self._finalize_tasks()

        elif msg_type == "semantic_ready":
            chunk_count = data.get("chunk_count", 0)
            logger.debug("Semantic search ready: %s chunks indexed", chunk_count)
            self._vector_store_path = data.get("vector_store_path")
            # Embeddings stay in worker subprocess (not picklable)
            self._semantic_ready = True
            self._key_sentences_pending = True  # Daemon thread extracting in subprocess
            # Enable search input whenever search index is ready; button
            # reflects whether the entry has any text (empty → disabled).
            self.followup_entry.configure(
                state="normal",
                placeholder_text="Search your documents...",
                placeholder_text_color="white",
            )
            self._update_followup_btn_state()
            self.set_status(
                f"Search index ready ({chunk_count} passages). Running default searches..."
            )

        elif msg_type == "semantic_error":
            error_msg = (
                data.get("error", "Unknown search error") if isinstance(data, dict) else str(data)
            )
            logger.warning("Semantic indexing error (full): %s", error_msg)
            # Truncate at word boundary to avoid mid-word cuts
            if len(error_msg) > 80:
                error_msg = error_msg[:77].rsplit(" ", 1)[0] + "..."
            self.set_status_error(f"Search unavailable: {error_msg}")
            # Search won't be available but vocab extraction can continue
            self._semantic_answering_active = False
            self._semantic_failed = True
            # Disable follow-up controls with explanatory message
            self.followup_entry.configure(
                state="disabled",
                placeholder_text="Search unavailable \u2014 index failed to build.",
                placeholder_text_color=COLORS["placeholder_red"],
            )
            self.followup_btn.configure(state="disabled")
            if self._pending_tasks.get("semantic"):
                self._failed_tasks.add("semantic")
            # Key excerpts won't arrive if vector store failed
            self._failed_tasks.add("key_excerpts")
            if self._all_tasks_complete():
                self._finalize_tasks()

        elif msg_type == "trigger_default_semantic_started":
            # SemanticWorker was auto-spawned in the worker subprocess
            self._semantic_answering_active = True
            if not self.ask_default_questions_check.get():
                logger.debug("Default questions disabled, skipping display update")
                self.set_status("Ready. Type a search below to query your documents.")
            else:
                # Update workflow phase for tab status
                from src.ui.workflow_status import WorkflowPhase

                self.output_display.set_workflow_phase(WorkflowPhase.SEMANTIC_SEARCHING)
                logger.debug("Default questions worker started in subprocess")

        # Search result handlers (messages from default searches worker)
        elif msg_type == "semantic_progress":
            current, total, _question = data
            answered = current + 1
            logger.debug("Semantic search progress: %s/%s", answered, total)
            if answered < total:
                self.set_status(f"Running search {answered}/{total}...")
            else:
                self.set_status(f"Completed {answered}/{total} searches")

        elif msg_type == "semantic_result":
            # Individual search result - add to results and update display
            logger.debug("Semantic search result received")
            with self._semantic_results_lock:  # Thread-safe access
                self._semantic_results.append(data)
                self.output_display.update_outputs(semantic_results=self._semantic_results)

        elif msg_type == "semantic_complete":
            # All default searches answered
            semantic_results = data if data else []
            logger.debug("Semantic search complete: %s answers", len(semantic_results))
            with self._semantic_results_lock:  # Thread-safe access
                self._semantic_results = semantic_results
            if semantic_results:
                self.output_display.update_outputs(semantic_results=semantic_results)
                self.set_status(f"Default searches complete: {len(semantic_results)} responses")
            else:
                self.set_status("Default searches complete: no matches found")
            self._update_followup_btn_state()
            self._semantic_answering_active = False
            if self._pending_tasks.get("semantic"):
                self._completed_tasks.add("semantic")
            if self._all_tasks_complete():
                self._finalize_tasks()

        elif msg_type == "key_sentences_result":
            # Key excerpts extracted via K-means clustering on chunk embeddings
            self._key_sentences_pending = False
            if data:
                self.output_display.update_key_sentences(data)
                logger.debug("Key excerpts displayed: %d passages", len(data))
            self._completed_tasks.add("key_excerpts")
            if self._all_tasks_complete():
                self._finalize_tasks()

        elif msg_type == "key_sentences_error":
            self._key_sentences_pending = False
            logger.error("Key excerpts extraction failed: %s", data)
            self.set_status_error(f"Key excerpts failed: {data}")
            self._failed_tasks.add("key_excerpts")
            if self._all_tasks_complete():
                self._finalize_tasks()

        elif msg_type == "semantic_followup_result":
            # _poll_queue() and _poll_followup_result() both drain the same queue, so
            # _poll_queue() may steal a tab-followup result when _key_sentences_pending
            # keeps it running after processing completes.
            if self._followup_pending:
                # Tab followup is waiting — stash so _poll_followup_result() can pick it up
                self._tab_followup_result = data
                logger.debug("Tab followup result stashed (intercepted by _poll_queue)")
            else:
                logger.debug("Followup result received but no consumer waiting")

        elif msg_type == "command_ack":
            cmd = data.get("cmd", "unknown") if isinstance(data, dict) else data
            logger.debug("Worker acknowledged command: %s", cmd)

        elif msg_type == "status_error":
            self.set_status_error(str(data))

        else:
            # Log unhandled messages for debugging
            logger.debug("Unhandled message type: %s", msg_type)

    def _on_preprocessing_complete(self, results: list[dict]):
        """Handle preprocessing completion."""
        # Stop timer
        self._stop_timer()

        # Stop queue polling
        if self._queue_poll_id:
            self.after_cancel(self._queue_poll_id)
            self._queue_poll_id = None

        # Merge new results into processing_results (cumulative)
        # The results contain 'preprocessed_text' added by ProcessingWorker
        # which wasn't present in the individual 'file_processed' messages
        if results:
            # Build set of filenames already in processing_results
            existing_filenames = {r.get("filename") for r in self.processing_results}
            for r in results:
                fname = r.get("filename")
                if fname and fname not in existing_filenames:
                    self.processing_results.append(r)
                elif fname in existing_filenames:
                    # Update existing entry with new preprocessed data
                    for i, existing in enumerate(self.processing_results):
                        if existing.get("filename") == fname:
                            self.processing_results[i] = r
                            break
            logger.debug(
                "Merged batch (%d new) → %d total documents",
                len(results),
                len(self.processing_results),
            )

        # Clear preprocessing flag so _update_generate_button_state sees correct state
        self._preprocessing_active = False

        # Re-enable controls
        self.add_files_btn.configure(state="normal")
        self.clear_files_btn.configure(state="normal")
        self._update_generate_button_state()

        # Count results
        success_count = sum(1 for r in results if r.get("status") == "success")
        failed_count = len(results) - success_count

        status = f"Processed {len(results)} {'file' if len(results) == 1 else 'files'}: {success_count} ready"
        if failed_count > 0:
            status += f", {failed_count} failed"
            self.set_status_error(status)
        else:
            self.set_status(status)
        self._update_session_stats()  # Show document stats
        self._update_doc_count_badge()

    # =========================================================================
    # Task Execution
    # =========================================================================

    def _update_generate_button_state(self):
        """Update the generate button state based on whether files are loaded."""
        if self._processing_active:
            self.generate_btn.configure(state="disabled")
            return

        # Ensure complete button is swapped back to generate button
        if self.complete_btn.winfo_ismapped():
            self._show_generate_button()

        has_files = len(self.processing_results) > 0

        if not has_files:
            self.generate_btn.configure(text="Process Documents", state="disabled")
        else:
            self.generate_btn.configure(text="Process Documents", state="normal")

    def _load_default_question_count(self) -> tuple[int, int]:
        """
        Get count of enabled and total default questions.

        Uses DefaultQuestionsManager for enable/disable support.

        Returns:
            Tuple of (enabled_count, total_count)
        """
        try:
            from src.services import SemanticService

            manager = SemanticService().get_default_questions_manager()
            return (manager.get_enabled_count(), manager.get_total_count())

        except Exception as e:
            logger.debug("Error loading default question count: %s", e)
            return (0, 0)

    def _update_default_questions_label(self):
        """Update checkbox text and state based on enabled question count."""
        enabled, total = self._load_default_question_count()
        search_word = "search" if enabled == 1 else "searches"

        if enabled == 0:
            # No default searches configured — grey out checkbox
            self.ask_default_questions_check.configure(
                text="No default searches configured", state="disabled"
            )
            self.ask_default_questions_check.deselect()
        elif enabled == total:
            self.ask_default_questions_check.configure(
                text=f"Run {enabled:,} default {search_word}", state="normal"
            )
        else:
            self.ask_default_questions_check.configure(
                text=f"Run {enabled:,}/{total:,} default {search_word}", state="normal"
            )

    def _save_task_checkbox_states(self):
        """Save default questions checkbox state to user preferences."""
        try:
            from src.user_preferences import get_user_preferences

            prefs = get_user_preferences()
            prefs.set("task_default_questions", self.ask_default_questions_check.get())
        except Exception as e:
            logger.warning("Could not save task checkbox states: %s", e)

    def _restore_task_checkbox_states(self):
        """Restore default questions checkbox state from user preferences."""
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()

        if not prefs.get("task_default_questions", True):
            self.ask_default_questions_check.deselect()

        self._update_generate_button_state()

    def _on_default_questions_toggled(self):
        """Handle default questions checkbox state change."""
        self._save_task_checkbox_states()
        state = "enabled" if self.ask_default_questions_check.get() else "disabled"
        logger.debug("Default questions %s", state)

    def refresh_default_questions_label(self):
        """Refresh the default questions checkbox label. Called after editing questions."""
        self._update_default_questions_label()

    def _update_session_stats(self, extraction_stats: dict | None = None):
        """
        Update the session stats display.

        Shows document stats (file count, pages, size) and extraction stats
        (term count, person count, semantic search count) after processing.

        Args:
            extraction_stats: Optional dict with extraction results:
                - vocab_count: Total vocabulary terms
                - person_count: Terms marked as persons
                - semantic_count: Number of semantic search results
                - processing_time: Time in seconds
        """
        if not self.processing_results:
            self.stats_label.configure(text="")
            return

        # Document stats from processing_results
        file_count = len(self.processing_results)
        success_count = sum(1 for r in self.processing_results if r.get("status") == "success")
        total_pages = sum(r.get("page_count", 0) or 0 for r in self.processing_results)
        total_size = sum(r.get("file_size", 0) or 0 for r in self.processing_results)
        total_words = sum(
            len((r.get("preprocessed_text") or r.get("extracted_text", "")).split())
            for r in self.processing_results
        )

        # Format file size
        if total_size >= 1024 * 1024:
            size_str = f"{total_size / (1024 * 1024):.1f} MB"
        elif total_size >= 1024:
            size_str = f"{total_size / 1024:.0f} KB"
        else:
            size_str = f"{total_size} B"

        # Format word count
        if total_words >= 1000:
            words_str = f"{total_words:,} words"
        else:
            words_str = f"{total_words} words"

        # Build stats text
        parts = [f"{success_count}/{file_count} files"]
        if total_pages > 0:
            parts.append(f"{total_pages} pages")
        parts.append(size_str)
        if total_words > 0:
            parts.append(words_str)

        stats_text = " · ".join(parts)

        # Add extraction stats if available
        if extraction_stats:
            ext_parts = []
            if extraction_stats.get("vocab_count", 0) > 0:
                v = extraction_stats["vocab_count"]
                p = extraction_stats.get("person_count", 0)
                ext_parts.append(f"{v} terms ({p} persons)")
            if extraction_stats.get("semantic_count", 0) > 0:
                ext_parts.append(f"{extraction_stats['semantic_count']} searches")
            if extraction_stats.get("processing_time"):
                t = extraction_stats["processing_time"]
                if t >= 60:
                    ext_parts.append(f"{t / 60:.1f}m")
                else:
                    ext_parts.append(f"{t:.1f}s")

            if ext_parts:
                stats_text += "\n" + " · ".join(ext_parts)

        self.stats_label.configure(text=stats_text)

        logger.debug("Session stats: %s", stats_text.replace(chr(10), " | "))

    def _perform_tasks(self):
        """Execute document processing: vocabulary, semantic search, and key excerpts."""
        if self._processing_active:
            logger.warning("_perform_tasks called while already processing — ignored")
            return

        if not self.processing_results or all(
            r.get("status") != "success" for r in self.processing_results
        ):
            title, message = _pick_no_valid_files_msg(self.selected_files, self.processing_results)
            messagebox.showwarning(title, message)
            logger.info(
                "Aborting _perform_tasks: %s (selected=%d, results=%d)",
                title,
                len(self.selected_files),
                len(self.processing_results),
            )
            return

        # Disable controls during processing
        self._processing_active = True
        self.generate_btn.configure(state="disabled")
        self._show_stop_button()
        self.add_files_btn.configure(state="disabled")
        self.clear_files_btn.configure(state="disabled")

        # Start timer
        self._start_timer()

        # Track pending tasks (vocab + search + key excerpts)
        self._pending_tasks = {"vocab": True, "semantic": True, "key_excerpts": True}
        self._completed_tasks = set()
        self._failed_tasks = set()
        self._degraded_algorithms = []
        with self._semantic_results_lock:
            self._semantic_results.clear()
        self._semantic_ready = False
        self._semantic_answering_active = False
        self._semantic_failed = False
        self._key_sentences_pending = False

        # Set initial workflow phase
        from src.ui.workflow_status import WorkflowPhase

        self.output_display.set_workflow_phase(WorkflowPhase.VOCAB_RUNNING)

        self._start_progressive_extraction()

    # =========================================================================
    # Stop / Cancel
    # =========================================================================

    def _on_stop_clicked(self):
        """Handle stop button click — confirm then cancel all active work."""
        if not self._processing_active:
            return

        result = messagebox.askyesno(
            "Stop Processing",
            "Are you sure you want to stop?\n\n"
            "Completed results (vocabulary, search results) will be kept,\n"
            "but any work still in progress will be lost.",
            icon="warning",
        )
        if not result:
            return

        logger.info("User requested stop — cancelling active work")

        # Send cancel to worker subprocess
        if self._worker_manager and self._worker_manager.is_alive():
            self._worker_manager.cancel()

        # Reset processing state
        self._semantic_answering_active = False
        self._preprocessing_active = False
        self._key_sentences_pending = False
        self._followup_pending = False
        self._followup_poll_count = 0

        # Finalize as a partial completion
        completed = len(self._completed_tasks)
        if completed:
            msg = f"Stopped by user ({completed} task(s) completed)"
        else:
            msg = "Stopped by user"
        self._on_tasks_complete(True, msg)
        self.set_status(msg)

    def _show_stop_button(self):
        """Swap generate/complete button for stop button during processing."""
        self.generate_btn.grid_remove()
        self.complete_btn.grid_remove()
        self.stop_btn.grid(row=6, column=0, sticky="ew", padx=10, pady=(15, 5))

    def _hide_stop_button(self, partial: bool = False, degraded: bool = False):
        """Swap stop button to complete/incomplete/degraded button."""
        from src.ui.theme import BUTTON_STYLES

        self.stop_btn.grid_remove()
        if partial:
            self.complete_btn.configure(text="Incomplete", **BUTTON_STYLES["warning"])
        elif degraded:
            self.complete_btn.configure(text="Complete*", **BUTTON_STYLES["degraded"])
        else:
            self.complete_btn.configure(text="Complete", **BUTTON_STYLES["success"])
        self.complete_btn.grid(row=6, column=0, sticky="ew", padx=10, pady=(15, 5))

    def _show_generate_button(self):
        """Swap complete button back to generate button."""
        self.complete_btn.grid_remove()
        self.generate_btn.grid(row=6, column=0, sticky="ew", padx=10, pady=(15, 5))

    # =========================================================================
    # Progressive Extraction
    # =========================================================================

    def _start_progressive_extraction(self):
        """
        Start progressive two-phase extraction.

        Phase 1 (NER): Fast, displays results in ~5 seconds
        Phase 2 (Semantic): Builds vector store, enables semantic search panel
        """
        from src.config import (
            LEGAL_EXCLUDE_LIST_PATH,
            MEDICAL_TERMS_LIST_PATH,
            USER_VOCAB_EXCLUDE_PATH,
        )

        self.set_status("Starting vocabulary extraction...")

        # Combine text from all processed documents
        from src.services import DocumentService

        combined_text = DocumentService().combine_document_texts(self.processing_results)

        logger.debug(
            "Progressive extraction: %s chars from %s docs",
            len(combined_text),
            len(self.processing_results),
        )

        if not combined_text.strip():
            logger.debug("WARNING: No text after combining documents!")
            self._on_tasks_complete(False, "No text to analyze")
            return

        # Calculate aggregate document confidence
        doc_confidence = self._calculate_aggregate_confidence(self.processing_results)
        logger.debug("Aggregate document confidence: %.1f%%", doc_confidence)

        # Ensure worker subprocess is ready before sending work
        if not self._worker_manager.is_ready():
            self._worker_ready_retries += 1
            if self._worker_ready_retries > 20:  # ~60s of retries
                self._worker_ready_retries = 0
                self._on_tasks_complete(
                    False, "Processing engine failed to start. Please restart the app."
                )
                return
            retries = self._worker_ready_retries
            self.set_status(f"Processing engine starting up, please wait... ({retries}/20)")
            self._cancel_and_reschedule(
                "_extraction_retry_id", 3000, self._start_progressive_extraction
            )
            return

        self._worker_ready_retries = 0  # Reset retry counter on success
        self._extraction_retry_id = None  # Retry completed / not needed

        # Send extraction command to worker subprocess
        logger.debug(
            "Sending extract: %d doc(s), confidence=%.0f%%",
            len(self.processing_results),
            doc_confidence,
        )
        self._worker_manager.send_command(
            "extract",
            {
                "documents": self.processing_results,
                "combined_text": combined_text,
                "exclude_list_path": str(LEGAL_EXCLUDE_LIST_PATH),
                "medical_terms_path": str(MEDICAL_TERMS_LIST_PATH),
                "user_exclude_path": str(USER_VOCAB_EXCLUDE_PATH),
                "doc_confidence": doc_confidence,
                "ask_default_questions": bool(self.ask_default_questions_check.get()),
            },
        )

        # Timer was already started by _perform_tasks() — no need to restart here.
        # (Calling it again would create a parallel timer loop.)

        # Ensure queue polling is running (may have stopped after preprocessing)
        if self._queue_poll_id:
            self.after_cancel(self._queue_poll_id)
        logger.debug("Polling started (extraction)")
        self._poll_queue()

    def _start_semantic_task(self):
        """Start Semantic Search task via worker subprocess."""
        self.set_status("Semantic Search: Loading embeddings model (this may take a moment)...")

        # Send run_qa command to worker subprocess
        # The subprocess handles embeddings loading and vector store creation
        logger.debug("Sending run_qa: mode=extraction")
        self._worker_manager.send_command(
            "run_qa",
            {
                "answer_mode": "extraction",
            },
        )

        # Ensure queue polling is active
        if self._queue_poll_id:
            self.after_cancel(self._queue_poll_id)
        logger.debug("Polling started (semantic search)")
        self._poll_queue()

    def _all_tasks_complete(self) -> bool:
        """Check if all pending tasks are done (completed or failed)."""
        for task_name, is_pending in self._pending_tasks.items():
            if is_pending and task_name not in self._completed_tasks | self._failed_tasks:
                return False
        return not self._semantic_answering_active

    def _finalize_tasks(self):
        """Finalize all tasks and update UI."""
        # Guard against duplicate finalization (e.g. two semantic_error messages)
        if not self._processing_active:
            logger.debug("Skipping finalization: already complete")
            return
        if self._semantic_answering_active:
            logger.debug("Deferring finalization: semantic search still active")
            return
        # Guard against race: semantic search is pending but trigger_default_semantic_started
        # hasn't arrived yet (subprocess is still building the index).
        done = self._completed_tasks | self._failed_tasks
        semantic_pending_not_started = (
            self._pending_tasks.get("semantic") and "semantic" not in done
        )
        if semantic_pending_not_started and not self._semantic_failed:
            logger.debug("Deferring finalization: semantic search pending but not yet started")
            return
        completed = len(self._completed_tasks)
        failed = len(self._failed_tasks)
        if failed:
            msg = f"Completed {completed} task(s), {failed} failed"
            self._on_tasks_complete(True, msg, partial=True)
        elif self._degraded_algorithms:
            names = ", ".join(self._degraded_algorithms)
            msg = f"Completed {completed} task(s) (unavailable: {names})"
            self._on_tasks_complete(True, msg, degraded=True)
        else:
            self._on_tasks_complete(True, f"Completed {completed} task(s)")

    def _on_tasks_complete(
        self, success: bool, message: str, partial: bool = False, degraded: bool = False
    ):
        """Handle task completion."""
        self._stop_timer()
        self._processing_active = False
        self._hide_stop_button(partial=partial, degraded=degraded)
        self.output_display.set_extraction_in_progress(False)

        # Update workflow phase for tab status
        from src.ui.workflow_status import WorkflowPhase

        if partial:
            self.output_display.set_workflow_phase(WorkflowPhase.COMPLETE_WITH_ERRORS)
        else:
            self.output_display.set_workflow_phase(WorkflowPhase.COMPLETE)

        # Re-enable controls
        self.add_files_btn.configure(state="normal")
        self.clear_files_btn.configure(state="normal")
        self._update_generate_button_state()

        # Enable follow-up if search succeeded (empty entry keeps button disabled)
        if success and not self._semantic_failed:
            self._update_followup_btn_state()

        # Show Export All button after successful processing
        if success and not self._export_all_visible:
            self.export_all_btn.pack(side="right", padx=10, pady=3)
            self._export_all_visible = True

        # Update session stats with extraction results
        if success:
            extraction_stats = self._gather_extraction_stats()
            self._update_session_stats(extraction_stats)

        if success and degraded:
            self.set_status_error(message, hold_seconds=8.0)
        elif success:
            self.set_status(message)
        else:
            self.set_status_error(message)

    def _gather_extraction_stats(self) -> dict:
        """
        Gather extraction statistics after task completion.

        Returns:
            Dict with vocab_count, person_count, semantic_count, processing_time
        """
        stats = {}

        # Vocabulary stats from output display
        vocab_data = (
            self.output_display._get_filtered_vocab_data()
            if hasattr(self.output_display, "_get_filtered_vocab_data")
            else None
        )
        if vocab_data:
            from src.config import VF

            stats["vocab_count"] = len(vocab_data)
            stats["person_count"] = sum(1 for v in vocab_data if v.get(VF.IS_PERSON) == VF.YES)

        # Search stats
        if self._semantic_results:
            stats["semantic_count"] = len(self._semantic_results)

        # Processing time
        if self._processing_start_time:
            stats["processing_time"] = time.time() - self._processing_start_time

        return stats

    def _update_followup_btn_state(self, _event=None):
        """
        Enable or disable the search button based on entry content.

        Called from <KeyRelease> / <FocusIn> bindings on the follow-up entry,
        and once at widget creation. Does not touch the button when the entry
        itself is disabled (no search index yet) — that state is managed by
        the semantic_ready / semantic_failed flows, and we must not override
        their intent. Whitespace-only text counts as empty.
        """
        entry = getattr(self, "followup_entry", None)
        btn = getattr(self, "followup_btn", None)
        if entry is None or btn is None:
            return
        # Respect disabled entry (search not ready / search in progress)
        try:
            entry_state = str(entry.cget("state"))
        except Exception:
            entry_state = "normal"
        if entry_state == "disabled":
            return
        text = entry.get().strip() if entry.get() else ""
        btn.configure(state="normal" if text else "disabled")

    def _ask_followup(self):
        """Ask a follow-up search using the semantic search system (async version)."""
        question = self.followup_entry.get().strip()
        if not question:
            return

        # Check prerequisites (vector_store_path tracked locally, embeddings in subprocess)
        if not self._vector_store_path or not self._semantic_ready:
            messagebox.showwarning(
                "Search Not Ready",
                "Search system is not initialized yet.\n\n"
                "To search your documents:\n"
                "1. Add document files\n"
                "2. Click 'Process Documents'\n"
                "3. Wait for the search index to finish building\n\n"
                "The search system will be ready once the vector index is built.",
            )
            return

        # Prevent duplicate submissions
        if getattr(self, "_followup_pending", False):
            logger.debug("Follow-up already in progress, ignoring")
            return

        # Clear entry and disable controls while processing
        self.followup_entry.delete(0, "end")
        self.followup_btn.configure(state="disabled", text="Searching...")
        self.followup_entry.configure(state="disabled")

        # Truncate at word boundary to avoid mid-word cuts
        if len(question) > 60:
            display_q = question[:57].rsplit(" ", 1)[0] + "..."
        else:
            display_q = question
        self.set_status(f"Searching: {display_q}")

        # Show pending result immediately in search panel

        SemanticResult = SemanticService().get_semantic_result_class()
        pending_result = SemanticResult(
            question=question,
            quick_answer=PENDING_ANSWER_TEXT,
            citation="",
            is_followup=True,
            include_in_export=False,
        )
        with self._semantic_results_lock:
            self._semantic_results.append(pending_result)
            self._pending_followup_index = len(self._semantic_results) - 1
        self.output_display.update_outputs(semantic_results=list(self._semantic_results))
        # Switch to Search tab so user sees the pending search
        self.output_display.tabview.set("Search")

        # Send follow-up command to worker subprocess
        logger.debug("Sending followup: %.80s", question)
        self._followup_pending = True
        self._followup_poll_count = 0
        self._worker_manager.send_command("followup", {"question": question})

        # Start polling for followup result (comes via semantic_followup_result message)
        self._poll_followup_result()

    def _poll_followup_result(self):
        """Poll for follow-up result from worker subprocess."""
        if self._destroying:
            return

        # Dead-worker detection: if subprocess crashed, stop polling immediately
        if not self._worker_manager.is_alive():
            logger.error("Worker process died during followup search")
            self._followup_pending = False
            self._followup_poll_count = 0
            self.followup_btn.configure(text="Search")
            self.followup_entry.configure(state="normal")
            self._update_followup_btn_state()
            self.set_status_error("Search failed — worker process crashed. Please restart the app.")
            return

        # Timeout guard: stop polling after 5 minutes (safety net for stuck subprocess)
        self._followup_poll_count += 1
        if self._followup_poll_count >= self._FOLLOWUP_TIMEOUT_POLLS:
            logger.warning("Follow-up polling timed out after %d polls", self._followup_poll_count)
            self._followup_pending = False
            self._followup_poll_count = 0
            self.followup_btn.configure(text="Search")
            self.followup_entry.configure(state="normal")
            self._update_followup_btn_state()
            self.set_status_error("Search timed out — worker may have crashed")
            return

        # Check for semantic_followup_result in subprocess messages
        messages = self._worker_manager.check_for_messages()
        followup_result = None
        other_messages = []

        for msg in messages:
            try:
                msg_type, data = msg
                if msg_type == "semantic_followup_result":
                    followup_result = data
                else:
                    # Re-handle non-followup messages normally
                    other_messages.append(msg)
            except (TypeError, ValueError):
                logger.debug("Malformed message in followup poll: %s", msg)
            except Exception:
                logger.error("Unhandled error in followup poll: %s", msg, exc_info=True)

        # Process any non-followup messages that arrived
        for msg in other_messages:
            try:
                msg_type, data = msg
                self._handle_queue_message(msg_type, data)
            except (TypeError, ValueError):
                logger.debug("Malformed message in queue processing: %s", msg)
            except Exception:
                logger.error("Unhandled error processing message: %s", msg, exc_info=True)

        # Also check if _poll_queue() already consumed the message and stashed it
        if followup_result is None and self._tab_followup_result is not None:
            followup_result = self._tab_followup_result
            self._tab_followup_result = None
            logger.debug("Tab followup result recovered from stash")

        if followup_result is None and not messages:
            # No result yet, keep polling
            self.after(100, self._poll_followup_result)
            return

        if followup_result is None:
            # Got messages but no followup result yet
            self.after(100, self._poll_followup_result)
            return

        # Got a result - re-enable controls (empty entry keeps button disabled)
        self._followup_pending = False
        self._followup_poll_count = 0
        self.followup_btn.configure(text="Search")
        self.followup_entry.configure(state="normal")
        self._update_followup_btn_state()
        self.followup_entry.focus()

        try:
            if followup_result is not None and hasattr(followup_result, "quick_answer"):
                # Replace pending result with actual answer
                with self._semantic_results_lock:
                    pending_idx = getattr(self, "_pending_followup_index", None)
                    if pending_idx is not None and 0 <= pending_idx < len(self._semantic_results):
                        if self._semantic_results[pending_idx].quick_answer == PENDING_ANSWER_TEXT:
                            self._semantic_results[pending_idx] = followup_result
                        else:
                            self._semantic_results.append(followup_result)
                    else:
                        self._semantic_results.append(followup_result)
                    self._pending_followup_index = None
                    self.output_display.update_outputs(
                        semantic_results=list(self._semantic_results)
                    )
                answer_len = len(followup_result.citation) if followup_result.citation else 0
                self.set_status(f"Follow-up answered: {answer_len} chars")
                logger.debug("Follow-up result displayed successfully")
            else:
                # None result = error in subprocess
                with self._semantic_results_lock:
                    pending_idx = getattr(self, "_pending_followup_index", None)
                    if pending_idx is not None and 0 <= pending_idx < len(self._semantic_results):
                        if self._semantic_results[pending_idx].quick_answer == PENDING_ANSWER_TEXT:
                            self._semantic_results.pop(pending_idx)
                    self._pending_followup_index = None
                    self.output_display.update_outputs(
                        semantic_results=list(self._semantic_results)
                    )
                self.set_status_error("Follow-up search could not be completed")
                messagebox.showerror("Error", "Failed to process follow-up search")
        except Exception as e:
            logger.debug("Error processing follow-up result: %s", e)
            self.set_status_error("Follow-up error. Try rephrasing your search.")
            messagebox.showerror("Error", f"Error displaying result: {e!s}")

    # =========================================================================
    # Settings
    # =========================================================================

    def _open_settings(self):
        """Open the settings dialog."""
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        font_size_before = prefs.get("font_size", "medium")

        def _after_settings():
            self._refresh_corpus_dropdown()
            self.refresh_default_questions_label()
            font_size_after = prefs.get("font_size", "medium")
            if font_size_after != font_size_before:
                messagebox.showinfo(
                    "Restart Required",
                    "Font size changed. Please restart the application for the new size to take full effect.",
                )

        from src.ui.settings.settings_dialog import SettingsDialog

        self._open_modal_dialog(
            self.settings_btn,
            lambda: SettingsDialog(parent=self),
            after=_after_settings,
        )

    def _open_search_settings(self):
        """Open the settings dialog directly to the Search tab."""
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        font_size_before = prefs.get("font_size", "medium")

        def _after_settings():
            self._refresh_corpus_dropdown()
            self.refresh_default_questions_label()
            font_size_after = prefs.get("font_size", "medium")
            if font_size_after != font_size_before:
                messagebox.showinfo(
                    "Restart Required",
                    "Font size changed. Please restart the application for the new size to take full effect.",
                )

        from src.ui.settings.settings_dialog import SettingsDialog

        self._open_modal_dialog(
            self.set_defaults_btn,
            lambda: SettingsDialog(parent=self, initial_tab="Search"),
            after=_after_settings,
        )

    # =========================================================================
    # Export All (tabbed HTML)
    # =========================================================================

    def _export_all(self):
        """
        Export all results (vocabulary, semantic search, summary) to a single file.

        Opens a save dialog offering HTML, Word, and PDF formats. Gathers
        score-filtered vocab, answered searches, and summary text into one document.
        """
        if self._exporting_all:
            return
        self._exporting_all = True

        try:
            self._export_all_impl()
        except Exception:
            logger.error("Export all failed", exc_info=True)
        finally:
            self._exporting_all = False

    def _export_all_impl(self):
        """Implementation of _export_all, guarded by _exporting_all flag."""
        from datetime import datetime

        from src.services import DocumentService, get_export_service
        from src.user_preferences import get_user_preferences

        # Gather filtered vocabulary data
        vocab_data = self.output_display._get_filtered_vocab_data()

        # Gather answered search results only
        semantic_results = []
        if self._semantic_results:
            semantic_panel = self.output_display._semantic_panel
            if semantic_panel and semantic_panel._results:
                semantic_results = [r for r in semantic_panel._results if r.is_exportable]

        # Gather key excerpts text
        summary = self.output_display._outputs.get("Key Excerpts", "")
        summary_text = summary.strip() if summary else ""

        # Check we have something to export
        if not vocab_data and not semantic_results and not summary_text:
            messagebox.showwarning("No Data", "No results to export yet.")
            return

        # Get initial directory (last export path or Documents)
        prefs = get_user_preferences()
        doc_service = DocumentService()
        initial_dir = prefs.get("last_export_path") or doc_service.get_default_documents_folder()

        # Open save dialog with all supported formats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[
                ("HTML files", "*.html"),
                ("Word documents", "*.docx"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*"),
            ],
            initialfile=f"case_report_{timestamp}.html",
            initialdir=initial_dir,
            title="Export All Results",
        )

        if not filepath:
            return

        # Route to format-specific export
        export_service = get_export_service()
        ext = Path(filepath).suffix.lower()

        if ext == ".docx":
            success, error_detail = export_service.export_combined_to_word(
                vocab_data=vocab_data,
                semantic_results=semantic_results,
                file_path=filepath,
                summary_text=summary_text,
            )
            fmt_label = "Word"
        elif ext == ".pdf":
            success, error_detail = export_service.export_combined_to_pdf(
                vocab_data=vocab_data,
                semantic_results=semantic_results,
                file_path=filepath,
                summary_text=summary_text,
            )
            fmt_label = "PDF"
        else:
            visible_columns = self.output_display._get_visible_columns()
            success, error_detail = export_service.export_combined_html(
                vocab_data=vocab_data,
                semantic_results=semantic_results,
                summary_text=summary_text,
                file_path=filepath,
                visible_columns=visible_columns,
            )
            fmt_label = "HTML"

        if success:
            # Remember export folder
            prefs.set("last_export_path", str(Path(filepath).parent))

            # Flash button and status
            self.export_all_btn.configure(text="Exported!")

            def _reset_export_btn():
                try:
                    self.export_all_btn.configure(text="Export All")
                except Exception:
                    pass  # Widget destroyed during delay

            self.after(1500, _reset_export_btn)

            filename = Path(filepath).name
            parts = []
            if vocab_data:
                parts.append(f"{len(vocab_data)} terms")
            if semantic_results:
                parts.append(f"{len(semantic_results)} searches")
            if summary_text:
                parts.append("summary")
            self.set_status(f"Exported {' + '.join(parts)} to {filename}", duration_ms=5000)
            logger.debug("Export All %s: %s", fmt_label, filepath)
        else:
            detail = f"\n\n{error_detail}" if error_detail else ""
            messagebox.showerror("Export Failed", f"Failed to create {fmt_label} report.{detail}")

    # =========================================================================
    # Timer
    # =========================================================================

    def _start_timer(self):
        """Start the processing timer and activity indicator."""
        # Cancel any existing timer loop to prevent orphaned parallel loops
        if self._timer_after_id:
            self.after_cancel(self._timer_after_id)
            self._timer_after_id = None
        self._processing_start_time = time.time()
        self._update_timer()
        self._start_activity_indicator()

    def _stop_timer(self):
        """Stop the processing timer and activity indicator."""
        if self._timer_after_id:
            self.after_cancel(self._timer_after_id)
            self._timer_after_id = None

        # Keep final time displayed
        if self._processing_start_time:
            elapsed = time.time() - self._processing_start_time
            self._format_timer(elapsed)

        # Clear start time so any orphaned _update_timer callback won't reschedule
        self._processing_start_time = None

        self._stop_activity_indicator()

    def _start_activity_indicator(self):
        """Show and start the animated activity indicator."""
        if hasattr(self, "activity_indicator"):
            if not self._activity_indicator_visible:
                self.activity_indicator.pack(side="right", padx=(0, 5), pady=5)
                self._activity_indicator_visible = True
            self.activity_indicator.start()
            # Tk redraws on next idle pass -- no update_idletasks() needed

    def _stop_activity_indicator(self):
        """Stop and hide the activity indicator."""
        if hasattr(self, "activity_indicator"):
            self.activity_indicator.stop()
            if self._activity_indicator_visible:
                self.activity_indicator.pack_forget()
                self._activity_indicator_visible = False

    def _update_timer(self):
        """Update the timer display."""
        if self._processing_start_time:
            elapsed = time.time() - self._processing_start_time
            self._format_timer(elapsed)
            self._timer_after_id = self.after(1000, self._update_timer)

    def _format_timer(self, seconds: float):
        """Format and display the timer."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        self.timer_label.configure(text=f"⏱ {minutes}:{secs:02d}")

    def _calculate_aggregate_confidence(self, documents: list[dict]) -> float:
        """
        Calculate aggregate confidence from processed documents.

        Uses minimum confidence across all documents because terms extracted
        from any document could be affected by that document's OCR quality.
        The ML model will learn to weight this signal appropriately.

        Args:
            documents: List of document dicts with 'confidence' field (0-100)

        Returns:
            Minimum confidence value, or 100.0 if no documents have confidence
        """
        confidences = []
        for doc in documents:
            conf = doc.get("confidence")
            if conf is not None:
                confidences.append(float(conf))

        if not confidences:
            return 100.0  # Default to 100% if no confidence data

        min_conf = min(confidences)
        logger.debug("Document confidences: %s -> min=%.1f%%", confidences, min_conf)
        return min_conf

    # =========================================================================
    # Status Bar
    # =========================================================================

    def set_status(self, message: str, duration_ms: int | None = None):
        """
        Update the status bar message with optional auto-clear.

        Args:
            message: Status message to display
            duration_ms: If set, clear to default status after this many milliseconds.
                         Use for temporary confirmations (e.g., "Exported 10 terms").
        """
        # If an error is being displayed with a hold timer, defer this message
        if hasattr(self, "_status_error_hold_until") and self._status_error_hold_until:
            import time

            if time.time() < self._status_error_hold_until:
                # Cancel any previous deferred status (only the latest matters)
                if hasattr(self, "_deferred_status_id") and self._deferred_status_id:
                    self.after_cancel(self._deferred_status_id)
                remaining_ms = int((self._status_error_hold_until - time.time()) * 1000)
                self._deferred_status_id = self.after(
                    remaining_ms, lambda: self.set_status(message, duration_ms)
                )
                return

        # Cancel any pending status clear
        if hasattr(self, "_status_clear_id") and self._status_clear_id:
            self.after_cancel(self._status_clear_id)
            self._status_clear_id = None

        # Reset to default text color (in case previous was an error)
        from src.ui.theme import COLORS

        self.status_label.configure(text=message, text_color=COLORS["text_primary"])

        self._status_error_hold_until = None
        self._deferred_status_id = None

        logger.debug("Status: %s", message)

        # Schedule auto-clear if duration specified
        if duration_ms:
            self._status_clear_id = self.after(duration_ms, lambda: self._clear_status_to_default())

    def set_status_error(self, message: str, hold_seconds: float = 5.0):
        """
        Display an error/warning message in orange text on the status bar.

        Convention: all pipeline failures should use this method so users can
        see that something went wrong. The message is held for hold_seconds
        before normal status messages can overwrite it.

        Args:
            message: Error message to display.
            hold_seconds: Minimum seconds to display before other messages
                          can overwrite (default 5s).
        """
        import time

        from src.ui.theme import COLORS

        # Cancel any pending status clear
        if hasattr(self, "_status_clear_id") and self._status_clear_id:
            self.after_cancel(self._status_clear_id)
            self._status_clear_id = None

        self.status_label.configure(text=message, text_color=COLORS["status_error"])
        self._status_error_hold_until = time.time() + hold_seconds

        logger.warning("Status (error): %s", message)

    def _clear_status_to_default(self):
        """Clear status bar to default 'Ready' message."""
        from src.ui.theme import COLORS

        self._status_clear_id = None
        self._status_error_hold_until = None
        self.status_label.configure(text="Ready", text_color=COLORS["text_secondary"])

    # =========================================================================
    # Startup Checks
    # =========================================================================

    def _check_corpus_limit(self):
        """Check if corpus exceeds the maximum document limit on startup."""
        try:
            from src.services import VocabularyService

            vocab_service = VocabularyService()
            max_docs = vocab_service.get_max_corpus_docs()

            if vocab_service.is_corpus_disabled():
                reason = vocab_service.get_corpus_disabled_reason()
                logger.warning("Corpus disabled on startup: %s", reason)

                messagebox.showwarning(
                    "Corpus Disabled",
                    f"Corpus disabled: document count exceeds {max_docs}.\n\n"
                    "The corpus folder has too many documents. Corpus features "
                    "will be disabled for this session.\n\n"
                    "To re-enable:\n"
                    "1. Open the corpus folder via the Manage button (or Settings > Corpus)\n"
                    "2. Remove files until you have 25 or fewer documents\n"
                    "3. Restart the application",
                )
        except Exception as e:
            logger.debug("Error checking corpus limit: %s", e)

    # =========================================================================
    # Document Count Badge
    # =========================================================================

    def _update_doc_count_badge(self):
        """Update the document count badge in the header."""
        count = len(self.processing_results)
        if count == 0:
            self.doc_count_label.configure(text="")
        else:
            label = "document" if count == 1 else "documents"
            self.doc_count_label.configure(text=f"\U0001f4c4 {count} {label} loaded")

    # =========================================================================
    # Cleanup
    # =========================================================================

    def _cancel_and_reschedule(self, attr_name: str, delay_ms: int, callback):
        """
        Cancel any previously-scheduled after() for `attr_name` and schedule a new one.

        Stores the new after-id back on `self` under `attr_name`. Safe to call
        repeatedly — prevents orphaned retry callbacks from stacking.

        Args:
            attr_name: Instance attribute that holds the tracked after-id.
            delay_ms: Delay in milliseconds before invoking `callback`.
            callback: Zero-arg callable to invoke after the delay.
        """
        existing = getattr(self, attr_name, None)
        if existing:
            try:
                self.after_cancel(existing)
            except Exception:
                pass
        new_id = self.after(delay_ms, callback)
        setattr(self, attr_name, new_id)

    def destroy(self):
        """Clean up resources before destroying window."""
        self._destroying = True

        # Cancel all tracked after() callbacks
        if self._queue_poll_id:
            self.after_cancel(self._queue_poll_id)
            self._queue_poll_id = None
        if hasattr(self, "_status_clear_id") and self._status_clear_id:
            self.after_cancel(self._status_clear_id)
            self._status_clear_id = None
        if hasattr(self, "_timer_after_id") and self._timer_after_id:
            self.after_cancel(self._timer_after_id)
            self._timer_after_id = None
        if self._resize_debounce_id is not None:
            self.after_cancel(self._resize_debounce_id)
            self._resize_debounce_id = None
        if hasattr(self, "_deferred_status_id") and self._deferred_status_id:
            self.after_cancel(self._deferred_status_id)
            self._deferred_status_id = None
        if getattr(self, "_preprocessing_retry_id", None):
            try:
                self.after_cancel(self._preprocessing_retry_id)
            except Exception:
                pass
            self._preprocessing_retry_id = None
        if getattr(self, "_extraction_retry_id", None):
            try:
                self.after_cancel(self._extraction_retry_id)
            except Exception:
                pass
            self._extraction_retry_id = None

        # Shut down the worker subprocess (non-blocking to avoid GUI hang)
        if self._worker_manager:
            logger.debug("Shutting down worker subprocess...")
            self._worker_manager.shutdown(blocking=False)

        # Stop timer
        self._stop_timer()

        super().destroy()
