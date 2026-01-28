"""
CasePrepd - Main Window (CustomTkinter)
Session 29: Two-Panel Q&A-First Layout
Session 33: Refactored - Layout extracted to WindowLayoutMixin
Session 82: Related mixin modules created for reference (not inherited)

Main application window with:
- Header: Corpus dropdown + Settings button
- No-corpus warning banner
- Two-panel layout: Left (Session Documents + Tasks), Right (Results)
- Status bar with processing timer

Architecture:
    MainWindow inherits from:
    - WindowLayoutMixin: UI creation methods (_create_header, _create_main_panels, etc.)
    - ctk.CTk: CustomTkinter main window base class

    Layout code is in: src/ui/window_layout.py
    Business logic is in: This file (main_window.py)

    Related helper modules (for reference/future refactoring):
    - src/ui/main_window_helpers/ollama_mixin.py - Ollama status helpers
    - src/ui/main_window_helpers/file_mixin.py - File management helpers
    - src/ui/main_window_helpers/task_mixin.py - Task execution helpers
    - src/ui/main_window_helpers/export_mixin.py - Export helpers
    - src/ui/main_window_helpers/timer_mixin.py - Timer helpers
"""

import logging
import re
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from tkinter import filedialog, messagebox

import customtkinter as ctk

from src.services.workers import (
    ProcessingWorker,
    ProgressiveExtractionWorker,
    QAWorker,
    VocabularyWorker,
)
from src.ui.styles import initialize_all_styles
from src.ui.tooltip_manager import tooltip_manager
from src.ui.window_layout import WindowLayoutMixin

logger = logging.getLogger(__name__)

# Try to import tkinterdnd2 for drag-and-drop support (Session 73)
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    HAS_DND = True
except ImportError:
    HAS_DND = False

# PERF-001: Pre-compile regex at module level (after all imports)
_MODEL_PARAM_PATTERN = re.compile(r":(\d+\.?\d*)b")


class MainWindow(WindowLayoutMixin, ctk.CTk):
    """
    Main application window for CasePrepd.

    Session 29: Q&A-first two-panel layout with corpus management.
    Session 33: Layout methods moved to WindowLayoutMixin for better code organization.

    Layout:
    - Header row: App title, corpus dropdown, settings button
    - Warning banner: Shown when no corpus configured
    - Left panel: Session documents + task checkboxes + "Perform N Tasks" button
    - Right panel: Results display with output type selector
    - Status bar: Status text + corpus info + processing timer

    Layout methods (from WindowLayoutMixin):
    - _create_header, _create_warning_banner, _create_main_panels
    - _create_left_panel, _create_right_panel, _create_status_bar
    """

    def __init__(self):
        super().__init__()

        from src.config import APP_NAME

        self.title(APP_NAME)
        self.geometry("1200x750")
        icon_path = Path(__file__).parent.parent.parent / "assets" / "icon.ico"
        if icon_path.exists():
            self.iconbitmap(str(icon_path))
        self.minsize(900, 600)

        # State
        self.selected_files: list[str] = []
        self.processing_results: list[dict] = []
        self._processing_start_time: float | None = None
        self._timer_after_id: str | None = None

        # Managers (via service layer for pipeline architecture)
        from src.services import AIService, VocabularyService

        ai_service = AIService()
        self.model_manager = ai_service.get_ollama_manager()
        self.prompt_template_manager = ai_service.get_prompt_template_manager()
        self.corpus_registry = VocabularyService().get_corpus_registry()

        # Workers and queue
        self._processing_worker: ProcessingWorker | None = None
        self._vocabulary_worker: VocabularyWorker | None = None
        self._qa_worker: QAWorker | None = None
        self._progressive_worker: ProgressiveExtractionWorker | None = None  # Session 45
        self._ui_queue: Queue | None = None
        self._queue_poll_id: str | None = None

        # Q&A infrastructure
        self._embeddings = None  # Lazy-loaded HuggingFaceEmbeddings
        self._vector_store_path = None  # Path to current session's vector store
        self._qa_results: list = []  # Store QAResult objects
        self._qa_results_lock = threading.Lock()  # LOG-007: Thread-safe access
        self._qa_ready = False  # Session 45: Q&A becomes available after indexing

        # Initialize all ttk styles once at startup (prevents freeze on first view switch)
        initialize_all_styles()

        # Build UI
        self._create_header()
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
        self._update_default_questions_label()  # Set initial question count
        self._update_vocab_llm_checkbox_state()  # Set LLM checkbox based on settings/GPU
        self._update_qa_checkbox_state()  # Set Q&A checkbox based on model size
        self._update_summary_checkbox_state()  # Set Summary checkbox based on GPU/settings

        # Initialize drag-and-drop support (Session 73)
        self._setup_drag_drop()

        # Startup checks and status updates
        self._check_ollama_service()
        self._update_ollama_status()
        self._update_model_display()

        dnd_status = "enabled" if HAS_DND else "disabled (tkinterdnd2 not installed)"
        logger.debug("Initialized with two-panel layout, drag-drop %s", dnd_status)

    # =========================================================================
    # Ollama Status & Model Display
    # =========================================================================

    def _update_ollama_status(self):
        """
        Update the Ollama status indicator in the status bar.

        Shows green dot if connected, red dot if disconnected.
        Adds tooltip with troubleshooting info when disconnected.
        """
        from src.ui.theme import COLORS

        if self.model_manager.is_connected:
            # Connected - green dot
            self.ollama_status_dot.configure(text_color=COLORS["success"])
            self.ollama_status_label.configure(text="Ollama")
            # Remove tooltip bindings
            self._clear_ollama_tooltip()
        else:
            # Disconnected - red dot
            self.ollama_status_dot.configure(text_color=COLORS["danger"])
            self.ollama_status_label.configure(text="Ollama (disconnected)")
            # Add tooltip with troubleshooting info
            self._setup_ollama_tooltip()

        status = "connected" if self.model_manager.is_connected else "disconnected"
        logger.debug("Ollama status: %s", status)

    def _setup_ollama_tooltip(self):
        """Set up hover tooltip for disconnected Ollama status."""
        tooltip_text = (
            "Ollama is not running.\n\n"
            "To fix:\n"
            "• Ensure Ollama is installed (ollama.ai)\n"
            "• Run 'ollama serve' in a terminal\n"
            "• Check if port 11434 is available"
        )

        def show_tooltip(event):
            # Session 62b: Close any existing tooltip first via global manager
            tooltip_manager.close_active()

            # Create tooltip window
            self._ollama_tooltip = tooltip = ctk.CTkToplevel(self)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")

            label = ctk.CTkLabel(
                tooltip,
                text=tooltip_text,
                font=("Segoe UI", 10),
                fg_color=("#2b2b2b", "#2b2b2b"),
                corner_radius=6,
                padx=10,
                pady=8,
                justify="left",
            )
            label.pack()

            # Session 62b: Register with global manager
            tooltip_manager.register(tooltip, owner=self.ollama_status_dot)

        def hide_tooltip(event):
            if hasattr(self, "_ollama_tooltip") and self._ollama_tooltip:
                # Session 62b: Unregister from global manager
                tooltip_manager.unregister(self._ollama_tooltip)
                self._ollama_tooltip.destroy()
                self._ollama_tooltip = None

        # Bind to both dot and label
        self.ollama_status_dot.bind("<Enter>", show_tooltip)
        self.ollama_status_dot.bind("<Leave>", hide_tooltip)
        self.ollama_status_label.bind("<Enter>", show_tooltip)
        self.ollama_status_label.bind("<Leave>", hide_tooltip)

    def _clear_ollama_tooltip(self):
        """Remove tooltip bindings when Ollama is connected."""
        # Unbind events
        self.ollama_status_dot.unbind("<Enter>")
        self.ollama_status_dot.unbind("<Leave>")
        self.ollama_status_label.unbind("<Enter>")
        self.ollama_status_label.unbind("<Leave>")

        # Destroy any existing tooltip
        if hasattr(self, "_ollama_tooltip") and self._ollama_tooltip:
            # Session 62b: Unregister from global manager
            tooltip_manager.unregister(self._ollama_tooltip)
            self._ollama_tooltip.destroy()
            self._ollama_tooltip = None

    def _update_model_display(self):
        """
        Update the model display in the header.

        Shows model name and parameter count (e.g., "gemma3:1b (1B params)").
        """
        model_name = self.model_manager.model_name

        # Format display text with parameter count if available
        display_text = self._format_model_display(model_name)
        self.model_name_label.configure(text=display_text)

        logger.debug("Model display updated: %s", display_text)

    def _format_model_display(self, model_name: str) -> str:
        """
        Format model name with parameter count for display.

        Args:
            model_name: Raw model name (e.g., "gemma3:1b", "llama3.2:3b")

        Returns:
            Formatted string (e.g., "gemma3:1b (1B params)")
        """
        if not model_name:
            return "No model selected"

        # Extract parameter count from model name if present
        # Common patterns: gemma3:1b, llama3.2:3b, mistral:7b
        param_info = ""
        name_lower = model_name.lower()

        # PERF-001: Look for parameter patterns using pre-compiled regex
        param_match = _MODEL_PARAM_PATTERN.search(name_lower)
        if param_match:
            param_size = param_match.group(1)
            param_info = f" ({param_size}B params)"

        return f"{model_name}{param_info}"

    def _open_model_settings(self):
        """Open the settings dialog directly to the Questions tab (model config)."""
        from src.ui.settings.settings_dialog import SettingsDialog
        from src.user_preferences import get_user_preferences

        # Session 62: Capture current model to detect changes
        prefs = get_user_preferences()
        old_model = prefs.get("ollama_model", self.model_manager.model_name)

        try:
            dialog = SettingsDialog(parent=self, initial_tab="Questions")
            dialog.wait_window()
        except Exception as e:
            logger.debug("Failed to open settings dialog: %s", e)

        # Session 62: Check if model changed and reload if needed
        new_model = prefs.get("ollama_model", self.model_manager.model_name)
        if new_model and new_model != old_model:
            try:
                self.model_manager.load_model(new_model)
                logger.info("Model changed: %s -> %s", old_model, new_model)
            except Exception as e:
                logger.warning("Failed to load model %s: %s", new_model, e)

        # Refresh UI after settings change
        self._refresh_corpus_dropdown()
        self._update_model_display()
        self._update_ollama_status()
        self._update_vocab_llm_checkbox_state()  # Session 63b: Refresh LLM checkbox
        self._update_qa_checkbox_state()  # Session 90: Refresh Q&A checkbox for model size
        self._update_summary_checkbox_state()  # Session 93: Refresh Summary checkbox

    # =========================================================================
    # Corpus Management
    # =========================================================================

    def _refresh_corpus_dropdown(self):
        """Refresh the corpus dropdown with available corpora."""
        try:
            corpora = self.corpus_registry.list_corpora()
            names = [c.name for c in corpora]

            if names:
                self.corpus_dropdown.configure(values=names)
                active = self.corpus_registry.get_active_corpus()
                self.corpus_dropdown.set(active)

                # Update corpus document count badge (Session 67)
                active_info = next((c for c in corpora if c.name == active), None)
                if active_info and active_info.doc_count > 0:
                    doc_text = "doc" if active_info.doc_count == 1 else "docs"
                    self.corpus_doc_count_label.configure(
                        text=f"({active_info.doc_count} {doc_text})"
                    )
                    # Update status bar too
                    self.corpus_info_label.configure(
                        text=f"BM25 active: {active_info.doc_count}-document corpus"
                    )
                else:
                    self.corpus_doc_count_label.configure(text="(empty)")
                    self.corpus_info_label.configure(text="")
            else:
                self.corpus_dropdown.configure(values=["No corpora"])
                self.corpus_dropdown.set("No corpora")
                self.corpus_doc_count_label.configure(text="")
                self.corpus_info_label.configure(text="")

        except Exception as e:
            logger.debug("Error refreshing corpus dropdown: %s", e)
            self.corpus_dropdown.configure(values=["Error"])
            self.corpus_dropdown.set("Error")
            self.corpus_doc_count_label.configure(text="")

    def _on_corpus_changed(self, corpus_name: str):
        """Handle corpus selection change."""
        try:
            self.corpus_registry.set_active_corpus(corpus_name)
            self._refresh_corpus_dropdown()
            self.set_status(f"Active corpus: {corpus_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to switch corpus: {e}")

    def _open_corpus_dialog(self):
        """Open Settings dialog to the Corpus tab (Session 64)."""
        from src.ui.settings import SettingsDialog

        try:
            dialog = SettingsDialog(parent=self, initial_tab="Corpus")
            dialog.wait_window()
        except Exception as e:
            logger.debug("Failed to open settings dialog: %s", e)

        # Refresh after dialog closes
        self._refresh_corpus_dropdown()

    # =========================================================================
    # File Management
    # =========================================================================

    def _setup_drag_drop(self):
        """
        Initialize drag-and-drop file support (Session 73).

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

            logger.debug("Drag-drop enabled on file table area")

        except Exception as e:
            logger.debug("Failed to initialize drag-drop: %s", e)

    def _on_file_drop(self, event):
        """
        Handle files dropped onto the file table area (Session 73).

        Args:
            event: Drop event containing file paths
        """
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

        # Process the dropped files
        self.selected_files = valid_files
        self.set_status(f"Processing {len(valid_files)} dropped file(s)...")
        self._start_preprocessing()

    def _select_files(self):
        """Open file dialog to select documents for this session."""
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

        self.selected_files = list(files)
        self.set_status(f"Processing {len(files)} file(s)...")
        self._start_preprocessing()

    def _clear_files(self):
        """Clear all files from the session."""
        self.selected_files.clear()
        self.processing_results.clear()
        self.file_table.clear()
        # Reset Q&A state so old answers don't persist
        self._qa_ready = False
        self._qa_results.clear()
        self._vector_store_path = None
        if hasattr(self, "followup_btn"):
            self.followup_btn.configure(state="disabled")
        self._update_generate_button_state()
        self._update_session_stats()  # Clear stats display
        self.set_status("Files cleared")

    def _start_preprocessing(self):
        """Start preprocessing selected files."""
        if not self.selected_files:
            return

        # Disable controls during preprocessing
        self.add_files_btn.configure(state="disabled")
        self.generate_btn.configure(state="disabled")

        # Clear previous results
        self.file_table.clear()
        self.processing_results.clear()

        # Start timer
        self._start_timer()

        # Create queue for worker communication
        self._ui_queue = Queue()

        # Create and start worker
        self._processing_worker = ProcessingWorker(
            file_paths=self.selected_files, ui_queue=self._ui_queue
        )
        self._processing_worker.start()

        # Start polling the queue
        self._poll_queue()

    def _poll_queue(self):
        """Poll the UI queue for worker messages."""
        # Process up to 10 messages per poll to avoid blocking UI
        messages_processed = 0
        max_messages_per_poll = 10

        try:
            while messages_processed < max_messages_per_poll:
                msg_type, data = self._ui_queue.get_nowait()
                self._handle_queue_message(msg_type, data)
                messages_processed += 1
        except Empty:
            pass

        # Force GUI refresh to keep animations smooth (activity indicator, etc.)
        self.update_idletasks()

        # Continue polling if any worker is running OR if we hit the message limit (more messages may be waiting)
        processing_alive = self._processing_worker and self._processing_worker.is_alive()
        progressive_alive = self._progressive_worker and self._progressive_worker.is_alive()
        # Session 86: Also check default questions worker so Q&A results are received
        default_qa_alive = (
            hasattr(self, "_default_qa_worker")
            and self._default_qa_worker
            and self._default_qa_worker.is_alive()
        )
        more_messages_likely = messages_processed >= max_messages_per_poll

        if processing_alive or progressive_alive or default_qa_alive or more_messages_likely:
            self._queue_poll_id = self.after(33, self._poll_queue)  # ~30fps for smooth animation
        else:
            # Final poll to catch any remaining messages
            try:
                while True:
                    msg_type, data = self._ui_queue.get_nowait()
                    self._handle_queue_message(msg_type, data)
            except Empty:
                pass

    def _handle_queue_message(self, msg_type: str, data):
        """Handle a message from the worker queue."""
        if msg_type == "progress":
            _percentage, message = data
            # Append Q&A status if ready (prevents status from hiding Q&A readiness)
            if self._qa_ready and "Q&A ready" not in message and "Questions" not in message:
                message = f"{message} (Q&A ready)"
            self.set_status(message)

        elif msg_type == "file_processed":
            self.processing_results.append(data)
            self.file_table.add_result(data)

        elif msg_type == "processing_finished":
            self._on_preprocessing_complete(data)

        elif msg_type == "error":
            self.set_status(f"Error: {data}")
            messagebox.showerror("Processing Error", str(data))
            self._on_preprocessing_complete([])

        # Progressive Extraction handlers (Session 48, Session 85)
        elif msg_type == "extraction_started":
            # Session 85: Dim feedback buttons while extraction is in progress
            logger.debug("Extraction started - dimming feedback buttons")
            self.output_display.set_extraction_in_progress(True)

        elif msg_type == "extraction_complete":
            # Session 85: Re-enable feedback buttons after extraction completes
            logger.debug("Extraction complete - enabling feedback buttons")
            self.output_display.set_extraction_in_progress(False)

        elif msg_type == "partial_vocab_complete":
            # Session 85: Show BM25 + RAKE results before NER completes
            term_count = len(data) if data else 0
            logger.debug("Partial results: %s terms from BM25+RAKE", term_count)
            self.output_display.update_outputs(vocab_csv_data=data)
            self.output_display.set_extraction_source("partial")
            self.set_status(f"Found {term_count} terms (BM25+RAKE). Running NER...")

        elif msg_type == "ner_progress":
            # Session 85: Update status bar with NER chunk progress
            chunk_num = data.get("chunk_num", 0)
            total_chunks = data.get("total_chunks", 1)
            pct = int((chunk_num / total_chunks) * 100)
            self.set_status(f"NER: {pct}% complete (chunk {chunk_num}/{total_chunks})...")
            # Note: We don't update the vocab table with each chunk because raw NER
            # candidates need post-processing. The final merged results come with ner_complete.

        elif msg_type == "ner_complete":
            term_count = len(data) if data else 0
            logger.debug("NER complete: %s terms - displaying immediately", term_count)
            self.output_display.update_outputs(vocab_csv_data=data)
            self.output_display.set_extraction_source("ner")
            self.set_status(f"NER complete: {term_count} terms found. LLM enhancement starting...")

        elif msg_type == "qa_ready":
            chunk_count = data.get("chunk_count", 0)
            logger.debug("Q&A ready: %s chunks indexed", chunk_count)
            self._vector_store_path = data.get("vector_store_path")
            self._embeddings = data.get("embeddings")
            self._qa_ready = True
            if self._pending_tasks.get("qa"):
                self._completed_tasks.add("qa")
                self.followup_btn.configure(state="normal")
            self.set_status(
                f"Questions and answers ready ({chunk_count} chunks). LLM enhancement in progress..."
            )

        elif msg_type == "qa_error":
            error_msg = (
                data.get("error", "Unknown Q&A error") if isinstance(data, dict) else str(data)
            )
            logger.debug("Q&A indexing error: %s", error_msg)
            self.set_status(f"Questions and answers unavailable: {error_msg[:50]}...")
            # Q&A won't be available but vocab extraction can continue

        elif msg_type == "trigger_default_qa":
            # Check if default questions checkbox is enabled
            if not self.ask_default_questions_check.get():
                logger.debug("Default questions disabled, skipping")
            else:
                # Spawn QAWorker with default questions
                from src.services.workers import QAWorker
                from src.user_preferences import get_user_preferences

                logger.debug("Spawning QAWorker for default questions")
                prefs = get_user_preferences()

                # Session 86: Store as instance variable so _poll_queue() keeps polling
                self._default_qa_worker = QAWorker(
                    vector_store_path=data["vector_store_path"],
                    embeddings=data["embeddings"],
                    ui_queue=self._ui_queue,
                    answer_mode=prefs.get("qa_answer_mode", "ollama"),
                    questions=None,
                    use_default_questions=True,
                )
                self._default_qa_worker.start()
                logger.debug("Default questions worker started")

        # Q&A result handlers (Session 63c: handle messages from default questions worker)
        elif msg_type == "qa_progress":
            current, total, _question = data
            logger.debug("Q&A progress: %s/%s", current + 1, total)
            self.set_status(f"Answering default questions: {current + 1}/{total}...")

        elif msg_type == "qa_result":
            # Individual Q&A result - add to results and update display
            logger.debug("Q&A result received")
            with self._qa_results_lock:  # LOG-007: Thread-safe access
                self._qa_results.append(data)
                self.output_display.update_outputs(qa_results=self._qa_results)

        elif msg_type == "qa_complete":
            # All Q&A questions answered
            qa_results = data if data else []
            logger.debug("Q&A complete: %s answers", len(qa_results))
            with self._qa_results_lock:  # LOG-007: Thread-safe access
                self._qa_results = qa_results
            if qa_results:
                self.output_display.update_outputs(qa_results=qa_results)
                self.set_status(f"Default questions answered: {len(qa_results)} responses")
            self.followup_btn.configure(state="normal")

        elif msg_type == "llm_progress":
            current, total = data
            logger.debug("LLM progress: %s/%s", current, total)

        elif msg_type == "llm_complete":
            term_count = len(data) if data else 0
            logger.debug("LLM complete: %s reconciled terms", term_count)

            # Session 78: Only update vocab and show "Enhanced" if LLM actually returned results
            # When LLM is skipped/disabled, data is empty [] - keep NER-only results
            if data:
                self.output_display.update_outputs(vocab_csv_data=data)
                self.output_display.set_extraction_source("both")
                self.set_status(f"Complete: {term_count} names & vocabulary extracted")
            else:
                # LLM was skipped - keep extraction source as "ner" (already set)
                self.set_status("Complete: NER extraction only (LLM disabled)")

            self._completed_tasks.add("vocab")
            if self._pending_tasks.get("summary"):
                self._start_summary_task()
            else:
                self._finalize_tasks()

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

        # Update processing_results with final preprocessed data
        # The results contain 'preprocessed_text' added by ProcessingWorker
        # which wasn't present in the individual 'file_processed' messages
        if results:
            self.processing_results = results
            logger.debug("Updated processing_results with %s preprocessed documents", len(results))

        # Re-enable controls
        self.add_files_btn.configure(state="normal")
        self._update_generate_button_state()

        # Count results
        success_count = sum(1 for r in results if r.get("status") == "success")
        failed_count = len(results) - success_count

        status = f"Processed {len(results)} file(s): {success_count} ready"
        if failed_count > 0:
            status += f", {failed_count} failed"

        self.set_status(status)
        self._update_session_stats()  # Show document stats

    # =========================================================================
    # Task Execution
    # =========================================================================

    def _get_task_count(self) -> int:
        """Get the number of selected tasks."""
        count = 0
        if self.qa_check.get():
            count += 1
        if self.vocab_check.get():
            count += 1
        if self.summary_check.get():
            count += 1
        return count

    def _update_generate_button_state(self):
        """Update the generate button text and state."""
        task_count = self._get_task_count()
        has_files = len(self.processing_results) > 0

        if task_count == 0:
            self.generate_btn.configure(text="Select Tasks", state="disabled")
        elif not has_files:
            self.generate_btn.configure(text=f"Add Files ({task_count} tasks)", state="disabled")
        elif task_count == 1:
            self.generate_btn.configure(text="Perform 1 Task", state="normal")
        else:
            self.generate_btn.configure(text=f"Perform {task_count} Tasks", state="normal")

        # Update task preview label (Session 69)
        self._update_task_preview()

    def _update_task_preview(self):
        """
        Update the task preview label to show what will run (Session 69).

        Shows a concise preview like:
        "Will run: Vocabulary (NER+LLM), Q&A (6 questions)"
        """
        parts = []

        # Vocabulary task
        if self.vocab_check.get():
            if self.vocab_llm_check.get() and self.vocab_llm_check.cget("state") == "normal":
                parts.append("Vocabulary (NER+LLM)")
            else:
                parts.append("Vocabulary (NER)")

        # Q&A task
        if self.qa_check.get():
            if self.ask_default_questions_check.get():
                enabled, _total = self._load_default_question_count()
                if enabled > 0:
                    q_word = "question" if enabled == 1 else "questions"
                    parts.append(f"Q&A ({enabled} {q_word})")
                else:
                    parts.append("Q&A")
            else:
                parts.append("Q&A")

        # Summary task
        if self.summary_check.get():
            parts.append("Summary (slow)")

        # Build preview text
        preview = "Will run: " + ", ".join(parts) if parts else "Select tasks above"

        self.task_preview_label.configure(text=preview)

    def _on_summary_checked(self):
        """Handle summary checkbox toggle - show warning if enabling."""
        if self.summary_check.get():
            # Show warning dialog
            result = messagebox.askyesno(
                "Summary Warning",
                "Summary generation can take several hours without a dedicated GPU.\n\n"
                "For quick case familiarization, Q&A is recommended instead.\n\n"
                "Continue with summary?",
                icon="warning",
            )
            if not result:
                self.summary_check.deselect()

        self._update_generate_button_state()

    def _load_default_question_count(self) -> tuple[int, int]:
        """
        Get count of enabled and total default questions.

        Session 63c: Now uses DefaultQuestionsManager for enable/disable support.

        Returns:
            Tuple of (enabled_count, total_count)
        """
        try:
            from src.services import QAService

            manager = QAService().get_default_questions_manager()
            return (manager.get_enabled_count(), manager.get_total_count())

        except Exception as e:
            logger.debug("Error loading default question count: %s", e)
            return (0, 0)

    def _update_default_questions_label(self):
        """Update checkbox text with current enabled question count."""
        enabled, total = self._load_default_question_count()
        question_word = "question" if enabled == 1 else "questions"

        if enabled == total:
            # All questions enabled - simple display
            self.ask_default_questions_check.configure(
                text=f"Ask {enabled:,} default {question_word}"
            )
        else:
            # Some questions disabled - show enabled/total
            self.ask_default_questions_check.configure(
                text=f"Ask {enabled:,}/{total:,} default {question_word}"
            )

    def _on_default_questions_toggled(self):
        """Handle default questions checkbox state change."""
        # Just update button state - no other action needed here
        self._update_generate_button_state()

        state = "enabled" if self.ask_default_questions_check.get() else "disabled"
        logger.debug("Default questions %s", state)

    def _on_qa_check_changed(self):
        """Handle Q&A checkbox state change."""
        self._update_generate_button_state()
        self._update_default_questions_checkbox_state()

        state = "enabled" if self.qa_check.get() else "disabled"
        logger.debug("Q&A %s", state)

    def _update_default_questions_checkbox_state(self):
        """
        Update default questions sub-checkbox based on Q&A checkbox state.

        When Q&A is unchecked:
        - Disable the default questions checkbox
        - Auto-uncheck it (since we won't be asking questions)

        When Q&A is checked:
        - Enable the default questions checkbox
        """
        if self.qa_check.get():
            # Q&A is enabled - allow user to toggle default questions
            self.ask_default_questions_check.configure(state="normal")
        else:
            # Q&A is disabled - disable and uncheck default questions
            self.ask_default_questions_check.deselect()
            self.ask_default_questions_check.configure(state="disabled")

    # =========================================================================
    # LLM Enhancement Checkbox State Management (Session 63b)
    # =========================================================================

    def _update_vocab_llm_checkbox_state(self):
        """
        Update LLM Enhancement checkbox state based on GPU availability and settings.

        Called at startup, when settings change, and when vocab checkbox is toggled.

        Greying logic (in order of precedence):
        - If vocab checkbox is unchecked: disable LLM checkbox
        - If no dedicated GPU detected: disable (requires GPU)
        - If vocab_use_llm setting is "no": disable and uncheck
        - Otherwise: enable and set based on is_vocab_llm_enabled()
        """
        from src.services import AIService
        from src.user_preferences import get_user_preferences

        ai_svc = AIService()
        prefs = get_user_preferences()
        vocab_mode = prefs.get_vocab_llm_mode()  # "auto", "yes", or "no"
        has_gpu = ai_svc.has_dedicated_gpu()
        llm_enabled = prefs.is_vocab_llm_enabled()

        # Case 1: Vocab is unchecked - disable LLM checkbox
        if not self.vocab_check.get():
            self.vocab_llm_check.deselect()
            self.vocab_llm_check.configure(state="disabled")
            self._set_vocab_llm_tooltip("Enable 'Extract Vocabulary' first")
            return

        # Case 2: No GPU detected - always disable regardless of settings
        if not has_gpu:
            self.vocab_llm_check.deselect()
            self.vocab_llm_check.configure(state="disabled")
            gpu_status = ai_svc.get_gpu_status_text()
            self._set_vocab_llm_tooltip(
                f"LLM enhancement requires a dedicated GPU.\n\n"
                f"{gpu_status}\n\n"
                "NER-only extraction will be used."
            )
            return

        # Case 3: User explicitly disabled LLM in settings
        if vocab_mode == "no":
            self.vocab_llm_check.deselect()
            self.vocab_llm_check.configure(state="disabled")
            self._set_vocab_llm_tooltip(
                "LLM extraction disabled in Settings.\n\n"
                "To enable: Settings > Performance > 'LLM vocabulary extraction'"
            )
            return

        # Case 4: LLM is available - enable checkbox
        self.vocab_llm_check.configure(state="normal")

        if llm_enabled:
            self.vocab_llm_check.select()
            self._set_vocab_llm_tooltip(
                "LLM enhancement finds additional terms missed by NER.\n"
                "Slower but more thorough vocabulary extraction."
            )
        else:
            self.vocab_llm_check.deselect()
            self._set_vocab_llm_tooltip(
                "LLM enhancement is available.\n"
                "Check this box for more thorough vocabulary extraction."
            )

        logger.debug(
            "LLM checkbox: mode=%s, has_gpu=%s, enabled=%s", vocab_mode, has_gpu, llm_enabled
        )

    def _set_vocab_llm_tooltip(self, text: str):
        """
        Update the tooltip for the LLM Enhancement checkbox.

        Args:
            text: New tooltip text to display
        """
        from src.ui.tooltip_helper import create_tooltip

        # Remove existing tooltip bindings
        if hasattr(self, "_vocab_llm_tooltip_hide") and self._vocab_llm_tooltip_hide:
            try:
                self.vocab_llm_check.unbind("<Enter>")
                self.vocab_llm_check.unbind("<Leave>")
            except Exception:
                pass

        # Create new tooltip
        self._vocab_llm_tooltip_hide = create_tooltip(self.vocab_llm_check, text)

    # =========================================================================
    # Q&A Checkbox State Management (Session 90 - Model Size Requirements)
    # =========================================================================

    def _update_qa_checkbox_state(self):
        """
        Update Q&A checkbox state based on model size requirements.

        Q&A requires an 8B+ parameter model for quality answers.
        Users can override this in Settings.

        Called at startup and when model changes.
        """
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        model_name = prefs.get("ollama_model", "")

        # Check if Q&A is allowed for current model
        allowed, reason = prefs.is_qa_allowed_for_model(model_name)

        if allowed:
            # Enable Q&A checkbox
            self.qa_check.configure(state="normal")
            self._set_qa_tooltip("")  # Clear any warning tooltip
        else:
            # Disable Q&A checkbox and show reason
            self.qa_check.deselect()
            self.qa_check.configure(state="disabled")
            self._set_qa_tooltip(reason)
            # Also disable the sub-checkbox
            self._update_default_questions_checkbox_state()

        logger.debug("Q&A checkbox: model=%s, allowed=%s", model_name, allowed)

    def _set_qa_tooltip(self, text: str):
        """
        Update the tooltip for the Q&A checkbox.

        Args:
            text: New tooltip text to display (empty to remove)
        """
        if not text:
            # Remove tooltip if no text
            if hasattr(self, "_qa_tooltip_hide") and self._qa_tooltip_hide:
                try:
                    self.qa_check.unbind("<Enter>")
                    self.qa_check.unbind("<Leave>")
                except Exception:
                    pass
                self._qa_tooltip_hide = None
            return

        from src.ui.tooltip_helper import create_tooltip

        # Remove existing tooltip bindings
        if hasattr(self, "_qa_tooltip_hide") and self._qa_tooltip_hide:
            try:
                self.qa_check.unbind("<Enter>")
                self.qa_check.unbind("<Leave>")
            except Exception:
                pass

        # Create new tooltip
        self._qa_tooltip_hide = create_tooltip(self.qa_check, text)

    def _update_summary_checkbox_state(self):
        """
        Update Summary checkbox state based on GPU availability and settings.

        Called at startup, when settings change, and similar to
        _update_vocab_llm_checkbox_state().

        Greying logic:
        - If no dedicated GPU and setting is "auto": disable with tooltip
        - If setting is "yes": enable regardless of GPU
        - If GPU detected: enable
        """
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        allowed, reason = prefs.is_summary_allowed()

        if allowed:
            # GPU detected or user overrode the requirement
            self.summary_check.configure(state="normal")
            self._set_summary_tooltip(
                "Summary generation can take several hours without\n"
                "a dedicated GPU.\n\n"
                "For quick case familiarization, Q&A is recommended."
            )
        else:
            # No GPU and setting is "auto" - disable checkbox
            self.summary_check.deselect()
            self.summary_check.configure(state="disabled")
            self._set_summary_tooltip(reason)

        from src.services import AIService

        ai_svc = AIService()
        logger.debug(
            "Summary checkbox: allowed=%s, has_gpu=%s",
            allowed,
            ai_svc.has_dedicated_gpu(),
        )

    def _set_summary_tooltip(self, text: str):
        """
        Update the tooltip for the Summary checkbox.

        Args:
            text: New tooltip text to display
        """
        from src.ui.tooltip_helper import create_tooltip

        # Remove existing tooltip bindings
        if hasattr(self, "_summary_tooltip_hide") and self._summary_tooltip_hide:
            try:
                self.summary_check.unbind("<Enter>")
                self.summary_check.unbind("<Leave>")
            except Exception:
                pass

        # Create new tooltip
        self._summary_tooltip_hide = create_tooltip(self.summary_check, text)

    def _on_vocab_check_changed(self):
        """Handle Vocabulary checkbox state change."""
        self._update_generate_button_state()
        self._update_vocab_llm_checkbox_state()

        state = "enabled" if self.vocab_check.get() else "disabled"
        logger.debug("Vocabulary extraction %s", state)

    def _on_vocab_llm_check_changed(self):
        """Handle LLM Enhancement checkbox state change (user manually toggles)."""
        # Update task preview to reflect LLM change (Session 69)
        self._update_task_preview()

        state = "enabled" if self.vocab_llm_check.get() else "disabled"
        logger.debug("LLM enhancement %s", state)

    def refresh_default_questions_label(self):
        """Refresh the default questions checkbox label. Called after editing questions."""
        self._update_default_questions_label()

    def _update_session_stats(self, extraction_stats: dict | None = None):
        """
        Update the session stats display (Session 73).

        Shows document stats (file count, pages, size) and extraction stats
        (term count, person count, Q&A count) after processing.

        Args:
            extraction_stats: Optional dict with extraction results:
                - vocab_count: Total vocabulary terms
                - person_count: Terms marked as persons
                - qa_count: Number of Q&A results
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

        # Format file size
        if total_size >= 1024 * 1024:
            size_str = f"{total_size / (1024 * 1024):.1f} MB"
        elif total_size >= 1024:
            size_str = f"{total_size / 1024:.0f} KB"
        else:
            size_str = f"{total_size} B"

        # Build stats text
        parts = [f"{success_count}/{file_count} files"]
        if total_pages > 0:
            parts.append(f"{total_pages} pages")
        parts.append(size_str)

        stats_text = " · ".join(parts)

        # Add extraction stats if available
        if extraction_stats:
            ext_parts = []
            if extraction_stats.get("vocab_count", 0) > 0:
                v = extraction_stats["vocab_count"]
                p = extraction_stats.get("person_count", 0)
                ext_parts.append(f"{v} terms ({p} persons)")
            if extraction_stats.get("qa_count", 0) > 0:
                ext_parts.append(f"{extraction_stats['qa_count']} Q&A")
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
        """Execute the selected tasks using progressive three-phase architecture (Session 45)."""
        if not self.processing_results:
            messagebox.showwarning("No Files", "Please add files first.")
            return

        task_count = self._get_task_count()
        if task_count == 0:
            messagebox.showwarning("No Tasks", "Please select at least one task.")
            return

        # Disable controls during processing
        self.generate_btn.configure(state="disabled", text=f"Processing {task_count} tasks...")
        self.add_files_btn.configure(state="disabled")

        # Hide task preview - status bar now shows progress (Session 77)
        self.task_preview_label.configure(text="")

        # Start timer
        self._start_timer()

        # Get selected options
        do_qa = self.qa_check.get()
        do_vocab = self.vocab_check.get()
        do_summary = self.summary_check.get()

        # Track pending tasks
        self._pending_tasks = {"vocab": do_vocab, "qa": do_qa, "summary": do_summary}
        self._completed_tasks = set()
        self._qa_ready = False

        # Session 45: Use progressive extraction for vocabulary (includes Q&A indexing)
        if do_vocab:
            self._start_progressive_extraction()
        elif do_qa:
            # Q&A only (without vocabulary) - use legacy Q&A task
            self._start_qa_task()
        elif do_summary:
            self._start_summary_task()
        else:
            self._on_tasks_complete(True, "No tasks selected")

    def _start_vocabulary_extraction(self):
        """Start vocabulary extraction task."""
        from src.config import (
            LEGAL_EXCLUDE_LIST_PATH,
            MEDICAL_TERMS_LIST_PATH,
            USER_VOCAB_EXCLUDE_PATH,
        )

        self.set_status("Extracting vocabulary...")

        # Debug: Log what's in processing_results
        logger.debug("Vocabulary: %s documents in processing_results", len(self.processing_results))
        for i, doc in enumerate(self.processing_results):
            text_len = len(doc.get("extracted_text", "") or "")
            logger.debug(
                "Doc %s: %s - %s chars, status=%s",
                i,
                doc.get("filename", "unknown"),
                text_len,
                doc.get("status"),
            )

        # Combine text from all processed documents
        from src.services import DocumentService

        combined_text = DocumentService().combine_document_texts(self.processing_results)

        logger.debug("Combined text length: %s chars", len(combined_text))

        if not combined_text.strip():
            self.set_status("No text to analyze for vocabulary")
            logger.debug("WARNING: No text after combining documents!")
            self._on_vocab_complete([])
            return

        # Create queue for vocab worker
        self._vocab_queue = Queue()

        # Check if LLM extraction is enabled (Session 43)
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        use_llm = prefs.is_vocab_llm_enabled()
        logger.debug("Vocabulary extraction with LLM: %s", use_llm)

        # Calculate aggregate document confidence (Session 54)
        # Use minimum confidence - terms from the worst document are most suspect
        doc_confidence = self._calculate_aggregate_confidence(self.processing_results)
        logger.debug("Aggregate document confidence: %.1f%%", doc_confidence)

        # Start vocabulary worker
        self._vocabulary_worker = VocabularyWorker(
            combined_text=combined_text,
            ui_queue=self._vocab_queue,
            exclude_list_path=str(LEGAL_EXCLUDE_LIST_PATH),
            medical_terms_path=str(MEDICAL_TERMS_LIST_PATH),
            user_exclude_path=str(USER_VOCAB_EXCLUDE_PATH),
            doc_count=len(self.processing_results),
            use_llm=use_llm,  # Session 43: Enable LLM-based extraction
            doc_confidence=doc_confidence,  # Session 54: OCR quality for ML
        )
        self._vocabulary_worker.start()

        # Start polling vocab queue
        self._poll_vocab_queue()

    def _poll_vocab_queue(self):
        """Poll the vocabulary worker queue."""
        try:
            while True:
                msg_type, data = self._vocab_queue.get_nowait()
                if msg_type == "progress":
                    self.set_status(data[1] if isinstance(data, tuple) else str(data))
                elif msg_type == "vocab_csv_generated":
                    self._on_vocab_complete(data)
                    return
                elif msg_type == "error":
                    self.set_status(f"Vocabulary error: {data}")
                    self._on_vocab_complete([])
                    return
        except Empty:
            pass

        # Continue polling if worker is alive
        if self._vocabulary_worker and self._vocabulary_worker.is_alive():
            self.after(50, self._poll_vocab_queue)
        else:
            # Worker finished - do final poll
            try:
                while True:
                    msg_type, data = self._vocab_queue.get_nowait()
                    if msg_type == "vocab_csv_generated":
                        self._on_vocab_complete(data)
                        return
            except Empty:
                pass
            # If we get here, worker finished without sending results
            self._on_vocab_complete([])

    def _on_vocab_complete(self, vocab_data: list):
        """Handle vocabulary extraction completion (legacy - for non-progressive flow)."""
        self._completed_tasks.add("vocab")

        # Display results using update_outputs
        if vocab_data:
            self.output_display.update_outputs(vocab_csv_data=vocab_data)
            self.set_status(f"Vocabulary: {len(vocab_data)} terms found")
        else:
            self.set_status("Vocabulary extraction complete (no terms)")

        # Continue to next task
        if self._pending_tasks.get("summary"):
            self._start_summary_task()
        else:
            self._finalize_tasks()

    # =========================================================================
    # Progressive Extraction (Session 45)
    # =========================================================================

    def _start_progressive_extraction(self):
        """
        Start progressive three-phase extraction (Session 45).

        Phase 1 (NER): Fast, displays results in ~5 seconds
        Phase 2 (Q&A): Builds vector store, enables Q&A panel
        Phase 3 (LLM): Slow enhancement, updates table progressively
        """
        from src.config import (
            LEGAL_EXCLUDE_LIST_PATH,
            MEDICAL_TERMS_LIST_PATH,
            USER_VOCAB_EXCLUDE_PATH,
        )

        self.set_status("Starting extraction (NER first, then LLM enhancement)...")

        # Combine text from all processed documents
        from src.services import DocumentService

        combined_text = DocumentService().combine_document_texts(self.processing_results)

        logger.debug(
            "Progressive extraction: %s chars from %s docs",
            len(combined_text),
            len(self.processing_results),
        )

        if not combined_text.strip():
            self.set_status("No text to analyze")
            logger.debug("WARNING: No text after combining documents!")
            self._on_tasks_complete(False, "No text to analyze")
            return

        # Calculate aggregate document confidence (Session 54)
        doc_confidence = self._calculate_aggregate_confidence(self.processing_results)
        logger.debug("Aggregate document confidence: %.1f%%", doc_confidence)

        # Session 63b: Use checkbox state (which already reflects settings + GPU detection)
        # The checkbox is pre-configured by _update_vocab_llm_checkbox_state() at startup
        # and when settings change. We read the checkbox to respect user's in-session choice.
        use_llm = self.vocab_llm_check.get() and self.vocab_llm_check.cget("state") == "normal"
        logger.debug("LLM extraction from checkbox: %s", use_llm)

        # Start progressive extraction worker (uses shared ui_queue)
        self._progressive_worker = ProgressiveExtractionWorker(
            documents=self.processing_results,
            combined_text=combined_text,
            ui_queue=self._ui_queue,  # Use shared queue for unified message routing
            embeddings=self._embeddings,  # Pass existing if available
            exclude_list_path=str(LEGAL_EXCLUDE_LIST_PATH),
            medical_terms_path=str(MEDICAL_TERMS_LIST_PATH),
            user_exclude_path=str(USER_VOCAB_EXCLUDE_PATH),
            doc_confidence=doc_confidence,  # Session 54: OCR quality for ML
            use_llm=use_llm,  # Session 62b: Respect GPU auto-detect preference
        )
        self._progressive_worker.start()

        # Ensure queue polling is running (may have stopped after preprocessing)
        if self._queue_poll_id:
            self.after_cancel(self._queue_poll_id)
        self._poll_queue()

    def _start_qa_task(self):
        """Start Q&A task - build vector store then run questions."""
        import threading

        self.set_status(
            "Questions and answers: Loading embeddings model (this may take a moment)..."
        )

        # Run the heavy initialization in a background thread
        def initialize_qa():
            """Background thread for embeddings + vector store setup."""
            try:
                # Lazy-load embeddings model (slow first time, reused after)
                if self._embeddings is None:
                    logger.debug("Loading HuggingFaceEmbeddings model...")
                    from langchain_huggingface import HuggingFaceEmbeddings

                    self._embeddings = HuggingFaceEmbeddings(
                        model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
                    )
                    logger.debug("Embeddings model loaded")

                # Build vector store from documents
                logger.debug("Building vector store...")
                from src.services import QAService

                builder = QAService().get_vector_store_builder()
                result = builder.create_from_documents(
                    documents=self.processing_results, embeddings=self._embeddings
                )
                self._vector_store_path = result.persist_dir
                logger.debug(
                    "Vector store created: %s chunks at %s", result.chunk_count, result.persist_dir
                )

                # Signal main thread that initialization is complete
                self.after(0, lambda: self._qa_init_complete(True, None))

            except Exception as e:
                logger.debug("Q&A initialization error: %s", e)
                error_msg = str(e)
                self.after(0, lambda err=error_msg: self._qa_init_complete(False, err))

        # Start background thread
        init_thread = threading.Thread(target=initialize_qa, daemon=True)
        init_thread.start()

    def _qa_init_complete(self, success: bool, error: str | None):
        """Called when Q&A initialization (embeddings + vector store) completes."""
        if not success:
            self.set_status(f"Questions and answers error: {error[:50] if error else 'Unknown'}...")
            self._completed_tasks.add("qa")
            if self._pending_tasks.get("summary"):
                self._start_summary_task()
            else:
                self._finalize_tasks()
            return

        self.set_status("Questions and answers: Building vector store...")

        # Create Q&A queue and worker
        self._qa_queue = Queue()
        self._qa_worker = QAWorker(
            vector_store_path=self._vector_store_path,
            embeddings=self._embeddings,
            ui_queue=self._qa_queue,
            answer_mode="extraction",  # Fast extraction mode
        )
        self._qa_worker.start()

        # Start polling Q&A queue
        self.set_status("Questions and answers: Processing questions...")
        self._poll_qa_queue()

    def _poll_qa_queue(self):
        """Poll the Q&A worker queue for results."""
        try:
            while True:
                msg_type, data = self._qa_queue.get_nowait()
                if msg_type == "qa_progress":
                    current, total, _question = data
                    self.set_status(
                        f"Questions and answers: Processing question {current + 1}/{total}..."
                    )
                elif msg_type == "qa_result":
                    # Individual result - could update incrementally
                    pass
                elif msg_type == "qa_complete":
                    self._on_qa_complete(data)
                    return
                elif msg_type == "error":
                    self.set_status(f"Questions and answers error: {data}")
                    self._on_qa_complete([])
                    return
        except Empty:
            pass

        # Continue polling if worker is alive
        if self._qa_worker and self._qa_worker.is_alive():
            self.after(50, self._poll_qa_queue)
        else:
            # Worker finished - do final poll
            try:
                while True:
                    msg_type, data = self._qa_queue.get_nowait()
                    if msg_type == "qa_complete":
                        self._on_qa_complete(data)
                        return
            except Empty:
                pass
            # Worker finished without sending results
            self._on_qa_complete([])

    def _on_qa_complete(self, qa_results: list):
        """Handle Q&A completion."""
        self._completed_tasks.add("qa")
        self._qa_results = qa_results

        # Display results using update_outputs
        if qa_results:
            self.output_display.update_outputs(qa_results=qa_results)
            self.set_status(f"Questions and answers: {len(qa_results)} questions answered")
            # Enable follow-up question controls
            self.followup_btn.configure(state="normal")
            self.followup_entry.configure(state="normal")
            self.followup_entry.configure(placeholder_text="Type your question here...")
        else:
            self.set_status("Questions and answers complete (no results)")

        # Continue to next task
        if self._pending_tasks.get("summary"):
            self._start_summary_task()
        else:
            self._finalize_tasks()

    def _start_summary_task(self):
        """Start summary generation task."""
        self.set_status("Summary: This feature can take several hours...")

        # Summary is complex - show placeholder for now
        self._completed_tasks.add("summary")

        self.output_display.update_outputs(
            meta_summary="Summary generation can take several hours without a dedicated GPU. "
            "For quick case familiarization, use Q&A instead."
        )

        self._finalize_tasks()

    def _finalize_tasks(self):
        """Finalize all tasks and update UI."""
        completed = len(self._completed_tasks)
        self._on_tasks_complete(True, f"Completed {completed} task(s)")

    def _on_tasks_complete(self, success: bool, message: str):
        """Handle task completion."""
        self._stop_timer()

        # Re-enable controls
        self.add_files_btn.configure(state="normal")
        self._update_generate_button_state()

        # Enable follow-up if Q&A was run
        if self.qa_check.get() and success:
            self.followup_btn.configure(state="normal")

        # Show Export All button after successful processing
        if success and not self._export_all_visible:
            self.export_all_btn.pack(side="right", padx=10, pady=3)
            self._export_all_visible = True

        # Update session stats with extraction results (Session 73)
        if success:
            extraction_stats = self._gather_extraction_stats()
            self._update_session_stats(extraction_stats)

        self.set_status(message)

    def _gather_extraction_stats(self) -> dict:
        """
        Gather extraction statistics after task completion (Session 73).

        Returns:
            Dict with vocab_count, person_count, qa_count, processing_time
        """
        stats = {}

        # Vocabulary stats from output display
        vocab_data = getattr(self.output_display, "_vocab_csv_data", None)
        if vocab_data:
            stats["vocab_count"] = len(vocab_data)
            stats["person_count"] = sum(
                1 for v in vocab_data if v.get("Is Person", "").lower() in ("yes", "true", "1")
            )

        # Q&A stats
        if self._qa_results:
            stats["qa_count"] = len(self._qa_results)

        # Processing time
        if self._processing_start_time:
            stats["processing_time"] = time.time() - self._processing_start_time

        return stats

    def _ask_followup(self):
        """Ask a follow-up question using the Q&A system (async version)."""
        question = self.followup_entry.get().strip()
        if not question:
            return

        # Check prerequisites
        if not self._vector_store_path or not self._embeddings:
            messagebox.showwarning(
                "Questions Not Ready",
                "Question system is not initialized yet.\n\n"
                "To ask questions:\n"
                "1. Add document files\n"
                "2. Ensure the 'Ask Questions' checkbox is checked\n"
                "3. Click 'Perform Tasks'\n"
                "4. Wait for 'Questions and answers ready' status message\n\n"
                "The question system will be ready once the vector index is built.",
            )
            return

        # Prevent duplicate submissions
        if (
            hasattr(self, "_followup_thread")
            and self._followup_thread is not None
            and self._followup_thread.is_alive()
        ):
            logger.debug("Follow-up already in progress, ignoring")
            return

        # Clear entry and disable controls while processing
        self.followup_entry.delete(0, "end")
        self.followup_btn.configure(state="disabled", text="Asking...")
        self.followup_entry.configure(state="disabled")

        self.set_status(f"Asking: {question[:40]}...")

        # Run Q&A in background thread to keep GUI responsive
        import queue
        import threading

        self._followup_queue = queue.Queue()

        def run_followup():
            try:
                from src.services import QAService
                from src.user_preferences import get_user_preferences

                prefs = get_user_preferences()
                orchestrator = QAService().create_orchestrator(
                    vector_store_path=self._vector_store_path,
                    embeddings=self._embeddings,
                    answer_mode=prefs.get("qa_answer_mode", "ollama"),
                )
                result = orchestrator.ask_followup(question)
                self._followup_queue.put(("success", result))
            except Exception as e:
                self._followup_queue.put(("error", str(e)))
                logger.debug("Follow-up thread error: %s", e)

        self._followup_thread = threading.Thread(target=run_followup, daemon=True)
        self._followup_thread.start()

        # Start polling for results
        self._poll_followup_result()

    def _poll_followup_result(self):
        """Poll for follow-up result from background thread."""
        import queue

        try:
            msg_type, data = self._followup_queue.get_nowait()
        except queue.Empty:
            # No result yet, keep polling
            self.after(100, self._poll_followup_result)
            return

        # Got a result - re-enable controls
        self.followup_btn.configure(state="normal", text="Ask")
        self.followup_entry.configure(state="normal")
        self.followup_entry.focus()

        try:
            if msg_type == "success" and data is not None:
                # Add to existing results and refresh display
                with self._qa_results_lock:  # LOG-007: Thread-safe access
                    self._qa_results.append(data)
                    self.output_display.update_outputs(qa_results=self._qa_results)
                # Note: QAResult uses 'quick_answer', not 'answer'
                answer_len = len(data.quick_answer) if data.quick_answer else 0
                self.set_status(f"Follow-up answered: {answer_len} chars")
                logger.debug("Follow-up result displayed successfully")
            elif msg_type == "error":
                self.set_status("Follow-up failed")
                messagebox.showerror("Error", f"Failed to process follow-up: {data}")
        except Exception as e:
            # Catch any errors during result processing
            logger.debug("Error processing follow-up result: %s", e)
            self.set_status("Follow-up error - check logs")
            messagebox.showerror("Error", f"Error displaying result: {e!s}")

    def _ask_followup_for_qa_panel(self, question: str):
        """
        Ask a follow-up question from the QAPanel widget.

        This method is called by QAPanel's built-in follow-up input.
        It returns a QAResult directly (unlike _ask_followup which updates UI).

        Args:
            question: The follow-up question text

        Returns:
            QAResult object with the answer, or None on error
        """
        if not question:
            return None

        # Check prerequisites
        if not self._vector_store_path or not self._embeddings:
            logger.debug("Follow-up unavailable: no vector store or embeddings")
            return None

        # NOTE: This method runs in a background thread (from QAPanel._submit_followup)
        # Do NOT call GUI methods like set_status() here - it causes freezes!

        try:
            # Import and use QAOrchestrator for follow-up (via service layer)
            from src.services import QAService
            from src.user_preferences import get_user_preferences

            prefs = get_user_preferences()
            orchestrator = QAService().create_orchestrator(
                vector_store_path=self._vector_store_path,
                embeddings=self._embeddings,
                answer_mode=prefs.get("qa_answer_mode", "ollama"),
            )

            # Ask the follow-up question
            result = orchestrator.ask_followup(question)

            # Add to internal results list (so it persists across view changes)
            # LOG-007: Use lock for thread-safe access
            with self._qa_results_lock:
                self._qa_results.append(result)

            logger.debug("Follow-up answered: %s chars", len(result.answer))
            return result

        except Exception as e:
            logger.debug("Follow-up error: %s", e)
            return None

    # =========================================================================
    # Settings
    # =========================================================================

    def _open_settings(self):
        """Open the settings dialog."""
        from src.ui.settings.settings_dialog import SettingsDialog

        try:
            dialog = SettingsDialog(parent=self)
            dialog.wait_window()
        except Exception as e:
            logger.debug("Failed to open settings dialog: %s", e)

        # Refresh UI after settings change
        self._refresh_corpus_dropdown()
        self._update_model_display()
        self._update_ollama_status()
        self._update_vocab_llm_checkbox_state()  # Session 63b: Refresh LLM checkbox
        self._update_qa_checkbox_state()  # Session 90: Refresh Q&A checkbox for model size
        self._update_summary_checkbox_state()  # Session 93: Refresh Summary checkbox
        self.refresh_default_questions_label()  # Update question count after settings change

    # =========================================================================
    # Export All (tabbed HTML)
    # =========================================================================

    def _export_all(self):
        """
        Export all results (vocabulary, Q&A, summary) to a single tabbed HTML file.

        Opens a save dialog defaulting to .html. Gathers score-filtered vocab,
        answered Q&A, and summary text into one combined HTML document.
        """
        from datetime import datetime

        from src.services import DocumentService, get_export_service
        from src.user_preferences import get_user_preferences

        # Gather filtered vocabulary data
        vocab_data = self.output_display._get_filtered_vocab_data()

        # Gather answered Q&A results only
        qa_results = []
        if self._qa_results:
            qa_panel = self.output_display._qa_panel
            if qa_panel and qa_panel._results:
                qa_results = [r for r in qa_panel._results if r.is_exportable]

        # Gather summary text
        summary = self.output_display._outputs.get("Summary", "")
        if not summary:
            summary = self.output_display._outputs.get("Meta-Summary", "")
        summary_text = summary.strip() if summary else ""

        # Check we have something to export
        if not vocab_data and not qa_results and not summary_text:
            messagebox.showwarning("No Data", "No results to export yet.")
            return

        # Get initial directory (last export path or Documents)
        prefs = get_user_preferences()
        doc_service = DocumentService()
        initial_dir = prefs.get("last_export_path") or doc_service.get_default_documents_folder()

        # Open save dialog
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[
                ("HTML files", "*.html"),
                ("All files", "*.*"),
            ],
            initialfile=f"case_report_{timestamp}.html",
            initialdir=initial_dir,
            title="Export All Results",
        )

        if not filepath:
            return

        # Get visible columns from vocab display
        visible_columns = self.output_display._get_visible_columns()

        # Export combined HTML
        export_service = get_export_service()
        success = export_service.export_combined_html(
            vocab_data=vocab_data,
            qa_results=qa_results,
            summary_text=summary_text,
            file_path=filepath,
            visible_columns=visible_columns,
        )

        if success:
            # Remember export folder
            prefs.set("last_export_path", str(Path(filepath).parent))

            # Flash button and status
            self.export_all_btn.configure(text="Exported!")
            self.after(1500, lambda: self.export_all_btn.configure(text="Export All"))

            filename = Path(filepath).name
            parts = []
            if vocab_data:
                parts.append(f"{len(vocab_data)} terms")
            if qa_results:
                parts.append(f"{len(qa_results)} Q&A")
            if summary_text:
                parts.append("summary")
            self.set_status(f"Exported {' + '.join(parts)} to {filename}", duration_ms=5000)
            logger.debug("Export All HTML: %s", filepath)
        else:
            messagebox.showerror("Export Failed", "Failed to create HTML report.")

    # =========================================================================
    # Timer
    # =========================================================================

    def _start_timer(self):
        """Start the processing timer and activity indicator."""
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

        self._stop_activity_indicator()

    def _start_activity_indicator(self):
        """Show and start the animated activity indicator."""
        if hasattr(self, "activity_indicator"):
            if not self._activity_indicator_visible:
                self.activity_indicator.pack(side="right", padx=(0, 5), pady=5)
                self._activity_indicator_visible = True
            self.activity_indicator.start()
            self.update_idletasks()  # Force initial render for smooth animation

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
        Calculate aggregate confidence from processed documents (Session 54).

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
        # Cancel any pending status clear
        if hasattr(self, "_status_clear_id") and self._status_clear_id:
            self.after_cancel(self._status_clear_id)
            self._status_clear_id = None

        self.status_label.configure(text=message)

        logger.debug("Status: %s", message)

        # Schedule auto-clear if duration specified
        if duration_ms:
            self._status_clear_id = self.after(duration_ms, lambda: self._clear_status_to_default())

    def _clear_status_to_default(self):
        """Clear status bar to default 'Ready' message."""
        self._status_clear_id = None
        self.status_label.configure(text="Ready")

    # =========================================================================
    # Startup Checks
    # =========================================================================

    def _check_ollama_service(self):
        """Check if Ollama service is running on startup."""
        try:
            self.model_manager.health_check()
            logger.debug("Ollama service is accessible")
        except Exception as e:
            logger.debug("Ollama service not accessible: %s", e)

            # Show warning
            from src.config import APP_NAME

            messagebox.showwarning(
                "Ollama Not Found",
                "Ollama service is not running.\n\n"
                f"{APP_NAME} requires Ollama for Q&A and summaries.\n\n"
                "To install: Visit https://ollama.ai\n"
                "To start: Run 'ollama serve' in a terminal\n\n"
                "Vocabulary extraction will still work without Ollama.",
            )

    # =========================================================================
    # Cleanup
    # =========================================================================

    def destroy(self):
        """Clean up resources before destroying window."""
        # Stop queue polling
        if self._queue_poll_id:
            self.after_cancel(self._queue_poll_id)
            self._queue_poll_id = None

        # Stop any running workers
        if self._processing_worker and self._processing_worker.is_alive():
            # Worker is a daemon thread, will stop when main thread exits
            pass
        if (
            self._vocabulary_worker
            and hasattr(self._vocabulary_worker, "is_alive")
            and self._vocabulary_worker.is_alive()
        ):
            pass

        # Stop timer
        self._stop_timer()

        super().destroy()
