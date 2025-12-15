"""
LocalScribe - Main Window (CustomTkinter)
Session 29: Two-Panel Q&A-First Layout
Session 33: Refactored - Layout extracted to WindowLayoutMixin

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
"""

import time
from queue import Queue, Empty
from tkinter import filedialog, messagebox

import customtkinter as ctk

from src.config import DEBUG_MODE, PROMPTS_DIR
from src.logging_config import debug_log
from src.ai import OllamaModelManager
from src.prompting import PromptTemplateManager
from src.ui.workers import ProcessingWorker, VocabularyWorker, QAWorker, BriefingWorker, ProgressiveExtractionWorker
from src.ui.window_layout import WindowLayoutMixin
from src.vocabulary import get_corpus_registry
from src.vector_store import VectorStoreBuilder


class MainWindow(WindowLayoutMixin, ctk.CTk):
    """
    Main application window for LocalScribe.

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

        self.title("LocalScribe")
        self.geometry("1200x750")
        self.minsize(900, 600)

        # State
        self.selected_files: list[str] = []
        self.processing_results: list[dict] = []
        self._processing_start_time: float | None = None
        self._timer_after_id: str | None = None

        # Managers
        self.model_manager = OllamaModelManager()
        self.prompt_template_manager = PromptTemplateManager(PROMPTS_DIR)
        self.corpus_registry = get_corpus_registry()

        # Workers and queue
        self._processing_worker: ProcessingWorker | None = None
        self._vocabulary_worker: VocabularyWorker | None = None
        self._qa_worker: QAWorker | None = None
        self._briefing_worker: BriefingWorker | None = None
        self._progressive_worker: ProgressiveExtractionWorker | None = None  # Session 45
        self._ui_queue: Queue | None = None
        self._queue_poll_id: str | None = None

        # Q&A infrastructure
        self._embeddings = None  # Lazy-loaded HuggingFaceEmbeddings
        self._vector_store_path = None  # Path to current session's vector store
        self._qa_results: list = []  # Store QAResult objects
        self._qa_ready = False  # Session 45: Q&A becomes available after indexing

        # Build UI
        self._create_header()
        self._create_warning_banner()
        self._create_main_panels()
        self._create_status_bar()

        # Initialize state
        self._refresh_corpus_dropdown()
        self._update_corpus_banner()
        self._update_generate_button_state()

        # Startup checks
        self._check_ollama_service()

        if DEBUG_MODE:
            debug_log("[MainWindow] Initialized with two-panel layout")

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

                # Update status bar with corpus info
                active_info = next((c for c in corpora if c.name == active), None)
                if active_info:
                    self.corpus_info_label.configure(
                        text=f"Corpus: {active} ({active_info.doc_count} docs)"
                    )
            else:
                self.corpus_dropdown.configure(values=["No corpora"])
                self.corpus_dropdown.set("No corpora")
                self.corpus_info_label.configure(text="")

        except Exception as e:
            debug_log(f"[MainWindow] Error refreshing corpus dropdown: {e}")
            self.corpus_dropdown.configure(values=["Error"])
            self.corpus_dropdown.set("Error")

    def _update_corpus_banner(self):
        """Show or hide the no-corpus warning banner."""
        try:
            corpora = self.corpus_registry.list_corpora()
            total_docs = sum(c.doc_count for c in corpora)

            if total_docs == 0:
                # Show banner
                self.banner_frame.pack(fill="x", after=self.header_frame)
            else:
                # Hide banner
                self.banner_frame.pack_forget()

        except Exception:
            # On error, hide banner
            self.banner_frame.pack_forget()

    def _on_corpus_changed(self, corpus_name: str):
        """Handle corpus selection change."""
        try:
            self.corpus_registry.set_active_corpus(corpus_name)
            self._refresh_corpus_dropdown()
            self.set_status(f"Active corpus: {corpus_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to switch corpus: {e}")

    def _open_corpus_dialog(self):
        """Open the corpus management dialog."""
        from src.ui.corpus_dialog import CorpusDialog

        dialog = CorpusDialog(self)
        self.wait_window(dialog)

        # Refresh after dialog closes
        if dialog.corpus_changed:
            self._refresh_corpus_dropdown()
            self._update_corpus_banner()
            self.set_status("Corpus updated")

    # =========================================================================
    # File Management
    # =========================================================================

    def _select_files(self):
        """Open file dialog to select documents for this session."""
        files = filedialog.askopenfilenames(
            title="Select Documents for This Session",
            filetypes=[
                ("Documents", "*.pdf *.txt *.rtf"),
                ("PDF files", "*.pdf"),
                ("Text files", "*.txt"),
                ("RTF files", "*.rtf"),
                ("All files", "*.*")
            ]
        )

        if not files:
            return

        self.selected_files = list(files)
        self.set_status(f"Processing {len(files)} file(s)...")
        self._start_preprocessing()

    def _clear_files(self):
        """Clear all files from the session."""
        self.selected_files.clear()
        self.processing_results.clear()
        self.file_table.clear()
        self._update_generate_button_state()
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
            file_paths=self.selected_files,
            ui_queue=self._ui_queue
        )
        self._processing_worker.start()

        # Start polling the queue
        self._poll_queue()

    def _poll_queue(self):
        """Poll the UI queue for worker messages."""
        try:
            while True:
                msg_type, data = self._ui_queue.get_nowait()
                self._handle_queue_message(msg_type, data)
        except Empty:
            pass

        # Continue polling if worker is running
        if self._processing_worker and self._processing_worker.is_alive():
            self._queue_poll_id = self.after(50, self._poll_queue)
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
            percentage, message = data
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

    def _on_preprocessing_complete(self, results: list[dict]):
        """Handle preprocessing completion."""
        # Stop timer
        self._stop_timer()

        # Stop queue polling
        if self._queue_poll_id:
            self.after_cancel(self._queue_poll_id)
            self._queue_poll_id = None

        # Re-enable controls
        self.add_files_btn.configure(state="normal")
        self._update_generate_button_state()

        # Count results
        success_count = sum(1 for r in results if r.get('status') == 'success')
        failed_count = len(results) - success_count

        status = f"Processed {len(results)} file(s): {success_count} ready"
        if failed_count > 0:
            status += f", {failed_count} failed"

        self.set_status(status)

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

    def _on_summary_checked(self):
        """Handle summary checkbox toggle - show warning if enabling."""
        if self.summary_check.get():
            # Show warning dialog
            result = messagebox.askyesno(
                "Summary Warning",
                "Summary generation typically takes 30+ minutes and results depend "
                "heavily on your hardware.\n\n"
                "For quick case familiarization, Q&A is recommended instead.\n\n"
                "Continue with summary?",
                icon="warning"
            )
            if not result:
                self.summary_check.deselect()

        self._update_generate_button_state()

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

        # Start timer
        self._start_timer()

        # Get selected options
        do_qa = self.qa_check.get()
        do_vocab = self.vocab_check.get()
        do_summary = self.summary_check.get()

        # Track pending tasks
        self._pending_tasks = {
            'vocab': do_vocab,
            'qa': do_qa,
            'summary': do_summary
        }
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
        from src.utils.text_utils import combine_document_texts
        from src.config import LEGAL_EXCLUDE_LIST_PATH, MEDICAL_TERMS_LIST_PATH, USER_VOCAB_EXCLUDE_PATH

        self.set_status("Extracting vocabulary...")

        # Debug: Log what's in processing_results
        debug_log(f"[MainWindow] Vocabulary: {len(self.processing_results)} documents in processing_results")
        for i, doc in enumerate(self.processing_results):
            text_len = len(doc.get('extracted_text', '') or '')
            debug_log(f"[MainWindow] Doc {i}: {doc.get('filename', 'unknown')} - {text_len} chars, status={doc.get('status')}")

        # Combine text from all processed documents
        combined_text = combine_document_texts(self.processing_results)

        debug_log(f"[MainWindow] Combined text length: {len(combined_text)} chars")

        if not combined_text.strip():
            self.set_status("No text to analyze for vocabulary")
            debug_log("[MainWindow] WARNING: No text after combining documents!")
            self._on_vocab_complete([])
            return

        # Create queue for vocab worker
        self._vocab_queue = Queue()

        # Check if LLM extraction is enabled (Session 43)
        from src.user_preferences import get_user_preferences
        prefs = get_user_preferences()
        use_llm = prefs.is_vocab_llm_enabled()
        debug_log(f"[MainWindow] Vocabulary extraction with LLM: {use_llm}")

        # Start vocabulary worker
        self._vocabulary_worker = VocabularyWorker(
            combined_text=combined_text,
            ui_queue=self._vocab_queue,
            exclude_list_path=str(LEGAL_EXCLUDE_LIST_PATH),
            medical_terms_path=str(MEDICAL_TERMS_LIST_PATH),
            user_exclude_path=str(USER_VOCAB_EXCLUDE_PATH),
            doc_count=len(self.processing_results),
            use_llm=use_llm,  # Session 43: Enable LLM-based extraction
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
        self._completed_tasks.add('vocab')

        # Display results using update_outputs
        if vocab_data:
            self.output_display.update_outputs(vocab_csv_data=vocab_data)
            self.set_status(f"Vocabulary: {len(vocab_data)} terms found")
        else:
            self.set_status("Vocabulary extraction complete (no terms)")

        # Continue to next task
        if self._pending_tasks.get('qa'):
            # Use Case Briefing instead of legacy Q&A
            self._start_briefing_task()
        elif self._pending_tasks.get('summary'):
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
        from src.utils.text_utils import combine_document_texts
        from src.config import LEGAL_EXCLUDE_LIST_PATH, MEDICAL_TERMS_LIST_PATH, USER_VOCAB_EXCLUDE_PATH

        self.set_status("Starting extraction (NER first, then LLM enhancement)...")

        # Combine text from all processed documents
        combined_text = combine_document_texts(self.processing_results)

        debug_log(f"[MainWindow] Progressive extraction: {len(combined_text)} chars from {len(self.processing_results)} docs")

        if not combined_text.strip():
            self.set_status("No text to analyze")
            debug_log("[MainWindow] WARNING: No text after combining documents!")
            self._on_tasks_complete(False, "No text to analyze")
            return

        # Create queue for progressive worker
        self._progressive_queue = Queue()

        # Start progressive extraction worker
        self._progressive_worker = ProgressiveExtractionWorker(
            documents=self.processing_results,
            combined_text=combined_text,
            ui_queue=self._progressive_queue,
            embeddings=self._embeddings,  # Pass existing if available
            exclude_list_path=str(LEGAL_EXCLUDE_LIST_PATH),
            medical_terms_path=str(MEDICAL_TERMS_LIST_PATH),
            user_exclude_path=str(USER_VOCAB_EXCLUDE_PATH),
        )
        self._progressive_worker.start()

        # Start polling progressive queue
        self._poll_progressive_queue()

    def _poll_progressive_queue(self):
        """Poll the progressive extraction worker queue for messages."""
        try:
            while True:
                msg_type, data = self._progressive_queue.get_nowait()
                self._handle_progressive_message(msg_type, data)
        except Empty:
            pass

        # Continue polling if worker is alive
        if self._progressive_worker and self._progressive_worker.is_alive():
            self.after(50, self._poll_progressive_queue)
        else:
            # Worker finished - do final poll
            try:
                while True:
                    msg_type, data = self._progressive_queue.get_nowait()
                    self._handle_progressive_message(msg_type, data)
            except Empty:
                pass
            # Finalize if not already done
            if 'vocab' not in self._completed_tasks:
                self._completed_tasks.add('vocab')
                self._finalize_tasks()

    def _handle_progressive_message(self, msg_type: str, data):
        """Handle messages from the progressive extraction worker."""
        if msg_type == "progress":
            percentage, message = data
            self.set_status(message)

        elif msg_type == "ner_complete":
            # Phase 1 complete - display NER results immediately
            debug_log(f"[MainWindow] NER complete: {len(data)} terms - displaying immediately")
            self.output_display.update_outputs(vocab_csv_data=data)
            self.output_display.set_extraction_source("ner")  # Show "Initial results (NER only)"
            self.set_status(f"NER complete: {len(data)} terms found. LLM enhancement starting...")

        elif msg_type == "qa_ready":
            # Phase 2 complete - enable Q&A
            debug_log(f"[MainWindow] Q&A ready: {data.get('chunk_count', 0)} chunks indexed")
            self._vector_store_path = data.get('vector_store_path')
            self._embeddings = data.get('embeddings')
            self._qa_ready = True

            # Mark Q&A as complete if it was requested
            if self._pending_tasks.get('qa'):
                self._completed_tasks.add('qa')
                self.followup_btn.configure(state="normal")
                self.set_status(f"Q&A ready ({data.get('chunk_count', 0)} chunks). LLM enhancement in progress...")

        elif msg_type == "qa_error":
            debug_log(f"[MainWindow] Q&A indexing error: {data}")
            # Continue without Q&A - not fatal

        elif msg_type == "llm_progress":
            # Phase 3 progress - LLM is processing chunks
            current, total = data
            # Status already set by 'progress' message

        elif msg_type == "llm_complete":
            # Phase 3 complete - update with reconciled results
            debug_log(f"[MainWindow] LLM complete: {len(data)} reconciled terms")
            self.output_display.update_outputs(vocab_csv_data=data)
            self.output_display.set_extraction_source("both")  # Show "Enhanced results (NER + LLM)"
            self._completed_tasks.add('vocab')
            self.set_status(f"Complete: {len(data)} names & vocabulary extracted")

            # Continue to summary if requested
            if self._pending_tasks.get('summary'):
                self._start_summary_task()
            else:
                self._finalize_tasks()

        elif msg_type == "error":
            self.set_status(f"Error: {data}")
            messagebox.showerror("Extraction Error", str(data))
            self._on_tasks_complete(False, str(data))

    def _start_qa_task(self):
        """Start Q&A task - build vector store then run questions."""
        import threading

        self.set_status("Q&A: Loading embeddings model (this may take a moment)...")

        # Run the heavy initialization in a background thread
        def initialize_qa():
            """Background thread for embeddings + vector store setup."""
            try:
                # Lazy-load embeddings model (slow first time, reused after)
                if self._embeddings is None:
                    debug_log("[MainWindow] Loading HuggingFaceEmbeddings model...")
                    from langchain_huggingface import HuggingFaceEmbeddings
                    self._embeddings = HuggingFaceEmbeddings(
                        model_name="all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'}
                    )
                    debug_log("[MainWindow] Embeddings model loaded")

                # Build vector store from documents
                debug_log("[MainWindow] Building vector store...")
                builder = VectorStoreBuilder()
                result = builder.create_from_documents(
                    documents=self.processing_results,
                    embeddings=self._embeddings
                )
                self._vector_store_path = result.persist_dir
                debug_log(f"[MainWindow] Vector store created: {result.chunk_count} chunks at {result.persist_dir}")

                # Signal main thread that initialization is complete
                self.after(0, lambda: self._qa_init_complete(True, None))

            except Exception as e:
                debug_log(f"[MainWindow] Q&A initialization error: {e}")
                self.after(0, lambda: self._qa_init_complete(False, str(e)))

        # Start background thread
        init_thread = threading.Thread(target=initialize_qa, daemon=True)
        init_thread.start()

    def _qa_init_complete(self, success: bool, error: str | None):
        """Called when Q&A initialization (embeddings + vector store) completes."""
        if not success:
            self.set_status(f"Q&A error: {error[:50] if error else 'Unknown'}...")
            self._completed_tasks.add('qa')
            if self._pending_tasks.get('summary'):
                self._start_summary_task()
            else:
                self._finalize_tasks()
            return

        self.set_status("Q&A: Building vector store...")

        # Create Q&A queue and worker
        self._qa_queue = Queue()
        self._qa_worker = QAWorker(
            vector_store_path=self._vector_store_path,
            embeddings=self._embeddings,
            ui_queue=self._qa_queue,
            answer_mode="extraction"  # Fast extraction mode
        )
        self._qa_worker.start()

        # Start polling Q&A queue
        self.set_status("Q&A: Processing questions...")
        self._poll_qa_queue()

    def _poll_qa_queue(self):
        """Poll the Q&A worker queue for results."""
        try:
            while True:
                msg_type, data = self._qa_queue.get_nowait()
                if msg_type == "qa_progress":
                    current, total, question = data
                    self.set_status(f"Q&A: Processing question {current + 1}/{total}...")
                elif msg_type == "qa_result":
                    # Individual result - could update incrementally
                    pass
                elif msg_type == "qa_complete":
                    self._on_qa_complete(data)
                    return
                elif msg_type == "error":
                    self.set_status(f"Q&A error: {data}")
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
        self._completed_tasks.add('qa')
        self._qa_results = qa_results

        # Display results using update_outputs
        if qa_results:
            self.output_display.update_outputs(qa_results=qa_results)
            self.set_status(f"Q&A: {len(qa_results)} questions answered")
            # Enable follow-up button
            self.followup_btn.configure(state="normal")
        else:
            self.set_status("Q&A complete (no results)")

        # Continue to next task
        if self._pending_tasks.get('summary'):
            self._start_summary_task()
        else:
            self._finalize_tasks()

    def _start_summary_task(self):
        """Start summary generation task."""
        self.set_status("Summary: This feature takes 30+ minutes...")

        # Summary is complex - show placeholder for now
        self._completed_tasks.add('summary')

        self.output_display.update_outputs(
            meta_summary="Summary generation is a long-running task (30+ minutes). "
            "For quick case familiarization, use Q&A instead."
        )

        self._finalize_tasks()

    def _finalize_tasks(self):
        """Finalize all tasks and update UI."""
        completed = len(self._completed_tasks)
        self._on_tasks_complete(True, f"Completed {completed} task(s)")

    # =========================================================================
    # Case Briefing Task (replaces Q&A for structured extraction)
    # =========================================================================

    def _start_briefing_task(self):
        """Start case briefing generation task."""
        self.set_status("Case Briefing: Starting document analysis...")

        # Create queue for briefing worker
        self._briefing_queue = Queue()

        # Start briefing worker
        self._briefing_worker = BriefingWorker(
            documents=self.processing_results,
            ui_queue=self._briefing_queue
        )
        self._briefing_worker.start()

        # Start polling briefing queue
        self._poll_briefing_queue()

    def _poll_briefing_queue(self):
        """Poll the briefing worker queue for results."""
        try:
            while True:
                msg_type, data = self._briefing_queue.get_nowait()
                if msg_type == "briefing_progress":
                    phase, current, total, message = data
                    self.set_status(f"Case Briefing: {message}")
                elif msg_type == "briefing_complete":
                    self._on_briefing_complete(data)
                    return
                elif msg_type == "error":
                    self.set_status(f"Briefing error: {data}")
                    self._on_briefing_complete(None)
                    return
        except Empty:
            pass

        # Continue polling if worker is alive
        if self._briefing_worker and self._briefing_worker.is_alive():
            self.after(100, self._poll_briefing_queue)  # 100ms for longer task
        else:
            # Worker finished - do final poll
            try:
                while True:
                    msg_type, data = self._briefing_queue.get_nowait()
                    if msg_type == "briefing_complete":
                        self._on_briefing_complete(data)
                        return
            except Empty:
                pass
            # Worker finished without sending results
            self._on_briefing_complete(None)

    def _on_briefing_complete(self, briefing_data: dict | None):
        """Handle briefing generation completion."""
        self._completed_tasks.add('qa')  # Count as Q&A task for compatibility

        if briefing_data and briefing_data.get('formatted'):
            formatted = briefing_data['formatted']
            result = briefing_data.get('result')

            # Store briefing result for export
            self._briefing_result = result
            self._formatted_briefing = formatted

            # Display the briefing text in the output widget
            self.output_display.update_outputs(
                briefing_text=formatted.text,
                briefing_sections=formatted.sections
            )

            time_str = f"{result.total_time_seconds:.1f}s" if result else ""
            self.set_status(f"Case Briefing complete ({time_str})")
        else:
            self.set_status("Briefing generation failed")

        # Continue to next task
        if self._pending_tasks.get('summary'):
            self._start_summary_task()
        else:
            self._finalize_tasks()

    def _on_tasks_complete(self, success: bool, message: str):
        """Handle task completion."""
        self._stop_timer()

        # Re-enable controls
        self.add_files_btn.configure(state="normal")
        self._update_generate_button_state()

        # Enable follow-up if Q&A was run
        if self.qa_check.get() and success:
            self.followup_btn.configure(state="normal")

        self.set_status(message)

    def _ask_followup(self):
        """Ask a follow-up question using the Q&A system."""
        question = self.followup_entry.get().strip()
        if not question:
            return

        # Check prerequisites
        if not self._vector_store_path or not self._embeddings:
            messagebox.showwarning("Not Ready", "Please run Q&A first to enable follow-up questions.")
            return

        # Clear entry
        self.followup_entry.delete(0, "end")

        self.set_status(f"Asking: {question[:40]}...")

        try:
            # Import and use QAOrchestrator for follow-up
            from src.qa import QAOrchestrator

            orchestrator = QAOrchestrator(
                vector_store_path=self._vector_store_path,
                embeddings=self._embeddings,
                answer_mode="extraction"
            )

            # Ask the follow-up question
            result = orchestrator.ask_followup(question)

            # Add to existing results and refresh display
            self._qa_results.append(result)
            self.output_display.update_outputs(qa_results=self._qa_results)
            self.set_status(f"Follow-up answered: {len(result.answer)} chars")

        except Exception as e:
            debug_log(f"[MainWindow] Follow-up error: {e}")
            messagebox.showerror("Error", f"Failed to process follow-up: {str(e)}")

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
            debug_log("[MainWindow] Follow-up unavailable: no vector store or embeddings")
            return None

        self.set_status(f"Asking: {question[:40]}...")

        try:
            # Import and use QAOrchestrator for follow-up
            from src.qa import QAOrchestrator

            orchestrator = QAOrchestrator(
                vector_store_path=self._vector_store_path,
                embeddings=self._embeddings,
                answer_mode="extraction"
            )

            # Ask the follow-up question
            result = orchestrator.ask_followup(question)

            # Add to internal results list (so it persists across view changes)
            self._qa_results.append(result)
            self.set_status(f"Follow-up answered: {len(result.answer)} chars")

            return result

        except Exception as e:
            debug_log(f"[MainWindow] Follow-up error: {e}")
            self.set_status(f"Error: {str(e)[:50]}")
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
            from src.logging_config import error as log_error
            log_error(f"Failed to open settings dialog: {e}")
            import traceback
            traceback.print_exc()

        # Refresh UI after settings change
        self._refresh_corpus_dropdown()
        self._update_corpus_banner()

    # =========================================================================
    # Timer
    # =========================================================================

    def _start_timer(self):
        """Start the processing timer."""
        self._processing_start_time = time.time()
        self._update_timer()

    def _stop_timer(self):
        """Stop the processing timer."""
        if self._timer_after_id:
            self.after_cancel(self._timer_after_id)
            self._timer_after_id = None

        # Keep final time displayed
        if self._processing_start_time:
            elapsed = time.time() - self._processing_start_time
            self._format_timer(elapsed)

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

    # =========================================================================
    # Status Bar
    # =========================================================================

    def set_status(self, message: str):
        """Update the status bar message."""
        self.status_label.configure(text=message)
        if DEBUG_MODE:
            debug_log(f"[MainWindow] Status: {message}")

    # =========================================================================
    # Startup Checks
    # =========================================================================

    def _check_ollama_service(self):
        """Check if Ollama service is running on startup."""
        try:
            self.model_manager.health_check()
            debug_log("[MainWindow] Ollama service is accessible")
        except Exception as e:
            debug_log(f"[MainWindow] Ollama service not accessible: {e}")

            # Show warning
            messagebox.showwarning(
                "Ollama Not Found",
                "Ollama service is not running.\n\n"
                "LocalScribe requires Ollama for Q&A and summaries.\n\n"
                "To install: Visit https://ollama.ai\n"
                "To start: Run 'ollama serve' in a terminal\n\n"
                "Vocabulary extraction will still work without Ollama."
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
        if self._vocabulary_worker and hasattr(self._vocabulary_worker, 'is_alive') and self._vocabulary_worker.is_alive():
            pass

        # Stop timer
        self._stop_timer()

        super().destroy()
