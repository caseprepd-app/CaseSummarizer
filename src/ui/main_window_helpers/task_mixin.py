"""
Task Execution Mixin.

Session 82: Extracted from main_window.py for modularity.

Contains:
- Task state management (task count, button state)
- Queue message handling
- Vocabulary extraction workflow
- Progressive extraction (NER + LLM)
- Q&A task execution
- Briefing task execution
- Follow-up question handling
"""

import threading
import time
from queue import Empty, Queue
from tkinter import messagebox

from src.logging_config import debug_log


class TaskMixin:
    """
    Mixin class providing task execution functionality.

    Requires parent class to have:
    - self.processing_results: List of processing result dicts
    - self.qa_check, self.vocab_check, self.summary_check: Checkbox vars
    - self.vocab_llm_check: LLM checkbox var
    - self.ask_default_questions_check: Default questions checkbox var
    - self.generate_btn: Generate button
    - self.followup_btn: Follow-up button
    - self.followup_entry: Follow-up entry widget
    - self.output_display: DynamicOutputWidget
    - self._ui_queue: Queue for worker communication
    - Various worker attributes
    """

    # =========================================================================
    # Task Count & Button State
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
                "Summary generation typically takes 30+ minutes and results depend "
                "heavily on your hardware.\n\n"
                "For quick case familiarization, Q&A is recommended instead.\n\n"
                "Continue with summary?",
                icon="warning",
            )
            if not result:
                self.summary_check.deselect()

        self._update_generate_button_state()

    # =========================================================================
    # Queue Message Handling
    # =========================================================================

    def _handle_queue_message(self, msg_type: str, data):
        """Handle a message from the worker queue."""
        if msg_type == "progress":
            _percentage, message = data
            # Append Q&A status if ready
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

        # Progressive Extraction handlers (Session 48)
        elif msg_type == "ner_complete":
            term_count = len(data) if data else 0
            debug_log(f"[MainWindow] NER complete: {term_count} terms")
            self.output_display.update_outputs(vocab_csv_data=data)
            self.output_display.set_extraction_source("ner")
            self.set_status(f"NER complete: {term_count} terms found. LLM enhancement starting...")

        elif msg_type == "qa_ready":
            chunk_count = data.get("chunk_count", 0)
            debug_log(f"[MainWindow] Q&A ready: {chunk_count} chunks indexed")
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
            debug_log(f"[MainWindow] Q&A indexing error: {error_msg}")
            self.set_status(f"Questions and answers unavailable: {error_msg[:50]}...")

        elif msg_type == "trigger_default_qa":
            self._handle_trigger_default_qa(data)

        elif msg_type == "qa_progress":
            current, total, _question = data
            debug_log(f"[MainWindow] Q&A progress: {current + 1}/{total}")
            self.set_status(f"Answering default questions: {current + 1}/{total}...")

        elif msg_type == "qa_result":
            debug_log("[MainWindow] Q&A result received")
            with self._qa_results_lock:
                self._qa_results.append(data)
                self.output_display.update_outputs(qa_results=self._qa_results)

        elif msg_type == "qa_complete":
            qa_results = data if data else []
            debug_log(f"[MainWindow] Q&A complete: {len(qa_results)} answers")
            with self._qa_results_lock:
                self._qa_results = qa_results
            if qa_results:
                self.output_display.update_outputs(qa_results=qa_results)
                self.set_status(f"Default questions answered: {len(qa_results)} responses")
            self.followup_btn.configure(state="normal")

        elif msg_type == "llm_progress":
            current, total = data
            debug_log(f"[MainWindow] LLM progress: {current}/{total}")

        elif msg_type == "llm_complete":
            term_count = len(data) if data else 0
            debug_log(f"[MainWindow] LLM complete: {term_count} reconciled terms")

            if data:
                self.output_display.update_outputs(vocab_csv_data=data)
                self.output_display.set_extraction_source("both")
                self.set_status(f"Complete: {term_count} names & vocabulary extracted")
            else:
                self.set_status("Complete: NER extraction only (LLM disabled)")

            self._completed_tasks.add("vocab")
            if self._pending_tasks.get("summary"):
                self._start_summary_task()
            else:
                self._finalize_tasks()

        else:
            debug_log(f"[MainWindow] Unhandled message type: {msg_type}")

    def _handle_trigger_default_qa(self, data):
        """Handle trigger_default_qa message - spawn QAWorker for default questions."""
        if not self.ask_default_questions_check.get():
            debug_log("[MainWindow] Default questions disabled, skipping")
            return

        from src.ui.workers import QAWorker
        from src.user_preferences import get_user_preferences

        debug_log("[MainWindow] Spawning QAWorker for default questions")
        prefs = get_user_preferences()

        qa_worker = QAWorker(
            vector_store_path=data["vector_store_path"],
            embeddings=data["embeddings"],
            ui_queue=self._ui_queue,
            answer_mode=prefs.get("qa_answer_mode", "extraction"),
            questions=None,
            use_default_questions=True,
        )
        qa_worker.start()
        debug_log("[MainWindow] Default questions worker started")

    # =========================================================================
    # Task Execution
    # =========================================================================

    def _perform_tasks(self):
        """Execute the selected tasks using progressive three-phase architecture."""
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

        # Hide task preview - status bar now shows progress
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

        # Use progressive extraction for vocabulary (includes Q&A indexing)
        if do_vocab:
            self._start_progressive_extraction()
        elif do_qa:
            self._start_qa_task()
        elif do_summary:
            self._start_summary_task()
        else:
            self._on_tasks_complete(True, "No tasks selected")

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
        from src.core.utils.text_utils import combine_document_texts
        from src.ui.workers import ProgressiveExtractionWorker

        self.set_status("Starting extraction (NER first, then LLM enhancement)...")

        # Combine text from all processed documents
        combined_text = combine_document_texts(self.processing_results)

        debug_log(
            f"[MainWindow] Progressive extraction: {len(combined_text)} chars "
            f"from {len(self.processing_results)} docs"
        )

        if not combined_text.strip():
            self.set_status("No text to analyze")
            debug_log("[MainWindow] WARNING: No text after combining documents!")
            self._on_tasks_complete(False, "No text to analyze")
            return

        # Calculate aggregate document confidence (Session 54)
        doc_confidence = self._calculate_aggregate_confidence(self.processing_results)
        debug_log(f"[MainWindow] Aggregate document confidence: {doc_confidence:.1f}%")

        # Use checkbox state (which reflects settings + GPU detection)
        use_llm = self.vocab_llm_check.get() and self.vocab_llm_check.cget("state") == "normal"
        debug_log(f"[MainWindow] LLM extraction from checkbox: {use_llm}")

        # Start progressive extraction worker
        self._progressive_worker = ProgressiveExtractionWorker(
            documents=self.processing_results,
            combined_text=combined_text,
            ui_queue=self._ui_queue,
            embeddings=self._embeddings,
            exclude_list_path=str(LEGAL_EXCLUDE_LIST_PATH),
            medical_terms_path=str(MEDICAL_TERMS_LIST_PATH),
            user_exclude_path=str(USER_VOCAB_EXCLUDE_PATH),
            doc_confidence=doc_confidence,
            use_llm=use_llm,
        )
        self._progressive_worker.start()

        # Ensure queue polling is running
        if self._queue_poll_id:
            self.after_cancel(self._queue_poll_id)
        self._poll_queue()

    # =========================================================================
    # Q&A Task
    # =========================================================================

    def _start_qa_task(self):
        """Start Q&A task - build vector store then run questions."""
        from src.core.vector_store import VectorStoreBuilder

        self.set_status(
            "Questions and answers: Loading embeddings model (this may take a moment)..."
        )

        def initialize_qa():
            """Background thread for embeddings + vector store setup."""
            try:
                # Lazy-load embeddings model
                if self._embeddings is None:
                    debug_log("[MainWindow] Loading HuggingFaceEmbeddings model...")
                    from langchain_huggingface import HuggingFaceEmbeddings

                    self._embeddings = HuggingFaceEmbeddings(
                        model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
                    )
                    debug_log("[MainWindow] Embeddings model loaded")

                # Build vector store
                debug_log("[MainWindow] Building vector store...")
                builder = VectorStoreBuilder()
                result = builder.create_from_documents(
                    documents=self.processing_results, embeddings=self._embeddings
                )
                self._vector_store_path = result.persist_dir
                debug_log(
                    f"[MainWindow] Vector store created: {result.chunk_count} chunks "
                    f"at {result.persist_dir}"
                )

                self.after(0, lambda: self._qa_init_complete(True, None))

            except Exception as e:
                debug_log(f"[MainWindow] Q&A initialization error: {e}")
                error_msg = str(e)
                self.after(0, lambda err=error_msg: self._qa_init_complete(False, err))

        init_thread = threading.Thread(target=initialize_qa, daemon=True)
        init_thread.start()

    def _qa_init_complete(self, success: bool, error: str | None):
        """Called when Q&A initialization completes."""
        from src.ui.workers import QAWorker

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
            answer_mode="extraction",
        )
        self._qa_worker.start()

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
                    pass  # Could update incrementally
                elif msg_type == "qa_complete":
                    self._on_qa_complete(data)
                    return
                elif msg_type == "error":
                    self.set_status(f"Questions and answers error: {data}")
                    self._on_qa_complete([])
                    return
        except Empty:
            pass

        if self._qa_worker and self._qa_worker.is_alive():
            self.after(50, self._poll_qa_queue)
        else:
            # Final poll
            try:
                while True:
                    msg_type, data = self._qa_queue.get_nowait()
                    if msg_type == "qa_complete":
                        self._on_qa_complete(data)
                        return
            except Empty:
                pass
            self._on_qa_complete([])

    def _on_qa_complete(self, qa_results: list):
        """Handle Q&A completion."""
        self._completed_tasks.add("qa")
        self._qa_results = qa_results

        if qa_results:
            self.output_display.update_outputs(qa_results=qa_results)
            self.set_status(f"Questions and answers: {len(qa_results)} questions answered")
            self.followup_btn.configure(state="normal")
        else:
            self.set_status("Questions and answers complete (no results)")

        if self._pending_tasks.get("summary"):
            self._start_summary_task()
        else:
            self._finalize_tasks()

    # =========================================================================
    # Summary & Briefing Tasks
    # =========================================================================

    def _start_summary_task(self):
        """Start summary generation task."""
        self.set_status("Summary: This feature takes 30+ minutes...")

        self._completed_tasks.add("summary")
        self.output_display.update_outputs(
            meta_summary="Summary generation is a long-running task (30+ minutes). "
            "For quick case familiarization, use Q&A instead."
        )
        self._finalize_tasks()

    def _start_briefing_task(self):
        """Start case briefing generation task."""
        from src.ui.workers import BriefingWorker

        self.set_status("Case Briefing: Starting document analysis...")

        self._briefing_queue = Queue()
        self._briefing_worker = BriefingWorker(
            documents=self.processing_results, ui_queue=self._briefing_queue
        )
        self._briefing_worker.start()
        self._poll_briefing_queue()

    def _poll_briefing_queue(self):
        """Poll the briefing worker queue for results."""
        try:
            while True:
                msg_type, data = self._briefing_queue.get_nowait()
                if msg_type == "briefing_progress":
                    _phase, _current, _total, message = data
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

        if self._briefing_worker and self._briefing_worker.is_alive():
            self.after(100, self._poll_briefing_queue)
        else:
            try:
                while True:
                    msg_type, data = self._briefing_queue.get_nowait()
                    if msg_type == "briefing_complete":
                        self._on_briefing_complete(data)
                        return
            except Empty:
                pass
            self._on_briefing_complete(None)

    def _on_briefing_complete(self, briefing_data: dict | None):
        """Handle briefing generation completion."""
        self._completed_tasks.add("qa")

        if briefing_data and briefing_data.get("formatted"):
            formatted = briefing_data["formatted"]
            result = briefing_data.get("result")

            self._briefing_result = result
            self._formatted_briefing = formatted

            self.output_display.update_outputs(
                briefing_text=formatted.text, briefing_sections=formatted.sections
            )

            time_str = f"{result.total_time_seconds:.1f}s" if result else ""
            self.set_status(f"Case Briefing complete ({time_str})")
        else:
            self.set_status("Briefing generation failed")

        if self._pending_tasks.get("summary"):
            self._start_summary_task()
        else:
            self._finalize_tasks()

    # =========================================================================
    # Task Completion
    # =========================================================================

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

        # Show Export buttons after successful processing
        if success and not self._export_all_visible:
            self.export_all_btn.pack(side="right", padx=10, pady=3)
            self._export_all_visible = True
        if success and not self._combined_report_visible:
            self.combined_report_btn.pack(side="right", padx=5, pady=3)
            self._combined_report_visible = True

        # Update session stats
        if success:
            extraction_stats = self._gather_extraction_stats()
            self._update_session_stats(extraction_stats)

        self.set_status(message)

    def _gather_extraction_stats(self) -> dict:
        """Gather extraction statistics after task completion."""
        stats = {}

        vocab_data = getattr(self.output_display, "_vocab_csv_data", None)
        if vocab_data:
            stats["vocab_count"] = len(vocab_data)
            stats["person_count"] = sum(
                1 for v in vocab_data if v.get("Is Person", "").lower() in ("yes", "true", "1")
            )

        if self._qa_results:
            stats["qa_count"] = len(self._qa_results)

        if self._processing_start_time:
            stats["processing_time"] = time.time() - self._processing_start_time

        return stats

    # =========================================================================
    # Follow-up Questions
    # =========================================================================

    def _ask_followup(self):
        """Ask a follow-up question using the Q&A system (async version)."""
        import queue

        question = self.followup_entry.get().strip()
        if not question:
            return

        if not self._vector_store_path or not self._embeddings:
            messagebox.showwarning(
                "Questions Not Ready",
                "Question system is not initialized yet.\n\n"
                "To ask questions:\n"
                "1. Add document files\n"
                "2. Ensure the 'Ask Questions' checkbox is checked\n"
                "3. Click 'Perform Tasks'\n"
                "4. Wait for 'Questions and answers ready' status message",
            )
            return

        # Prevent duplicate submissions
        if (
            hasattr(self, "_followup_thread")
            and self._followup_thread is not None
            and self._followup_thread.is_alive()
        ):
            debug_log("[MainWindow] Follow-up already in progress, ignoring")
            return

        # Clear entry and disable controls
        self.followup_entry.delete(0, "end")
        self.followup_btn.configure(state="disabled", text="Asking...")
        self.followup_entry.configure(state="disabled")

        self.set_status(f"Asking: {question[:40]}...")

        self._followup_queue = queue.Queue()

        def run_followup():
            try:
                from src.core.qa import QAOrchestrator

                orchestrator = QAOrchestrator(
                    vector_store_path=self._vector_store_path,
                    embeddings=self._embeddings,
                    answer_mode="extraction",
                )
                result = orchestrator.ask_followup(question)
                self._followup_queue.put(("success", result))
            except Exception as e:
                self._followup_queue.put(("error", str(e)))
                debug_log(f"[MainWindow] Follow-up thread error: {e}")

        self._followup_thread = threading.Thread(target=run_followup, daemon=True)
        self._followup_thread.start()
        self._poll_followup_result()

    def _poll_followup_result(self):
        """Poll for follow-up result from background thread."""
        import queue

        try:
            msg_type, data = self._followup_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_followup_result)
            return

        # Re-enable controls
        self.followup_btn.configure(state="normal", text="Ask")
        self.followup_entry.configure(state="normal")
        self.followup_entry.focus()

        try:
            if msg_type == "success" and data is not None:
                with self._qa_results_lock:
                    self._qa_results.append(data)
                    self.output_display.update_outputs(qa_results=self._qa_results)
                answer_len = len(data.quick_answer) if data.quick_answer else 0
                self.set_status(f"Follow-up answered: {answer_len} chars")
                debug_log("[MainWindow] Follow-up result displayed successfully")
            elif msg_type == "error":
                self.set_status("Follow-up failed")
                messagebox.showerror("Error", f"Failed to process follow-up: {data}")
        except Exception as e:
            debug_log(f"[MainWindow] Error processing follow-up result: {e}")
            self.set_status("Follow-up error - check logs")
            messagebox.showerror("Error", f"Error displaying result: {e!s}")

    def _ask_followup_for_qa_panel(self, question: str):
        """
        Ask a follow-up question from the QAPanel widget.

        This runs in a background thread (from QAPanel._submit_followup).
        Do NOT call GUI methods here.

        Args:
            question: The follow-up question text

        Returns:
            QAResult object with the answer, or None on error
        """
        if not question:
            return None

        if not self._vector_store_path or not self._embeddings:
            debug_log("[MainWindow] Follow-up unavailable: no vector store or embeddings")
            return None

        try:
            from src.core.qa import QAOrchestrator

            orchestrator = QAOrchestrator(
                vector_store_path=self._vector_store_path,
                embeddings=self._embeddings,
                answer_mode="extraction",
            )

            result = orchestrator.ask_followup(question)

            with self._qa_results_lock:
                self._qa_results.append(result)

            debug_log(f"[MainWindow] Follow-up answered: {len(result.answer)} chars")
            return result

        except Exception as e:
            debug_log(f"[MainWindow] Follow-up error: {e}")
            return None
