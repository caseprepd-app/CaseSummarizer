"""
Task Execution Helpers.

Standalone utility methods extracted from main_window.py.
These are called directly by MainWindow, not inherited as a mixin.

Contains:
- Task state management (task count, button state)
- Queue message handling
- Vocabulary extraction workflow
- Progressive extraction (NER + LLM)
- Q&A task execution
- Follow-up question handling
"""

import logging
import time
from tkinter import messagebox

logger = logging.getLogger(__name__)

# Placeholder text shown while waiting for Q&A answer
PENDING_ANSWER_TEXT = "Searching documents..."


class TaskMixin:
    """
    Mixin class providing task execution functionality.

    Requires parent class to have:
    - self.processing_results: List of processing result dicts
    - self.qa_check, self.vocab_check, self.summary_check: Checkbox vars
    - self.ask_default_questions_check: Default questions checkbox var
    - self.generate_btn: Generate button
    - self.followup_btn: Follow-up button
    - self.followup_entry: Follow-up entry widget
    - self.output_display: DynamicOutputWidget
    - self._worker_manager: WorkerProcessManager for subprocess communication
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

        # Update task preview label
        self._update_task_preview()

    def _update_task_preview(self):
        """
        Update the task preview label to show what will run.

        Shows a concise preview like:
        "Will run: Vocabulary (NER+LLM), Q&A (6 questions)"
        """
        parts = []

        # Vocabulary task
        if self.vocab_check.get():
            from src.user_preferences import get_user_preferences

            if get_user_preferences().is_vocab_llm_enabled():
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
            parts.append("Summary")

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

        # Show pipeline indicator with enabled steps
        if hasattr(self, "pipeline_indicator"):
            enabled_steps = ["Extract"]  # Always starts with extraction
            if do_vocab:
                enabled_steps.append("Vocabulary")
            if do_qa:
                enabled_steps.append("Q&A")
            if do_summary:
                enabled_steps.append("Summary")
            self.pipeline_indicator.set_enabled_steps(enabled_steps)
            self.pipeline_indicator.set_step_state("Extract", "active")
            if not self._pipeline_indicator_visible:
                self.pipeline_indicator.pack(fill="x", padx=10, pady=(0, 5), before=self.main_frame)
                self._pipeline_indicator_visible = True

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
    # Progressive Extraction
    # =========================================================================

    def _start_progressive_extraction(self):
        """
        Start progressive three-phase extraction.

        Phase 1 (NER): Fast, displays results in ~5 seconds
        Phase 2 (Q&A): Builds vector store, enables Q&A panel
        Phase 3 (LLM): Slow enhancement, updates table progressively
        """
        from src.config import (
            LEGAL_EXCLUDE_LIST_PATH,
            MEDICAL_TERMS_LIST_PATH,
            USER_VOCAB_EXCLUDE_PATH,
        )
        from src.services import DocumentService

        self.set_status("Starting extraction (NER first, then LLM enhancement)...")

        # Combine text from all processed documents
        doc_service = DocumentService()
        combined_text = doc_service.combine_document_texts(self.processing_results)

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

        # Calculate aggregate document confidence
        doc_confidence = self._calculate_aggregate_confidence(self.processing_results)
        logger.debug("Aggregate document confidence: %.1f%%", doc_confidence)

        # Read LLM preference directly from settings (no main-window checkbox)
        from src.user_preferences import get_user_preferences

        use_llm = get_user_preferences().is_vocab_llm_enabled()
        logger.debug("LLM extraction from preference: %s", use_llm)

        # Send extraction command to worker subprocess
        ask_defaults = bool(self.qa_check.get() and self.ask_default_questions_check.get())
        self._worker_manager.send_command(
            "extract",
            {
                "documents": self.processing_results,
                "combined_text": combined_text,
                "exclude_list_path": str(LEGAL_EXCLUDE_LIST_PATH),
                "medical_terms_path": str(MEDICAL_TERMS_LIST_PATH),
                "user_exclude_path": str(USER_VOCAB_EXCLUDE_PATH),
                "doc_confidence": doc_confidence,
                "use_llm": use_llm,
                "ask_default_questions": ask_defaults,
            },
        )

        # Ensure queue polling is running
        if self._queue_poll_id:
            self.after_cancel(self._queue_poll_id)
        self._poll_queue()

    # =========================================================================
    # Q&A Task
    # =========================================================================

    def _start_qa_task(self):
        """Start Q&A task via worker subprocess."""
        self.set_status(
            "Questions and answers: Loading embeddings model (this may take a moment)..."
        )

        # Send run_qa command to worker subprocess
        self._worker_manager.send_command(
            "run_qa",
            {
                "answer_mode": "extraction",
            },
        )

        # Ensure queue polling is active
        if self._queue_poll_id:
            self.after_cancel(self._queue_poll_id)
        self._poll_queue()

    def _on_qa_complete(self, qa_results: list):
        """Handle Q&A completion."""
        self._completed_tasks.add("qa")
        self._qa_results = qa_results

        if qa_results:
            self.output_display.update_outputs(qa_results=qa_results)
            self.set_status(f"Questions and answers: {len(qa_results)} questions answered")
            # Enable follow-up question controls
            self.followup_btn.configure(state="normal")
            self.followup_entry.configure(state="normal")
            self.followup_entry.configure(placeholder_text="Type your question here...")
        else:
            self.set_status("Questions and answers complete (no results)")

        if self._pending_tasks.get("summary"):
            self._start_summary_task()
        else:
            self._finalize_tasks()

    # =========================================================================
    # Summary Task
    # =========================================================================

    def _start_summary_task(self):
        """Start summary generation task."""
        if hasattr(self, "pipeline_indicator"):
            self.pipeline_indicator.set_step_state("Summary", "active")
        self.set_status("Summary: This feature can take several hours...")

        self._completed_tasks.add("summary")
        self.output_display.update_outputs(
            meta_summary="Summary generation can take several hours without a dedicated GPU. "
            "For quick case familiarization, use Q&A instead."
        )
        self._finalize_tasks()

    # =========================================================================
    # Task Completion
    # =========================================================================

    def _finalize_tasks(self):
        """Finalize all tasks and update UI."""
        # Mark remaining pipeline steps as done
        if hasattr(self, "pipeline_indicator"):
            for step in ["Extract", "Vocabulary", "Q&A", "Summary"]:
                if self.pipeline_indicator._step_states.get(step) == "active":
                    self.pipeline_indicator.set_step_state(step, "done")
        completed = len(self._completed_tasks)
        self._on_tasks_complete(True, f"Completed {completed} task(s)")

    def _on_tasks_complete(self, success: bool, message: str):
        """Handle task completion."""
        self._stop_timer()

        # Re-enable controls
        self.add_files_btn.configure(state="normal")
        self._update_generate_button_state()

        # Enable follow-up if Q&A was run AND vector store is ready
        if self.qa_check.get() and success and self._qa_ready:
            self.followup_btn.configure(state="normal")
            self.followup_entry.configure(
                state="normal", placeholder_text="Ask a follow-up question..."
            )

        # Show Export All button after successful processing
        if success and not self._export_all_visible:
            self.export_all_btn.pack(side="right", padx=10, pady=3)
            self._export_all_visible = True

        # Update session stats
        if success:
            extraction_stats = self._gather_extraction_stats()
            self._update_session_stats(extraction_stats)

        # Success celebration: brief green flash on status bar
        if success:
            from src.ui.theme import COLORS

            self.status_frame.configure(fg_color=COLORS["monitor_bg"])
            self.status_label.configure(text=f"\u2713 {message}", text_color=COLORS["success"])
            # Restore normal status bar after 2 seconds
            self.after(
                2000,
                lambda: self._restore_status_bar_color(message),
            )
        else:
            self.set_status(message)

    def _restore_status_bar_color(self, message: str):
        """Restore status bar to normal colors after success celebration."""
        from src.ui.theme import COLORS

        self.status_frame.configure(fg_color=COLORS["status_bar_bg"])
        self.status_label.configure(text=message, text_color=COLORS["text_secondary"])

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
        """Ask a follow-up question using progressive retrieval then generation."""

        question = self.followup_entry.get().strip()
        if not question:
            return

        if not self._vector_store_path or not self._qa_ready:
            messagebox.showwarning(
                "Questions Not Ready",
                "Question system is not initialized yet.\n\n"
                "To ask questions:\n"
                "1. Add document files\n"
                "2. Ensure the 'Ask Questions' checkbox is checked\n"
                "3. Click 'Perform Tasks'\n"
                "4. Wait for the search index to finish building",
            )
            return

        # Prevent duplicate submissions
        if getattr(self, "_followup_pending", False):
            logger.debug("Follow-up already in progress, ignoring")
            return

        # Clear entry and disable controls
        self.followup_entry.delete(0, "end")
        self.followup_btn.configure(state="disabled", text="Asking...")
        self.followup_entry.configure(state="disabled")

        self.set_status(f"Searching: {question[:40]}...")

        # Create pending QAResult and add to display immediately
        from src.services import QAService

        QAResult = QAService().get_qa_result_class()
        pending_result = QAResult(
            question=question,
            quick_answer=PENDING_ANSWER_TEXT,
            citation="",
            is_followup=True,
            include_in_export=False,
        )
        with self._qa_results_lock:
            self._qa_results.append(pending_result)
            self._pending_followup_index = len(self._qa_results) - 1
            self.output_display.update_outputs(qa_results=list(self._qa_results))

        # Send follow-up command to worker subprocess
        self._followup_pending = True
        self._worker_manager.send_command("followup", {"question": question})
        self._poll_followup_result()

    def _restore_followup_controls(self):
        """Re-enable follow-up input controls after completion."""
        self.followup_btn.configure(state="normal", text="Ask")
        self.followup_entry.configure(state="normal")
        self.followup_entry.focus()

    def _poll_followup_result(self):
        """Poll for follow-up result from worker subprocess."""
        if self._destroying:
            return

        messages = self._worker_manager.check_for_messages()
        followup_result = None

        for msg in messages:
            try:
                msg_type, data = msg
                if msg_type == "qa_followup_result":
                    followup_result = data
                else:
                    self._handle_queue_message(msg_type, data)
            except (TypeError, ValueError):
                pass

        if followup_result is None:
            self.after(100, self._poll_followup_result)
            return

        self._followup_pending = False
        self._restore_followup_controls()

        try:
            if followup_result is not None and hasattr(followup_result, "quick_answer"):
                with self._qa_results_lock:
                    pending_idx = getattr(self, "_pending_followup_index", None)
                    if pending_idx is not None and pending_idx < len(self._qa_results):
                        self._qa_results[pending_idx] = followup_result
                        self._pending_followup_index = None
                    else:
                        self._qa_results.append(followup_result)
                    self.output_display.update_outputs(qa_results=list(self._qa_results))
                answer_len = (
                    len(followup_result.quick_answer) if followup_result.quick_answer else 0
                )
                self.set_status(f"Follow-up answered: {answer_len} chars")
            else:
                with self._qa_results_lock:
                    pending_idx = getattr(self, "_pending_followup_index", None)
                    if pending_idx is not None and pending_idx < len(self._qa_results):
                        if self._qa_results[pending_idx].quick_answer == PENDING_ANSWER_TEXT:
                            self._qa_results.pop(pending_idx)
                    self._pending_followup_index = None
                    self.output_display.update_outputs(qa_results=list(self._qa_results))
                self.set_status("Follow-up failed")
        except Exception as e:
            logger.debug("Error processing follow-up result: %s", e)
            self.set_status("Follow-up error - check logs")

    def _ask_followup_for_qa_panel(self, question: str):
        """
        Ask a follow-up question from the QAPanel widget.

        This runs in a background thread (from QAPanel._submit_followup).
        Sends command to worker subprocess and polls synchronously.

        Args:
            question: The follow-up question text

        Returns:
            QAResult object with the answer, or None on error
        """
        if not question:
            return None

        if not self._vector_store_path or not self._qa_ready:
            logger.debug("Follow-up unavailable: no vector store or Q&A not ready")
            return None

        try:
            import time

            self._worker_manager.send_command("followup", {"question": question})

            timeout = 120
            start = time.time()
            while time.time() - start < timeout:
                messages = self._worker_manager.check_for_messages()
                for msg in messages:
                    try:
                        msg_type, data = msg
                        if msg_type == "qa_followup_result":
                            if data is not None:
                                with self._qa_results_lock:
                                    self._qa_results.append(data)
                                logger.debug("Follow-up answered: %s chars", len(data.answer))
                                return data
                            return None
                        self._worker_manager.result_queue.put(msg)
                    except (TypeError, ValueError):
                        pass
                time.sleep(0.1)

            logger.warning("Follow-up timed out after %ss", timeout)
            return None

        except Exception as e:
            logger.debug("Follow-up error: %s", e)
            return None
