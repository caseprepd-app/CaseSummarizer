"""
Queue Message Handler Module

Routes inter-thread messages from worker threads to appropriate UI update handlers.
This module is responsible for UI updates ONLY - workflow orchestration logic
is delegated to WorkflowOrchestrator.

Design Principle: Single Responsibility
- QueueMessageHandler: Routes messages and updates UI widgets
- WorkflowOrchestrator: Decides what workflow steps to execute next

Message Types Handled:
- progress: Update progress bar and status label
- file_processed: Add/update file in results table
- meta_summary_generated: Display generated meta-summary
- vocab_csv_generated: Store vocabulary CSV data
- processing_finished: Delegate to orchestrator, then update UI
- summary_result: Display AI-generated summary
- error: Show error dialog and reset UI
- ner_complete: Display initial NER vocabulary results (Session 48)
- qa_ready: Enable Q&A panel after vector store built (Session 48)
- llm_progress: Track LLM chunk processing progress (Session 48)
- llm_complete: Display reconciled NER+LLM results (Session 48)

Performance Optimizations (Session 14):
- Explicit garbage collection after processing completes
- Worker reference cleanup to prevent memory leaks
"""

import gc
from tkinter import messagebox
from src.logging_config import debug_log
from src.ui.processing_timer import format_duration
from src.ui.queue_messages import MessageType


class QueueMessageHandler:
    """
    Routes queue messages to appropriate UI update handlers.

    This class encapsulates all UI update logic for worker thread messages,
    making it easier to test and maintain message-handling behavior.

    For workflow orchestration (deciding what to do next), see WorkflowOrchestrator.

    Attributes:
        main_window: Reference to MainWindow instance
        orchestrator: Reference to WorkflowOrchestrator instance
    """

    def __init__(self, main_window):
        """
        Initialize the message handler.

        Args:
            main_window: Reference to MainWindow instance (for widget updates)
        """
        self.main_window = main_window
        # Orchestrator is set by main_window after construction
        self.orchestrator = None

    def set_orchestrator(self, orchestrator):
        """
        Set the workflow orchestrator reference.

        Called by MainWindow after both handler and orchestrator are created.

        Args:
            orchestrator: WorkflowOrchestrator instance
        """
        self.orchestrator = orchestrator

    def handle_progress(self, data):
        """
        Handle 'progress' message - update progress bar and status.

        Args:
            data: Tuple of (percentage: int, message: str)
        """
        percentage, message = data
        self.main_window.progress_bar.set(percentage / 100.0)
        self.main_window.status_label.configure(text=message)

    def handle_file_processed(self, data):
        """
        Handle 'file_processed' message - update file table with result.

        Args:
            data: Result dictionary with filename, status, confidence, etc.
        """
        self.main_window.processed_results.append(data)
        self.main_window.file_table.add_result(data)

        # Display individual document summary if available
        if data.get("summary"):
            self.main_window.summary_results.update_outputs(
                document_summaries={data["filename"]: data["summary"]}
            )

    def handle_meta_summary_generated(self, data):
        """
        Handle 'meta_summary_generated' message - display meta-summary.

        Args:
            data: The meta-summary text string
        """
        self.main_window.summary_results.update_outputs(meta_summary=data)

    def handle_vocab_csv_generated(self, data):
        """
        Handle 'vocab_csv_generated' message - store vocabulary data.

        Args:
            data: List of vocabulary term dictionaries
        """
        self.main_window.summary_results.update_outputs(vocab_csv_data=data)
        if self.orchestrator:
            self.orchestrator.on_vocab_complete()

    def handle_processing_finished(self, data):
        """
        Handle 'processing_finished' message - document extraction complete.

        Delegates workflow decisions to the orchestrator, then updates UI
        based on what actions were taken.

        Args:
            data: List of extracted document result dictionaries
        """
        extracted_documents = data

        if self.orchestrator and self.main_window.pending_ai_generation:
            # Delegate to orchestrator for workflow decisions
            actions = self.orchestrator.on_extraction_complete(
                extracted_documents, self.main_window.pending_ai_generation
            )

            # If workflow completed without AI (shouldn't happen normally)
            if actions.get("workflow_complete"):
                self._reset_ui_after_processing()
                self.main_window.status_label.configure(text="Processing complete.")
        elif self.main_window.pending_ai_generation:
            # Fallback if orchestrator not set (backward compatibility)
            debug_log("[QUEUE HANDLER] WARNING: Orchestrator not set, using fallback.")
            self.main_window._start_ai_generation(
                extracted_documents, self.main_window.pending_ai_generation
            )
        else:
            # No AI generation requested
            self._reset_ui_after_processing()
            self.main_window.status_label.configure(text="Processing complete.")

    def handle_summary_result(self, data):
        """
        Handle 'summary_result' message - AI summary generated (single doc).

        Args:
            data: Dictionary with 'summary' key containing the generated text
        """
        debug_log(f"[QUEUE HANDLER] Summary result received.")
        self.main_window.summary_results.update_outputs(meta_summary=data.get("summary", ""))
        self.main_window.progress_bar.set(1.0)
        self.main_window.status_label.configure(text="Summary generation complete!")
        self._reset_ui_after_processing()
        self.main_window.pending_ai_generation = None

        if self.orchestrator:
            self.orchestrator.on_summary_complete()

    def handle_multi_doc_result(self, data):
        """
        Handle 'multi_doc_result' message - multi-document summarization complete.

        Displays both individual document summaries and the combined meta-summary.

        Args:
            data: MultiDocumentSummaryResult with individual_summaries and meta_summary
        """
        # Import here to avoid circular imports
        from src.core.summarization import MultiDocumentSummaryResult

        debug_log(
            f"[QUEUE HANDLER] Multi-doc result received: "
            f"{data.documents_processed} processed, {data.documents_failed} failed"
        )

        # Extract individual summaries as dict[filename, summary_text]
        individual_summaries = {}
        for filename, result in data.individual_summaries.items():
            if result.success:
                individual_summaries[filename] = result.summary
            else:
                individual_summaries[filename] = f"[Error: {result.error_message}]"

        # Update UI with both individual summaries and meta-summary
        self.main_window.summary_results.update_outputs(
            meta_summary=data.meta_summary, document_summaries=individual_summaries
        )

        # Update progress and status
        self.main_window.progress_bar.set(1.0)
        duration_str = format_duration(data.total_processing_time_seconds)
        status_msg = (
            f"Multi-document summarization complete! "
            f"{data.documents_processed} documents in {duration_str}"
        )
        if data.documents_failed > 0:
            status_msg += f" ({data.documents_failed} failed)"
        self.main_window.status_label.configure(text=status_msg)

        # Reset UI
        self._reset_ui_after_processing()
        self.main_window.pending_ai_generation = None

        # Clean up worker reference
        if self.orchestrator and hasattr(self.orchestrator, "multi_doc_worker"):
            self.orchestrator.multi_doc_worker = None

        if self.orchestrator:
            self.orchestrator.on_summary_complete()

    def handle_error(self, error_message):
        """
        Handle 'error' message - show error dialog and reset UI.

        Args:
            error_message: Description of the error to display
        """
        messagebox.showerror("Processing Error", error_message)
        self._reset_ui_after_processing()
        self.main_window.pending_ai_generation = None

    def _reset_ui_after_processing(self):
        """Reset UI buttons and progress bar to post-processing state."""
        debug_log("[QUEUE HANDLER] Resetting UI after processing complete...")

        # Stop timer and log metrics to CSV
        if hasattr(self.main_window, "processing_timer"):
            # Update document metadata with actual page counts from processed results
            if self.main_window.processing_timer._job_metadata:
                docs_meta = self.main_window.processing_timer._job_metadata.get("documents", [])
                for doc_meta in docs_meta:
                    # Find matching processed result to get actual page count
                    for result in self.main_window.processed_results:
                        if result.get("filename") == doc_meta.get("filename"):
                            doc_meta["page_count"] = result.get("page_count", 0)
                            break

            self.main_window.processing_timer.stop_and_log()

        self.main_window.select_files_btn.configure(state="normal")
        self.main_window.generate_outputs_btn.configure(state="normal")
        self.main_window.output_options.unlock_controls()  # Unlock slider and checkboxes

        # Reset button text from "Generating..." back to normal
        self.main_window.output_options.set_generating_state(False)

        # Disable cancel button (grey it out instead of hiding)
        self.main_window.cancel_btn.configure(
            state="disabled", fg_color="#6c757d", hover_color="#5a6268"  # Grey when disabled
        )
        debug_log("[QUEUE HANDLER] Cancel button disabled (greyed out)")

        self.main_window.progress_bar.grid_remove()

        # Clear worker references to allow garbage collection
        self.main_window.worker = None
        if self.orchestrator:
            self.orchestrator.vocab_worker = None

        # Force garbage collection after heavy processing
        gc.collect()
        debug_log("[QUEUE HANDLER] Worker references cleared, garbage collected.")

        # Force immediate UI update to ensure button disappears
        self.main_window.update_idletasks()
        debug_log("[QUEUE HANDLER] UI reset complete.")

    # =========================================================================
    # Vector Store Q&A Handlers (Session 24)
    # =========================================================================

    def handle_vector_store_ready(self, data: dict):
        """
        Handle 'vector_store_ready' message - vector store is ready for Q&A.

        Notifies the orchestrator and updates UI to enable Q&A tab.

        Args:
            data: Dictionary with 'path', 'case_id', 'chunk_count', 'creation_time_ms'
        """
        debug_log(
            f"[QUEUE HANDLER] Vector store ready: {data.get('case_id')} "
            f"({data.get('chunk_count')} chunks)"
        )

        # Update orchestrator state
        if self.orchestrator:
            self.orchestrator.on_vector_store_complete(data)

        # Update status to indicate Q&A is available
        chunk_count = data.get("chunk_count", 0)
        creation_time = data.get("creation_time_ms", 0)
        self.main_window.status_label.configure(
            text=f"Q&A Ready ({chunk_count} chunks indexed in {creation_time:.0f}ms)"
        )

        # Store vector store info on main window for Q&A access
        self.main_window.vector_store_path = data.get("path")
        self.main_window.vector_store_case_id = data.get("case_id")

        debug_log("[QUEUE HANDLER] Vector store handler complete - Q&A available")

    def handle_vector_store_error(self, data: dict):
        """
        Handle 'vector_store_error' message - vector store creation failed.

        Logs the error but doesn't show a modal (Q&A is optional).

        Args:
            data: Dictionary with 'error' message
        """
        error_msg = data.get("error", "Unknown error")
        debug_log(f"[QUEUE HANDLER] Vector store error: {error_msg}")

        # Update status to indicate Q&A is not available
        self.main_window.status_label.configure(text="Q&A unavailable (indexing failed)")

        # Mark as not available
        self.main_window.vector_store_path = None
        self.main_window.vector_store_case_id = None

    # =========================================================================
    # Q&A Handlers (Session 28)
    # =========================================================================

    def handle_qa_progress(self, data: tuple):
        """
        Handle 'qa_progress' message - Q&A question being processed.

        Args:
            data: Tuple of (current, total, question_text)
        """
        current, total, question = data
        debug_log(f"[QUEUE HANDLER] Q&A progress: {current}/{total} - {question[:50]}...")
        self.main_window.status_label.configure(
            text=f"Q&A: Processing question {current}/{total}..."
        )

    def handle_qa_result(self, result):
        """
        Handle 'qa_result' message - single Q&A result ready.

        Args:
            result: QAResult object with question, answer, metadata
        """
        debug_log(f"[QUEUE HANDLER] Q&A result: {result.question[:40]}...")
        # Results are accumulated by QAWorker, just update status
        self.main_window.status_label.configure(
            text=f"Q&A: Got answer for '{result.question[:30]}...'"
        )

    def handle_qa_complete(self, results: list):
        """
        Handle 'qa_complete' message - all Q&A processing finished.

        Args:
            results: List of QAResult objects
        """
        result_count = len(results) if results else 0
        debug_log(f"[QUEUE HANDLER] Q&A complete: {result_count} results")

        # Update status
        self.main_window.status_label.configure(
            text=f"Q&A Complete: {result_count} questions answered"
        )

        # Store results on main window for display
        self.main_window.qa_results = results

        # Update the summary results widget to show Q&A results
        if hasattr(self.main_window, "summary_results") and results:
            self.main_window.summary_results.update_outputs(qa_results=results)

        debug_log("[QUEUE HANDLER] Q&A results delivered to UI")

    def handle_qa_followup_result(self, result):
        """
        Handle 'qa_followup_result' message - follow-up question answered.

        Adds the result to the existing Q&A results and updates the QAPanel.

        Args:
            result: QAResult object for the follow-up question
        """
        # Add to existing results
        if not hasattr(self.main_window, "qa_results") or self.main_window.qa_results is None:
            self.main_window.qa_results = []
        self.main_window.qa_results.append(result)

        # Update QAPanel if visible
        if (
            hasattr(self.main_window, "summary_results")
            and self.main_window.summary_results._qa_panel
        ):
            qa_panel = self.main_window.summary_results._qa_panel
            qa_panel.display_results(self.main_window.qa_results)

        self.main_window.status_label.configure(
            text=f"Follow-up answered: {result.question[:30]}..."
        )
        debug_log(f"[QUEUE HANDLER] Follow-up Q&A result delivered: {result.question[:30]}...")

    def handle_qa_error(self, data: dict):
        """
        Handle 'qa_error' message - Q&A processing error.

        Args:
            data: Dictionary with 'error' key containing error message
        """
        error_msg = data.get("error", "Unknown Q&A error")
        self.main_window.status_label.configure(text=f"Q&A Error: {error_msg}")
        debug_log(f"[QUEUE HANDLER] Q&A error: {error_msg}")

    # =========================================================================
    # Progressive Extraction Handlers (Session 48)
    # =========================================================================

    def handle_ner_complete(self, data):
        """
        Handle 'ner_complete' message - Phase 1 local algorithm extraction complete.

        Phase 1 runs NER, RAKE, and BM25 (if corpus available).
        Displays initial vocabulary results immediately while LLM enhancement
        continues in background.

        Args:
            data: List of vocabulary term dictionaries from local algorithms
        """
        term_count = len(data) if data else 0
        debug_log(f"[QUEUE HANDLER] Phase 1 complete: {term_count} terms - displaying immediately")

        # Update vocab display with local algorithm results
        self.main_window.output_display.update_outputs(vocab_csv_data=data)
        self.main_window.output_display.set_extraction_source("ner")

        # Update status
        self.main_window.status_label.configure(
            text=f"Phase 1 complete: {term_count} terms found. LLM enhancement starting..."
        )

    def handle_qa_ready(self, data: dict):
        """
        Handle 'qa_ready' message - Phase 2 Q&A vector store ready.

        Enables Q&A functionality now that the vector store is built.

        Args:
            data: Dictionary with 'vector_store_path', 'embeddings', 'chunk_count'
        """
        chunk_count = data.get("chunk_count", 0)
        debug_log(f"[QUEUE HANDLER] Q&A ready: {chunk_count} chunks indexed")

        # Store vector store info on main window
        self.main_window._vector_store_path = data.get("vector_store_path")
        self.main_window._embeddings = data.get("embeddings")
        self.main_window._qa_ready = True

        # Refresh tabs to show Q&A option now that it's ready (Session 51)
        self.main_window.output_display._refresh_tabs()
        debug_log("[QUEUE HANDLER] Refreshed tabs - Q&A tab should now be accessible")

        # Mark Q&A as complete if it was requested
        if self.main_window._pending_tasks.get("qa"):
            self.main_window._completed_tasks.add("qa")
            self.main_window.followup_btn.configure(state="normal")

        # Update status
        self.main_window.status_label.configure(
            text=f"Q&A ready ({chunk_count} chunks). LLM enhancement in progress..."
        )

    def handle_trigger_default_qa(self, data: dict):
        """
        Handle 'trigger_default_qa' message - Auto-trigger default Q&A questions.

        Checks if default questions checkbox is enabled and spawns QAWorker if so.

        Args:
            data: Dictionary with 'vector_store_path' and 'embeddings'
        """
        # Check if checkbox is enabled
        if not self.main_window.ask_default_questions_check.get():
            # LOG-013: debug_log always exists; just call it directly
            debug_log("[QUEUE HANDLER] Default questions disabled, skipping")
            return

        # Spawn QAWorker with default questions
        from src.ui.workers import QAWorker
        from src.user_preferences import get_user_preferences

        vector_store_path = data["vector_store_path"]
        embeddings = data["embeddings"]

        debug_log("[QUEUE HANDLER] Spawning QAWorker for default questions")
        prefs = get_user_preferences()

        qa_worker = QAWorker(
            vector_store_path=vector_store_path,
            embeddings=embeddings,
            ui_queue=self.main_window._ui_queue,
            answer_mode=prefs.get("qa_answer_mode", "extraction"),
            questions=None,
            use_default_questions=True,
        )

        qa_worker.start()
        debug_log("[QUEUE HANDLER] Default questions worker started")

    def handle_llm_progress(self, data: tuple):
        """
        Handle 'llm_progress' message - Phase 3 LLM chunk processing progress.

        Args:
            data: Tuple of (current_chunk, total_chunks)
        """
        current, total = data
        debug_log(f"[QUEUE HANDLER] LLM progress: {current}/{total}")
        # Status is already updated by 'progress' message, just log here

    def handle_llm_complete(self, data):
        """
        Handle 'llm_complete' message - Phase 3 LLM extraction complete.

        Updates vocabulary display with reconciled NER + LLM results and
        triggers summary task if pending.

        Args:
            data: List of reconciled vocabulary term dictionaries
        """
        term_count = len(data) if data else 0
        debug_log(f"[QUEUE HANDLER] LLM complete: {term_count} reconciled terms")

        # Update vocab display with enhanced results
        self.main_window.output_display.update_outputs(vocab_csv_data=data)
        self.main_window.output_display.set_extraction_source("both")

        # Mark vocab task complete
        self.main_window._completed_tasks.add("vocab")

        # Update status
        self.main_window.status_label.configure(
            text=f"Complete: {term_count} names & vocabulary extracted"
        )

        # Continue to summary if requested, otherwise finalize
        if self.main_window._pending_tasks.get("summary"):
            self.main_window._start_summary_task()
        else:
            self.main_window._finalize_tasks()

    def process_message(self, message_type: str, data) -> bool:
        """
        Route a message to the appropriate handler.

        Args:
            message_type: Type of message (e.g., 'progress', 'error')
            data: Message payload (type varies by message_type)

        Returns:
            True if message was handled successfully, False otherwise
        """
        handlers = {
            MessageType.PROGRESS: self.handle_progress,
            MessageType.FILE_PROCESSED: self.handle_file_processed,
            MessageType.META_SUMMARY_GENERATED: self.handle_meta_summary_generated,
            MessageType.VOCAB_CSV_GENERATED: self.handle_vocab_csv_generated,
            MessageType.PROCESSING_FINISHED: self.handle_processing_finished,
            MessageType.SUMMARY_RESULT: self.handle_summary_result,
            MessageType.MULTI_DOC_RESULT: self.handle_multi_doc_result,
            MessageType.ERROR: self.handle_error,
            # Vector Store Q&A handlers (Session 24)
            MessageType.VECTOR_STORE_READY: self.handle_vector_store_ready,
            MessageType.VECTOR_STORE_ERROR: self.handle_vector_store_error,
            # Q&A handlers (Session 28)
            MessageType.QA_PROGRESS: self.handle_qa_progress,
            MessageType.QA_RESULT: self.handle_qa_result,
            MessageType.QA_COMPLETE: self.handle_qa_complete,
            MessageType.QA_FOLLOWUP_RESULT: self.handle_qa_followup_result,
            MessageType.QA_ERROR: self.handle_qa_error,
            # Progressive Extraction handlers (Session 48)
            MessageType.NER_COMPLETE: self.handle_ner_complete,
            MessageType.QA_READY: self.handle_qa_ready,
            MessageType.LLM_PROGRESS: self.handle_llm_progress,
            MessageType.LLM_COMPLETE: self.handle_llm_complete,
        }

        handler = handlers.get(message_type)
        if handler:
            try:
                handler(data)
                return True
            except Exception as e:
                debug_log(f"[QUEUE HANDLER] Error handling {message_type}: {e}")
                return False

        debug_log(f"[QUEUE HANDLER] Unknown message type: {message_type}")
        return False
