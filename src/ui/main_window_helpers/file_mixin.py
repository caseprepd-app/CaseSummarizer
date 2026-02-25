"""
File Management Mixin.

Contains:
- File selection dialog
- Drag-and-drop support
- Preprocessing workflow
- Queue handling for file processing
"""

import logging
from pathlib import Path
from tkinter import filedialog

logger = logging.getLogger(__name__)

# Try to import tkinterdnd2 for drag-and-drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    HAS_DND = True
except ImportError:
    HAS_DND = False


class FileMixin:
    """
    Mixin class providing file management functionality.

    Requires parent class to have:
    - self.selected_files: List of file paths
    - self.processing_results: List of processing result dicts
    - self.file_table: File table widget
    - self.add_files_btn: Add files button
    - self.generate_btn: Generate button
    - self._worker_manager: WorkerProcessManager for subprocess communication
    """

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

            logger.debug("Drag-drop enabled on file table area")

        except Exception as e:
            logger.debug("Failed to initialize drag-drop: %s", e)

    def _on_file_drop(self, event):
        """
        Handle files dropped onto the file table area.

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
                    end = raw_data.index("}", i)
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
        self._update_generate_button_state()
        self._update_session_stats()  # Clear stats display
        self.set_status("Files cleared")

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

    def _start_preprocessing(self):
        """Start preprocessing selected files."""
        if not self.selected_files:
            return

        # Check OCR availability before starting (runs once per batch)
        ocr_allowed = self._check_ocr_availability()

        # Disable controls during preprocessing
        self.add_files_btn.configure(state="disabled")
        self.generate_btn.configure(state="disabled")

        # Clear previous results
        self.file_table.clear()
        self.processing_results.clear()

        # Start timer
        self._start_timer()

        # Send command to worker subprocess
        self._worker_manager.send_command(
            "process_files",
            {
                "file_paths": self.selected_files,
                "ocr_allowed": ocr_allowed,
            },
        )
        self._preprocessing_active = True

        # Start polling the queue
        self._poll_queue()

    def _poll_queue(self):
        """Poll the worker subprocess result queue for messages."""
        if self._destroying:
            return

        messages = self._worker_manager.check_for_messages()
        for msg in messages:
            try:
                msg_type, data = msg
                self._handle_queue_message(msg_type, data)
            except (TypeError, ValueError):
                logger.warning("Invalid message from worker subprocess: %s", msg)

        if self._processing_active or self._preprocessing_active or messages:
            self._queue_poll_id = self.after(33, self._poll_queue)
        else:
            self._queue_poll_id = None

    def _on_preprocessing_complete(self, results: list[dict]):
        """Handle preprocessing completion."""
        # Stop timer
        self._stop_timer()

        # Stop queue polling
        if self._queue_poll_id:
            self.after_cancel(self._queue_poll_id)
            self._queue_poll_id = None

        # Clear preprocessing flag so _update_generate_button_state sees correct state
        self._preprocessing_active = False

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
