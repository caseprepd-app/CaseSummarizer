"""
Processing Timer Module

Provides a visible timer widget and CSV logging for processing metrics.
The timer displays elapsed time during document processing and AI generation.
Metrics are logged to CSV for future ML-based duration prediction.

CSV columns logged:
- timestamp: ISO format datetime when job completed
- duration_seconds: Total processing time
- document_count: Number of documents processed
- total_pages: Sum of all page counts
- total_size_bytes: Sum of all file sizes
- avg_pages: Average pages per document
- avg_size_bytes: Average file size
- model_name: Ollama model used
- outputs_requested: Comma-separated list of requested outputs
"""

import csv
import time
from datetime import datetime
from pathlib import Path

import customtkinter as ctk

from src.config import PROCESSING_METRICS_CSV
from src.logging_config import debug_log
from src.ui.theme import FONTS


def format_duration(seconds: float) -> str:
    """
    Format seconds into human-readable duration string.

    This is a module-level utility function for use across the codebase
    whenever durations need to be displayed to users.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string like "45s", "2m 30s", or "1h 23m 45s"

    Examples:
        >>> format_duration(45)
        '45s'
        >>> format_duration(150)
        '2m 30s'
        >>> format_duration(3725)
        '1h 2m 5s'
    """
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class ProcessingTimer(ctk.CTkLabel):
    """
    A label widget that displays elapsed time during processing.

    Updates every second while active, showing time in MM:SS or HH:MM:SS format.
    Also handles logging processing metrics to CSV when processing completes.

    Attributes:
        start_time: Unix timestamp when timer started
        is_running: Whether the timer is currently active
        _after_id: ID of the scheduled after() callback for cancellation
    """

    def __init__(self, master, **kwargs):
        """
        Initialize the processing timer.

        Args:
            master: Parent widget
            **kwargs: Additional arguments passed to CTkLabel
        """
        # Set default styling for visibility
        kwargs.setdefault("text", "")
        kwargs.setdefault("font", FONTS["heading"])
        kwargs.setdefault("text_color", "#00D4FF")  # Cyan - visible on dark bg

        super().__init__(master, **kwargs)

        self.start_time: float | None = None
        self.is_running = False
        self._after_id: str | None = None

        # Job metadata for CSV logging
        self._job_metadata: dict | None = None

    def start(self, job_metadata: dict | None = None):
        """
        Start the timer and store job metadata for later logging.

        Args:
            job_metadata: Dictionary with job info for CSV logging:
                - document_count: Number of documents
                - documents: List of document dicts with page_count, file_size
                - model_name: Ollama model being used
                - outputs_requested: List of output types requested
        """
        self.start_time = time.time()
        self.is_running = True
        self._job_metadata = job_metadata or {}

        debug_log("[TIMER] Started processing timer")
        self._update_display()

    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.

        Returns:
            Elapsed time in seconds (0 if timer wasn't running)
        """
        if not self.is_running:
            return 0.0

        self.is_running = False

        # Cancel any pending update
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None

        elapsed = time.time() - self.start_time if self.start_time else 0.0
        debug_log(f"[TIMER] Stopped after {elapsed:.1f}s")

        return elapsed

    def stop_and_log(self) -> float:
        """
        Stop the timer and log metrics to CSV.

        Returns:
            Elapsed time in seconds
        """
        elapsed = self.stop()

        if elapsed > 0 and self._job_metadata:
            self._log_metrics(elapsed)

        return elapsed

    def reset(self):
        """Reset the timer display and clear state."""
        self.stop()
        self.start_time = None
        self._job_metadata = None
        self.configure(text="")

    def get_elapsed(self) -> float:
        """
        Get current elapsed time without stopping.

        Returns:
            Elapsed time in seconds (0 if not running)
        """
        if not self.is_running or not self.start_time:
            return 0.0
        return time.time() - self.start_time

    def _update_display(self):
        """Update the timer display with current elapsed time."""
        if not self.is_running:
            return

        elapsed = self.get_elapsed()
        self.configure(text=self._format_time(elapsed))

        # Schedule next update in 1 second
        self._after_id = self.after(1000, self._update_display)

    def _format_time(self, seconds: float) -> str:
        """
        Format seconds into human-readable time string.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted string like "0:45" or "1:23:45"
        """
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60

        if hours > 0:
            return f"⏱ {hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"⏱ {minutes}:{secs:02d}"

    def _log_metrics(self, duration_seconds: float):
        """
        Log processing metrics to CSV file.

        Args:
            duration_seconds: Total processing time
        """
        try:
            metadata = self._job_metadata
            documents = metadata.get("documents", [])

            # Calculate aggregates
            total_pages = sum(d.get("page_count", 0) or 0 for d in documents)
            total_size = sum(d.get("file_size", 0) or 0 for d in documents)
            doc_count = len(documents)

            avg_pages = total_pages / doc_count if doc_count > 0 else 0
            avg_size = total_size / doc_count if doc_count > 0 else 0

            outputs = metadata.get("outputs_requested", [])
            outputs_str = ",".join(outputs) if outputs else ""

            row = {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": round(duration_seconds, 2),
                "document_count": doc_count,
                "total_pages": total_pages,
                "total_size_bytes": total_size,
                "avg_pages": round(avg_pages, 1),
                "avg_size_bytes": round(avg_size, 0),
                "model_name": metadata.get("model_name", "unknown"),
                "outputs_requested": outputs_str,
            }

            # Write to CSV (create with headers if doesn't exist)
            csv_path = Path(PROCESSING_METRICS_CSV)
            file_exists = csv_path.exists()

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

            debug_log(
                f"[TIMER] Logged metrics to {csv_path}: "
                f"{doc_count} docs, {total_pages} pages, {duration_seconds:.1f}s"
            )

        except Exception as e:
            debug_log(f"[TIMER] Error logging metrics: {e}")
