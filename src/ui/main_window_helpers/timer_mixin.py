"""
Timer Management Mixin.

Session 82: Extracted from main_window.py for modularity.

Contains:
- Processing timer display
- Activity indicator management
- Aggregate confidence calculation
"""

import time

from src.logging_config import debug_log


class TimerMixin:
    """
    Mixin class providing timer functionality.

    Requires parent class to have:
    - self._processing_start_time: Float timestamp or None
    - self._timer_after_id: After ID for timer updates
    - self.timer_label: CTkLabel for timer display
    - self.activity_indicator: Activity indicator widget
    - self._activity_indicator_visible: Boolean visibility state
    """

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
        debug_log(f"[MainWindow] Document confidences: {confidences} -> min={min_conf:.1f}%")
        return min_conf
