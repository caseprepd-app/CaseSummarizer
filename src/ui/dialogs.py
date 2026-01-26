"""
CasePrepd - Custom Dialogs (CustomTkinter Refactor)
"""

import time

import customtkinter as ctk

from src.ui.theme import FONTS


class ModelLoadProgressDialog(ctk.CTkToplevel):
    """
    Progress dialog for model loading, refactored for CustomTkinter.
    """

    def __init__(self, model_name, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.start_time = time.time()
        self._finished = False

        self.title("Loading Model")
        self.geometry("400x200")
        self.grab_set()  # Make modal

        self.grid_columnconfigure(0, weight=1)

        title_label = ctk.CTkLabel(
            self, text=f"Loading {self.model_name} Model", font=FONTS["heading"]
        )
        title_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        info_label = ctk.CTkLabel(
            self, text="This may take 30-60 seconds depending on your hardware.", wraplength=350
        )
        info_label.grid(row=1, column=0, padx=20, pady=5)

        self.progress_bar = ctk.CTkProgressBar(self, mode="indeterminate")
        self.progress_bar.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.progress_bar.start()

        self.timer_label = ctk.CTkLabel(self, text="Elapsed time: 0.0 seconds")
        self.timer_label.grid(row=3, column=0, padx=20, pady=5)

        self.status_label = ctk.CTkLabel(self, text="Initializing...")
        self.status_label.grid(row=4, column=0, padx=20, pady=5)

        self._update_timer()

    def _update_timer(self):
        """Update the elapsed time display."""
        if self._finished:
            return
        elapsed = time.time() - self.start_time
        self.timer_label.configure(text=f"Elapsed time: {elapsed:.1f} seconds")
        self.after(100, self._update_timer)  # Schedule next update

    def finish_success(self):
        """Mark loading as complete and close dialog."""
        self._finished = True
        self.progress_bar.stop()
        self.progress_bar.set(1)
        self.status_label.configure(text="Model loaded successfully!", text_color="green")
        self.after(1000, self.destroy)  # Auto-close after 1s


class SimpleProgressDialog(ctk.CTkToplevel):
    """
    Simple progress dialog, refactored for CustomTkinter.
    """

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title(title)
        self.geometry("400x150")
        self.grab_set()

        self.grid_columnconfigure(0, weight=1)

        self.message_label = ctk.CTkLabel(self, text="Starting...", wraplength=350)
        self.message_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.progress_bar = ctk.CTkProgressBar(self, mode="determinate")
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

    def update_progress(self, percentage, message):
        """Update progress display."""
        self.progress_bar.set(percentage / 100.0)
        self.message_label.configure(text=message)
        self.update_idletasks()  # Force UI update
