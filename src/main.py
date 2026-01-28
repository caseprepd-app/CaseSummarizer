"""
CasePrepd - Main Application Entry Point
Phase 2.1: CustomTkinter UI

This module initializes the CustomTkinter application and launches the main window.
"""

# Set environment variables BEFORE any imports that might trigger torch loading.
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent HuggingFace tokenizer deadlocks
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Suppress HuggingFace Hub symlink warning

import multiprocessing
import sys
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import contextlib

import customtkinter as ctk

# CRITICAL: Import src.core.ai BEFORE UI framework to avoid DirectML DLL conflicts on Windows
# This pre-loads onnxruntime_genai before UI framework initializes
import src.core.ai  # noqa: F401
from src.config import LOGS_DIR
from src.ui.main_window import MainWindow


def setup_file_logging():
    """Redirects stdout and stderr to a log file."""
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = LOGS_DIR / f"main_log_{timestamp}.txt"

    class Logger:
        """LOG-002, LOG-003: Added error handling and close method."""

        def __init__(self, filepath):
            self.terminal = sys.stdout
            try:
                self.logfile = open(filepath, "w", encoding="utf-8")  # noqa: SIM115
            except OSError as e:
                print(f"Warning: Could not open log file {filepath}: {e}", file=sys.stderr)
                self.logfile = None

        def write(self, message):
            self.terminal.write(message)
            if self.logfile:
                try:
                    self.logfile.write(message)
                    self.flush()
                except OSError:
                    pass  # Log file may have been closed

        def flush(self):
            self.terminal.flush()
            if self.logfile:
                with contextlib.suppress(OSError):
                    self.logfile.flush()

        def close(self):
            """Close the log file when done."""
            if self.logfile:
                with contextlib.suppress(OSError):
                    self.logfile.close()
                self.logfile = None

    sys.stdout = Logger(log_filename)
    sys.stderr = sys.stdout  # Redirect stderr to the same file
    print(f"--- Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"Logging to: {log_filename}")


def main():
    """
    Main entry point for CasePrepd desktop application.
    """
    # Setup stdout/stderr crash log (separate from structured logging)
    setup_file_logging()

    # Configure structured logging (RotatingFileHandler → caseprepd.log)
    from src.logging_config import setup_logging

    setup_logging()

    # Enable multiprocessing support for Windows frozen executables
    multiprocessing.freeze_support()

    # Set appearance mode (light/dark/system)
    ctk.set_appearance_mode("System")  # Options: "System", "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

    # Create and run the application
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
