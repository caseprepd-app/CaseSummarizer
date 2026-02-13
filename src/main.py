"""
CasePrepd - Main Application Entry Point
Phase 2.1: CustomTkinter UI

This module initializes the CustomTkinter application and launches the main window.
"""

# Set environment variables BEFORE any imports that might trigger torch loading.
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent HuggingFace tokenizer deadlocks
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Suppress HuggingFace Hub symlink warning

# Support --debug CLI flag — MUST be set before src.config is imported,
# because DEBUG_MODE is evaluated at import time.
if "--debug" in sys.argv:
    os.environ["DEBUG"] = "true"

import multiprocessing
import traceback
from datetime import datetime
from pathlib import Path

# Add project root to Python path (skip in frozen mode — PyInstaller handles it)
if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import contextlib

# Early crash log for import failures (before Logger redirect is set up).
# In windowed mode sys.stdout/stderr are None, so errors would be silent.
_CRASH_LOG = Path(os.environ.get("APPDATA", ".")) / "CasePrepd" / "crash.log"

try:
    import customtkinter as ctk

    # CRITICAL: Import src.core.ai BEFORE UI framework to avoid DLL load order conflicts
    import src.core.ai  # noqa: F401
    from src.config import LOGS_DIR
    from src.ui.main_window import MainWindow
except Exception:
    _CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
    _CRASH_LOG.write_text(traceback.format_exc(), encoding="utf-8")
    raise


def setup_file_logging():
    """Redirects stdout and stderr to a log file."""
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = LOGS_DIR / f"main_log_{timestamp}.txt"

    class Logger:
        """LOG-002, LOG-003: Added error handling and close method."""

        def __init__(self, filepath):
            # In PyInstaller windowed mode (--noconsole), sys.stdout is None
            self.terminal = sys.stdout
            try:
                self.logfile = open(filepath, "w", encoding="utf-8")  # noqa: SIM115
            except OSError as e:
                if self.terminal is not None:
                    print(f"Warning: Could not open log file {filepath}: {e}", file=sys.stderr)
                self.logfile = None

        def write(self, message):
            if self.terminal is not None:
                try:
                    self.terminal.write(message)
                except (UnicodeEncodeError, OSError):
                    # Windows terminal uses cp1252 which can't encode all Unicode.
                    # Replace unencodable characters rather than crashing the app.
                    try:
                        self.terminal.write(
                            message.encode("ascii", errors="replace").decode("ascii")
                        )
                    except OSError:
                        pass
            if self.logfile:
                try:
                    self.logfile.write(message)
                    self.flush()
                except OSError:
                    pass  # Log file may have been closed

        def flush(self):
            if self.terminal is not None:
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
    from src.logging_config import purge_old_logs, setup_logging

    setup_logging()
    purge_old_logs()

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
