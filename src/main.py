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

import contextlib
import logging
import multiprocessing

multiprocessing.freeze_support()  # Must run before ANY subprocess code in frozen builds

# On Windows, force multiprocessing to use pythonw.exe (no console window)
# to prevent blank CLI windows appearing alongside the GUI.
if sys.platform == "win32":
    _pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    if os.path.exists(_pythonw):
        multiprocessing.set_executable(_pythonw)

# ============================================================
# Splash subprocess check: in frozen builds, the main exe
# spawns itself with _CASEPREPD_SPLASH=1 env var. Detect that
# here and show the splash window, then exit immediately.
# This MUST run before any heavy imports.
# ============================================================
if os.environ.get("_CASEPREPD_SPLASH") == "1":
    from src.splash import run_splash_window

    run_splash_window()  # Shows tkinter splash, then sys.exit(0)

import threading
import traceback
from datetime import datetime
from pathlib import Path

# Add project root to Python path (skip in frozen mode — PyInstaller handles it)
if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Early crash log for import failures (before Logger redirect is set up).
# In windowed mode sys.stdout/stderr are None, so errors would be silent.
_CRASH_LOG = Path(os.environ.get("APPDATA", ".")) / "CasePrepd" / "crash.log"

# Tell Windows we are DPI-aware so winfo_screenwidth/height return real pixels.
# Must be called before any Tk() window is created.
with contextlib.suppress(Exception):
    import ctypes

    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware


def setup_file_logging(logs_dir):
    """Redirects stdout and stderr to a log file.

    Args:
        logs_dir: Path to the log output directory.
    """
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"main_log_{timestamp}.txt"

    class Logger:
        """Tee stdout/stderr to a log file with error handling."""

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
                    except Exception:
                        pass  # Terminal write completely failed
                except Exception:
                    pass  # Unexpected terminal error — don't crash the app
            if self.logfile:
                try:
                    self.logfile.write(message)
                    self.flush()
                except Exception:
                    pass  # Log file may have been closed or inaccessible

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

    All heavy imports (customtkinter, torch/AI libs, MainWindow) live here
    rather than at module level.  This prevents the worker subprocess — which
    re-imports __main__ as __mp_main__ on Windows "spawn" — from launching a
    second splash screen or wasting 20-30 s on imports it never uses.
    """
    from src.splash import kill, launch

    # 1. Show splash while heavy imports load
    splash_proc = launch()

    # 2. Heavy imports (wrapped so crash log is written on failure)
    try:
        import customtkinter as ctk

        # CRITICAL: Import src.core.ai BEFORE UI framework to avoid DLL load order conflicts
        import src.core.ai  # noqa: F401
        from src.config import LOGS_DIR
        from src.ui.main_window import MainWindow
    except Exception:
        kill(splash_proc)
        try:
            _CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
            _CRASH_LOG.write_text(traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass  # Best-effort crash log; original exception re-raised below
        raise

    # 3. Setup stdout/stderr crash log (separate from structured logging)
    try:
        setup_file_logging(LOGS_DIR)

        # 4. Configure structured logging (RotatingFileHandler -> caseprepd.log)
        from src.logging_config import purge_old_logs, setup_logging

        setup_logging()
        purge_old_logs()

        # 4b. Log where every asset loads from (catches "works on my machine" issues)
        from src.services.asset_audit import run_asset_audit

        run_asset_audit()

        # 5. Install global exception hooks so unhandled errors are logged
        # (especially important in PyInstaller --noconsole where stdout is None)
        _logger = logging.getLogger(__name__)

        def _uncaught_exception(exc_type, exc_value, exc_tb):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_tb)
                return
            _logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

        def _uncaught_thread_exception(args):
            if args.exc_type is SystemExit:
                return

            _logger.critical(
                "Uncaught exception in thread %s",
                args.thread.name if args.thread else "unknown",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )

        sys.excepthook = _uncaught_exception

        threading.excepthook = _uncaught_thread_exception

        # 6. Set appearance mode from user preference (default: Dark)
        from src.user_preferences import get_user_preferences

        _prefs = get_user_preferences()
        _appearance = _prefs.get("appearance_mode", "Dark")
        ctk.set_appearance_mode(_appearance)
        ctk.set_default_color_theme("blue")

        # Apply display scaling (font offset + UI scale) before creating any widgets
        from src.ui.scaling import apply_scaling

        apply_scaling()

        # 8. Start persistent worker subprocess for pipeline tasks (GIL-free)
        from src.services.worker_manager import WorkerProcessManager

        worker_manager = WorkerProcessManager()
        worker_manager.start()

        # 9. Create main window
        app = MainWindow(worker_manager=worker_manager)
    finally:
        # Kill splash even if startup fails (prevents orphaned splash window)
        kill(splash_proc)

    # Force main window to front (splash had -topmost, so GUI may be behind it)
    app.lift()
    app.focus_force()
    app.attributes("-topmost", True)
    app.after(200, lambda: app.attributes("-topmost", False))

    app.mainloop()

    # 11. Clean up worker subprocess after GUI closes
    worker_manager.shutdown(blocking=True)


if __name__ == "__main__":
    main()
