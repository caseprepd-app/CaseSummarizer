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
import subprocess
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

# tkinter.PhotoImage supports .png (Tk 8.6+, bundled since Python 3.1) and .gif natively
SPLASH_EXTENSIONS = {".png", ".gif"}

# Tell Windows we are DPI-aware so winfo_screenwidth/height return real pixels.
# Must be called before any Tk() window is created.
with contextlib.suppress(Exception):
    import ctypes

    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware

# ============================================================
# Splash-only mode: used in frozen builds where the main .exe
# spawns itself with --splash-only to show a tkinter splash in
# a separate process (avoiding the dual-Tk-root hang).
# This block MUST run before any heavy imports.
# ============================================================
if "--splash-only" in sys.argv:
    import random
    import tkinter as tk

    _splash_exts = {".png", ".gif"}

    if getattr(sys, "frozen", False):
        _splash_dir = Path(sys._MEIPASS) / "assets" / "splash"
    else:
        _splash_dir = Path(__file__).parent.parent / "assets" / "splash"

    if _splash_dir.is_dir():
        _images = [f for f in _splash_dir.iterdir() if f.suffix.lower() in _splash_exts]
        if _images:
            _img_path = random.choice(_images)
            _root = tk.Tk()
            _root.overrideredirect(True)
            _root.attributes("-topmost", True)

            _img = tk.PhotoImage(file=str(_img_path))
            _sw = _root.winfo_screenwidth()
            _sh = _root.winfo_screenheight()
            _x = (_sw - _img.width()) // 2
            _y = (_sh - _img.height()) // 2
            _root.geometry(f"{_img.width()}x{_img.height() + 24}+{_x}+{_y}")

            _lbl = tk.Label(_root, image=_img, borderwidth=0)
            _lbl.image = _img
            _lbl.pack()

            _st = tk.Label(
                _root,
                text="Loading...",
                bg="#1a1a2e",
                fg="#ffffff",
                font=("Segoe UI", 10),
            )
            _st.pack(fill="x")

            # Safety net: auto-close after 60s if parent process crashes
            _root.after(60000, _root.quit)

            _root.mainloop()

    sys.exit(0)


def _splash_log(msg):
    """Log splash messages safely (stdout is None in windowed/noconsole mode)."""
    if sys.stdout is not None:
        print(msg)


def _get_splash_dir():
    """Return the path to the splash image directory.

    Returns:
        Path or None: Path to splash dir, or None if not found.
    """
    if getattr(sys, "frozen", False):
        splash_dir = Path(sys._MEIPASS) / "assets" / "splash"
    else:
        splash_dir = Path(__file__).parent.parent / "assets" / "splash"

    if not splash_dir.is_dir():
        _splash_log(f"[Splash] Directory not found: {splash_dir}")
        return None
    return splash_dir


def _launch_splash():
    """Launch splash screen in a separate process to avoid Tk root conflicts.

    The main process needs ctk.CTk() (which creates tk.Tk() internally).
    A second tk.Tk() in the same process causes hangs on Windows.
    Running the splash as a subprocess with pythonw.exe avoids this entirely
    and also hides the console window from end users.

    Returns:
        subprocess.Popen or None: The splash process, or None on failure.
    """
    splash_dir = _get_splash_dir()
    if splash_dir is None:
        return None

    images = [f for f in splash_dir.iterdir() if f.suffix.lower() in SPLASH_EXTENSIONS]
    if not images:
        _splash_log(f"[Splash] No images found in {splash_dir}")
        return None

    if getattr(sys, "frozen", False):
        # Frozen mode: spawn ourselves with --splash-only flag.
        # The subprocess only imports tkinter (fast) and shows the splash.
        try:
            proc = subprocess.Popen(
                [sys.executable, "--splash-only"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _splash_log(f"[Splash] Launched frozen splash subprocess (PID: {proc.pid})")
            return proc
        except Exception as e:
            _splash_log(f"[Splash] Failed to launch frozen splash: {e}")
            return None

    # Dev mode: use pythonw.exe (no console window) with inline script
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.exists():
        pythonw = Path(sys.executable)  # Fallback (will show console)
        _splash_log("[Splash] pythonw.exe not found, falling back to python.exe")

    # Inline script for the splash subprocess
    splash_script = f"""
import tkinter as tk
import random
import contextlib
from pathlib import Path

with contextlib.suppress(Exception):
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(2)

splash_dir = Path(r"{splash_dir}")
exts = {{".png", ".gif"}}
images = [f for f in splash_dir.iterdir() if f.suffix.lower() in exts]
if not images:
    raise SystemExit()

img_path = random.choice(images)
root = tk.Tk()
root.overrideredirect(True)
root.attributes("-topmost", True)

img = tk.PhotoImage(file=str(img_path))
sw = root.winfo_screenwidth()
sh = root.winfo_screenheight()
x = (sw - img.width()) // 2
y = (sh - img.height()) // 2
root.geometry(f"{{img.width()}}x{{img.height() + 24}}+{{x}}+{{y}}")

lbl = tk.Label(root, image=img, borderwidth=0)
lbl.image = img
lbl.pack()

st = tk.Label(root, text="Loading...", bg="#1a1a2e", fg="#ffffff", font=("Segoe UI", 10))
st.pack(fill="x")

root.after(60000, root.quit)
root.mainloop()
"""

    try:
        proc = subprocess.Popen(
            [str(pythonw), "-c", splash_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _splash_log(f"[Splash] Launched subprocess (PID: {proc.pid})")
        return proc
    except Exception as e:
        _splash_log(f"[Splash] Failed to launch subprocess: {e}")
        return None


def _kill_splash(proc):
    """Terminate the splash screen subprocess.

    Args:
        proc: subprocess.Popen or None
    """
    if proc is not None:
        _splash_log("[Splash] Terminating splash subprocess")
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _splash_log("[Splash] terminate() timed out, escalating to kill()")
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception as e:
                _splash_log(f"[Splash] kill() failed: {e}")
        except Exception as e:
            _splash_log(f"[Splash] terminate() failed: {e}")


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

    All heavy imports (customtkinter, torch/AI libs, MainWindow) live here
    rather than at module level.  This prevents the worker subprocess — which
    re-imports __main__ as __mp_main__ on Windows "spawn" — from launching a
    second splash screen or wasting 20-30 s on imports it never uses.
    """
    # 1. Show splash while heavy imports load
    splash_proc = _launch_splash()

    # 2. Heavy imports (wrapped so crash log is written on failure)
    try:
        import customtkinter as ctk

        # CRITICAL: Import src.core.ai BEFORE UI framework to avoid DLL load order conflicts
        import src.core.ai  # noqa: F401
        from src.config import LOGS_DIR
        from src.ui.main_window import MainWindow
    except Exception:
        _kill_splash(splash_proc)
        _CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
        _CRASH_LOG.write_text(traceback.format_exc(), encoding="utf-8")
        raise

    # 3. Setup stdout/stderr crash log (separate from structured logging)
    setup_file_logging(LOGS_DIR)

    # 4. Configure structured logging (RotatingFileHandler -> caseprepd.log)
    from src.logging_config import purge_old_logs, setup_logging

    setup_logging()
    purge_old_logs()

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

    # 6. Enable multiprocessing support for Windows frozen executables
    multiprocessing.freeze_support()

    # 7. Set appearance mode (light/dark/system)
    ctk.set_appearance_mode("System")  # Options: "System", "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

    # Apply display scaling (font offset + UI scale) before creating any widgets
    from src.ui.scaling import apply_scaling

    apply_scaling()

    # 8. Start persistent worker subprocess for pipeline tasks (GIL-free)
    from src.services.worker_manager import WorkerProcessManager

    worker_manager = WorkerProcessManager()
    worker_manager.start()

    # 9. Create main window
    app = MainWindow(worker_manager=worker_manager)

    # 10. Kill splash now that the main window is ready
    _kill_splash(splash_proc)

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
