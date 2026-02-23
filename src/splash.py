"""
CasePrepd - Splash Screen Utilities

Standalone module for splash screen management. No src.* imports allowed —
this runs before the application is fully initialized.

In frozen (PyInstaller) builds, the main exe spawns itself as a subprocess
with the _CASEPREPD_SPLASH=1 environment variable. That subprocess calls
run_splash_window() and exits. Environment variables are passed directly
via CreateProcess on Windows — no argv parsing by PyInstaller's bootloader.
"""

import os
import subprocess
import sys
from pathlib import Path

# tkinter.PhotoImage supports .png (Tk 8.6+) and .gif natively
SPLASH_EXTENSIONS = {".png", ".gif"}


def get_splash_dir():
    """Return the path to the splash image directory.

    Returns:
        Path or None: Path to splash dir, or None if not found.
    """
    if getattr(sys, "frozen", False):
        splash_dir = Path(sys._MEIPASS) / "assets" / "splash"
    else:
        splash_dir = Path(__file__).parent.parent / "assets" / "splash"

    if not splash_dir.is_dir():
        splash_log(f"[Splash] Directory not found: {splash_dir}")
        return None
    return splash_dir


def splash_log(msg):
    """Log splash messages safely (stdout is None in windowed/noconsole mode).

    Args:
        msg: Message string to print.
    """
    if sys.stdout is not None:
        print(msg)


def launch():
    """Launch splash screen in a separate process to avoid Tk root conflicts.

    The main process needs ctk.CTk() (which creates tk.Tk() internally).
    A second tk.Tk() in the same process causes hangs on Windows.
    Running the splash as a subprocess avoids this entirely.

    Returns:
        subprocess.Popen or None: The splash process, or None on failure.
    """
    splash_dir = get_splash_dir()
    if splash_dir is None:
        return None

    images = [f for f in splash_dir.iterdir() if f.suffix.lower() in SPLASH_EXTENSIONS]
    if not images:
        splash_log(f"[Splash] No images found in {splash_dir}")
        return None

    if getattr(sys, "frozen", False):
        # Frozen mode: spawn ourselves with _CASEPREPD_SPLASH env var.
        # The subprocess imports only tkinter (fast) and shows the splash.
        # We use an env var instead of --splash-only argv because PyInstaller's
        # bootloader does not reliably pass argv to the Python runtime.
        try:
            env = os.environ.copy()
            env["_CASEPREPD_SPLASH"] = "1"
            proc = subprocess.Popen(
                [sys.executable],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            splash_log(f"[Splash] Launched frozen splash subprocess (PID: {proc.pid})")
            return proc
        except Exception as e:
            splash_log(f"[Splash] Failed to launch frozen splash: {e}")
            return None

    # Dev mode: use pythonw.exe (no console window) with inline script
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.exists():
        pythonw = Path(sys.executable)  # Fallback (will show console)
        splash_log("[Splash] pythonw.exe not found, falling back to python.exe")

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
        splash_log(f"[Splash] Launched subprocess (PID: {proc.pid})")
        return proc
    except Exception as e:
        splash_log(f"[Splash] Failed to launch subprocess: {e}")
        return None


def kill(proc):
    """Terminate the splash screen subprocess.

    Args:
        proc: subprocess.Popen or None
    """
    if proc is not None:
        splash_log("[Splash] Terminating splash subprocess")
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            splash_log("[Splash] terminate() timed out, escalating to kill()")
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception as e:
                splash_log(f"[Splash] kill() failed: {e}")
        except Exception as e:
            splash_log(f"[Splash] terminate() failed: {e}")


def run_splash_window():
    """Show a tkinter splash window and exit. Called in the splash subprocess.

    This function is the entry point for the frozen splash subprocess.
    It imports only tkinter (fast), shows a random splash image, and
    calls sys.exit(0) when done. Must run before any heavy imports.
    """
    import random
    import tkinter as tk

    splash_dir = get_splash_dir()
    if splash_dir is None:
        sys.exit(0)

    images = [f for f in splash_dir.iterdir() if f.suffix.lower() in SPLASH_EXTENSIONS]
    if not images:
        sys.exit(0)

    img_path = random.choice(images)
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)

    img = tk.PhotoImage(file=str(img_path))
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - img.width()) // 2
    y = (sh - img.height()) // 2
    root.geometry(f"{img.width()}x{img.height() + 24}+{x}+{y}")

    lbl = tk.Label(root, image=img, borderwidth=0)
    lbl.image = img
    lbl.pack()

    st = tk.Label(
        root,
        text="Loading...",
        bg="#1a1a2e",
        fg="#ffffff",
        font=("Segoe UI", 10),
    )
    st.pack(fill="x")

    # Safety net: auto-close after 60s if parent process crashes
    root.after(60000, root.quit)

    root.mainloop()
    sys.exit(0)
