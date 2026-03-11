"""
Menu Handler for MainWindow

Encapsulates menu creation and menu-related callbacks.
Separates menu logic from main window to keep main_window.py focused on layout.
"""

from tkinter import Menu

from src.ui.help_about_dialogs import AboutDialog, HelpDialog


def create_menus(window, select_files_callback, show_settings_callback, quit_callback):
    """
    Create menubar with File and Help menus.

    Args:
        window: The Tk root window
        select_files_callback: Function to call when "Select Files" is clicked
        show_settings_callback: Function to call when "Settings" is clicked
        quit_callback: Function to call when "Exit" is clicked
    """
    from src.ui.theme import get_color

    bg_color = get_color("menu_bg")
    fg_color = get_color("menu_fg")
    active_bg = get_color("menu_active_bg")
    active_fg = get_color("menu_active_fg")
    disabled_fg = get_color("menu_disabled_fg")

    menubar = Menu(
        window,
        bg=bg_color,
        fg=fg_color,
        activebackground=active_bg,
        activeforeground=active_fg,
        borderwidth=1,
        relief="flat",
        disabledforeground=disabled_fg,
    )
    window.config(menu=menubar)

    # File menu
    file_menu = Menu(
        menubar,
        tearoff=0,
        bg=bg_color,
        fg=fg_color,
        activebackground=active_bg,
        activeforeground=active_fg,
        borderwidth=0,
        relief="flat",
        disabledforeground=disabled_fg,
    )
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(
        label="Select Files...", command=select_files_callback, accelerator="Ctrl+O"
    )
    file_menu.add_separator()

    # Export submenu
    menu_style = dict(
        tearoff=0,
        bg=bg_color,
        fg=fg_color,
        activebackground=active_bg,
        activeforeground=active_fg,
        borderwidth=0,
        relief="flat",
        disabledforeground=disabled_fg,
    )

    export_menu = Menu(file_menu, **menu_style)
    export_menu.add_command(
        label="Vocabulary Model...",
        command=lambda: _export_model(window),
    )
    export_menu.add_command(
        label="Feedback History...",
        command=lambda: _export_feedback(window),
    )
    file_menu.add_cascade(label="Export", menu=export_menu)

    import_menu = Menu(file_menu, **menu_style)
    import_menu.add_command(
        label="Vocabulary Model...",
        command=lambda: _import_model(window),
    )
    import_menu.add_command(
        label="Feedback History...",
        command=lambda: _import_feedback(window),
    )
    file_menu.add_cascade(label="Import", menu=import_menu)

    file_menu.add_separator()
    file_menu.add_command(label="Settings", command=show_settings_callback, accelerator="Ctrl+,")
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=quit_callback, accelerator="Ctrl+Q")

    # Help menu
    help_menu = Menu(
        menubar,
        tearoff=0,
        bg=bg_color,
        fg=fg_color,
        activebackground=active_bg,
        activeforeground=active_fg,
        borderwidth=0,
        relief="flat",
        disabledforeground=disabled_fg,
    )
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(
        label="Getting Started...",
        command=lambda: HelpDialog(window),
    )
    help_menu.add_separator()
    from src.config import APP_NAME

    help_menu.add_command(
        label=f"About {APP_NAME}",
        command=lambda: AboutDialog(window),
    )

    # Bind keyboard shortcuts
    window.bind("<Control-o>", lambda e: select_files_callback())
    window.bind("<Control-comma>", lambda e: show_settings_callback())
    window.bind("<Control-q>", lambda e: quit_callback())


def _export_model(window):
    """Export user vocabulary model via file dialog."""
    from tkinter import filedialog, messagebox

    from src.services.model_io_service import export_user_model

    dest = filedialog.asksaveasfilename(
        parent=window,
        title="Export Vocabulary Model",
        defaultextension=".pkl",
        filetypes=[("Pickle files", "*.pkl")],
        initialfile="vocab_model.pkl",
    )
    if not dest:
        return
    from pathlib import Path

    ok, msg = export_user_model(Path(dest))
    if ok:
        messagebox.showinfo("Export Complete", msg, parent=window)
    else:
        messagebox.showerror("Export Failed", msg, parent=window)


def _import_model(window):
    """Import vocabulary model with trust warning and validation."""
    from tkinter import filedialog, messagebox

    # Mandatory trust warning
    messagebox.showwarning(
        "Security Warning",
        "Only load model files from sources you trust.\n"
        "Model files can contain executable code.\n\n"
        "Press OK to continue.",
        parent=window,
    )

    src = filedialog.askopenfilename(
        parent=window,
        title="Import Vocabulary Model",
        filetypes=[("Pickle files", "*.pkl")],
    )
    if not src:
        return
    from pathlib import Path

    from src.services.model_io_service import import_user_model

    ok, msg = import_user_model(Path(src))
    if ok:
        messagebox.showinfo("Import Complete", msg, parent=window)
    else:
        messagebox.showerror("Import Failed", msg, parent=window)


def _export_feedback(window):
    """Export user feedback history via file dialog."""
    from tkinter import filedialog, messagebox

    from src.services import VocabularyService
    from src.services.model_io_service import export_user_feedback

    feedback_mgr = VocabularyService().get_feedback_manager()
    dest = filedialog.asksaveasfilename(
        parent=window,
        title="Export Feedback History",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        initialfile="feedback_history.csv",
    )
    if not dest:
        return
    from pathlib import Path

    ok, msg = export_user_feedback(Path(dest), feedback_mgr)
    if ok:
        messagebox.showinfo("Export Complete", msg, parent=window)
    else:
        messagebox.showerror("Export Failed", msg, parent=window)


def _import_feedback(window):
    """Import feedback history with mode selection and optional retrain."""
    from tkinter import filedialog, messagebox

    src = filedialog.askopenfilename(
        parent=window,
        title="Import Feedback History",
        filetypes=[("CSV files", "*.csv")],
    )
    if not src:
        return

    # Ask replace or append
    result = messagebox.askyesnocancel(
        "Import Mode",
        "How should the imported feedback be combined with existing data?\n\n"
        "Yes = Replace (old data backed up)\n"
        "No = Append to existing\n"
        "Cancel = Abort import",
        parent=window,
    )
    if result is None:
        return
    mode = "replace" if result else "append"

    from pathlib import Path

    from src.services import VocabularyService
    from src.services.model_io_service import import_user_feedback

    feedback_mgr = VocabularyService().get_feedback_manager()
    ok, msg, count = import_user_feedback(Path(src), mode, feedback_mgr)

    if not ok:
        messagebox.showerror("Import Failed", msg, parent=window)
        return

    # Ask about retraining
    retrain = messagebox.askyesno(
        "Retrain Model?",
        f"{msg}\n\nRetrain the vocabulary model with the new feedback?",
        parent=window,
    )
    if retrain:
        learner = VocabularyService().get_meta_learner()
        learner.train()
        messagebox.showinfo(
            "Import Complete",
            f"Imported {count} feedback records. Model retrained.",
            parent=window,
        )
    else:
        messagebox.showinfo("Import Complete", msg, parent=window)
