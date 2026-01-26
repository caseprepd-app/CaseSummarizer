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
    # Darker colors to blend seamlessly with CustomTkinter dark theme
    bg_color = "#212121"  # Very dark (blends with UI)
    fg_color = "#ffffff"  # White text
    active_bg = "#333333"  # Slightly lighter for hover
    active_fg = "#ffffff"  # White text on hover

    menubar = Menu(
        window,
        bg=bg_color,
        fg=fg_color,
        activebackground=active_bg,
        activeforeground=active_fg,
        borderwidth=1,
        relief="flat",
        disabledforeground="#666666",
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
        disabledforeground="#666666",
    )
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(
        label="Select Files...", command=select_files_callback, accelerator="Ctrl+O"
    )
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
        disabledforeground="#666666",
    )
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(
        label="Getting Started...",
        command=lambda: HelpDialog(window),
    )
    help_menu.add_command(
        label="Ollama Setup Guide",
        command=lambda: __import__("webbrowser").open(
            "https://sites.google.com/view/caseprepd/ollama-instructions"
        ),
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
