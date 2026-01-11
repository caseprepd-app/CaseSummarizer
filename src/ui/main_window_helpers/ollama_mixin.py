"""
Ollama Status and Model Management Mixin.

Session 82: Extracted from main_window.py for modularity.

Contains:
- Ollama status indicator (connected/disconnected)
- Ollama tooltips with troubleshooting info
- Model display formatting
- Settings dialog integration
"""

import re

import customtkinter as ctk

from src.config import DEBUG_MODE
from src.logging_config import debug_log
from src.ui.tooltip_manager import tooltip_manager

# PERF-001: Pre-compile regex at module level
_MODEL_PARAM_PATTERN = re.compile(r":(\d+\.?\d*)b")


class OllamaMixin:
    """
    Mixin class providing Ollama status and model management.

    Requires parent class to have:
    - self.model_manager: OllamaModelManager instance
    - self.ollama_status_dot: CTkLabel for status indicator
    - self.ollama_status_label: CTkLabel for status text
    - self.model_name_label: CTkLabel for model display
    """

    def _update_ollama_status(self):
        """
        Update the Ollama status indicator in the status bar.

        Shows green dot if connected, red dot if disconnected.
        Adds tooltip with troubleshooting info when disconnected.
        """
        from src.ui.theme import COLORS

        if self.model_manager.is_connected:
            # Connected - green dot
            self.ollama_status_dot.configure(text_color=COLORS["success"])
            self.ollama_status_label.configure(text="Ollama")
            # Remove tooltip bindings
            self._clear_ollama_tooltip()
        else:
            # Disconnected - red dot
            self.ollama_status_dot.configure(text_color=COLORS["danger"])
            self.ollama_status_label.configure(text="Ollama (disconnected)")
            # Add tooltip with troubleshooting info
            self._setup_ollama_tooltip()

        if DEBUG_MODE:
            status = "connected" if self.model_manager.is_connected else "disconnected"
            debug_log(f"[MainWindow] Ollama status: {status}")

    def _setup_ollama_tooltip(self):
        """Set up hover tooltip for disconnected Ollama status."""
        tooltip_text = (
            "Ollama is not running.\n\n"
            "To fix:\n"
            "• Ensure Ollama is installed (ollama.ai)\n"
            "• Run 'ollama serve' in a terminal\n"
            "• Check if port 11434 is available"
        )

        def show_tooltip(event):
            # Session 62b: Close any existing tooltip first via global manager
            tooltip_manager.close_active()

            # Create tooltip window
            self._ollama_tooltip = tooltip = ctk.CTkToplevel(self)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")

            label = ctk.CTkLabel(
                tooltip,
                text=tooltip_text,
                font=("Segoe UI", 10),
                fg_color=("#2b2b2b", "#2b2b2b"),
                corner_radius=6,
                padx=10,
                pady=8,
                justify="left",
            )
            label.pack()

            # Session 62b: Register with global manager
            tooltip_manager.register(tooltip, owner=self.ollama_status_dot)

        def hide_tooltip(event):
            if hasattr(self, "_ollama_tooltip") and self._ollama_tooltip:
                # Session 62b: Unregister from global manager
                tooltip_manager.unregister(self._ollama_tooltip)
                self._ollama_tooltip.destroy()
                self._ollama_tooltip = None

        # Bind to both dot and label
        self.ollama_status_dot.bind("<Enter>", show_tooltip)
        self.ollama_status_dot.bind("<Leave>", hide_tooltip)
        self.ollama_status_label.bind("<Enter>", show_tooltip)
        self.ollama_status_label.bind("<Leave>", hide_tooltip)

    def _clear_ollama_tooltip(self):
        """Remove tooltip bindings when Ollama is connected."""
        # Unbind events
        self.ollama_status_dot.unbind("<Enter>")
        self.ollama_status_dot.unbind("<Leave>")
        self.ollama_status_label.unbind("<Enter>")
        self.ollama_status_label.unbind("<Leave>")

        # Destroy any existing tooltip
        if hasattr(self, "_ollama_tooltip") and self._ollama_tooltip:
            # Session 62b: Unregister from global manager
            tooltip_manager.unregister(self._ollama_tooltip)
            self._ollama_tooltip.destroy()
            self._ollama_tooltip = None

    def _update_model_display(self):
        """
        Update the model display in the header.

        Shows model name and parameter count (e.g., "gemma3:1b (1B params)").
        """
        model_name = self.model_manager.model_name

        # Format display text with parameter count if available
        display_text = self._format_model_display(model_name)
        self.model_name_label.configure(text=display_text)

        if DEBUG_MODE:
            debug_log(f"[MainWindow] Model display updated: {display_text}")

    def _format_model_display(self, model_name: str) -> str:
        """
        Format model name with parameter count for display.

        Args:
            model_name: Raw model name (e.g., "gemma3:1b", "llama3.2:3b")

        Returns:
            Formatted string (e.g., "gemma3:1b (1B params)")
        """
        if not model_name:
            return "No model selected"

        # Extract parameter count from model name if present
        # Common patterns: gemma3:1b, llama3.2:3b, mistral:7b
        param_info = ""
        name_lower = model_name.lower()

        # PERF-001: Look for parameter patterns using pre-compiled regex
        param_match = _MODEL_PARAM_PATTERN.search(name_lower)
        if param_match:
            param_size = param_match.group(1)
            param_info = f" ({param_size}B params)"

        return f"{model_name}{param_info}"

    def _open_model_settings(self):
        """Open the settings dialog directly to the Questions tab (model config)."""
        from src.ui.settings.settings_dialog import SettingsDialog
        from src.user_preferences import get_user_preferences

        # Session 62: Capture current model to detect changes
        prefs = get_user_preferences()
        old_model = prefs.get("ollama_model", self.model_manager.model_name)

        try:
            dialog = SettingsDialog(parent=self, initial_tab="Questions")
            dialog.wait_window()
        except Exception as e:
            # LOG-006: Use debug_log instead of traceback.print_exc()
            debug_log(f"Failed to open settings dialog: {e}")

        # Session 62: Check if model changed and reload if needed
        new_model = prefs.get("ollama_model", self.model_manager.model_name)
        if new_model and new_model != old_model:
            try:
                self.model_manager.load_model(new_model)
                from src.logging_config import info as log_info

                log_info(f"Model changed: {old_model} → {new_model}")
            except Exception as e:
                from src.logging_config import warning as log_warning

                log_warning(f"Failed to load model {new_model}: {e}")

        # Refresh UI after settings change
        self._refresh_corpus_dropdown()
        self._update_model_display()
        self._update_ollama_status()
        self._update_vocab_llm_checkbox_state()  # Session 63b: Refresh LLM checkbox

    def _check_ollama_service(self):
        """Check if Ollama service is running on startup."""
        from tkinter import messagebox

        try:
            self.model_manager.health_check()
            debug_log("[MainWindow] Ollama service is accessible")
        except Exception as e:
            debug_log(f"[MainWindow] Ollama service not accessible: {e}")

            # Show warning
            messagebox.showwarning(
                "Ollama Not Found",
                "Ollama service is not running.\n\n"
                "CasePrepd requires Ollama for Q&A and summaries.\n\n"
                "To install: Visit https://ollama.ai\n"
                "To start: Run 'ollama serve' in a terminal\n\n"
                "Vocabulary extraction will still work without Ollama.",
            )
