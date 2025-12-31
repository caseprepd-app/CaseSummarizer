"""
Settings widgets with integrated tooltips.

Each widget includes an info icon that shows explanatory text on hover.
Uses CustomTkinter for modern appearance consistent with LocalScribe.

Widget Types:
- TooltipIcon: Info icon with hover tooltip
- SettingRow: Base class with label + tooltip layout
- SliderSetting: Numeric range (int/float)
- CheckboxSetting: Boolean toggle
- DropdownSetting: Selection from options
- SpinboxSetting: Integer with +/- buttons

Layout Standard (Session 62b):
─────────────────────────────────────────────────────────────────────────
All settings follow a consistent 2/3 - 1/3 layout:

    ┌─────────────────────────────────┬───────────────────┐
    │  Label Text  ⓘ                  │   [Control]       │
    │  (LABEL_AREA_WIDTH pixels)      │   (CONTROL_WIDTH) │
    └─────────────────────────────────┴───────────────────┘

- Labels + tooltip icons occupy the LEFT 2/3 of each row
- Interactive controls occupy the RIGHT 1/3 of each row
- All controls (sliders, dropdowns, spinboxes) have IDENTICAL widths
- This ensures sliders at the same % value appear at the same X position

To add a new widget type:
1. Subclass SettingRow
2. Use CONTROL_WIDTH for the widget width
3. Grid the widget in column 2 with sticky="e" (right-aligned)
─────────────────────────────────────────────────────────────────────────
"""

from typing import Any, Callable

import customtkinter as ctk

from src.ui.theme import FONTS, COLORS
from src.ui.tooltip_manager import tooltip_manager


# =============================================================================
# LAYOUT CONSTANTS — Change these to adjust all settings uniformly
# =============================================================================

# Width of the label + tooltip area (left portion of each row)
# This should be wide enough for the longest label + tooltip icon
LABEL_AREA_WIDTH = 280

# Width of the interactive control area (right portion of each row)
# All sliders, dropdowns, spinboxes will have this exact width
CONTROL_WIDTH = 220

# Width for value display labels (e.g., "0.50" next to sliders)
VALUE_LABEL_WIDTH = 50

# Padding between elements
CONTROL_PADDING_X = 10


class TooltipIcon(ctk.CTkLabel):
    """
    Info icon that shows a tooltip popup on hover.

    Uses CTkToplevel for the tooltip to avoid z-order issues with
    other widgets. The tooltip appears near the icon and disappears
    when the mouse leaves.

    Session 62b Fix: Uses global TooltipManager to ensure only ONE tooltip
    is visible at a time across the ENTIRE application (not just TooltipIcon).

    Attributes:
        tooltip_text: The help text to display on hover.
        tooltip_window: Reference to the popup window (if visible).
    """

    def __init__(self, parent, tooltip_text: str, **kwargs):
        """
        Initialize the tooltip icon.

        Args:
            parent: Parent widget.
            tooltip_text: Help text shown on hover.
            **kwargs: Additional CTkLabel arguments.
        """
        super().__init__(
            parent,
            text="\u24d8",  # Unicode circled i
            font=FONTS["heading"],
            text_color=COLORS["text_secondary"],
            cursor="hand2",
            **kwargs
        )
        self.tooltip_text = tooltip_text
        self.tooltip_window = None

        self.bind("<Enter>", self._show_tooltip)
        self.bind("<Leave>", self._hide_tooltip)

    def _show_tooltip(self, event=None):
        """Display tooltip popup near the icon."""
        # Session 62b: Close any existing tooltip from ANYWHERE in the app
        tooltip_manager.close_active()

        if self.tooltip_window:
            return

        # Position tooltip to the right of the icon
        x = self.winfo_rootx() + 25
        y = self.winfo_rooty() - 5

        self.tooltip_window = ctk.CTkToplevel(self)
        self.tooltip_window.wm_overrideredirect(True)  # No window decorations
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        self.tooltip_window.attributes("-topmost", True)

        # Tooltip content
        label = ctk.CTkLabel(
            self.tooltip_window,
            text=self.tooltip_text,
            wraplength=300,
            justify="left",
            corner_radius=6,
            fg_color=("gray85", "gray25"),
            text_color=("gray10", "gray90"),
        )
        label.pack(padx=10, pady=8)

        # Session 62b: Register with global manager
        tooltip_manager.register(self.tooltip_window, owner=self)

        # Also bind leave event to the tooltip window itself
        self.tooltip_window.bind("<Leave>", self._check_hide_tooltip)

    def _check_hide_tooltip(self, event=None):
        """Hide tooltip if mouse has left both icon and tooltip window."""
        if self.tooltip_window:
            # Get mouse position
            mouse_x = self.winfo_pointerx()
            mouse_y = self.winfo_pointery()

            # Check if mouse is over the icon
            icon_x1 = self.winfo_rootx()
            icon_y1 = self.winfo_rooty()
            icon_x2 = icon_x1 + self.winfo_width()
            icon_y2 = icon_y1 + self.winfo_height()

            over_icon = (icon_x1 <= mouse_x <= icon_x2 and
                        icon_y1 <= mouse_y <= icon_y2)

            # Check if mouse is over the tooltip
            over_tooltip = False
            if self.tooltip_window and self.tooltip_window.winfo_exists():
                tip_x1 = self.tooltip_window.winfo_rootx()
                tip_y1 = self.tooltip_window.winfo_rooty()
                tip_x2 = tip_x1 + self.tooltip_window.winfo_width()
                tip_y2 = tip_y1 + self.tooltip_window.winfo_height()
                over_tooltip = (tip_x1 <= mouse_x <= tip_x2 and
                               tip_y1 <= mouse_y <= tip_y2)

            # Hide if mouse is over neither
            if not over_icon and not over_tooltip:
                self._force_hide_tooltip()

    def _hide_tooltip(self, event=None):
        """Hide tooltip when mouse leaves the icon."""
        # Use after() to delay slightly - allows _show_tooltip on next element to run first
        if self.tooltip_window:
            self.after(50, self._delayed_hide_check)

    def _delayed_hide_check(self):
        """Check if we should hide after a short delay."""
        # If another tooltip became active, the manager already closed us
        if not tooltip_manager.is_active(self.tooltip_window):
            self.tooltip_window = None
            return
        # Otherwise, check mouse position
        self._check_hide_tooltip()

    def _force_hide_tooltip(self):
        """Unconditionally hide the tooltip."""
        if self.tooltip_window:
            # Session 62b: Unregister from global manager
            tooltip_manager.unregister(self.tooltip_window)
            try:
                self.tooltip_window.destroy()
            except Exception:
                pass  # Window may already be destroyed
            self.tooltip_window = None


class SettingRow(ctk.CTkFrame):
    """
    Base class for setting rows with label + tooltip + widget.

    Provides consistent layout and tooltip handling for all setting types.
    Subclasses implement get_value() and set_value() for their widget.

    Layout (Session 62b - Standardized):
    ┌────────────────────────────────┬──────────────────────┬──────┐
    │  Label + Tooltip               │   Control            │ Value│
    │  (LABEL_AREA_WIDTH)            │   (CONTROL_WIDTH)    │      │
    └────────────────────────────────┴──────────────────────┴──────┘

    All controls are RIGHT-ALIGNED within a fixed-width area, ensuring:
    - Sliders at the same % appear at the same horizontal position
    - Dropdowns, spinboxes, checkboxes align consistently
    """

    def __init__(
        self,
        parent,
        label: str,
        tooltip: str,
        on_change: Callable[[Any], None] = None,
        **kwargs
    ):
        """
        Initialize the setting row.

        Args:
            parent: Parent widget.
            label: Display name for the setting.
            tooltip: Help text shown on icon hover.
            on_change: Optional callback when value changes.
            **kwargs: Additional CTkFrame arguments.
        """
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.on_change = on_change

        # Session 62b: Standardized layout with fixed column widths
        # Column 0: Label area (fixed width)
        # Column 1: Control area (fixed width, controls right-aligned within)
        # Column 2: Value display (fixed width, for sliders)
        self.grid_columnconfigure(0, weight=0, minsize=LABEL_AREA_WIDTH)
        self.grid_columnconfigure(1, weight=1)  # Flexible spacer
        self.grid_columnconfigure(2, weight=0, minsize=CONTROL_WIDTH)
        self.grid_columnconfigure(3, weight=0, minsize=VALUE_LABEL_WIDTH)

        # Label frame (contains label + tooltip icon together)
        label_frame = ctk.CTkFrame(self, fg_color="transparent")
        label_frame.grid(row=0, column=0, sticky="w")

        # Label
        self.label_widget = ctk.CTkLabel(
            label_frame,
            text=label,
            anchor="w",
            font=FONTS["heading_sm"]
        )
        self.label_widget.pack(side="left", padx=(0, 5))

        # Tooltip icon
        self.tooltip_icon = TooltipIcon(label_frame, tooltip)
        self.tooltip_icon.pack(side="left")

    def get_value(self) -> Any:
        """Return the current value. Override in subclass."""
        raise NotImplementedError

    def set_value(self, value: Any) -> None:
        """Set the current value. Override in subclass."""
        raise NotImplementedError


class SliderSetting(SettingRow):
    """
    Slider widget for numeric range settings.

    Displays a horizontal slider with a value label showing the current
    value. Supports integer steps for clean display.

    Session 62b: Slider uses CONTROL_WIDTH for consistent positioning.
    """

    def __init__(
        self,
        parent,
        label: str,
        tooltip: str,
        min_value: float,
        max_value: float,
        step: float = 1,
        initial_value: float = None,
        on_change: Callable[[float], None] = None,
        **kwargs
    ):
        """
        Initialize the slider setting.

        Args:
            parent: Parent widget.
            label: Display name.
            tooltip: Help text.
            min_value: Minimum slider value.
            max_value: Maximum slider value.
            step: Increment between values.
            initial_value: Starting value (defaults to min_value).
            on_change: Callback when value changes.
        """
        super().__init__(parent, label, tooltip, on_change, **kwargs)

        self.min_value = min_value
        self.max_value = max_value
        self.step = step

        # Calculate number of steps
        num_steps = int((max_value - min_value) / step) if step else None

        # Session 62b: Slider with fixed width for consistent positioning
        self.slider = ctk.CTkSlider(
            self,
            from_=min_value,
            to=max_value,
            number_of_steps=num_steps,
            width=CONTROL_WIDTH,  # Fixed width for all sliders
            command=self._on_slider_change
        )
        self.slider.grid(row=0, column=2, sticky="e", padx=(CONTROL_PADDING_X, 0))

        # Value display label (shows current value) - right of slider
        self.value_label = ctk.CTkLabel(
            self,
            text="",
            width=VALUE_LABEL_WIDTH,
            anchor="e",
            font=FONTS["heading_sm"]
        )
        self.value_label.grid(row=0, column=3, sticky="e", padx=(CONTROL_PADDING_X, 0))

        # Set initial value
        if initial_value is not None:
            self.set_value(initial_value)
        else:
            self.set_value(min_value)

    def _on_slider_change(self, value):
        """Handle slider value change."""
        # Format floats with decimals, integers without
        if isinstance(value, float) and value != int(value):
            self.value_label.configure(text=f"{value:.2f}")
        else:
            self.value_label.configure(text=str(int(value)))
        if self.on_change:
            self.on_change(value)

    def get_value(self) -> float:
        """Return current slider value."""
        return self.slider.get()

    def set_value(self, value: float) -> None:
        """Set slider to specified value."""
        self.slider.set(value)
        # Format floats with decimals, integers without
        if isinstance(value, float) and value != int(value):
            self.value_label.configure(text=f"{value:.2f}")
        else:
            self.value_label.configure(text=str(int(value)))


class CheckboxSetting(SettingRow):
    """
    Checkbox widget for boolean settings.

    Simple toggle with no additional text (label is in the row).

    Session 62b: Checkbox positioned at start of control area for consistency.
    """

    def __init__(
        self,
        parent,
        label: str,
        tooltip: str,
        initial_value: bool = False,
        on_change: Callable[[bool], None] = None,
        **kwargs
    ):
        """
        Initialize the checkbox setting.

        Args:
            parent: Parent widget.
            label: Display name.
            tooltip: Help text.
            initial_value: Starting state (True/False).
            on_change: Callback when value changes.
        """
        super().__init__(parent, label, tooltip, on_change, **kwargs)

        self.var = ctk.BooleanVar(value=initial_value)
        self.checkbox = ctk.CTkCheckBox(
            self,
            text="",
            variable=self.var,
            command=self._on_checkbox_change,
            width=24,
            checkbox_width=20,
            checkbox_height=20
        )
        # Session 62b: Left-align checkbox within the control column
        # Checkboxes don't need the full CONTROL_WIDTH but should start at same position
        self.checkbox.grid(row=0, column=2, sticky="w", padx=(CONTROL_PADDING_X, 0))

    def _on_checkbox_change(self):
        """Handle checkbox state change."""
        if self.on_change:
            self.on_change(self.var.get())

    def get_value(self) -> bool:
        """Return current checkbox state."""
        return self.var.get()

    def set_value(self, value: bool) -> None:
        """Set checkbox to specified state."""
        self.var.set(value)


class DropdownSetting(SettingRow):
    """
    Dropdown (combobox) widget for selection settings.

    Displays options as text labels but stores/returns actual values.
    Options are provided as (display_text, value) tuples.

    Session 62b: Dropdown uses CONTROL_WIDTH for consistent positioning.
    """

    def __init__(
        self,
        parent,
        label: str,
        tooltip: str,
        options: list[tuple[str, Any]],
        initial_value: Any = None,
        on_change: Callable[[Any], None] = None,
        **kwargs
    ):
        """
        Initialize the dropdown setting.

        Args:
            parent: Parent widget.
            label: Display name.
            tooltip: Help text.
            options: List of (display_text, value) tuples.
            initial_value: Starting value (matched against option values).
            on_change: Callback when selection changes.
        """
        super().__init__(parent, label, tooltip, on_change, **kwargs)

        self.options = options
        # Maps: display_text -> value, value -> display_text
        self.value_map = {text: val for text, val in options}
        self.text_map = {val: text for text, val in options}

        display_values = [text for text, _ in options]
        initial_text = self.text_map.get(initial_value, display_values[0] if display_values else "")

        # Session 62b: Use CONTROL_WIDTH for consistent positioning
        self.dropdown = ctk.CTkComboBox(
            self,
            values=display_values,
            command=self._on_dropdown_change,
            state="readonly",
            width=CONTROL_WIDTH
        )
        self.dropdown.set(initial_text)
        self.dropdown.grid(row=0, column=2, sticky="e", padx=(CONTROL_PADDING_X, 0))

    def _on_dropdown_change(self, selected_text):
        """Handle dropdown selection change."""
        if self.on_change:
            self.on_change(self.value_map.get(selected_text))

    def get_value(self) -> Any:
        """Return value for current selection."""
        return self.value_map.get(self.dropdown.get())

    def set_value(self, value: Any) -> None:
        """Set dropdown to option with specified value."""
        text = self.text_map.get(value, "")
        if text:
            self.dropdown.set(text)


class SpinboxSetting(SettingRow):
    """
    Spinbox widget for integer settings with +/- buttons.

    Provides a compact control for integer values within a range.
    Uses buttons instead of a slider for precise control.

    Session 62b: Spinbox positioned consistently with other controls.
    """

    def __init__(
        self,
        parent,
        label: str,
        tooltip: str,
        min_value: int,
        max_value: int,
        initial_value: int = None,
        on_change: Callable[[int], None] = None,
        **kwargs
    ):
        """
        Initialize the spinbox setting.

        Args:
            parent: Parent widget.
            label: Display name.
            tooltip: Help text.
            min_value: Minimum value.
            max_value: Maximum value.
            initial_value: Starting value (defaults to min_value).
            on_change: Callback when value changes.
        """
        super().__init__(parent, label, tooltip, on_change, **kwargs)

        self.min_value = min_value
        self.max_value = max_value
        self.value = initial_value if initial_value is not None else min_value

        # Session 62b: Container with fixed width for consistent positioning
        spinbox_frame = ctk.CTkFrame(self, fg_color="transparent", width=CONTROL_WIDTH)
        spinbox_frame.grid(row=0, column=2, sticky="w", padx=(CONTROL_PADDING_X, 0))
        spinbox_frame.grid_propagate(False)  # Maintain fixed width

        # Minus button
        self.minus_btn = ctk.CTkButton(
            spinbox_frame,
            text="-",
            width=32,
            height=28,
            command=self._decrement,
            font=FONTS["heading_lg"]
        )
        self.minus_btn.pack(side="left")

        # Value display
        self.value_label = ctk.CTkLabel(
            spinbox_frame,
            text=str(self.value),
            width=45,
            font=FONTS["heading"]
        )
        self.value_label.pack(side="left", padx=8)

        # Plus button
        self.plus_btn = ctk.CTkButton(
            spinbox_frame,
            text="+",
            width=32,
            height=28,
            command=self._increment,
            font=FONTS["heading_lg"]
        )
        self.plus_btn.pack(side="left")

    def _decrement(self):
        """Decrease value by 1 if above minimum."""
        if self.value > self.min_value:
            self.value -= 1
            self._update_display()

    def _increment(self):
        """Increase value by 1 if below maximum."""
        if self.value < self.max_value:
            self.value += 1
            self._update_display()

    def _update_display(self):
        """Update value label and trigger callback."""
        self.value_label.configure(text=str(self.value))
        if self.on_change:
            self.on_change(self.value)

    def get_value(self) -> int:
        """Return current value."""
        return self.value

    def set_value(self, value: int) -> None:
        """Set to specified value (clamped to range)."""
        self.value = max(self.min_value, min(self.max_value, value))
        self.value_label.configure(text=str(self.value))

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the spinbox controls."""
        state = "normal" if enabled else "disabled"
        self.minus_btn.configure(state=state)
        self.plus_btn.configure(state=state)
        # Grey out text when disabled
        text_color = ("gray10", "gray90") if enabled else ("gray50", "gray50")
        self.value_label.configure(text_color=text_color)
        self.label_widget.configure(text_color=text_color)


class ButtonSetting(SettingRow):
    """
    Button widget for action settings.

    Displays a button that triggers an action when clicked.
    Useful for opening folders, running calibrations, etc.

    Session 62b: Button uses CONTROL_WIDTH for consistent positioning.
    """

    def __init__(
        self,
        parent,
        label: str,
        tooltip: str,
        action: callable,
        button_text: str = None,
        **kwargs
    ):
        """
        Initialize the button setting.

        Args:
            parent: Parent widget.
            label: Display name (shown as button text if button_text is None).
            tooltip: Help text.
            action: Function to call when button is clicked.
            button_text: Optional custom button text.
        """
        super().__init__(parent, label, tooltip, None, **kwargs)

        self.action = action

        # Session 62b: Use CONTROL_WIDTH for consistent positioning
        self.button = ctk.CTkButton(
            self,
            text=button_text or label,
            command=self._on_click,
            width=CONTROL_WIDTH,
            height=28,
            font=FONTS["body"]
        )
        self.button.grid(row=0, column=2, sticky="w", padx=(CONTROL_PADDING_X, 0))

    def _on_click(self):
        """Handle button click."""
        if self.action:
            self.action()

    def get_value(self):
        """Buttons don't have a value."""
        return None

    def set_value(self, value) -> None:
        """Buttons don't have a value."""
        pass


class DefaultQuestionsWidget(ctk.CTkFrame):
    """
    Widget for managing default Q&A questions with checkboxes.

    Session 63c: Provides a scrollable list of questions where each can be
    enabled/disabled via checkbox. Also supports add, edit, delete, and reorder.

    Layout:
        ┌─────────────────────────────────────────────────────────────┐
        │  Default Questions                                      ⓘ  │
        ├─────────────────────────────────────────────────────────────┤
        │  ☑ What is this case about?                            [✕] │
        │  ☑ What are the main allegations?                      [✕] │
        │  ☐ Who are the plaintiffs?                             [✕] │
        │  ...                                                        │
        ├─────────────────────────────────────────────────────────────┤
        │  [+ Add Question]                                           │
        └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, parent, **kwargs):
        """
        Initialize the default questions widget.

        Args:
            parent: Parent widget.
            **kwargs: Additional CTkFrame arguments.
        """
        super().__init__(parent, fg_color="transparent", **kwargs)

        # Get manager instance
        from src.core.qa.default_questions_manager import get_default_questions_manager
        self.manager = get_default_questions_manager()

        # Track checkbox variables
        self._checkboxes: list[tuple[ctk.CTkCheckBox, ctk.BooleanVar]] = []

        self._setup_ui()

    def _setup_ui(self):
        """Create the widget layout."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header row with label and tooltip
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        header_label = ctk.CTkLabel(
            header_frame,
            text="Default Questions",
            font=FONTS["heading_sm"],
            anchor="w"
        )
        header_label.pack(side="left", padx=(0, 5))

        tooltip = TooltipIcon(
            header_frame,
            tooltip_text=(
                "Questions that are automatically asked after document processing.\n\n"
                "• Check/uncheck to enable/disable questions\n"
                "• Click '✕' to delete a question\n"
                "• Click '+ Add Question' to add new questions\n\n"
                "Disabled questions are saved but won't be asked."
            )
        )
        tooltip.pack(side="left")

        # Scrollable frame for questions
        self.scroll_frame = ctk.CTkScrollableFrame(
            self,
            height=180,
            fg_color=("gray90", "gray17"),
            corner_radius=6
        )
        self.scroll_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        self.scroll_frame.grid_columnconfigure(0, weight=1)

        # Populate with existing questions
        self._refresh_question_list()

        # Add question button
        add_frame = ctk.CTkFrame(self, fg_color="transparent")
        add_frame.grid(row=2, column=0, sticky="w")

        self.add_btn = ctk.CTkButton(
            add_frame,
            text="+ Add Question",
            command=self._add_question,
            width=140,
            height=28,
            font=FONTS["body"]
        )
        self.add_btn.pack(side="left")

    def _refresh_question_list(self):
        """Rebuild the question list from manager."""
        # Clear existing widgets
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        self._checkboxes.clear()

        # Add each question
        questions = self.manager.get_all_questions()
        for idx, q in enumerate(questions):
            self._add_question_row(idx, q.text, q.enabled)

    def _add_question_row(self, index: int, text: str, enabled: bool):
        """Add a single question row."""
        row_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        row_frame.grid(row=index, column=0, sticky="ew", pady=2, padx=5)
        row_frame.grid_columnconfigure(1, weight=1)

        # Checkbox with question text
        var = ctk.BooleanVar(value=enabled)
        checkbox = ctk.CTkCheckBox(
            row_frame,
            text="",
            variable=var,
            command=lambda i=index: self._on_toggle(i),
            width=24,
            checkbox_width=18,
            checkbox_height=18
        )
        checkbox.grid(row=0, column=0, sticky="w")

        # Question text label (clickable to edit)
        text_label = ctk.CTkLabel(
            row_frame,
            text=text,
            anchor="w",
            font=FONTS["body"],
            cursor="hand2"
        )
        text_label.grid(row=0, column=1, sticky="ew", padx=(5, 10))
        text_label.bind("<Button-1>", lambda e, i=index: self._edit_question(i))

        # Delete button
        delete_btn = ctk.CTkButton(
            row_frame,
            text="✕",
            command=lambda i=index: self._delete_question(i),
            width=24,
            height=24,
            fg_color="transparent",
            hover_color=("gray70", "gray30"),
            text_color=("gray40", "gray60"),
            font=FONTS["body"]
        )
        delete_btn.grid(row=0, column=2, sticky="e")

        self._checkboxes.append((checkbox, var))

    def _on_toggle(self, index: int):
        """Handle checkbox toggle."""
        if index < len(self._checkboxes):
            _, var = self._checkboxes[index]
            self.manager.set_enabled(index, var.get())

    def _add_question(self):
        """Show dialog to add a new question."""
        dialog = ctk.CTkInputDialog(
            text="Enter a new question:",
            title="Add Question"
        )
        text = dialog.get_input()

        if text and text.strip():
            self.manager.add_question(text.strip())
            self._refresh_question_list()

    def _edit_question(self, index: int):
        """Show dialog to edit a question."""
        questions = self.manager.get_all_questions()
        if index >= len(questions):
            return

        current_text = questions[index].text
        # Show current text in the prompt since CTkInputDialog doesn't support pre-fill
        display_text = current_text[:60] + "..." if len(current_text) > 60 else current_text
        dialog = ctk.CTkInputDialog(
            text=f"Current: \"{display_text}\"\n\nEnter new text:",
            title="Edit Question"
        )

        text = dialog.get_input()

        if text and text.strip():
            self.manager.update_question(index, text.strip())
            self._refresh_question_list()

    def _delete_question(self, index: int):
        """Delete a question after confirmation."""
        from tkinter import messagebox

        questions = self.manager.get_all_questions()
        if index >= len(questions):
            return

        question_text = questions[index].text
        # Truncate for display
        display_text = question_text[:50] + "..." if len(question_text) > 50 else question_text

        if messagebox.askyesno(
            "Delete Question",
            f"Delete this question?\n\n\"{display_text}\""
        ):
            self.manager.remove_question(index)
            self._refresh_question_list()

    def get_value(self):
        """
        Return enabled state - not needed since changes are saved immediately.

        Returns:
            None (changes are persisted via manager)
        """
        return None

    def set_value(self, value):
        """
        Set value - not needed since we load from manager.

        Args:
            value: Ignored
        """
        pass


class CorpusSettingsWidget(ctk.CTkFrame):
    """
    Widget for corpus management within Settings dialog.

    Session 64: Provides corpus status overview and quick access to full
    corpus management dialog. Shows active corpus, document count, and
    BM25 algorithm status.

    Layout:
        ┌─────────────────────────────────────────────────────────────┐
        │  Corpus Management                                       ⓘ  │
        ├─────────────────────────────────────────────────────────────┤
        │  📚 What is a Corpus?                                       │
        │  A corpus is a collection of YOUR past transcripts that     │
        │  helps LocalScribe understand which words are common in     │
        │  your work vs. unusual for a specific case.                 │
        │                                                             │
        │  ✓ 100% local and offline - never leaves your machine       │
        │  ✓ Powers the BM25 vocabulary algorithm                     │
        ├─────────────────────────────────────────────────────────────┤
        │  Status:                                                    │
        │  • Active corpus: My Transcripts                            │
        │  • Documents: 23 files                                      │
        │  • BM25 Algorithm: ✓ Active (5+ documents)                  │
        ├─────────────────────────────────────────────────────────────┤
        │  [Manage Corpus...]                                         │
        └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, parent, **kwargs):
        """
        Initialize the corpus settings widget.

        Args:
            parent: Parent widget.
            **kwargs: Additional CTkFrame arguments.
        """
        super().__init__(parent, fg_color="transparent", **kwargs)

        # Get corpus registry
        from src.core.vocabulary import get_corpus_registry
        self.registry = get_corpus_registry()

        self._setup_ui()

    def _setup_ui(self):
        """Create the widget layout."""
        self.grid_columnconfigure(0, weight=1)

        # Header with tooltip
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        header_label = ctk.CTkLabel(
            header_frame,
            text="Corpus Management",
            font=FONTS["heading"],
            anchor="w"
        )
        header_label.pack(side="left", padx=(0, 5))

        tooltip = TooltipIcon(
            header_frame,
            tooltip_text=(
                "A corpus is your collection of past transcripts used to identify "
                "case-specific vocabulary.\n\n"
                "The BM25 algorithm compares current documents against your corpus "
                "to find unusual terms - words that appear often in this case but "
                "rarely in your typical work.\n\n"
                "Requires 5+ documents to activate. More documents = better results."
            )
        )
        tooltip.pack(side="left")

        # Educational section
        edu_frame = ctk.CTkFrame(self, fg_color=("gray90", "gray17"), corner_radius=6)
        edu_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        edu_title = ctk.CTkLabel(
            edu_frame,
            text="📚 What is a Corpus?",
            font=FONTS["heading_sm"],
            anchor="w"
        )
        edu_title.pack(anchor="w", padx=15, pady=(10, 5))

        edu_text = ctk.CTkLabel(
            edu_frame,
            text=(
                "A corpus is a collection of YOUR past transcripts that helps LocalScribe\n"
                "understand which words are common in your work vs. unusual for a specific case."
            ),
            font=FONTS["body"],
            anchor="w",
            justify="left"
        )
        edu_text.pack(anchor="w", padx=15, pady=(0, 5))

        check1 = ctk.CTkLabel(
            edu_frame,
            text="✓ 100% local and offline - never leaves your machine",
            font=FONTS["body"],
            text_color=(COLORS["success"], COLORS["success_light"]),
            anchor="w"
        )
        check1.pack(anchor="w", padx=15, pady=(0, 2))

        check2 = ctk.CTkLabel(
            edu_frame,
            text="✓ Powers the BM25 vocabulary algorithm",
            font=FONTS["body"],
            text_color=(COLORS["success"], COLORS["success_light"]),
            anchor="w"
        )
        check2.pack(anchor="w", padx=15, pady=(0, 10))

        # Status section
        status_frame = ctk.CTkFrame(self, fg_color=("gray90", "gray17"), corner_radius=6)
        status_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        status_title = ctk.CTkLabel(
            status_frame,
            text="Current Status",
            font=FONTS["heading_sm"],
            anchor="w"
        )
        status_title.pack(anchor="w", padx=15, pady=(10, 5))

        # Active corpus line
        self.active_corpus_label = ctk.CTkLabel(
            status_frame,
            text="• Active corpus: Loading...",
            font=FONTS["body"],
            anchor="w"
        )
        self.active_corpus_label.pack(anchor="w", padx=15, pady=(0, 2))

        # Document count line
        self.doc_count_label = ctk.CTkLabel(
            status_frame,
            text="• Documents: Loading...",
            font=FONTS["body"],
            anchor="w"
        )
        self.doc_count_label.pack(anchor="w", padx=15, pady=(0, 2))

        # BM25 status line
        self.bm25_status_label = ctk.CTkLabel(
            status_frame,
            text="• BM25 Algorithm: Loading...",
            font=FONTS["body"],
            anchor="w"
        )
        self.bm25_status_label.pack(anchor="w", padx=15, pady=(0, 10))

        # Manage button
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=3, column=0, sticky="w", pady=(5, 0))

        self.manage_btn = ctk.CTkButton(
            btn_frame,
            text="Manage Corpus...",
            command=self._open_corpus_dialog,
            width=160,
            height=32
        )
        self.manage_btn.pack(side="left")

        # Refresh status on load
        self._refresh_status()

    def _refresh_status(self):
        """Update status labels from corpus registry."""
        try:
            active_name = self.registry.get_active_corpus()
            corpora = self.registry.list_corpora()

            # Find active corpus info
            active_corpus = None
            for c in corpora:
                if c.name == active_name:
                    active_corpus = c
                    break

            # Active corpus
            if active_corpus:
                self.active_corpus_label.configure(
                    text=f"• Active corpus: {active_corpus.name}"
                )
                doc_count = active_corpus.doc_count
            else:
                self.active_corpus_label.configure(
                    text="• Active corpus: (none)"
                )
                doc_count = 0

            # Document count
            self.doc_count_label.configure(
                text=f"• Documents: {doc_count} files"
            )

            # BM25 status
            from src.config import CORPUS_MIN_DOCUMENTS
            if doc_count >= CORPUS_MIN_DOCUMENTS:
                self.bm25_status_label.configure(
                    text=f"• BM25 Algorithm: ✓ Active ({doc_count}/{CORPUS_MIN_DOCUMENTS}+ documents)",
                    text_color=(COLORS["success"], COLORS["success_light"])
                )
            else:
                needed = CORPUS_MIN_DOCUMENTS - doc_count
                self.bm25_status_label.configure(
                    text=f"• BM25 Algorithm: ○ Inactive (need {needed} more documents)",
                    text_color=(COLORS["warning"], COLORS["warning"])
                )

        except Exception as e:
            from src.logging_config import debug_log
            debug_log(f"[CorpusSettingsWidget] Error refreshing status: {e}")
            self.active_corpus_label.configure(text="• Active corpus: (error)")
            self.doc_count_label.configure(text="• Documents: (error)")
            self.bm25_status_label.configure(text="• BM25 Algorithm: (error)")

    def _open_corpus_dialog(self):
        """Open the full corpus management dialog."""
        from src.ui.corpus_dialog import CorpusDialog

        # Find the settings dialog (our grandparent) to use as parent
        parent = self.winfo_toplevel()

        dialog = CorpusDialog(parent)
        parent.wait_window(dialog)

        # Refresh status after dialog closes
        if dialog.corpus_changed:
            self._refresh_status()

            # Also notify main window to refresh its corpus dropdown
            try:
                main_window = parent.master
                if main_window and hasattr(main_window, '_refresh_corpus_dropdown'):
                    main_window._refresh_corpus_dropdown()
            except Exception:
                pass  # Main window refresh is best-effort

    def get_value(self):
        """No value to return - changes are made via dialog."""
        return None

    def set_value(self, value):
        """No value to set - loads from registry."""
        pass
