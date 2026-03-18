"""
Settings widgets with integrated tooltips.

Each widget includes an info icon that shows explanatory text on hover.
Uses CustomTkinter for modern appearance consistent with CasePrepd.

Widget Types:
- TooltipIcon: Info icon with hover tooltip
- SettingRow: Base class with label + tooltip layout
- SliderSetting: Numeric range (int/float)
- CheckboxSetting: Boolean toggle
- DropdownSetting: Selection from options
- SpinboxSetting: Integer with +/- buttons

Layout Standard:
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

import tkinter as tk
from collections.abc import Callable
from typing import Any

import customtkinter as ctk

from src.ui.theme import COLORS, FONTS
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
    immediately when the mouse leaves — no straddling logic.

    Best practices (v1.0.19 refactor):
    - 400ms show delay prevents flicker during mouse traversal
    - Immediate hide on <Leave> (no delayed position checks)
    - <ButtonPress> hides tooltip so it doesn't obscure click targets
    - <Destroy> cleans up tooltip when icon widget is destroyed
    - Timer IDs stored and cancelled on every event entry
    - All bindings use add="+" to preserve existing handlers

    Attributes:
        tooltip_text: The help text to display on hover.
        tooltip_window: Reference to the popup window (if visible).
    """

    _SHOW_DELAY_MS = 400

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
            **kwargs,
        )
        self.tooltip_text = tooltip_text
        self.tooltip_window = None
        self._show_timer = None

        self.bind("<Enter>", self._on_enter, add="+")
        self.bind("<Leave>", self._on_leave, add="+")
        self.bind("<ButtonPress>", self._on_leave, add="+")
        self.bind("<Destroy>", self._on_destroy, add="+")

    def _cancel_show(self):
        """Cancel any pending show timer."""
        if self._show_timer is not None:
            try:
                self.after_cancel(self._show_timer)
            except (tk.TclError, RuntimeError):
                pass
            self._show_timer = None

    def _on_enter(self, event=None):
        """Schedule tooltip display after delay."""
        self._cancel_show()
        self._show_timer = self.after(self._SHOW_DELAY_MS, self._show_tooltip)

    def _on_leave(self, event=None):
        """Hide tooltip immediately on mouse leave or click."""
        self._cancel_show()
        self._force_hide_tooltip()

    def _on_destroy(self, event=None):
        """Clean up tooltip when icon widget is destroyed."""
        self._cancel_show()
        self._force_hide_tooltip()

    def _show_tooltip(self):
        """Display tooltip popup near the icon."""
        self._show_timer = None

        if self.tooltip_window:
            return

        try:
            if not self.winfo_exists():
                return
        except (tk.TclError, RuntimeError):
            return

        tooltip_manager.close_active()

        x = self.winfo_rootx() + 25
        y = self.winfo_rooty() - 5

        try:
            self.tooltip_window = ctk.CTkToplevel(self)
        except (tk.TclError, RuntimeError):
            return

        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        self.tooltip_window.attributes("-topmost", True)

        try:
            self.tooltip_window.wm_attributes("-toolwindow", True)
        except (tk.TclError, RuntimeError):
            pass

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

        tooltip_manager.register(self.tooltip_window, owner=self)

    def _force_hide_tooltip(self):
        """Unconditionally hide and destroy the tooltip."""
        if self.tooltip_window:
            tooltip_manager.unregister(self.tooltip_window)
            try:
                self.tooltip_window.destroy()
            except (tk.TclError, RuntimeError):
                pass
            self.tooltip_window = None


class SettingRow(ctk.CTkFrame):
    """
    Base class for setting rows with label + tooltip + widget.

    Provides consistent layout and tooltip handling for all setting types.
    Subclasses implement get_value() and set_value() for their widget.

    Layout (Standardized):
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
        on_change: Callable[[Any], None] | None = None,
        **kwargs,
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

        # Standardized layout with fixed column widths
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
            label_frame, text=label, anchor="w", font=FONTS["heading_sm"]
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

    Slider uses CONTROL_WIDTH for consistent positioning.
    """

    def __init__(
        self,
        parent,
        label: str,
        tooltip: str,
        min_value: float,
        max_value: float,
        step: float = 1,
        initial_value: float | None = None,
        on_change: Callable[[float], None] | None = None,
        **kwargs,
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

        # Slider with fixed width for consistent positioning
        self.slider = ctk.CTkSlider(
            self,
            from_=min_value,
            to=max_value,
            number_of_steps=num_steps,
            width=CONTROL_WIDTH,  # Fixed width for all sliders
            command=self._on_slider_change,
        )
        self.slider.grid(row=0, column=2, sticky="e", padx=(CONTROL_PADDING_X, 0))

        # Value display label (shows current value) - right of slider
        self.value_label = ctk.CTkLabel(
            self, text="", width=VALUE_LABEL_WIDTH, anchor="e", font=FONTS["heading_sm"]
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

    Checkbox is positioned at start of control area for consistency.
    """

    def __init__(
        self,
        parent,
        label: str,
        tooltip: str,
        initial_value: bool = False,
        on_change: Callable[[bool], None] | None = None,
        **kwargs,
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
            checkbox_height=20,
        )
        # Left-align checkbox within the control column
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

    Dropdown uses CONTROL_WIDTH for consistent positioning.
    """

    def __init__(
        self,
        parent,
        label: str,
        tooltip: str,
        options: list[tuple[str, Any]],
        initial_value: Any = None,
        on_change: Callable[[Any], None] | None = None,
        **kwargs,
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
        self.value_map = dict(options)
        self.text_map = {val: text for text, val in options}

        display_values = [text for text, _ in options]
        initial_text = self.text_map.get(initial_value, display_values[0] if display_values else "")

        # Use CONTROL_WIDTH for consistent positioning
        self.dropdown = ctk.CTkComboBox(
            self,
            values=display_values,
            command=self._on_dropdown_change,
            state="readonly",
            width=CONTROL_WIDTH,
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

    Spinbox is positioned consistently with other controls.
    """

    def __init__(
        self,
        parent,
        label: str,
        tooltip: str,
        min_value: int,
        max_value: int,
        initial_value: int | None = None,
        on_change: Callable[[int], None] | None = None,
        **kwargs,
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

        # Container with fixed width for consistent positioning
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
            font=FONTS["heading_lg"],
        )
        self.minus_btn.pack(side="left")

        # Value display
        self.value_label = ctk.CTkLabel(
            spinbox_frame, text=str(self.value), width=45, font=FONTS["heading"]
        )
        self.value_label.pack(side="left", padx=8)

        # Plus button
        self.plus_btn = ctk.CTkButton(
            spinbox_frame,
            text="+",
            width=32,
            height=28,
            command=self._increment,
            font=FONTS["heading_lg"],
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

    Button uses CONTROL_WIDTH for consistent positioning.
    """

    def __init__(
        self,
        parent,
        label: str,
        tooltip: str,
        action: callable,
        button_text: str | None = None,
        **kwargs,
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

        # Use CONTROL_WIDTH for consistent positioning
        self.button = ctk.CTkButton(
            self,
            text=button_text or label,
            command=self._on_click,
            width=CONTROL_WIDTH,
            height=28,
            font=FONTS["body"],
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


# Specialized widgets live in separate files for modularity:
# - DefaultQuestionsWidget → questions_widget.py
# - CorpusSettingsWidget → corpus_widget.py
# - ColumnVisibilityWidget → columns_widget.py
