"""
Settings Dialog for CasePrepd.

Dynamically generates a tabbed settings interface by reading from
the SettingsRegistry. Each category becomes a tab, and each setting
gets an appropriate widget based on its type.

Features:
- Tabbed interface (one tab per category)
- Auto-generated UI from registry metadata
- Tooltip icons for each setting
- Apply immediately on save
- Resizable dialog

Usage:
    from src.ui.settings import SettingsDialog
    dialog = SettingsDialog(parent=root, on_save_callback=my_callback)
"""

import customtkinter as ctk

from src.ui.base_dialog import BaseModalDialog
from src.ui.theme import COLORS, FONTS

from .settings_registry import SettingsRegistry, SettingType
from .settings_widgets import (
    ButtonSetting,
    CheckboxSetting,
    DropdownSetting,
    SliderSetting,
    SpinboxSetting,
)


class SettingsDialog(BaseModalDialog):
    """
    Tabbed settings dialog with auto-generated UI.

    Reads setting definitions from SettingsRegistry and creates
    appropriate widgets for each. Settings are applied immediately
    when the user clicks Save.

    Attributes:
        on_save_callback: Optional function called after saving.
        widgets: Dict mapping setting keys to widget instances.
    """

    def __init__(self, parent=None, on_save_callback=None, initial_tab: str | None = None):
        """
        Initialize the settings dialog.

        Args:
            parent: Parent window (dialog will be modal to this).
            on_save_callback: Optional callback after settings are saved.
            initial_tab: Optional tab name to open initially (e.g., "Questions").
        """
        super().__init__(
            parent=parent,
            title="Settings",
            width=700,
            height=520,
            min_width=550,
            min_height=420,
        )

        self.on_save_callback = on_save_callback
        self.widgets: dict[str, ctk.CTkFrame] = {}
        self._initial_tab = initial_tab

        self._setup_ui()
        self._load_current_values()
        self._setup_dependencies()

    def _setup_ui(self):
        """Create the tabbed interface with all settings."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Title
        title_frame = ctk.CTkFrame(self, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 5))

        title = ctk.CTkLabel(title_frame, text="Application Settings", font=FONTS["heading_xl"])
        title.pack(anchor="w")

        subtitle = ctk.CTkLabel(
            title_frame,
            text="Configure CasePrepd behavior and performance",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"],
        )
        subtitle.pack(anchor="w")

        # Tab view - with more prominent styling
        self.tabview = ctk.CTkTabview(
            self,
            corner_radius=8,
            segmented_button_fg_color=("gray75", "gray30"),
            segmented_button_selected_color=("#3B8ED0", "#1F6AA5"),
            segmented_button_selected_hover_color=("#36719F", "#144870"),
            text_color=("gray10", "gray90"),
        )
        self.tabview.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        # Make tab buttons larger and bolder
        self.tabview._segmented_button.configure(font=FONTS["heading"], height=36)

        # Create tabs from registry
        categories = SettingsRegistry.get_categories()
        if not categories:
            # Fallback if registry is empty (shouldn't happen)
            self._show_empty_state()
            return

        for category in categories:
            tab = self.tabview.add(category)
            tab.grid_columnconfigure(0, weight=1)
            self._populate_tab(tab, category)

        # Set initial tab (user-specified or first tab)
        if categories:
            if self._initial_tab and self._initial_tab in categories:
                self.tabview.set(self._initial_tab)
            else:
                self.tabview.set(categories[0])

        # Button frame
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(10, 20))
        btn_frame.grid_columnconfigure(0, weight=1)

        cancel_btn = ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=self.destroy,
            width=100,
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
        )
        cancel_btn.grid(row=0, column=1, padx=(0, 10))

        save_btn = ctk.CTkButton(btn_frame, text="Save", command=self._save, width=100)
        save_btn.grid(row=0, column=2)

    def _show_empty_state(self):
        """Show message when no settings are registered."""
        empty_label = ctk.CTkLabel(
            self,
            text="No settings available.",
            font=FONTS["heading"],
            text_color=COLORS["text_secondary"],
        )
        empty_label.grid(row=1, column=0, pady=50)

    def _populate_tab(self, tab: ctk.CTkFrame, category: str):
        """
        Add setting widgets to a tab.

        For tabs with section-grouped settings (like Advanced), creates
        collapsible sections. Otherwise renders flat list of widgets.

        Args:
            tab: The tab frame to populate.
            category: Category name to get settings for.
        """
        settings = SettingsRegistry.get_settings_for_category(category)

        # Check if this tab uses collapsible sections
        has_sections = any(s.section and s.section != "_header" for s in settings)

        # Create a scrollable frame for many settings
        scroll_frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)
        scroll_frame.grid_columnconfigure(0, weight=1)

        if not has_sections:
            # Standard flat layout
            for idx, setting in enumerate(settings):
                widget = self._create_widget(scroll_frame, setting)
                widget.grid(row=idx, column=0, sticky="ew", pady=10, padx=5)
                self.widgets[setting.key] = widget
        else:
            self._populate_sectioned_tab(scroll_frame, settings)

    def _populate_sectioned_tab(self, scroll_frame, settings):
        """
        Populate a tab with collapsible sections.

        Groups settings by their `section` field and renders each group
        inside a CollapsibleSection widget. Header items (section="_header")
        are rendered at the top without a section wrapper.

        Args:
            scroll_frame: Scrollable frame to add sections to.
            settings: List of SettingDefinition objects.
        """
        from .advanced_registry import reset_section
        from .advanced_sections import CollapsibleSection

        row_idx = 0

        # Render header items first (e.g., "Restore All Defaults" button)
        for setting in settings:
            if setting.section == "_header":
                widget = self._create_widget(scroll_frame, setting)
                widget.grid(row=row_idx, column=0, sticky="ew", pady=(5, 10), padx=5)
                self.widgets[setting.key] = widget
                row_idx += 1

        # Group remaining settings by section (preserve order)
        sections_ordered = []
        section_settings = {}
        for setting in settings:
            if setting.section and setting.section != "_header":
                if setting.section not in section_settings:
                    sections_ordered.append(setting.section)
                    section_settings[setting.section] = []
                section_settings[setting.section].append(setting)

        # Create collapsible sections
        for section_name in sections_ordered:
            section_items = section_settings[section_name]

            def make_reset_fn(name):
                def _reset():
                    from tkinter import messagebox

                    result = messagebox.askyesno(
                        "Reset Section",
                        f"Reset all '{name}' settings to their defaults?",
                        icon="question",
                    )
                    if result:
                        reset_section(name)
                        # Reload widget values
                        for s in section_settings.get(name, []):
                            w = self.widgets.get(s.key)
                            if w and s.getter:
                                w.set_value(s.getter())

                return _reset

            section = CollapsibleSection(
                scroll_frame,
                title=section_name,
                on_reset=make_reset_fn(section_name),
            )
            section.grid(row=row_idx, column=0, sticky="ew", pady=2, padx=2)
            row_idx += 1

            # Add setting widgets to the section content
            for widget_idx, setting in enumerate(section_items):
                widget = self._create_widget(section.content_frame, setting)
                widget.grid(row=widget_idx, column=0, sticky="ew", pady=6, padx=5)
                self.widgets[setting.key] = widget

    def _create_widget(self, parent, setting) -> ctk.CTkFrame:
        """
        Create appropriate widget based on setting type.

        Args:
            parent: Parent frame for the widget.
            setting: SettingDefinition with metadata.

        Returns:
            The created widget (subclass of SettingRow).

        Raises:
            ValueError: If setting type is unknown.
        """
        # Get initial value from getter or use default
        initial_value = setting.getter() if setting.getter else setting.default

        # Resolve callable tooltip/options at dialog-open time (not import time)
        tooltip = setting.tooltip() if callable(setting.tooltip) else setting.tooltip
        options = setting.options() if callable(setting.options) else setting.options

        if setting.setting_type == SettingType.SLIDER:
            return SliderSetting(
                parent,
                label=setting.label,
                tooltip=tooltip,
                min_value=setting.min_value,
                max_value=setting.max_value,
                step=setting.step,
                initial_value=initial_value,
            )

        elif setting.setting_type == SettingType.CHECKBOX:
            return CheckboxSetting(
                parent,
                label=setting.label,
                tooltip=tooltip,
                initial_value=initial_value,
            )

        elif setting.setting_type == SettingType.DROPDOWN:
            return DropdownSetting(
                parent,
                label=setting.label,
                tooltip=tooltip,
                options=options,
                initial_value=initial_value,
            )

        elif setting.setting_type == SettingType.SPINBOX:
            return SpinboxSetting(
                parent,
                label=setting.label,
                tooltip=tooltip,
                min_value=int(setting.min_value),
                max_value=int(setting.max_value),
                initial_value=initial_value,
            )

        elif setting.setting_type == SettingType.BUTTON:
            return ButtonSetting(
                parent,
                label=setting.label,
                tooltip=tooltip,
                action=setting.action,
            )

        elif setting.setting_type == SettingType.CUSTOM:
            # CUSTOM type uses a widget_factory to create the widget
            if setting.widget_factory:
                return setting.widget_factory(parent)
            else:
                raise ValueError(f"CUSTOM setting '{setting.key}' requires widget_factory")

        else:
            raise ValueError(f"Unknown setting type: {setting.setting_type}")

    def _load_current_values(self):
        """Load current values from getters into widgets."""
        for setting in SettingsRegistry.get_all_settings():
            widget = self.widgets.get(setting.key)
            if widget and setting.getter:
                try:
                    value = setting.getter()
                    widget.set_value(value)
                except Exception as e:
                    # Use default if getter fails (e.g., missing preference)
                    from src.logging_config import debug_log

                    debug_log(f"[Settings] Getter failed for {setting.key}: {e}")
                    widget.set_value(setting.default)

    def _save(self):
        """
        Apply all settings immediately and close dialog.

        If validation fails for any setting, shows an error message
        and does NOT close the dialog, allowing user to correct the value.
        """
        from tkinter import messagebox

        from src.logging_config import debug_log

        errors = []

        for setting in SettingsRegistry.get_all_settings():
            widget = self.widgets.get(setting.key)
            if widget and setting.setter:
                try:
                    value = widget.get_value()
                    setting.setter(value)
                except ValueError as ve:
                    # Validation error - collect for user display
                    errors.append(f"• {setting.label}: {ve}")
                    debug_log(f"[Settings] Validation error for {setting.key}: {ve}")
                except Exception as e:
                    # Other errors - log but continue
                    debug_log(f"[Settings] Error saving {setting.key}: {e}")

        # If there were validation errors, show them and stay open
        if errors:
            error_msg = "Please correct the following:\n\n" + "\n".join(errors)
            messagebox.showerror("Invalid Settings", error_msg)
            return  # Don't close dialog

        # Call the callback if provided
        if self.on_save_callback:
            try:
                self.on_save_callback()
            except Exception as e:
                debug_log(f"[Settings] Error in save callback: {e}")

        self.destroy()

    def _setup_dependencies(self):
        """
        Set up dependencies between settings.

        For example, the manual worker count should be disabled
        when auto-detect CPU cores is enabled.
        """
        # Link auto-detect checkbox to worker count spinbox
        auto_detect_widget = self.widgets.get("parallel_workers_auto")
        worker_count_widget = self.widgets.get("parallel_workers_count")

        if (
            auto_detect_widget
            and worker_count_widget
            and hasattr(worker_count_widget, "set_enabled")
        ):
            # Set initial state based on current auto-detect value
            auto_enabled = auto_detect_widget.get_value()
            worker_count_widget.set_enabled(not auto_enabled)

            # Update when checkbox changes
            original_on_change = auto_detect_widget.on_change

            def on_auto_detect_change(value):
                worker_count_widget.set_enabled(not value)
                if original_on_change:
                    original_on_change(value)

            auto_detect_widget.on_change = on_auto_detect_change
            # Also update the checkbox command
            auto_detect_widget.checkbox.configure(
                command=lambda: on_auto_detect_change(auto_detect_widget.get_value())
            )

        # Link logging level dropdown to customize button visibility
        logging_widget = self.widgets.get("logging_level")
        customize_widget = self.widgets.get("customize_logging")

        if logging_widget and customize_widget:
            # Hide button initially if not in custom mode
            if logging_widget.get_value() != "custom":
                customize_widget.grid_remove()

            original_log_change = logging_widget.on_change

            def on_level_change(value):
                if value == "custom":
                    customize_widget.grid()
                else:
                    customize_widget.grid_remove()
                if original_log_change:
                    original_log_change(value)

            logging_widget.on_change = on_level_change
            logging_widget.dropdown.configure(
                command=lambda v: on_level_change(logging_widget.value_map.get(v))
            )
