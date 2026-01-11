"""
Settings UI Package for CasePrepd.

Provides an extensible settings dialog with:
- Tabbed interface organized by category
- Auto-generated UI from SettingsRegistry
- Tooltips with info icons for each setting
- Immediate apply on save

Usage:
    from src.ui.settings import SettingsDialog
    SettingsDialog(parent=root, on_save_callback=my_callback)

Adding new settings:
    Add a SettingsRegistry.register() call in settings_registry.py.
    The UI will automatically include it - no other changes needed.

Layout Standard (Session 62b):
    All settings use a consistent 2/3 - 1/3 layout where:
    - Labels + tooltip icons occupy the left portion (LABEL_AREA_WIDTH)
    - Controls occupy the right portion (CONTROL_WIDTH)

    To create a custom widget, subclass SettingRow and use CONTROL_WIDTH
    for your widget's width to ensure consistent alignment.

    Available layout constants:
    - LABEL_AREA_WIDTH: Width of label area (280px)
    - CONTROL_WIDTH: Width of control area (220px)
    - VALUE_LABEL_WIDTH: Width for value displays (50px)
    - CONTROL_PADDING_X: Padding between elements (10px)
"""

from .settings_dialog import SettingsDialog
from .settings_registry import (
    SettingDefinition,
    SettingsRegistry,
    SettingType,
)
from .settings_widgets import (
    CONTROL_PADDING_X,
    CONTROL_WIDTH,
    # Layout constants for custom widget creation
    LABEL_AREA_WIDTH,
    VALUE_LABEL_WIDTH,
    CheckboxSetting,
    DropdownSetting,
    SettingRow,
    SliderSetting,
    SpinboxSetting,
    TooltipIcon,
)

__all__ = [
    "CONTROL_PADDING_X",
    "CONTROL_WIDTH",
    # Layout constants (Session 62b)
    "LABEL_AREA_WIDTH",
    "VALUE_LABEL_WIDTH",
    "CheckboxSetting",
    "DropdownSetting",
    "SettingDefinition",
    "SettingRow",
    "SettingType",
    # Main dialog
    "SettingsDialog",
    # Registry
    "SettingsRegistry",
    "SliderSetting",
    "SpinboxSetting",
    # Widgets
    "TooltipIcon",
]
