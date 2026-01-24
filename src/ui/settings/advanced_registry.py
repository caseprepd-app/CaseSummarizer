"""
Advanced settings registration for CasePrepd.

Registers all Advanced tab settings with the SettingsRegistry,
reading default values from config_defaults.py and tooltip metadata
from config_defaults_meta.py. Includes a "Restore All Defaults"
button at the top of the tab.
"""

from src.config_defaults import DEFAULTS
from src.config_defaults_meta import DESCRIPTIONS

from .settings_registry import SettingDefinition, SettingsRegistry, SettingType

_CATEGORY = "Advanced"


def _map_setting_type(entry: dict) -> SettingType:
    """
    Map a config_defaults type string to a SettingType enum.

    Args:
        entry: Dict from DEFAULTS with 'type' and optional 'options'.

    Returns:
        Appropriate SettingType for widget rendering.
    """
    dtype = entry["type"]
    if dtype == "bool":
        return SettingType.CHECKBOX
    if dtype in ("dropdown", "dropdown_int"):
        return SettingType.DROPDOWN
    if dtype == "int" and (entry.get("max", 0) - entry.get("min", 0)) <= 20:
        return SettingType.SPINBOX
    return SettingType.SLIDER


def _register_advanced_settings():
    """
    Register all advanced settings from DEFAULTS with the SettingsRegistry.

    Called by settings_registry.py on import.
    """
    from src.user_preferences import get_user_preferences

    prefs = get_user_preferences()

    # --- Restore All Defaults button ---
    def _restore_all_defaults():
        """Reset all Advanced settings to their default values."""
        from tkinter import messagebox

        result = messagebox.askyesno(
            "Restore All Defaults",
            "Reset all Advanced settings to their default values?\n\n"
            "This will undo any customizations you've made in the Advanced tab.",
            icon="question",
        )
        if result:
            for key, entry in DEFAULTS.items():
                prefs.set(key, entry["value"])
            messagebox.showinfo(
                "Defaults Restored",
                "All Advanced settings have been restored to defaults.\n\n"
                "Click Save to apply, or Cancel to discard.",
            )

    SettingsRegistry.register(
        SettingDefinition(
            key="advanced_restore_all_defaults",
            label="Restore All Defaults",
            category=_CATEGORY,
            setting_type=SettingType.BUTTON,
            tooltip="Reset all Advanced settings to their factory default values.",
            default=None,
            action=_restore_all_defaults,
            section="_header",
        )
    )

    # --- Register each setting ---
    for key, entry in DEFAULTS.items():
        meta = DESCRIPTIONS.get(key, {})
        label = meta.get("label", key.replace("_", " ").title())
        tooltip = meta.get("tooltip", "")
        section = entry["category"]
        setting_type = _map_setting_type(entry)
        default_val = entry["value"]

        kwargs = {
            "key": key,
            "label": label,
            "category": _CATEGORY,
            "setting_type": setting_type,
            "tooltip": tooltip,
            "default": default_val,
            "section": section,
        }

        # Type-specific attributes
        if setting_type == SettingType.SLIDER:
            kwargs["min_value"] = entry["min"]
            kwargs["max_value"] = entry["max"]
            kwargs["step"] = entry.get("step", 1 if entry["type"] == "int" else 0.05)
        elif setting_type == SettingType.SPINBOX:
            kwargs["min_value"] = entry["min"]
            kwargs["max_value"] = entry["max"]
        elif setting_type == SettingType.DROPDOWN:
            kwargs["options"] = entry.get("options", [])

        # Getter/setter using prefs with default fallback
        if setting_type == SettingType.CHECKBOX:
            _key, _default = key, default_val
            kwargs["getter"] = lambda _k=_key, _d=_default: prefs.get(_k, _d)
            kwargs["setter"] = lambda v, _k=_key: prefs.set(_k, v)
        elif setting_type == SettingType.DROPDOWN:
            _key, _default = key, default_val
            if entry["type"] == "dropdown_int":
                kwargs["getter"] = lambda _k=_key, _d=_default: prefs.get(_k, _d)
                kwargs["setter"] = lambda v, _k=_key: prefs.set(
                    _k, int(v) if isinstance(v, str) and v.isdigit() else v
                )
            else:
                kwargs["getter"] = lambda _k=_key, _d=_default: prefs.get(_k, _d)
                kwargs["setter"] = lambda v, _k=_key: prefs.set(_k, v)
        elif entry["type"] == "int":
            _key, _default = key, default_val
            kwargs["getter"] = lambda _k=_key, _d=_default: prefs.get(_k, _d)
            kwargs["setter"] = lambda v, _k=_key: prefs.set(_k, int(v))
        elif entry["type"] == "float":
            _key, _default = key, default_val
            kwargs["getter"] = lambda _k=_key, _d=_default: prefs.get(_k, _d)
            kwargs["setter"] = lambda v, _k=_key: prefs.set(_k, float(v))

        SettingsRegistry.register(SettingDefinition(**kwargs))


def _get_section_keys(section_name: str) -> list[str]:
    """
    Get all setting keys belonging to a section.

    Args:
        section_name: Section name (matches category in DEFAULTS).

    Returns:
        List of setting keys in that section.
    """
    return [k for k, v in DEFAULTS.items() if v["category"] == section_name]


def reset_section(section_name: str) -> None:
    """
    Reset all settings in a section to their defaults.

    Args:
        section_name: Section name to reset.
    """
    from src.user_preferences import get_user_preferences

    prefs = get_user_preferences()
    for key in _get_section_keys(section_name):
        prefs.set(key, DEFAULTS[key]["value"])
