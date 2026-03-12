"""
Base Settings Widget

Abstract base class for custom settings widgets used in the
Settings dialog. Provides a consistent interface that SettingsDialog
can interact with polymorphically.

All custom settings widgets (ColumnVisibilityWidget, DefaultQuestionsWidget,
CorpusSettingsWidget, IndicatorPatternWidget, CustomPatternsWidget) inherit
from this base.

Usage:
    class MyWidget(BaseSettingsWidget):
        def _setup_ui(self):
            # Build widget UI here
            pass

        def get_value(self) -> Any:
            return self._my_data

        def set_value(self, value: Any) -> None:
            self._my_data = value
"""

from abc import ABC, abstractmethod
from typing import Any

import customtkinter as ctk


class BaseSettingsWidget(ctk.CTkFrame, ABC):
    """
    Abstract base for custom settings widgets in the Settings dialog.

    Provides a consistent interface with get_value(), set_value(),
    and optional validate(). All widgets use transparent background.

    Subclasses must implement:
        - _setup_ui(): Build the widget's UI
        - get_value(): Return current widget value
        - set_value(value): Set widget to given value
    """

    def __init__(self, parent, **kwargs):
        """
        Initialize the settings widget with transparent background.

        Args:
            parent: Parent widget.
            **kwargs: Additional CTkFrame arguments.
        """
        super().__init__(parent, fg_color="transparent", **kwargs)
        self._setup_ui()

    @abstractmethod
    def _setup_ui(self) -> None:
        """Build the widget's UI. Called during __init__."""
        pass

    @abstractmethod
    def get_value(self) -> Any:
        """
        Return the current widget value for persistence.

        Returns:
            The widget's current value (type varies by subclass).
        """
        pass

    @abstractmethod
    def set_value(self, value: Any) -> None:
        """
        Set the widget to the given value.

        Args:
            value: Value to set (type varies by subclass).
        """
        pass

    def validate(self) -> str | None:
        """
        Validate the current widget state.

        Override in subclasses that need validation.

        Returns:
            None if valid, error message string if invalid.
        """
        return None
