"""
Base Named Component

Shared abstract base for all named, toggleable pipeline components.
Eliminates repeated name/enabled/get_config/repr boilerplate across:
- BasePreprocessor
- BaseExtractionAlgorithm
- BaseRetrievalAlgorithm
- BaseVocabularyFilter

Each subclass adds its own domain-specific abstract methods.
"""

from abc import ABC
from typing import Any


class BaseNamedComponent(ABC):
    """
    Abstract base for named, toggleable pipeline components.

    Provides shared attributes and methods that all pipeline
    components need: a human-readable name, an enabled flag,
    serializable config, and a readable repr.

    Attributes:
        name: Human-readable name for logging and display.
        enabled: Whether this component is active.
    """

    name: str = "BaseComponent"
    enabled: bool = True

    def get_config(self) -> dict[str, Any]:
        """
        Return component configuration for serialization/logging.

        Override in subclass to include component-specific settings.

        Returns:
            Dictionary of configuration values.
        """
        return {
            "name": self.name,
            "enabled": self.enabled,
        }

    def __repr__(self) -> str:
        """Return readable representation with class name and state."""
        extras = self._repr_extras()
        extra_str = ", ".join(f"{k}={v!r}" for k, v in extras.items())
        base = f"enabled={self.enabled}"
        if extra_str:
            base = f"{base}, {extra_str}"
        return f"{self.__class__.__name__}({base})"

    def _repr_extras(self) -> dict[str, Any]:
        """
        Return extra fields for repr. Override in subclasses.

        Returns:
            Dict of field_name -> value to include in repr.
        """
        return {}
