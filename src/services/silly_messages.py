"""
Silly Messages Module

Provides fun, random status messages for quick processing steps
to add personality to the status bar during brief transitions.
"""

import logging
import random

from src.core.paths import get_config_dir

logger = logging.getLogger(__name__)

_FALLBACK_MESSAGES = [
    "Rearranging electromagnetic fields...",
    "Combobulating the data...",
    "Teaching the algorithm to read...",
    "Convincing electrons to cooperate...",
    "Herding digital cats...",
]

_messages: list[str] = []

# Load messages from config file at import time
_config_path = get_config_dir() / "silly_messages.txt"
try:
    _messages = [
        line.strip()
        for line in _config_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logger.debug("Loaded %s silly messages from %s", len(_messages), _config_path)
except FileNotFoundError:
    logger.debug("Silly messages file not found, using fallback list")
    _messages = _FALLBACK_MESSAGES
except Exception as e:
    logger.error("Failed to load silly messages: %s", e, exc_info=True)
    _messages = _FALLBACK_MESSAGES


def get_silly_message() -> str:
    """
    Return a random silly message for status bar flavor.

    Frequency is controlled by callers (e.g. StatusReporter.SILLY_ODDS = 25
    gives ~4% chance).

    Returns:
        str: A random humorous status message.
    """
    return random.choice(_messages)
