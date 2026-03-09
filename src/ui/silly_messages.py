"""
Silly Messages Module

Provides fun, random status messages for quick processing steps
to add personality to the status bar during brief transitions.
"""

import logging
import random
from pathlib import Path

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
_config_path = Path(__file__).parent.parent.parent / "config" / "silly_messages.txt"
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


def get_silly_message(chance: float = 0.2) -> str:
    """
    Return a random silly message for status bar flavor.

    Only returns a silly message some of the time (default 20%).
    Otherwise returns a straightforward status message.

    Args:
        chance: Probability (0.0-1.0) of returning a silly message.

    Returns:
        str: A silly or straightforward status message.
    """
    if random.random() < chance:
        return random.choice(_messages)
    return "Finalizing documents..."
