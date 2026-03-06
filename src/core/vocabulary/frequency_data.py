"""
Google word frequency data loader.

Shared loader for the Google 333K word frequency TSV file.
Provides thread-safe cached loading of raw {word: count} data.
Callers do their own post-processing (percentile scaling, rank map, top-N set).
"""

import logging
import threading

from src.config import GOOGLE_WORD_FREQUENCY_FILE

logger = logging.getLogger(__name__)

_raw_frequencies: dict[str, int] | None = None
_raw_lock = threading.Lock()


def load_raw_frequency_data() -> dict[str, int]:
    """
    Load the Google word frequency TSV file.

    Thread-safe with double-check locking. Cached after first load.

    Returns:
        Dict mapping lowercase word to frequency count.
        Empty dict if file not found or parse error.
    """
    global _raw_frequencies

    if _raw_frequencies is not None:
        return _raw_frequencies

    with _raw_lock:
        if _raw_frequencies is not None:
            return _raw_frequencies

        if not GOOGLE_WORD_FREQUENCY_FILE.exists():
            logger.debug("Frequency file not found: %s", GOOGLE_WORD_FREQUENCY_FILE)
            _raw_frequencies = {}
            return _raw_frequencies

        result: dict[str, int] = {}
        try:
            with open(GOOGLE_WORD_FREQUENCY_FILE, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        try:
                            result[parts[0].lower()] = int(parts[1])
                        except ValueError:
                            logger.debug("Skipping bad frequency line: %s", line.strip()[:80])
                            continue

            logger.debug("Loaded %s words from frequency dataset", len(result))
        except Exception as e:
            logger.debug("Error loading frequency dataset: %s", e)

        _raw_frequencies = result
        return _raw_frequencies
