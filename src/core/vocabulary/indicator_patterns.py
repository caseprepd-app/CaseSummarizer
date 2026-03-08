"""
User-Defined Indicator Pattern Compiler for Vocabulary ML.

Compiles user-defined string lists into OR regexes for matching
vocabulary terms against positive/negative indicator patterns.
These become boolean ML features that the preference learner trains on.

Usage:
    from src.core.vocabulary.indicator_patterns import matches_positive, matches_negative

    if matches_positive("Dr. Smith"):
        ...  # User-defined positive indicator matched
"""

import logging
import re

from src.user_preferences import get_user_preferences

logger = logging.getLogger(__name__)

# Cache: (frozen_strings_tuple, override_str) -> compiled regex
_cache: dict[str, tuple[tuple, str, re.Pattern | None]] = {}


def _compile_pattern(strings: list[str], regex_override: str) -> re.Pattern | None:
    """
    Compile a list of strings into a case-insensitive OR regex.

    If regex_override is non-empty, uses that instead of auto-generating.
    Returns None if the list is empty and no override is set.

    Args:
        strings: List of indicator strings (e.g., ["dr.", "plaintiff"]).
        regex_override: Raw regex string to use instead of auto-generating.

    Returns:
        Compiled regex pattern, or None if no patterns defined.

    Raises:
        No exceptions — logs warnings for invalid regex and returns None.
    """
    if regex_override and regex_override.strip():
        try:
            return re.compile(regex_override.strip(), re.IGNORECASE)
        except re.error as e:
            logger.warning("Invalid regex override '%s': %s", regex_override, e)
            return None
        except Exception as e:
            logger.error("Unexpected error compiling regex override: %s", e, exc_info=True)
            return None

    if not strings:
        return None

    # Escape each string for literal matching, join with OR
    escaped = [re.escape(s) for s in strings if s.strip()]
    if not escaped:
        return None

    pattern = f"(?i)(?:{'|'.join(escaped)})"
    try:
        return re.compile(pattern)
    except re.error as e:
        logger.warning("Failed to compile indicator pattern: %s", e)
        return None
    except Exception as e:
        logger.error("Unexpected error compiling indicator pattern: %s", e, exc_info=True)
        return None


def _get_cached_pattern(cache_key: str, strings: list[str], override: str) -> re.Pattern | None:
    """
    Get a compiled pattern from cache, recompiling only when inputs change.

    Args:
        cache_key: "positive" or "negative".
        strings: Current indicator strings.
        override: Current regex override string.

    Returns:
        Compiled regex pattern, or None.
    """
    frozen = tuple(strings)
    cached = _cache.get(cache_key)
    if cached and cached[0] == frozen and cached[1] == override:
        return cached[2]

    pattern = _compile_pattern(strings, override)
    _cache[cache_key] = (frozen, override, pattern)
    return pattern


def invalidate_cache() -> None:
    """Clear compiled regex cache. Call when user changes indicator patterns."""
    _cache.clear()


def matches_positive(term: str) -> bool:
    """
    Check if term matches any user-defined positive indicator pattern.

    Args:
        term: Vocabulary term to check.

    Returns:
        True if the term matches a positive indicator pattern.
    """
    prefs = get_user_preferences()
    from src.config import DEFAULT_POSITIVE_INDICATORS, DEFAULT_POSITIVE_REGEX_OVERRIDE

    strings = prefs.get("vocab_positive_indicators", DEFAULT_POSITIVE_INDICATORS)
    override = prefs.get("vocab_positive_regex_override", DEFAULT_POSITIVE_REGEX_OVERRIDE)
    pattern = _get_cached_pattern("positive", strings, override)
    if pattern is None:
        return False
    return bool(pattern.search(term))


def matches_negative(term: str) -> bool:
    """
    Check if term matches any user-defined negative indicator pattern.

    Args:
        term: Vocabulary term to check.

    Returns:
        True if the term matches a negative indicator pattern.
    """
    from src.config import DEFAULT_NEGATIVE_INDICATORS, DEFAULT_NEGATIVE_REGEX_OVERRIDE

    prefs = get_user_preferences()
    strings = prefs.get("vocab_negative_indicators", DEFAULT_NEGATIVE_INDICATORS)
    override = prefs.get("vocab_negative_regex_override", DEFAULT_NEGATIVE_REGEX_OVERRIDE)
    pattern = _get_cached_pattern("negative", strings, override)
    if pattern is None:
        return False
    return bool(pattern.search(term))


def build_regex_preview(strings: list[str]) -> str:
    """
    Build a preview of the auto-generated regex from a list of strings.

    Args:
        strings: List of indicator strings.

    Returns:
        The regex pattern string (without compilation), or empty string.
    """
    escaped = [re.escape(s) for s in strings if s.strip()]
    if not escaped:
        return ""
    return f"(?i)(?:{'|'.join(escaped)})"


def validate_regex(regex_str: str) -> str | None:
    """
    Validate a regex string.

    Args:
        regex_str: The regex to validate.

    Returns:
        None if valid, or an error message string if invalid.
    """
    if not regex_str or not regex_str.strip():
        return None
    try:
        re.compile(regex_str.strip())
        return None
    except re.error as e:
        return str(e)
