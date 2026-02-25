"""
Regex Exclusion Filter

User-configurable regex patterns for filtering transcript artifacts.
Reads patterns from config/vocab_exclude_patterns.txt and removes matching terms.

This allows users to add domain-specific patterns without code changes.
"""

import logging
import re
from pathlib import Path

from src.core.vocabulary.filters.base import BaseVocabularyFilter, FilterResult

logger = logging.getLogger(__name__)


class RegexExclusionFilter(BaseVocabularyFilter):
    """
    Filters vocabulary terms matching user-defined regex patterns.

    Reads patterns from a config file (one pattern per line).
    Comments (lines starting with #) and blank lines are ignored.
    Invalid regexes are skipped with a warning.

    Person names are NOT exempt - artifacts like "Q. Smith" should be removed.

    Attributes:
        patterns_file: Path to the regex patterns file
        patterns: List of compiled regex patterns
    """

    name = "Regex Exclusion Filter"
    priority = 15  # Run early, before artifact filter
    exempt_persons = False  # Artifacts can look like person names

    def __init__(self, patterns_file: Path | str | None = None):
        """
        Initialize filter with patterns file path.

        Args:
            patterns_file: Path to regex patterns file.
                          Defaults to config/vocab_exclude_patterns.txt
        """
        if patterns_file is None:
            # Default to config directory
            patterns_file = (
                Path(__file__).parent.parent.parent.parent.parent
                / "config"
                / "vocab_exclude_patterns.txt"
            )
        self.patterns_file = Path(patterns_file)
        self.patterns: list[re.Pattern] = []
        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load and compile regex patterns from file."""
        self.patterns = []

        if not self.patterns_file.exists():
            logger.debug("Patterns file not found: %s", self.patterns_file)
            return

        try:
            with open(self.patterns_file, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # Skip comments and blank lines
                    if not line or line.startswith("#"):
                        continue

                    # Try to compile the regex
                    try:
                        pattern = re.compile(line, re.IGNORECASE)
                        self.patterns.append(pattern)
                    except re.error as e:
                        logger.debug(
                            "Invalid regex on line %s: '%s' - %s",
                            line_num,
                            line,
                            e,
                        )

            logger.debug(
                "Loaded %s patterns from %s",
                len(self.patterns),
                self.patterns_file.name,
            )

        except Exception as e:
            logger.debug("Error loading patterns file: %s", e)

    def filter(self, vocabulary: list[dict]) -> FilterResult:
        """Filter terms matching any regex pattern."""
        if not self.patterns:
            # No patterns loaded, pass through
            return FilterResult(vocabulary=vocabulary, removed_count=0)

        filtered = []
        removed_count = 0
        removed_terms = []

        for term_data in vocabulary:
            term = term_data.get("Term", "")

            # Check against all patterns
            should_remove = False
            for pattern in self.patterns:
                if pattern.search(term):
                    should_remove = True
                    break

            if should_remove:
                removed_count += 1
                removed_terms.append(term)
                logger.debug("Removed: '%s'", term)
            else:
                filtered.append(term_data)

        if removed_count > 0:
            logger.debug(
                "Removed %s terms matching %s patterns",
                removed_count,
                len(self.patterns),
            )

        return FilterResult(
            vocabulary=filtered,
            removed_count=removed_count,
            removed_terms=removed_terms,
        )

    def reload_patterns(self) -> None:
        """Reload patterns from file (for runtime updates)."""
        self._load_patterns()
