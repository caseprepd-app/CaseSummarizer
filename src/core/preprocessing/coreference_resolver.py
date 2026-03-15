"""
Coreference Resolution Preprocessor

Replaces pronouns (he, she, they, etc.) with the actual names they refer to.
This improves search accuracy since chunks containing only pronouns become
findable by name.

Note: fastcoref was removed (Mar 2026) — this module is now always a no-op.
The class remains to avoid breaking the preprocessing pipeline. _ensure_model()
will always return False since fastcoref is not installed.
"""

import logging
import re

from src.core.preprocessing.base import BasePreprocessor, PreprocessingResult

logger = logging.getLogger(__name__)

# Pronouns eligible for replacement (lowercase)
_PRONOUNS = {
    "he",
    "she",
    "they",
    "him",
    "her",
    "them",
    "his",
    "its",
    "their",
    "himself",
    "herself",
    "themselves",
    "hers",
    "theirs",
}

# Default max text size per processing chunk
_DEFAULT_CHUNK_SIZE = 15_000

# Default max document size — skip coreference for very large documents
_DEFAULT_MAX_CHARS = 200_000


class CoreferenceResolver(BasePreprocessor):
    """
    Resolves coreference chains, replacing pronouns with named antecedents.

    fastcoref was removed (Mar 2026). This class is now always a no-op —
    _ensure_model() returns False and process() returns text unchanged.
    Retained to avoid breaking the preprocessing pipeline.
    """

    name = "Coreference Resolver"

    def __init__(self):
        """Initialize resolver. Model loading is deferred to first use."""
        super().__init__()
        self._nlp = None
        self._available = None  # None = not checked, True/False after check

    def _ensure_model(self) -> bool:
        """
        Always returns False — fastcoref was removed (Mar 2026).

        Returns:
            False always.
        """
        self._available = False
        return False

    def _resolve_text(self, text: str) -> tuple[str, int, list[dict]]:
        """
        Run coreference resolution on a single text chunk.

        Args:
            text: Input text to resolve.

        Returns:
            Tuple of (resolved_text, replacement_count, replacements_list).
        """
        doc = self._nlp(text)

        if not doc._.coref_clusters:
            return text, 0, []

        replacements = []

        for cluster in doc._.coref_clusters:
            # Find the best antecedent: longest mention that isn't a pronoun
            mentions = [doc[span[0] : span[1]] for span in cluster]
            named_mentions = [
                m
                for m in mentions
                if m.text.lower() not in _PRONOUNS and not _is_pronoun_like(m.text)
            ]

            if not named_mentions:
                continue

            # Pick the longest named mention as the canonical reference
            best = max(named_mentions, key=lambda m: len(m.text))
            canonical = best.text

            # Replace only pronoun mentions with the canonical name
            for mention in mentions:
                if mention.text.lower() in _PRONOUNS:
                    replacements.append(
                        {
                            "pronoun": mention.text,
                            "resolved_to": canonical,
                            "start": mention.start_char,
                            "end": mention.end_char,
                        }
                    )

        if not replacements:
            return text, 0, []

        # Apply replacements in reverse order to preserve character offsets
        replacements.sort(key=lambda r: r["start"], reverse=True)
        result = text
        for rep in replacements:
            result = result[: rep["start"]] + rep["resolved_to"] + result[rep["end"] :]

        return result, len(replacements), replacements

    def _get_settings(self) -> tuple[int, int]:
        """
        Read coreference settings from user preferences.

        Returns:
            Tuple of (max_chars, chunk_size) from preferences or defaults.
        """
        try:
            from src.user_preferences import get_user_preferences

            prefs = get_user_preferences()
            max_chars = int(prefs.get("coreference_max_chars", _DEFAULT_MAX_CHARS))
            chunk_size = int(prefs.get("coreference_chunk_size", _DEFAULT_CHUNK_SIZE))
            return max_chars, chunk_size
        except Exception:
            return _DEFAULT_MAX_CHARS, _DEFAULT_CHUNK_SIZE

    def process(self, text: str) -> PreprocessingResult:
        """
        Replace pronouns with their resolved antecedents throughout the text.

        Args:
            text: Full document text after earlier preprocessing.

        Returns:
            PreprocessingResult with resolved text and replacement metadata.
        """
        if not text:
            return PreprocessingResult(text=text, changes_made=0)

        max_chars, chunk_size = self._get_settings()

        # Skip coreference for documents exceeding max size
        if len(text) > max_chars:
            logger.info(
                "Skipping coreference: document size %d chars exceeds limit %d",
                len(text),
                max_chars,
            )
            return PreprocessingResult(text=text, changes_made=0)

        if not self._ensure_model():
            return PreprocessingResult(text=text, changes_made=0)

        # Process in chunks for large documents
        if len(text) <= chunk_size:
            resolved, count, replacements = self._resolve_text(text)
        else:
            resolved, count, replacements = self._process_in_chunks(text, chunk_size)

        if count > 0:
            # Log a few examples for transparency
            examples = replacements[:5]
            example_strs = [f"'{r['pronoun']}' -> '{r['resolved_to']}'" for r in examples]
            logger.info(
                "Coreference: %d pronoun(s) resolved. Examples: %s",
                count,
                ", ".join(example_strs),
            )

        return PreprocessingResult(
            text=resolved,
            changes_made=count,
            metadata={
                "resolutions": count,
                "examples": replacements[:10],
            },
        )

    def _process_in_chunks(self, text: str, chunk_size: int) -> tuple[str, int, list[dict]]:
        """
        Process large text in paragraph-boundary chunks.

        Args:
            text: Text exceeding chunk_size.
            chunk_size: Max characters per processing chunk.

        Returns:
            Tuple of (resolved_text, total_count, all_replacements).
        """
        # Split on double newlines (paragraph boundaries)
        paragraphs = re.split(r"(\n\n+)", text)
        chunks = []
        current_chunk = ""

        for part in paragraphs:
            if len(current_chunk) + len(part) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = part
            else:
                current_chunk += part

        if current_chunk:
            chunks.append(current_chunk)

        all_resolved = []
        total_count = 0
        all_replacements = []

        for chunk in chunks:
            resolved, count, replacements = self._resolve_text(chunk)
            all_resolved.append(resolved)
            total_count += count
            all_replacements.extend(replacements)

        return "".join(all_resolved), total_count, all_replacements


def _is_pronoun_like(text: str) -> bool:
    """
    Check if text is a pronoun or pronoun-like phrase.

    Args:
        text: Mention text to check.

    Returns:
        True if the text is pronoun-like and should not be used as canonical.
    """
    clean = text.strip().lower()
    # Single-word pronouns
    if clean in _PRONOUNS:
        return True
    # Short demonstratives that aren't useful as replacements
    if clean in {"this", "that", "these", "those", "it", "who", "which"}:
        return True
    return False
