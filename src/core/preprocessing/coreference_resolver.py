"""
Coreference Resolution Preprocessor

Replaces pronouns (he, she, they, etc.) with the actual names they refer to,
using the fastcoref library (F-coref mode). This improves search accuracy
since chunks containing only pronouns become findable by name.

F-coref: 78.5% F1 on OntoNotes, 16x faster than LingMess, ~350MB model.

Uses spaCy + fastcoref integration. Model is loaded lazily on first use.
Falls back gracefully if fastcoref is not installed.
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

# Max text size per processing chunk (50KB, same pattern as NER)
_MAX_CHUNK_SIZE = 50_000


class CoreferenceResolver(BasePreprocessor):
    """
    Resolves coreference chains, replacing pronouns with named antecedents.

    Uses fastcoref (F-coref, 78.5 F1) via spaCy pipeline integration.
    Loads model lazily on first call. If fastcoref is unavailable, returns
    text unchanged with a logged warning.
    """

    name = "Coreference Resolver"

    def __init__(self):
        """Initialize resolver. Model loading is deferred to first use."""
        super().__init__()
        self._nlp = None
        self._available = None  # None = not checked, True/False after check

    def _ensure_model(self) -> bool:
        """
        Load spaCy + fastcoref model on first use.

        Returns:
            True if model is ready, False if fastcoref unavailable.
        """
        if self._available is not None:
            return self._available

        try:
            import spacy
            from fastcoref import spacy_component  # noqa: F401 — registers the pipe

            from src.config import (
                COREF_MODEL_LOCAL_PATH,
                COREF_MODEL_NAME,
                HF_CACHE_DIR,
                SPACY_EN_CORE_WEB_SM_PATH,
            )
        except ImportError:
            logger.warning(
                "fastcoref not installed — coreference resolution disabled. "
                "Install with: pip install fastcoref"
            )
            self._available = False
            return False

        try:
            # Bundled spaCy model (Windows installer) or installed package
            spacy_model = (
                str(SPACY_EN_CORE_WEB_SM_PATH)
                if SPACY_EN_CORE_WEB_SM_PATH.exists()
                else "en_core_web_sm"
            )
            nlp = spacy.load(spacy_model, exclude=["ner", "lemmatizer"])

            # Determine model path: bundled first, then HuggingFace fallback
            model_path = (
                str(COREF_MODEL_LOCAL_PATH) if COREF_MODEL_LOCAL_PATH.exists() else COREF_MODEL_NAME
            )

            if not COREF_MODEL_LOCAL_PATH.exists():
                import os

                os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
                logger.info("Bundled coref model not found, downloading %s", COREF_MODEL_NAME)

            nlp.add_pipe(
                "fastcoref",
                config={
                    "model_architecture": "FCoref",
                    "model_path": model_path,
                    "device": "cpu",
                },
            )

            self._nlp = nlp
            self._available = True
            logger.info("Coreference model loaded: %s", model_path)
            return True

        except Exception as e:
            logger.error("Failed to load coreference model: %s", e)
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

        if not self._ensure_model():
            return PreprocessingResult(text=text, changes_made=0)

        # Process in chunks for large documents
        if len(text) <= _MAX_CHUNK_SIZE:
            resolved, count, replacements = self._resolve_text(text)
        else:
            resolved, count, replacements = self._process_in_chunks(text)

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

    def _process_in_chunks(self, text: str) -> tuple[str, int, list[dict]]:
        """
        Process large text in paragraph-boundary chunks.

        Args:
            text: Text exceeding _MAX_CHUNK_SIZE.

        Returns:
            Tuple of (resolved_text, total_count, all_replacements).
        """
        # Split on double newlines (paragraph boundaries)
        paragraphs = re.split(r"(\n\n+)", text)
        chunks = []
        current_chunk = ""

        for part in paragraphs:
            if len(current_chunk) + len(part) > _MAX_CHUNK_SIZE and current_chunk:
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
