"""
GLiNER Zero-Shot NER Algorithm

Uses GLiNER (Generalist and Lightweight model for Named Entity Recognition)
to extract entities based on user-defined labels. The model matches labels
to text spans by meaning, enabling extraction of arbitrary entity types
without training.

Model: urchade/gliner_medium-v2.1 (209M params, Apache 2.0)

CRITICAL: GLiNER silently truncates input to ~384 words (~512 subtokens).
This algorithm chunks documents into ~300-word sentence-aligned segments
with ~50-word overlap, runs prediction on each chunk, and deduplicates
across chunks.
"""

import logging
import threading
import time
from typing import Any

from src.config import GLINER_DEFAULT_LABELS, VOCAB_ALGORITHM_WEIGHTS
from src.core.vocabulary.algorithms import register_algorithm
from src.core.vocabulary.algorithms.base import (
    AlgorithmResult,
    BaseExtractionAlgorithm,
    CandidateTerm,
)

logger = logging.getLogger(__name__)

# Max input size: 1 MB (same cap as other algorithms)
_MAX_TEXT_BYTES = 1_024 * 1_024

# Chunking parameters (words)
_CHUNK_SIZE_WORDS = 300
_OVERLAP_WORDS = 50

# Timeout for waiting on background model warm-up (seconds)
_WARMUP_TIMEOUT_SEC = 120

# Label-to-type mapping keywords
_TYPE_MAPPING = {
    "Person": ["person"],
    "Medical": [
        "medical",
        "medication",
        "drug",
        "disease",
        "condition",
        "anatomical",
        "body part",
        "procedure",
    ],
    "Organization": ["court", "organization", "company"],
    "Place": ["place", "location", "city", "state", "country"],
}


def _map_label_to_type(label: str) -> str:
    """
    Map a GLiNER label to a CandidateTerm suggested_type.

    Args:
        label: The entity label (e.g., "medical condition")

    Returns:
        Suggested type: Person, Medical, Organization, Place, or Technical
    """
    lower_label = label.lower()
    for type_name, keywords in _TYPE_MAPPING.items():
        if any(kw in lower_label for kw in keywords):
            return type_name
    return "Technical"


def _chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping sentence-aligned chunks for GLiNER processing.

    Uses the legal-aware NUPunkt sentence splitter to avoid cutting entities
    mid-span. Accumulates sentences until reaching ~300 words, then backs up
    by ~50 words of sentences for the next chunk's overlap.

    Args:
        text: Full document text

    Returns:
        List of text chunks, each ~300 words with ~50-word sentence overlap
    """
    from src.core.utils.sentence_splitter import split_sentences

    words = text.split()
    if len(words) <= _CHUNK_SIZE_WORDS:
        return [text]

    sentences = split_sentences(text)
    if not sentences:
        return [text]

    # Build list of (sentence_text, word_count) for accumulation
    sent_words = [(s, len(s.split())) for s in sentences]

    chunks = []
    sent_idx = 0

    while sent_idx < len(sent_words):
        # Accumulate sentences until we reach the target word count
        chunk_sents = []
        chunk_word_count = 0
        start_idx = sent_idx

        while sent_idx < len(sent_words) and chunk_word_count < _CHUNK_SIZE_WORDS:
            sent_text, wc = sent_words[sent_idx]
            chunk_sents.append(sent_text)
            chunk_word_count += wc
            sent_idx += 1

        chunks.append(" ".join(chunk_sents))

        # Back up by ~OVERLAP_WORDS worth of sentences for next chunk
        if sent_idx < len(sent_words):
            overlap_words = 0
            back = sent_idx
            while back > start_idx and overlap_words < _OVERLAP_WORDS:
                back -= 1
                overlap_words += sent_words[back][1]
            sent_idx = max(back, start_idx + 1)

    return chunks


@register_algorithm("GLiNER")
class GLiNERAlgorithm(BaseExtractionAlgorithm):
    """
    Zero-shot NER using GLiNER with user-defined entity labels.

    Extracts entities matching user-specified labels (e.g., "person",
    "medical condition", "case citation") without any model fine-tuning.
    """

    name = "GLiNER"
    weight = VOCAB_ALGORITHM_WEIGHTS.get("GLiNER", 0.75)

    def __init__(
        self,
        labels: list[str] | None = None,
        threshold: float = 0.5,
        max_candidates: int = 300,
    ):
        """
        Initialize GLiNER algorithm.

        Args:
            labels: Entity labels to search for. Defaults to GLINER_DEFAULT_LABELS.
            threshold: Minimum confidence threshold for predictions (0-1).
            max_candidates: Maximum entities to return.
        """
        self.labels = labels or list(GLINER_DEFAULT_LABELS)
        self.threshold = threshold
        self.max_candidates = max_candidates
        self._model = None
        self._model_ready = threading.Event()
        self._load_error: str | None = None

    def warm_up(self):
        """
        Start loading the model in a background thread.

        Call this early so the model is ready by the time extract() runs.
        If warm_up() is not called, extract() will load synchronously.
        """

        def _background_load():
            try:
                self._load_model()
            except Exception as e:
                self._load_error = str(e)
                logger.warning("GLiNER background warm-up failed: %s", e)
            finally:
                self._model_ready.set()

        thread = threading.Thread(target=_background_load, daemon=True)
        thread.start()
        logger.debug("GLiNER model warm-up started in background thread")

    def _load_model(self):
        """Load GLiNER model from bundled local path, falling back to HuggingFace."""
        from gliner import GLiNER

        from src.config import GLINER_MODEL_LOCAL_PATH, GLINER_MODEL_NAME

        if GLINER_MODEL_LOCAL_PATH.exists():
            self._model = GLiNER.from_pretrained(str(GLINER_MODEL_LOCAL_PATH))
            logger.debug("Loaded GLiNER model from bundled path: %s", GLINER_MODEL_LOCAL_PATH)
        else:
            self._model = GLiNER.from_pretrained(GLINER_MODEL_NAME)
            logger.debug("Loaded GLiNER model from HuggingFace (bundled not found)")

    def _wait_for_model(self) -> bool:
        """
        Wait for model to be ready (from warm_up or load synchronously).

        Returns:
            True if model is available, False if loading failed.
        """
        if self._model is not None:
            return True

        # If warm_up() was called, wait for it
        if self._model_ready.is_set() or self._load_error is not None:
            return self._model is not None

        # If warm_up thread is running but not done, wait with timeout
        if not self._model_ready.wait(timeout=_WARMUP_TIMEOUT_SEC):
            logger.warning("GLiNER warm-up timed out after %ds", _WARMUP_TIMEOUT_SEC)
            return False

        return self._model is not None

    def extract(self, text: str, **kwargs) -> AlgorithmResult:
        """
        Extract entities from text using GLiNER zero-shot NER.

        Args:
            text: Document text to analyze
            **kwargs: Not used by this algorithm

        Returns:
            AlgorithmResult with entity candidates
        """
        start_time = time.time()

        # Wait for warm-up or load synchronously
        if self._model is None:
            if self._model_ready.is_set() or not self._model_ready.wait(timeout=0):
                # No warm_up() was called — load synchronously
                try:
                    self._load_model()
                except Exception as e:
                    logger.warning("GLiNER unavailable: %s", e)
                    return AlgorithmResult(
                        candidates=[],
                        processing_time_ms=0.0,
                        metadata={"skipped": True, "reason": str(e)},
                    )
            elif not self._wait_for_model():
                reason = self._load_error or "warm-up timed out"
                return AlgorithmResult(
                    candidates=[],
                    processing_time_ms=0.0,
                    metadata={"skipped": True, "reason": reason},
                )

        # Truncate very long texts
        truncated = len(text) > _MAX_TEXT_BYTES
        process_text = text[:_MAX_TEXT_BYTES] if truncated else text

        # Chunk text to avoid GLiNER's silent 384-word truncation
        chunks = _chunk_text(process_text)
        logger.debug(
            "GLiNER processing %d chunks from %dKB text", len(chunks), len(process_text) // 1024
        )

        # Run prediction on each chunk, deduplicate by (lower_text, label)
        # Keep highest confidence for each unique entity
        best_entities: dict[tuple[str, str], dict] = {}

        for chunk in chunks:
            try:
                entities = self._model.predict_entities(
                    chunk, self.labels, threshold=self.threshold
                )
            except Exception as e:
                logger.debug("GLiNER chunk prediction failed: %s", e)
                continue

            for ent in entities:
                ent_text = ent["text"].strip()
                if len(ent_text) < 2:
                    continue
                if ent_text.replace(" ", "").isdigit():
                    continue

                key = (ent_text.lower(), ent["label"])
                score = ent.get("score", 0.5)

                if key not in best_entities:
                    best_entities[key] = {
                        "text": ent_text,
                        "label": ent["label"],
                        "score": score,
                        "hits": 1,
                    }
                else:
                    best_entities[key]["hits"] += 1
                    if score > best_entities[key]["score"]:
                        best_entities[key]["score"] = score

        # Convert to CandidateTerm list
        candidates = []
        for (lower_text, label), ent_data in best_entities.items():
            if len(candidates) >= self.max_candidates:
                break

            candidates.append(
                CandidateTerm(
                    term=ent_data["text"],
                    source_algorithm=self.name,
                    confidence=ent_data["score"],
                    suggested_type=_map_label_to_type(label),
                    frequency=1,
                    metadata={"gliner_label": label, "chunk_hits": ent_data["hits"]},
                )
            )

        processing_time_ms = (time.time() - start_time) * 1000

        logger.debug(
            "GLiNER extracted %d entities in %.1fms (%d chunks, truncated: %s)",
            len(candidates),
            processing_time_ms,
            len(chunks),
            truncated,
        )

        return AlgorithmResult(
            candidates=candidates,
            processing_time_ms=processing_time_ms,
            metadata={
                "total_raw_entities": len(best_entities),
                "filtered_candidates": len(candidates),
                "chunk_count": len(chunks),
                "text_truncated": truncated,
                "labels_used": self.labels,
            },
        )

    def get_config(self) -> dict[str, Any]:
        """Return algorithm configuration."""
        return {
            **super().get_config(),
            "labels": self.labels,
            "threshold": self.threshold,
            "max_candidates": self.max_candidates,
        }
