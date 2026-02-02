"""
GLiNER Zero-Shot NER Algorithm

Uses GLiNER (Generalist and Lightweight model for Named Entity Recognition)
to extract entities based on user-defined labels. The model matches labels
to text spans by meaning, enabling extraction of arbitrary entity types
without training.

Model: urchade/gliner_medium-v2.1 (209M params, Apache 2.0)

CRITICAL: GLiNER silently truncates input to ~384 words (~512 subtokens).
This algorithm chunks documents into ~300-word segments with ~50-word overlap,
runs prediction on each chunk, and deduplicates across chunks.
"""

import logging
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
    Split text into overlapping word-based chunks for GLiNER processing.

    Args:
        text: Full document text

    Returns:
        List of text chunks, each ~300 words with ~50-word overlap
    """
    words = text.split()
    if len(words) <= _CHUNK_SIZE_WORDS:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + _CHUNK_SIZE_WORDS
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += _CHUNK_SIZE_WORDS - _OVERLAP_WORDS
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

        # Lazy-load model
        if self._model is None:
            try:
                self._load_model()
            except Exception as e:
                logger.warning("GLiNER unavailable: %s", e)
                return AlgorithmResult(
                    candidates=[],
                    processing_time_ms=0.0,
                    metadata={"skipped": True, "reason": str(e)},
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

                if key not in best_entities or score > best_entities[key]["score"]:
                    best_entities[key] = {
                        "text": ent_text,
                        "label": ent["label"],
                        "score": score,
                    }

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
                    metadata={"gliner_label": label},
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
