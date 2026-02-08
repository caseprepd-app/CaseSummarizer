"""
scispaCy Medical NER Algorithm

Uses scispaCy's en_ner_bc5cdr_md model to detect drug/chemical names and
disease mentions. The BC5CDR corpus was annotated for biomedical named
entity recognition covering CHEMICAL and DISEASE entity types.

This complements the general-purpose NER pipeline:
- NER finds people, organizations, locations
- TextRank finds important keyphrases via graph centrality
- MedicalNER finds drug names, chemical compounds, and disease terms

Reference:
Li et al. (2016), "BioCreative V CDR task corpus"
"""

import logging
import time
from typing import Any

from src.config import VOCAB_ALGORITHM_WEIGHTS
from src.core.vocabulary.algorithms import register_algorithm
from src.core.vocabulary.algorithms.base import (
    AlgorithmResult,
    BaseExtractionAlgorithm,
    CandidateTerm,
)

logger = logging.getLogger(__name__)

# Max input size: 1 MB (same cap as TextRank)
_MAX_TEXT_BYTES = 1_024 * 1_024


@register_algorithm("MedicalNER")
class ScispaCyAlgorithm(BaseExtractionAlgorithm):
    """
    Medical named-entity recognition using scispaCy.

    Extracts CHEMICAL and DISEASE entities from text using the
    en_ner_bc5cdr_md model. Useful for legal-medical case documents
    that reference medications, substances, and medical conditions.
    """

    name = "MedicalNER"
    weight = VOCAB_ALGORITHM_WEIGHTS.get("MedicalNER", 0.75)

    def __init__(self, max_candidates: int = 200):
        """
        Initialize MedicalNER algorithm.

        Args:
            max_candidates: Maximum number of entities to return
        """
        self.max_candidates = max_candidates
        self._nlp = None

    def _load_nlp(self):
        """Load scispaCy en_ner_bc5cdr_md model from bundled path or package."""
        import spacy

        from src.config import SPACY_EN_NER_BC5CDR_MD_PATH

        model_path = (
            str(SPACY_EN_NER_BC5CDR_MD_PATH)
            if SPACY_EN_NER_BC5CDR_MD_PATH.exists()
            else "en_ner_bc5cdr_md"
        )
        self._nlp = spacy.load(model_path)
        logger.debug("Loaded scispaCy model: %s", model_path)

    def extract(self, text: str, **kwargs) -> AlgorithmResult:
        """
        Extract medical entities from text using scispaCy.

        Args:
            text: Document text to analyze
            **kwargs: Not used by this algorithm

        Returns:
            AlgorithmResult with medical entity candidates
        """
        start_time = time.time()

        # Lazy-load model
        if self._nlp is None:
            try:
                self._load_nlp()
            except Exception as e:
                logger.warning("MedicalNER unavailable: %s", e)
                return AlgorithmResult(
                    candidates=[],
                    processing_time_ms=0.0,
                    metadata={"skipped": True, "reason": str(e)},
                )

        # Truncate very long texts
        max_chars = _MAX_TEXT_BYTES
        truncated = len(text) > max_chars
        process_text = text[:max_chars] if truncated else text

        doc = self._nlp(process_text)

        candidates = []
        seen_terms: set[str] = set()

        for ent in doc.ents:
            if len(candidates) >= self.max_candidates:
                break

            ent_text = ent.text.strip()

            # Skip very short entities
            if len(ent_text) < 2:
                continue

            # Skip pure numbers
            if ent_text.replace(" ", "").isdigit():
                continue

            # Dedup (case-insensitive)
            lower_text = ent_text.lower()
            if lower_text in seen_terms:
                continue
            seen_terms.add(lower_text)

            # Use title case for consistency with other algorithms
            candidates.append(
                CandidateTerm(
                    term=ent_text.title(),
                    source_algorithm=self.name,
                    confidence=0.7,
                    suggested_type="Medical",
                    frequency=1,
                    metadata={
                        "entity_label": ent.label_,
                        "start_char": ent.start_char,
                    },
                )
            )

        processing_time_ms = (time.time() - start_time) * 1000

        logger.debug(
            "Extracted %d medical entities in %.1fms (truncated: %s)",
            len(candidates),
            processing_time_ms,
            truncated,
        )

        return AlgorithmResult(
            candidates=candidates,
            processing_time_ms=processing_time_ms,
            metadata={
                "total_entities_found": len(doc.ents) if doc else 0,
                "filtered_candidates": len(candidates),
                "text_truncated": truncated,
            },
        )

    def get_config(self) -> dict[str, Any]:
        """Return algorithm configuration."""
        return {
            **super().get_config(),
            "max_candidates": self.max_candidates,
        }
