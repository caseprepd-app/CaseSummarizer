"""
TopicRank Keyword Extraction Algorithm

Uses pytextrank's TopicRank (a spaCy pipeline component) to extract
keyphrases via graph-based ranking. TopicRank clusters candidate noun
phrases into topics, builds a topic-level graph, and runs PageRank on it.
It then selects one representative keyphrase per top-ranked topic.

Why TopicRank instead of TextRank:
    This pipeline originally used plain TextRank (Mihalcea & Tarau, 2004),
    which builds a word co-occurrence graph and runs PageRank on individual
    words. We switched to TopicRank for three reasons specific to legal
    document processing:

    1. Redundancy elimination. TextRank returns overlapping phrases like
       "Dr. Johnson", "Johnson", "the defendant Johnson" as separate entries.
       TopicRank clusters candidates into topics (via hierarchical agglomerative
       clustering with 25% lemma overlap) and selects one representative per
       topic, structurally preventing near-duplicate output.

    2. Long-document scaling. TextRank runs PageRank on a word graph with
       thousands of nodes. On JCDL 2020 benchmarks, TextRank F@10 collapses
       from 35.8 (short abstracts) to 1.8-2.7 on longer documents (PubMed,
       NYTime). TopicRank runs PageRank on a topic graph with dozens of
       nodes, scaling much better to our 50-200 page transcripts.

    3. No position bias. Unlike PositionRank (another candidate), TopicRank
       does not privilege terms appearing early in the document. In
       depositions, critical testimony can appear on any page.

    Both algorithms are implemented in pytextrank, so the switch required
    only changing the spaCy pipe name from "textrank" to "topicrank". The
    doc._.phrases API (text, rank, count) is identical.

References:
    Bougouin, Boudin & Daille (2013), "TopicRank: Graph-Based Topic
    Ranking for Keyphrase Extraction"
    https://aclanthology.org/I13-1062.pdf

    Mihalcea & Tarau (2004), "TextRank: Bringing Order into Text"
    https://aclanthology.org/W04-3252/

    Boudin (2018), "Unsupervised Keyphrase Extraction with Multipartite
    Graphs" (comparison of graph-based methods)
    https://arxiv.org/abs/1803.08721

    JCDL 2020 large-scale keyphrase extraction evaluation
    https://github.com/ygorg/JCDL_2020_KPE_Eval

This algorithm can share the NER pipeline's en_core_web_lg instance since
topicrank is a read-only analysis pipe. Falls back to loading its own
instance if no shared model is provided.
"""

import logging
import sys
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

# Components TopicRank does NOT need — disable per-call for speed.
# TopicRank requires: tagger (POS tags) + parser (noun_chunks) + topicrank.
# NER, attribute_ruler, lemmatizer are unused overhead (~30-50% savings).
_TOPICRANK_DISABLED = ["ner", "attribute_ruler", "lemmatizer"]

# pytextrank dependency chain (module_name, pip_package_name)
_PYTEXTRANK_DEPS = [
    ("scipy", "scipy"),
    ("scipy.cluster.hierarchy", "scipy (clustering)"),
    ("scipy.spatial.distance", "scipy (distance)"),
    ("networkx", "networkx"),
    ("graphviz", "graphviz"),
    ("icecream", "icecream"),
    ("pygments", "pygments"),
    ("gitdb", "gitdb"),
    ("smmap", "smmap"),
    ("git", "GitPython"),
    ("asttokens", "asttokens"),
    ("executing", "executing"),
    ("colorama", "colorama"),
]


def _log_pytextrank_diagnostics(original_error: Exception):
    """Log which pytextrank dependencies loaded vs failed."""
    import importlib

    logger.warning(
        "pytextrank import failed: %s: %s", type(original_error).__name__, original_error
    )
    logger.warning("Checking pytextrank dependency chain:")
    for module_name, package_name in _PYTEXTRANK_DEPS:
        try:
            mod = importlib.import_module(module_name)
            ver = getattr(mod, "__version__", "?")
            logger.warning("  OK: %s (%s) v%s", module_name, package_name, ver)
        except Exception as e:
            logger.error("  MISSING: %s (%s): %s", module_name, package_name, e)
    for submod in ["pytextrank", "pytextrank.version", "pytextrank.base"]:
        try:
            importlib.import_module(submod)
            logger.warning("  OK: %s", submod)
        except Exception as e:
            logger.error("  FAILED: %s: %s: %s", submod, type(e).__name__, e)


def _import_pytextrank():
    """Import pytextrank with diagnostic logging on failure."""
    logger.debug("Importing pytextrank...")
    try:
        import pytextrank

        ver = getattr(pytextrank, "__version__", "?")
        logger.info("pytextrank v%s imported successfully", ver)
        return pytextrank
    except Exception as e:
        _log_pytextrank_diagnostics(e)
        raise


@register_algorithm("TopicRank")
class TextRankAlgorithm(BaseExtractionAlgorithm):
    """
    TopicRank-based keyword extraction using pytextrank + spaCy.

    Clusters candidate phrases into topics, builds a topic graph, and
    applies PageRank to find the most important topics. Selects one
    representative keyphrase per topic, eliminating redundancy.

    Can share the NER pipeline's spaCy model to save memory,
    or loads its own instance if none is provided.
    """

    name = "TopicRank"
    weight = VOCAB_ALGORITHM_WEIGHTS.get("TopicRank", 0.6)

    def __init__(self, max_candidates: int = 150, nlp=None):
        """
        Initialize TopicRank algorithm.

        Args:
            max_candidates: Maximum number of phrases to return
            nlp: Optional shared spaCy model (e.g. from NER). If provided,
                 the topicrank pipe is added to it. If None, loads its own.
        """
        self.max_candidates = max_candidates
        self._nlp = None

        if nlp is not None:
            _import_pytextrank()
            self._nlp = nlp
            if "topicrank" not in self._nlp.pipe_names:
                self._nlp.add_pipe("topicrank")
                logger.info("Added topicrank pipe to shared spaCy model")
            else:
                logger.debug("Shared spaCy model already has topicrank pipe")

    def _load_nlp(self):
        """
        Load spaCy model with pytextrank TopicRank pipeline component.

        Fallback: loads en_core_web_lg with topicrank when no shared
        model was provided at init time.
        """
        _import_pytextrank()
        import spacy

        from src.config import SPACY_EN_CORE_WEB_LG_PATH

        if SPACY_EN_CORE_WEB_LG_PATH.exists():
            self._nlp = spacy.load(str(SPACY_EN_CORE_WEB_LG_PATH))
            logger.debug("Loaded bundled spaCy model: %s", SPACY_EN_CORE_WEB_LG_PATH)
        elif getattr(sys, "frozen", False):
            raise RuntimeError(
                f"Bundled spaCy model not found: {SPACY_EN_CORE_WEB_LG_PATH}\n"
                f"Please reinstall the application to restore model files."
            )
        else:
            try:
                self._nlp = spacy.load("en_core_web_lg")
                logger.warning("Using pip-installed spaCy model (not bundled)")
            except OSError:
                raise RuntimeError(
                    f"Language model not found. "
                    f"Expected at: {SPACY_EN_CORE_WEB_LG_PATH}\n"
                    f"For developers: python scripts/download_models.py"
                )
        self._nlp.add_pipe("topicrank")
        logger.debug("Loaded en_core_web_lg with pytextrank TopicRank pipeline")

    def extract(self, text: str, **kwargs) -> AlgorithmResult:
        """
        Extract keyphrases from text using TopicRank.

        Args:
            text: Document text to analyze
            **kwargs: Not used by this algorithm

        Returns:
            AlgorithmResult with candidate keyphrases
        """
        start_time = time.time()

        # Lazy-load model
        if self._nlp is None:
            try:
                logger.info("TopicRank: lazy-loading spaCy model + pytextrank...")
                self._load_nlp()
            except Exception as e:
                logger.warning("TopicRank unavailable: %s: %s", type(e).__name__, e, exc_info=True)
                return AlgorithmResult(
                    candidates=[],
                    processing_time_ms=0.0,
                    metadata={"skipped": True, "reason": str(e)},
                )

        # Truncate very long texts for performance
        from src.config import TOPICRANK_MAX_TEXT_KB

        max_chars = TOPICRANK_MAX_TEXT_KB * 1024
        truncated = len(text) > max_chars
        process_text = text[:max_chars] if truncated else text

        # Disable components TopicRank doesn't use (ner, attr_ruler, lemmatizer)
        active_pipes = self._nlp.pipe_names
        disable = [c for c in _TOPICRANK_DISABLED if c in active_pipes]
        doc = self._nlp(process_text, disable=disable)

        candidates = []
        seen_phrases: set[str] = set()

        for phrase in doc._.phrases:
            if len(candidates) >= self.max_candidates:
                break

            phrase_text = phrase.text.strip()

            # Skip single-char terms
            if len(phrase_text) < 2:
                continue

            # Skip pure numbers
            if phrase_text.replace(" ", "").isdigit():
                continue

            # Skip pure stopwords (single-word case)
            lower_text = phrase_text.lower()
            if " " not in phrase_text:
                from src.core.utils.tokenizer import STOPWORDS

                if lower_text in STOPWORDS:
                    continue

            # Dedup
            if lower_text in seen_phrases:
                continue
            seen_phrases.add(lower_text)

            # pytextrank TopicRank score is already 0-1 normalized
            confidence = min(phrase.rank, 1.0)

            # Count occurrences from pytextrank TopicRank's chunk count
            frequency = phrase.count

            candidates.append(
                CandidateTerm(
                    term=phrase_text,
                    source_algorithm=self.name,
                    confidence=confidence,
                    suggested_type="Technical",
                    frequency=max(frequency, 1),
                    metadata={
                        "topicrank_score": round(phrase.rank, 4),
                        "chunk_count": phrase.count,
                        "word_count": len(phrase_text.split()),
                    },
                )
            )

        processing_time_ms = (time.time() - start_time) * 1000

        logger.debug(
            "Extracted %d keyphrases in %.1fms (truncated: %s)",
            len(candidates),
            processing_time_ms,
            truncated,
        )

        return AlgorithmResult(
            candidates=candidates,
            processing_time_ms=processing_time_ms,
            metadata={
                "total_phrases_found": len(doc._.phrases) if doc else 0,
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
