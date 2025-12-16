"""
BM25 Corpus-Based Vocabulary Extraction Algorithm

Identifies terms that are statistically unusual compared to a user's corpus of
previous transcripts. Uses BM25 (Best Matching 25) scoring to find terms that
are frequent in the current document but rare across the corpus.

Key Insight:
Court reporters' past transcripts establish a "baseline" vocabulary. Terms that
are RARE in the corpus but FREQUENT in the current document are likely case-specific
(medical conditions, product names, unique proper nouns).

Privacy: All processing is local - no documents or data are sent externally.

Example:
    from src.vocabulary.algorithms import get_algorithm
    from src.vocabulary.corpus_manager import get_corpus_manager

    corpus_manager = get_corpus_manager()
    if corpus_manager.is_corpus_ready():
        bm25 = get_algorithm("BM25", corpus_manager=corpus_manager)
        result = bm25.extract(document_text)
"""

import time
from collections import Counter
from typing import Any

from src.config import BM25_K1, BM25_B, BM25_MIN_SCORE_THRESHOLD, VOCAB_ALGORITHM_WEIGHTS
from src.logging_config import debug_log
from src.utils.tokenizer import tokenize, STOPWORDS
from src.vocabulary.algorithms import register_algorithm
from src.vocabulary.algorithms.base import (
    AlgorithmResult,
    BaseExtractionAlgorithm,
    CandidateTerm,
)


@register_algorithm("BM25")
class BM25Algorithm(BaseExtractionAlgorithm):
    """
    Corpus-based term extraction using BM25 scoring.

    Identifies terms that are frequent in the current document but rare across
    the user's corpus of past transcripts. This complements NER (linguistic entities)
    and RAKE (key phrases) with cross-document statistical analysis.

    How BM25 Works:
        TF (Term Frequency) = count(term in doc) / len(doc)
        IDF (Inverse Document Frequency) = log((N - df + 0.5) / (df + 0.5) + 1)
            where N = total docs in corpus, df = docs containing term

        BM25 Score = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * doc_len / avg_len))

    High Score = term is frequent HERE but rare in CORPUS = case-specific term

    Attributes:
        name: Algorithm identifier ("BM25")
        weight: Relative weight for scoring (0.8, between NER and RAKE)
        corpus_manager: CorpusManager instance providing IDF lookups
    """

    name = "BM25"
    weight = VOCAB_ALGORITHM_WEIGHTS.get("BM25", 0.8)  # Corpus-based importance

    # BM25 tuning parameters (from unified config)
    K1 = BM25_K1  # Term frequency saturation parameter
    B = BM25_B    # Length normalization parameter

    # Filtering parameters
    MIN_TERM_LENGTH = 3  # Minimum characters for a term
    MAX_CANDIDATES = 100  # Maximum terms to return

    def __init__(
        self,
        corpus_manager=None,
        min_score_threshold: float = BM25_MIN_SCORE_THRESHOLD,
    ):
        """
        Initialize BM25 algorithm.

        Args:
            corpus_manager: CorpusManager instance for IDF lookups.
                           If None, will use global singleton.
            min_score_threshold: Minimum BM25 score to include term.
                                Terms below this are filtered out.
        """
        if corpus_manager is None:
            from src.vocabulary.corpus_manager import get_corpus_manager
            corpus_manager = get_corpus_manager()

        self.corpus_manager = corpus_manager
        self.min_score_threshold = min_score_threshold

    def extract(self, text: str, **kwargs) -> AlgorithmResult:
        """
        Extract terms that are unusually frequent compared to corpus.

        Args:
            text: Document text to analyze
            **kwargs: Additional arguments (unused, for API compatibility)

        Returns:
            AlgorithmResult with candidate terms sorted by BM25 score.
            Returns empty result if corpus has insufficient documents.
        """
        start_time = time.time()

        # Check if corpus is ready
        if not self.corpus_manager.is_corpus_ready():
            doc_count = self.corpus_manager.get_document_count()
            debug_log(
                f"[BM25] Skipped: insufficient corpus ({doc_count}/5 documents)"
            )
            return AlgorithmResult(
                candidates=[],
                processing_time_ms=0.0,
                metadata={
                    "skipped": True,
                    "reason": "insufficient_corpus",
                    "corpus_doc_count": doc_count,
                }
            )

        # Ensure IDF index is built
        self.corpus_manager.build_idf_index()

        # Tokenize current document
        tokens = self._tokenize(text)
        doc_length = len(tokens)
        term_freqs = Counter(tokens)

        debug_log(
            f"[BM25] Processing document: {doc_length} tokens, "
            f"{len(term_freqs)} unique terms"
        )

        # Get average document length from corpus
        avg_doc_length = self.corpus_manager.get_average_doc_length()

        # Score each unique term using BM25
        candidates = []
        for term, tf in term_freqs.items():
            # Apply filters
            if not self._is_valid_term(term):
                continue

            # Get IDF from corpus
            idf = self.corpus_manager.get_idf(term)

            # BM25 scoring formula
            # score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * L/avgL))
            length_norm = 1 - self.B + self.B * (doc_length / avg_doc_length)
            tf_component = (tf * (self.K1 + 1)) / (tf + self.K1 * length_norm)
            score = idf * tf_component

            if score >= self.min_score_threshold:
                # Normalize confidence to [0, 1] range
                # Scores typically range from 0 to ~15, so divide by 15
                normalized_confidence = min(score / 15.0, 1.0)

                candidates.append(CandidateTerm(
                    term=term,
                    source_algorithm=self.name,
                    confidence=normalized_confidence,
                    suggested_type="Technical",  # Default; NER may override
                    frequency=tf,
                    metadata={
                        "bm25_score": round(score, 4),
                        "idf": round(idf, 4),
                        "tf": tf,
                        "length_normalized": True,
                    }
                ))

        # Sort by BM25 score (highest first) and limit results
        candidates.sort(key=lambda x: x.metadata["bm25_score"], reverse=True)
        candidates = candidates[:self.MAX_CANDIDATES]

        processing_time_ms = (time.time() - start_time) * 1000

        corpus_stats = self.corpus_manager.get_corpus_stats()

        debug_log(
            f"[BM25] Found {len(candidates)} candidates "
            f"(threshold: {self.min_score_threshold}) in {processing_time_ms:.1f}ms"
        )

        return AlgorithmResult(
            candidates=candidates,
            processing_time_ms=processing_time_ms,
            metadata={
                "skipped": False,
                "corpus_doc_count": corpus_stats["doc_count"],
                "corpus_vocab_size": corpus_stats["vocab_size"],
                "document_tokens": doc_length,
                "unique_terms_analyzed": len(term_freqs),
                "candidates_above_threshold": len(candidates),
                "min_score_threshold": self.min_score_threshold,
            }
        )

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into lowercase words.

        Uses shared tokenizer for consistent processing across BM25 operations.

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase word tokens
        """
        return tokenize(text)

    def _is_valid_term(self, term: str) -> bool:
        """
        Check if a term should be considered for scoring.

        Args:
            term: Term to validate

        Returns:
            True if term should be scored, False to skip
        """
        # Too short
        if len(term) < self.MIN_TERM_LENGTH:
            return False

        # Pure digits
        if term.isdigit():
            return False

        # Stopword (already filtered in tokenization, but double-check)
        if term.lower() in STOPWORDS:
            return False

        # Contains only repeated characters (e.g., "aaa", "xxxxx")
        if len(set(term)) == 1:
            return False

        return True

    def get_config(self) -> dict[str, Any]:
        """Return algorithm configuration for logging/debugging."""
        corpus_stats = self.corpus_manager.get_corpus_stats()
        return {
            **super().get_config(),
            "min_score_threshold": self.min_score_threshold,
            "k1": self.K1,
            "b": self.B,
            "corpus_ready": corpus_stats["is_ready"],
            "corpus_doc_count": corpus_stats["doc_count"],
            "corpus_vocab_size": corpus_stats["vocab_size"],
        }
