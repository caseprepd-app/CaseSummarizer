"""
Vocabulary Extractor Orchestrator

Orchestrates multiple extraction algorithms and produces final vocabulary output.

FILTERING STRATEGY:
- Algorithm-level: Each algorithm (NER, RAKE, BM25) handles single-word filtering
  (stopwords, rarity threshold, exclude lists)
- Centralized: Multi-word phrase filtering is done ONCE by rarity_filter.py
  after all algorithms have contributed, before displaying to the user

This module coordinates:
1. Running multiple extraction algorithms (NER, RAKE, BM25)
2. Merging and deduplicating results via AlgorithmScoreMerger
3. Post-processing: frequency filtering, ML boost, role detection
4. Name deduplication (fuzzy matching for OCR variants)
5. Artifact filtering (substring containment)
6. Phrase rarity filtering (filter phrases with all-common words)
7. Quality scoring and final output formatting

The extraction algorithms are pluggable via dependency injection.

TEXT FLOW ARCHITECTURE:
All vocabulary algorithms (NER, RAKE, BM25) receive IDENTICAL preprocessed text.
The text flow is:
    preprocessed_text → All algorithms receive same input → Merged results
All algorithms run locally without external LLM calls.
"""

import logging
import math
import os
import sys
import threading
import time
from pathlib import Path

import spacy
from nltk.corpus import wordnet

from src.config import (
    VOCABULARY_MAX_TEXT_KB,
    VOCABULARY_MIN_OCCURRENCES,
    VOCABULARY_RARITY_THRESHOLD,
    VOCABULARY_SORT_METHOD,
)
from src.core.vocab_schema import VF
from src.core.vocabulary.algorithms.base import BaseExtractionAlgorithm
from src.core.vocabulary.preference_learner import get_meta_learner
from src.core.vocabulary.result_merger import AlgorithmScoreMerger, MergedTerm
from src.core.vocabulary.role_profiles import RoleDetectionProfile, StenographerProfile
from src.core.vocabulary.term_sources import TermSources
from src.user_preferences import get_user_preferences

logger = logging.getLogger(__name__)

# User-friendly status messages for each algorithm (non-debug mode)
_ALGO_FRIENDLY_NAMES = {
    "NER": "Extracting proper names...",
    "YAKE": "Identifying domain-specific terms...",
    "RAKE": "Scanning for key phrases...",
    "BM25": "Analyzing term relevance...",
    "TopicRank": "Finding topic keywords...",
    "MedicalNER": "Identifying medical terminology...",
}


def _algo_status_message(algo_name: str) -> str:
    """
    Return a status message for the given algorithm.

    In debug mode, shows the technical algorithm name.
    In normal mode, shows a user-friendly description.

    Args:
        algo_name: Internal algorithm name (e.g., "YAKE", "NER")

    Returns:
        Status message string for the status bar
    """
    from src.config import DEBUG_MODE

    if DEBUG_MODE:
        return f"{algo_name} extraction in progress..."
    return _ALGO_FRIENDLY_NAMES.get(algo_name, f"Processing ({algo_name})...")


# Organization indicator words for category detection
ORGANIZATION_INDICATORS = {
    "LLP",
    "PLLC",
    "P.C.",
    "LLC",
    "Inc",
    "Corp",
    "Corporation",
    "Law Firm",
    "Law Office",
    "Firm",
    "Hospital",
    "Medical",
    "Healthcare",
    "Health",
    "Clinic",
    "University",
    "College",
    "School",
    "Bank",
    "Insurance",
    "Services",
}


# Threshold: words ranked below this in Google 333K are considered "common"
_NAME_COMMON_RANK_THRESHOLD = 50000


def _get_real_name_words(term_words: list[str]) -> list[str]:
    """Filter out initials, suffixes, and titles from a name.

    Words ≤2 characters after stripping periods are discarded
    (e.g., "R.", "Jr", "II", "S."). This leaves only substantive
    name components for frequency evaluation.

    Args:
        term_words: Split words from the person name term.

    Returns:
        List of name words longer than 2 chars (stripped of periods).
    """
    return [w for w in term_words if len(w.strip(".")) > 2]


def _person_multi_word_boost(term_words: list[str], rank_map: dict[str, int]) -> float:
    """Calculate person boost for multi-word names using per-word frequency.

    Evaluates each substantive name word (>2 chars) against Google 333K
    to tier the bonus: rare names get more, common names less.

    Args:
        term_words: Split words from the person name term.
        rank_map: Google frequency rank map {word: rank}.

    Returns:
        Point bonus to add to the quality score.
    """
    real_words = _get_real_name_words(term_words)
    if not real_words:
        # All components are initials/titles — treat as single-word
        phrase = "".join(term_words).lower().strip(".")
        rank = rank_map.get(phrase, 0)
        return _person_single_word_boost(rank)
    common_count = 0
    for word in real_words:
        rank = rank_map.get(word.lower(), 0)
        if 0 < rank < _NAME_COMMON_RANK_THRESHOLD:
            common_count += 1
    if common_count == 0:
        return 15.0  # All rare — "Xiomara Bjelkengren"
    if common_count < len(real_words):
        return 10.0  # Mixed — "Xiomara Smith"
    return 6.0  # All common — "David Wilson"


def _person_single_word_boost(frequency_rank: int) -> float:
    """Calculate person boost for single-word names.

    Args:
        frequency_rank: Google 333K rank (0 = not in dataset).

    Returns:
        Point bonus to add to the quality score.
    """
    if frequency_rank == 0 or frequency_rank >= _NAME_COMMON_RANK_THRESHOLD:
        return 8.0  # Rare — "Bjelkengren"
    if frequency_rank >= 5000:
        return 4.0  # Moderate — could be name, could be noise
    return 2.0  # Very common — "Will", "Brown"


class VocabularyExtractor:
    """
    Orchestrates multiple extraction algorithms and produces final vocabulary.

    This class coordinates:
    1. Running multiple extraction algorithms
    2. Merging and deduplicating results
    3. Role detection (via RoleDetectionProfile)
    4. Quality scoring
    5. Final output formatting

    The extraction algorithms are pluggable via dependency injection.

    Attributes:
        algorithms: List of extraction algorithms to use
        role_profile: Role detection profile for profession-specific relevance
        merger: AlgorithmScoreMerger for combining algorithm outputs

    Example:
        >>> extractor = VocabularyExtractor()
        >>> vocab = extractor.extract("John Smith filed a HIPAA complaint.")
        >>> for term in vocab:
        ...     print(f"{term['Term']}: {term['Type']}")
    """

    def __init__(
        self,
        algorithms: list[BaseExtractionAlgorithm] | None = None,
        exclude_list_path: str | None = None,
        medical_terms_path: str | None = None,
        user_exclude_path: str | None = None,
        role_profile: RoleDetectionProfile | None = None,
    ):
        """
        Initialize vocabulary extractor.

        Args:
            algorithms: List of extraction algorithms to use.
                       If None, uses create_default_algorithms().
            exclude_list_path: Path to file containing words to exclude.
            medical_terms_path: Path to file containing known medical terms.
            user_exclude_path: Path to user's personal exclusion list.
            role_profile: Role detection profile. Defaults to StenographerProfile.
        """
        # Load shared resources first (these are passed to algorithms)
        self.exclude_list = self._load_word_list(exclude_list_path)
        self.user_exclude_list = self._load_word_list(user_exclude_path)
        self.medical_terms = self._load_word_list(medical_terms_path)

        # Log user exclusions for debugging
        if self.user_exclude_list:
            logger.debug("User exclusion list has %s terms", len(self.user_exclude_list))

        # Load common medical/legal words blacklist
        common_blacklist_path = (
            Path(__file__).parent.parent.parent.parent / "config" / "common_medical_legal.txt"
        )
        self.common_words_blacklist = self._load_word_list(common_blacklist_path)

        # Load frequency dataset
        self.frequency_dataset, self.frequency_rank_map = self._load_frequency_dataset()
        self.rarity_threshold = VOCABULARY_RARITY_THRESHOLD
        # Get sort method from user preferences (falls back to config default)
        prefs = get_user_preferences()
        self.sort_method = prefs.get("vocab_sort_method", VOCABULARY_SORT_METHOD)

        # Store user exclude path for adding new exclusions
        self.user_exclude_path = user_exclude_path

        # Set role detection profile
        self.role_profile = role_profile or StenographerProfile()

        # Initialize algorithms (pass shared resources)
        if algorithms is None:
            # Create default algorithms with shared resources
            from src.core.vocabulary.algorithms.ner_algorithm import NERAlgorithm
            from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

            ner = NERAlgorithm(
                exclude_list=self.exclude_list,
                user_exclude_list=self.user_exclude_list,
                medical_terms=self.medical_terms,
                common_words_blacklist=self.common_words_blacklist,
                frequency_dataset=self.frequency_dataset,
                frequency_rank_map=self.frequency_rank_map,
                rarity_threshold=self.rarity_threshold,
            )
            rake = RAKEAlgorithm()

            self.algorithms = [ner, rake]
            skipped: list[str] = []

            # Conditionally add BM25 if enabled and corpus is ready
            if self._should_enable_bm25():
                try:
                    from src.core.vocabulary.algorithms.bm25_algorithm import BM25Algorithm
                    from src.core.vocabulary.corpus_manager import get_corpus_manager

                    corpus_manager = get_corpus_manager()
                    bm25 = BM25Algorithm(corpus_manager=corpus_manager)
                    self.algorithms.append(bm25)
                    logger.debug(
                        "BM25 algorithm enabled (corpus: %s docs)",
                        corpus_manager.get_document_count(),
                    )
                except Exception as e:
                    logger.debug("Failed to initialize BM25: %s", e)

            # Conditionally add TopicRank if pytextrank is installed
            try:
                from src.core.vocabulary.algorithms.textrank_algorithm import TextRankAlgorithm

                topicrank = TextRankAlgorithm(nlp=ner.nlp)
                self.algorithms.append(topicrank)
                logger.info("TopicRank algorithm enabled (shared spaCy model)")
            except Exception as e:
                logger.warning("TopicRank unavailable: %s: %s", type(e).__name__, e, exc_info=True)
                skipped.append("TopicRank")

            # Conditionally add MedicalNER if scispacy is installed
            try:
                from src.core.vocabulary.algorithms.scispacy_algorithm import ScispaCyAlgorithm

                medical_ner = ScispaCyAlgorithm()
                self.algorithms.append(medical_ner)
                logger.debug("MedicalNER algorithm enabled")
            except ImportError:
                logger.debug("MedicalNER unavailable (scispacy not installed)")
                skipped.append("MedicalNER")

            # Conditionally add YAKE if installed
            try:
                from src.core.vocabulary.algorithms.yake_algorithm import YAKEAlgorithm

                yake_algo = YAKEAlgorithm()
                self.algorithms.append(yake_algo)
                logger.debug("YAKE algorithm enabled")
            except ImportError:
                logger.debug("YAKE unavailable (yake not installed)")
                skipped.append("YAKE")

        else:
            self.algorithms = algorithms
            skipped = []

        # Track algorithms that were unavailable at init time
        self.skipped_algorithms: list[str] = list(skipped)

        # Initialize merger with algorithm weights
        self.merger = AlgorithmScoreMerger(
            algorithm_weights={alg.name: alg.weight for alg in self.algorithms}
        )

        # Ensure NLTK data is available (wordnet used by NER rarity check)
        self._ensure_nltk_data()

        # Cache spaCy model reference for categorization (with lock for thread safety)
        self._nlp = None
        self._nlp_lock = threading.Lock()

        # Track original casing variants across documents for merge_document_results.
        # Populated by extract_from_document(), consumed by merge_document_results().
        # Maps lowercase_term → {original_casing: weighted_frequency}
        self._term_casing_variants: dict[str, dict[str, int]] = {}

        # Initialize meta-learner for ML-boosted quality scores
        self._meta_learner = get_meta_learner()

    @property
    def nlp(self):
        """Get spaCy model (from NER algorithm or load separately).

        Uses double-checked locking pattern for thread safety.
        """
        # Fast path: already loaded
        if self._nlp is not None:
            return self._nlp

        # Slow path: need to load (with lock to prevent concurrent loading)
        with self._nlp_lock:
            # Double-check after acquiring lock
            if self._nlp is None:
                # Try to get from NER algorithm
                for alg in self.algorithms:
                    if hasattr(alg, "nlp") and alg.nlp is not None:
                        self._nlp = alg.nlp
                        break
                # Fallback: load minimal model for categorization
                if self._nlp is None:
                    from src.config import (
                        SPACY_EN_CORE_WEB_LG_PATH,
                        SPACY_EN_CORE_WEB_SM_PATH,
                    )

                    if SPACY_EN_CORE_WEB_LG_PATH.exists():
                        self._nlp = spacy.load(str(SPACY_EN_CORE_WEB_LG_PATH))
                    elif SPACY_EN_CORE_WEB_SM_PATH.exists():
                        self._nlp = spacy.load(str(SPACY_EN_CORE_WEB_SM_PATH))
                    elif getattr(sys, "frozen", False):
                        raise RuntimeError(
                            f"Bundled spaCy model not found: "
                            f"{SPACY_EN_CORE_WEB_LG_PATH}\n"
                            f"Please reinstall the application to "
                            f"restore model files."
                        )
                    else:
                        try:
                            self._nlp = spacy.load("en_core_web_lg")
                            logger.warning("Using pip-installed spaCy (not bundled)")
                        except OSError:
                            try:
                                self._nlp = spacy.load("en_core_web_sm")
                                logger.warning("Using pip-installed en_core_web_sm")
                            except OSError:
                                raise RuntimeError(
                                    f"Language model not found. Expected "
                                    f"at: {SPACY_EN_CORE_WEB_LG_PATH}\n"
                                    f"For developers: python scripts/"
                                    f"download_models.py"
                                )
            return self._nlp

    def extract(
        self, text: str, doc_count: int = 1, doc_confidence: float = 100.0
    ) -> list[dict[str, str]]:
        """
        Extract vocabulary using all enabled algorithms.

        Args:
            text: The document text to analyze
            doc_count: Number of documents being processed (for frequency filtering)
            doc_confidence: Average/min confidence score of source documents (0-100).
                           Used as ML feature to weight terms from OCR-heavy documents.

        Returns:
            List of vocabulary dictionaries with standard schema:
            - Term: The extracted term
            - Type: Person/Place/Medical/Technical/Unknown
            - Role/Relevance: Context-specific role
            - Quality Score: 0-100 composite score
            - Occurrences: Term occurrence count
            - Google Rarity Rank: Google frequency rank
            - Definition: WordNet definition (Medical/Technical only)
            - Sources: Comma-separated algorithm names
        """
        original_kb = len(text) // 1024
        logger.debug("Starting multi-algorithm extraction on %sKB document", original_kb)

        # Limit text size (safety net — individual algorithms handle own limits)
        max_chars = VOCABULARY_MAX_TEXT_KB * 1024
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.warning(
                "Truncated to %sKB for processing (input was %sKB)",
                VOCABULARY_MAX_TEXT_KB,
                original_kb,
            )

        # 1. Run all enabled algorithms (parallel when possible)
        all_results = self._run_algorithms_parallel(text)

        # 2. Merge results from all algorithms
        logger.debug("Merging results from %s algorithms...", len(all_results))
        merged_terms = self.merger.merge(all_results)
        logger.debug("After merge: %s unique terms", len(merged_terms))

        # 3. Post-process: categorize, detect roles, add definitions
        logger.debug(
            "Post-processing (doc_count=%s, doc_confidence=%.1f%%)...",
            doc_count,
            doc_confidence,
        )
        vocabulary, filtered_terms = self._post_process(
            merged_terms, text, doc_count, doc_confidence
        )
        logger.debug(
            "After post-process: %s terms, %s filtered", len(vocabulary), len(filtered_terms)
        )

        # 4. Run vocabulary filter chain
        # Consolidates: name dedup, artifact filter, name regularizer,
        # rarity filter, corpus familiarity, and gibberish filter
        logger.debug("Running filter chain...")
        from src.core.vocabulary.filters import create_optimized_filter_chain

        filter_chain = create_optimized_filter_chain()
        filter_result = filter_chain.run(vocabulary)
        vocabulary = filter_result.vocabulary
        logger.debug(
            "Filter chain complete: %s removed, %s remaining",
            filter_result.removed_count,
            len(vocabulary),
        )

        # 8. Sort vocabulary based on configured method
        vocabulary = self._sort_vocabulary(vocabulary)

        return vocabulary, filtered_terms

    def extract_progressive(
        self,
        text: str,
        doc_count: int = 1,
        doc_confidence: float = 100.0,
        partial_callback=None,
        ner_progress_callback=None,
        status_callback=None,
    ) -> list[dict[str, str]]:
        """
        Extract vocabulary with progressive updates for better UX.

        Runs BM25 and RAKE first, sends partial results, then runs NER with
        progress updates for each chunk. This allows the GUI to show results
        quickly while NER processes large documents.

        Args:
            text: The document text to analyze
            doc_count: Number of documents being processed
            doc_confidence: Average confidence score of source documents (0-100)
            partial_callback: Optional callback(vocab_data) called after BM25+RAKE complete
            ner_progress_callback: Optional callback(vocab_data, chunk_num, total_chunks)
                                   called after each NER chunk completes
            status_callback: Optional callback(message) called before each algorithm
                            runs, e.g. "RAKE extraction in progress..."

        Returns:
            Final merged vocabulary list (same format as extract())
        """
        from src.system_resources import get_optimal_workers

        original_kb = len(text) // 1024
        logger.debug("Starting progressive extraction on %sKB document", original_kb)

        # Limit text size (safety net — individual algorithms handle own limits)
        max_chars = VOCABULARY_MAX_TEXT_KB * 1024
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.warning(
                "Truncated to %sKB for processing (input was %sKB)",
                VOCABULARY_MAX_TEXT_KB,
                original_kb,
            )

        # Separate algorithms into fast (BM25, RAKE) and slow (NER)
        fast_algorithms = []
        ner_algorithm = None
        for alg in self.algorithms:
            if not alg.enabled:
                continue
            if alg.name == "NER":
                ner_algorithm = alg
            else:
                fast_algorithms.append(alg)

        all_results = []

        # Phase 1: Run fast algorithms (BM25, RAKE) in parallel
        if fast_algorithms:
            logger.debug("Phase 1: Running %s fast algorithms...", len(fast_algorithms))
            workers = min(len(fast_algorithms), get_optimal_workers(task_ram_gb=0.5, max_workers=4))

            if len(fast_algorithms) == 1:
                # Sequential for single algorithm
                alg = fast_algorithms[0]
                if status_callback:
                    status_callback(_algo_status_message(alg.name))
                try:
                    result = alg.extract(text)
                    all_results.append(result)
                    logger.debug("%s: %s candidates", alg.name, len(result.candidates))
                except Exception as e:
                    logger.warning("Algorithm %s failed: %s", alg.name, e)
                    if alg.name not in self.skipped_algorithms:
                        self.skipped_algorithms.append(alg.name)
            else:
                # Sequential with status updates (so user sees per-algorithm messages)
                for alg in fast_algorithms:
                    if status_callback:
                        status_callback(_algo_status_message(alg.name))
                    try:
                        result = alg.extract(text)
                        all_results.append(result)
                        logger.debug("%s: %s candidates", alg.name, len(result.candidates))
                    except Exception as e:
                        logger.warning("Algorithm %s failed: %s", alg.name, e)
                        if alg.name not in self.skipped_algorithms:
                            self.skipped_algorithms.append(alg.name)

            # Send partial results if callback provided
            if partial_callback is not None and all_results:
                logger.debug("Sending partial results (BM25+RAKE)...")
                partial_merged = self.merger.merge(all_results)
                partial_vocab, _partial_filtered = self._post_process(
                    partial_merged, text, doc_count, doc_confidence
                )
                # Apply lightweight filter chain (skip rarity filter for partial results)
                # BM25 and RAKE find common keyphrases that rarity filter removes
                from src.core.vocabulary.filters import create_partial_results_filter_chain

                filter_chain = create_partial_results_filter_chain()
                filter_result = filter_chain.run(partial_vocab)
                partial_vocab = self._sort_vocabulary(filter_result.vocabulary)
                logger.debug(
                    "Partial filter chain: %s removed, %s remaining",
                    filter_result.removed_count,
                    len(partial_vocab),
                )
                try:
                    partial_callback(partial_vocab)
                except Exception as e:
                    logger.debug("Partial callback error: %s", e)

        # Phase 2: Run NER with progress callback
        if ner_algorithm is not None:
            if status_callback:
                status_callback(_algo_status_message("NER"))
            logger.debug("Phase 2: Running NER with progress updates...")

            # Wrapper to post-process NER chunk results before sending
            def ner_chunk_callback(chunk_candidates, chunk_num, total_chunks):
                if ner_progress_callback is None:
                    return
                # Convert candidates to vocab format for GUI
                # Note: These are raw candidates, not fully processed
                # The callback handler should merge with existing data
                ner_progress_callback(chunk_candidates, chunk_num, total_chunks)

            result = ner_algorithm.extract(text, progress_callback=ner_chunk_callback)
            all_results.append(result)
            logger.debug("NER complete: %s candidates", len(result.candidates))

        # Final merge and post-process
        logger.debug("Merging results from %s algorithms...", len(all_results))
        merged_terms = self.merger.merge(all_results)
        logger.debug("After merge: %s unique terms", len(merged_terms))

        vocabulary, filtered_terms = self._post_process(
            merged_terms, text, doc_count, doc_confidence
        )

        # Apply filter chain
        from src.core.vocabulary.filters import create_optimized_filter_chain

        filter_chain = create_optimized_filter_chain()
        filter_result = filter_chain.run(vocabulary)
        vocabulary = filter_result.vocabulary
        logger.debug(
            "Filter chain complete: %s removed, %s remaining",
            filter_result.removed_count,
            len(vocabulary),
        )

        vocabulary = self._sort_vocabulary(vocabulary)
        return vocabulary, filtered_terms

    def _run_algorithms_parallel(self, text: str) -> list:
        """
        Run extraction algorithms in parallel when beneficial.

        Uses ThreadPoolStrategy to run NER, RAKE, and BM25 concurrently.
        Falls back to sequential execution when:
        - Only 1 algorithm is enabled
        - System has only 1 CPU core

        Args:
            text: Document text to extract vocabulary from

        Returns:
            List of AlgorithmResult from all enabled algorithms
        """
        from src.core.parallel.executor_strategy import ThreadPoolStrategy
        from src.core.parallel.task_runner import ParallelTaskRunner
        from src.system_resources import get_optimal_workers

        # Get list of enabled algorithms
        enabled_algorithms = [alg for alg in self.algorithms if alg.enabled]

        if not enabled_algorithms:
            logger.debug("No algorithms enabled")
            return []

        # Log text hash for consistency verification
        # All algorithms receive IDENTICAL text - this hash verifies that guarantee
        import hashlib

        text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        logger.debug("Text hash for all algorithms: %s (%s chars)", text_hash, len(text))

        # Decide whether to parallelize
        # Skip parallelization for 1 algorithm or 1 CPU core
        cpu_count = os.cpu_count() or 1
        use_parallel = len(enabled_algorithms) > 1 and cpu_count > 1

        if not use_parallel:
            # Sequential fallback
            logger.debug("Running %s algorithm(s) sequentially", len(enabled_algorithms))
            all_results = []
            for algorithm in enabled_algorithms:
                logger.debug("Running %s algorithm...", algorithm.name)
                try:
                    result = algorithm.extract(text)
                    all_results.append(result)
                    logger.debug(
                        "%s: %s candidates in %.1fms",
                        algorithm.name,
                        len(result.candidates),
                        result.processing_time_ms,
                    )
                except Exception as e:
                    logger.warning("Algorithm %s failed: %s", algorithm.name, e)
                    if algorithm.name not in self.skipped_algorithms:
                        self.skipped_algorithms.append(algorithm.name)
            return all_results

        # Parallel execution
        # Each algorithm uses ~0.5GB RAM (spaCy model is shared)
        workers = min(len(enabled_algorithms), get_optimal_workers(task_ram_gb=0.5, max_workers=4))
        logger.debug(
            "Running %s algorithms in parallel (%s workers)",
            len(enabled_algorithms),
            workers,
        )

        start_time = time.time()
        strategy = ThreadPoolStrategy(max_workers=workers)

        def run_algorithm(algorithm):
            """Worker function to run a single algorithm."""
            result = algorithm.extract(text)
            return (algorithm.name, result)

        try:
            # Build items list: (task_id, payload)
            items = [(alg.name, alg) for alg in enabled_algorithms]

            runner = ParallelTaskRunner(strategy=strategy)
            task_results = runner.run(run_algorithm, items)

            # Collect results
            all_results = []
            for task_result in task_results:
                if task_result.success:
                    alg_name, result = task_result.result
                    all_results.append(result)
                    logger.debug(
                        "%s: %s candidates in %.1fms",
                        alg_name,
                        len(result.candidates),
                        result.processing_time_ms,
                    )
                else:
                    logger.debug(
                        "Algorithm %s failed: %s",
                        task_result.task_id,
                        task_result.error,
                    )
                    if task_result.task_id not in self.skipped_algorithms:
                        self.skipped_algorithms.append(task_result.task_id)

            total_time = (time.time() - start_time) * 1000
            logger.debug("Parallel extraction complete in %.1fms", total_time)

            return all_results

        finally:
            strategy.shutdown(wait=True)

    def _post_process(
        self,
        merged_terms: list[MergedTerm],
        full_text: str,
        doc_count: int,
        doc_confidence: float = 100.0,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """
        Post-process merged terms: categorize, detect roles, add definitions.

        Terms that fail frequency filters are collected into a separate
        filtered list instead of being discarded, so the UI can show them
        in a collapsed "Filtered" section.

        Args:
            merged_terms: List of MergedTerm from merger
            full_text: Complete document text for role detection
            doc_count: Number of documents being processed
            doc_confidence: Average/min confidence of source documents (0-100)

        Returns:
            (vocabulary, filtered_terms) — both are list[dict]
        """
        vocabulary = []
        filtered_terms = []
        seen_terms = set()
        # Estimate total pages across all docs (~3500 chars per transcript page)
        estimated_pages = max(len(full_text) // 3500, 1) * doc_count
        frequency_threshold = estimated_pages * 5
        # Total unique terms for ML occurrence_ratio feature
        total_unique_terms = len(merged_terms)

        # Read min occurrences once (user pref with config fallback)
        min_occurrences = get_user_preferences().get(
            "vocab_min_occurrences", VOCABULARY_MIN_OCCURRENCES
        )

        # Track iteration count for periodic GIL yield
        iteration_count = 0
        for merged in merged_terms:
            term = merged.term
            lower_term = term.lower()

            # Skip duplicates
            if lower_term in seen_terms:
                continue

            # Determine if NER detected this as a Person entity
            # This is the only reliable type information we track
            is_person = merged.final_type == "Person"

            # Detect role/relevance using profession-specific profile
            role_relevance = self._get_role_relevance(term, is_person, full_text)

            # Calculate quality score
            frequency_rank = self._get_term_frequency_rank(term)
            topicrank_score = float(merged.metadata.get("topicrank_score", 0.0))

            # Create legacy TermSources BEFORE quality score so confidence-based
            # adjustments (high-conf boost, all-low-conf penalty, single-source
            # penalty) actually fire instead of being dead code.
            sources_obj = TermSources.create_legacy(merged.frequency, doc_confidence / 100.0)

            base_quality_score = self._calculate_quality_score(
                is_person,
                merged.frequency,
                frequency_rank,
                len(merged.sources),
                term_sources=sources_obj,
                total_docs_in_session=doc_count,
                topicrank_score=topicrank_score,
                yake_score=float(merged.metadata.get("yake_score", 0.0)),
                rake_score=float(merged.metadata.get("rake_score", 0.0)),
                bm25_score=float(merged.metadata.get("bm25_score", 0.0)),
                term=term,
            )

            # Build term data for potential ML boost
            # Create per-algorithm detection flags for detailed tracking
            sources_upper = [s.upper() for s in merged.sources]
            algo_count = len(merged.sources)  # Number of algorithms that found this term

            # Build "Found By" display string from sources
            found_by = ", ".join(merged.sources)  # e.g., "NER, RAKE" or "NER, RAKE, BM25"

            term_data = {
                VF.TERM: term,
                VF.IS_PERSON: VF.YES if is_person else VF.NO,
                VF.FOUND_BY: found_by,  # Which algorithms found this term
                VF.ROLE_RELEVANCE: role_relevance,
                VF.QUALITY_SCORE: base_quality_score,
                VF.OCCURRENCES: merged.frequency,
                VF.GOOGLE_RARITY_RANK: frequency_rank,
                VF.SOURCES: ",".join(merged.sources),  # Keep for backward compatibility
                # Per-algorithm detection flags
                VF.NER: VF.YES if "NER" in sources_upper else VF.NO,
                VF.RAKE: VF.YES if "RAKE" in sources_upper else VF.NO,
                VF.BM25: VF.YES if "BM25" in sources_upper else VF.NO,
                VF.TOPICRANK: VF.YES if "TOPICRANK" in sources_upper else VF.NO,
                VF.MEDICALNER: VF.YES if "MEDICALNER" in sources_upper else VF.NO,
                VF.YAKE: VF.YES if "YAKE" in sources_upper else VF.NO,
                VF.ALGO_COUNT: algo_count,  # Sum of algorithms that found term
                # Display columns from TermSources
                VF.NUM_DOCS: sources_obj.num_documents,
                VF.OCR_CONFIDENCE: f"{sources_obj.median_confidence:.0%}",
                # TermSources object for ML/filters
                "sources": sources_obj,
                "total_docs_in_session": doc_count,
                # ML feature fields (from feedback CSV schema)
                "base_quality_score": base_quality_score,
                "occurrences": merged.frequency,
                "rarity_rank": frequency_rank,
                "algorithms": ",".join(merged.sources),
                "is_person": 1 if is_person else 0,  # Binary flag for ML
                "total_unique_terms": total_unique_terms,  # For ML occurrence_ratio
                "source_doc_confidence": doc_confidence,  # OCR quality for ML
                "topicrank_score": float(merged.metadata.get("topicrank_score", 0.0)),
                "yake_score": float(merged.metadata.get("yake_score", 0.0)),
                "rake_score": float(merged.metadata.get("rake_score", 0.0)),
                "bm25_score": float(merged.metadata.get("bm25_score", 0.0)),
                "total_word_count": len(full_text.split()),  # For freq_per_1k_words feature
            }

            # Apply ML boost if meta-learner is trained
            final_quality_score = self._apply_ml_boost(term_data, base_quality_score)
            term_data[VF.QUALITY_SCORE] = final_quality_score

            seen_terms.add(lower_term)

            # Frequency filters — divert to filtered list instead of discarding
            # High-scoring single-occurrence items (score >= 85) stay in main list
            if merged.frequency > frequency_threshold:
                term_data[VF.FILTER_REASON] = "too frequent"
                filtered_terms.append(term_data)
            elif merged.frequency < min_occurrences and final_quality_score >= 85:
                vocabulary.append(term_data)
            elif merged.frequency < min_occurrences:
                term_data[VF.FILTER_REASON] = "below min occurrences"
                filtered_terms.append(term_data)
            else:
                vocabulary.append(term_data)

            # Yield GIL every 50 terms to keep GUI responsive
            iteration_count += 1
            if iteration_count % 50 == 0:
                time.sleep(0)

        return vocabulary, filtered_terms

    def _get_role_relevance(self, term: str, is_person: bool, full_text: str) -> str:
        """Get role/relevance description for a term."""
        if is_person:
            return self.role_profile.detect_person_role(term, full_text)
        else:
            # For non-person terms, detect based on term characteristics
            return "Vocabulary term"

    def _calculate_quality_score(
        self,
        is_person: bool,
        term_count: int,
        frequency_rank: int,
        algorithm_count: int,
        term_sources: TermSources | None = None,
        total_docs_in_session: int = 1,
        topicrank_score: float = 0.0,
        yake_score: float = 0.0,
        rake_score: float = 0.0,
        bm25_score: float = 0.0,
        term: str = "",
    ) -> float:
        """
        Calculate composite quality score (0-100).

        Higher score = more likely to be a useful, high-quality term.

        Args:
            is_person: Whether NER detected this as a person name
            term_count: Number of occurrences
            frequency_rank: Google frequency rank
            algorithm_count: Number of algorithms that found this term
            term_sources: Optional TermSources for per-document confidence data
            total_docs_in_session: Total documents in this extraction session
            topicrank_score: TopicRank centrality score (0-1), 0 if not available
            yake_score: Raw YAKE score (lower = more important), 0 if not available
            rake_score: Raw RAKE score, 0 if not available
            bm25_score: Raw BM25 score, 0 if not available
            term: The term text (for artifact detection penalties)

        Returns:
            Quality score between 0.0 and 100.0
        """
        from src.config import (
            SCORE_ALGO_CONFIDENCE_BOOST,
            SCORE_ALL_LOW_CONF_PENALTY,
            SCORE_HIGH_CONF_BOOST,
            SCORE_MULTI_DOC_BOOST,
            SCORE_SINGLE_SOURCE_CONF_THRESHOLD,
            SCORE_SINGLE_SOURCE_MIN_DOCS,
            SCORE_SINGLE_SOURCE_PENALTY,
            SCORE_TOPICRANK_CENTRALITY_BOOST,
        )

        score = 50.0  # Base score

        # Boost for multiple occurrences (max +35)
        # Gentler curve with higher cap — more room to distinguish frequent terms
        # Examples: 1→+5, 4→+13, 10→+18, 50→+31, 300→+35
        occurrence_boost = min(math.log10(term_count + 1) * 18, 35)
        score += occurrence_boost

        # Boost for rare words (max +20)
        if frequency_rank == 0:
            score += 20  # Not in Google dataset - very rare
        elif frequency_rank > 200000:
            score += 15
        elif frequency_rank > 180000:
            score += 10

        # Tiered person name boost — uses per-word frequency to score
        # common names ("David Wilson") lower than rare ones ("Xiomara Bjelkengren").
        # Court reporters already have common names in their steno dictionaries.
        if is_person:
            term_words = term.split() if term else []
            is_multi_word = len(term_words) >= 2
            if is_multi_word:
                score += _person_multi_word_boost(term_words, self.frequency_rank_map)
            else:
                score += _person_single_word_boost(frequency_rank)

        # Boost for multi-algorithm agreement (non-linear tiers)
        # Terms found by multiple algorithms are more trustworthy
        if algorithm_count == 2:
            score += 4
        elif algorithm_count == 3:
            score += 8
        elif algorithm_count >= 4:
            score += 12

        # Boost for TopicRank centrality (scaled by score, capped at config max)
        # High centrality = term is central to document's topic graph
        if topicrank_score > 0:
            score += min(topicrank_score * 10, SCORE_TOPICRANK_CENTRALITY_BOOST)

        # Algorithm confidence boost — extra points when algorithms report high confidence
        # Normalize all to 0-1 higher=better, take best score
        algo_confidences = []
        if yake_score > 0:
            algo_confidences.append(1.0 / (1.0 + yake_score))  # Invert raw YAKE
        if rake_score > 0:
            algo_confidences.append(min(rake_score / 15.0, 1.0))
        if bm25_score > 0:
            algo_confidences.append(min(bm25_score / 15.0, 1.0))
        if algo_confidences:
            best_confidence = max(algo_confidences)
            score += min(best_confidence * 8, SCORE_ALGO_CONFIDENCE_BOOST)

        # === TermSources-based adjustments ===
        if term_sources is not None and term_sources.num_documents > 0:
            # Boost for terms found in multiple documents
            if term_sources.num_documents >= 2:
                score += SCORE_MULTI_DOC_BOOST

            # Boost for high-confidence sources
            if term_sources.high_conf_doc_ratio > 0.8:
                score += SCORE_HIGH_CONF_BOOST

            # Penalty if ALL sources are low confidence
            if term_sources.all_low_conf:
                score += SCORE_ALL_LOW_CONF_PENALTY

            # Conditional single-source penalty:
            # Only apply when session has 3+ docs and term is in only 1 low-conf doc
            if (
                total_docs_in_session >= SCORE_SINGLE_SOURCE_MIN_DOCS
                and term_sources.num_documents == 1
                and term_sources.mean_confidence < SCORE_SINGLE_SOURCE_CONF_THRESHOLD
            ):
                score += SCORE_SINGLE_SOURCE_PENALTY

        # === User-defined indicator patterns ===
        # Modest nudge from user's positive/negative indicator settings (+/-5).
        # Kept small because the rule-based score is poisoning-resistant.
        if term:
            from src.core.vocabulary.indicator_patterns import (
                matches_negative,
                matches_positive,
            )

            if matches_negative(term):
                score -= 5.0
            elif matches_positive(term):
                score += 5.0

        # === Artifact detection penalties ===
        # Toughened to match rules' 45% floor — these need teeth.
        if term:
            # All caps - headers/labels, rarely useful vocabulary (-12)
            alpha_chars = [c for c in term if c.isalpha()]
            if alpha_chars and all(c.isupper() for c in alpha_chars):
                score -= 12.0

            # Leading digit - line numbers attached to terms (-8)
            if term[0].isdigit():
                score -= 8.0

            # Single letter - "Q", "A" transcript artifacts (-15)
            stripped = term.strip()
            if len(stripped) == 1 and stripped.isalpha():
                score -= 15.0

            # Trailing punctuation - extraction boundary errors (-5)
            if term[-1] in ":;.,!?":
                score -= 5.0

        return min(100.0, max(0.0, round(score, 1)))

    def _apply_ml_boost(self, term_data: dict, base_score: float) -> float:
        """
        Apply graduated ML-based scoring if meta-learner is trained.

        Uses graduated weight based on user sample count.
        Formula: score = base_score * (1 - ml_weight) + ml_prob * 100 * ml_weight

        The ml_weight increases with training corpus size:
        - < 30 samples: 0% (rules only)
        - 30-40 samples: 25%
        - 41-60 samples: 35%
        - 61-100 samples: 45%
        - 100+ samples: 55% (cap) → rules floor = 45%

        Args:
            term_data: Dictionary with term features for ML prediction
            base_score: Rule-based quality score

        Returns:
            Final quality score blending rules and ML based on weight
        """
        if not self._meta_learner.is_trained:
            return base_score

        try:
            # Get ML weight based on user sample count
            ml_weight = self._meta_learner.get_ml_weight()

            # If weight is 0, just use base score (not enough training data)
            if ml_weight == 0:
                return base_score

            # Get ML prediction (probability of user approval)
            preference_prob = self._meta_learner.predict_preference(term_data)

            # Blend base score with ML prediction
            # ML prob * 100 converts probability to 0-100 scale
            ml_score = preference_prob * 100
            final_score = base_score * (1 - ml_weight) + ml_score * ml_weight
            final_score = min(100.0, max(0.0, round(final_score, 1)))

            # Log significant ML adjustments for debugging
            score_diff = final_score - base_score
            if abs(score_diff) > 5:
                term = term_data.get(VF.TERM, "?")
                logger.debug(
                    "'%s': prob=%.2f, weight=%.0f%%, base=%.1f -> final=%.1f (%+.1f)",
                    term,
                    preference_prob,
                    ml_weight * 100,
                    base_score,
                    final_score,
                    score_diff,
                )

            return final_score

        except Exception as e:
            logger.debug("Error applying boost: %s", e)
            return base_score

    # Definition lookup removed: WordNet generic definitions were unhelpful for
    # legal terms (e.g., "Hearing" → sense-organ, not legal proceeding).
    # To reinstate, uncomment and restore the Definition column in extract() and
    # _merge_term_across_docs().
    # def _get_definition(self, term: str, is_person: bool) -> str:
    #     """Get definition for non-person terms only."""
    #     if is_person:
    #         return "—"
    #     lower_term = term.lower()
    #     synsets = wordnet.synsets(lower_term)
    #     if synsets:
    #         definition = synsets[0].definition()
    #         if len(definition) > 100:
    #             definition = definition[:97] + "..."
    #         return definition
    #     return "—"

    def _select_best_casing(self, term_lower: str) -> str:
        """
        Select the best casing variant for a term from extraction data.

        Uses frequency-weighted variants tracked during extract_from_document().
        Prefers mixed-case over ALL-CAPS, falls back to lowercase if no
        casing data exists.

        Args:
            term_lower: Lowercase term key

        Returns:
            Best-cased display term (e.g., "radiculopathy", "John Smith")
        """
        variants = self._term_casing_variants.get(term_lower)
        if not variants:
            # No casing data — use lowercase as-is (no blind capitalization)
            return term_lower

        # Sort by frequency, with tie-breaker preferring mixed case over ALL-CAPS
        def sort_key(item):
            term, count = item
            is_mixed = not term.isupper() and not term.islower()
            return (count, is_mixed, term)

        sorted_variants = sorted(variants.items(), key=sort_key, reverse=True)
        return sorted_variants[0][0]

    def _get_term_frequency_rank(self, term: str) -> int:
        """Get Google frequency rank for a term."""
        return self.frequency_rank_map.get(term.lower(), 0)

    def _sort_vocabulary(self, vocabulary: list[dict]) -> list[dict]:
        """
        Sort vocabulary based on configured method.

        Args:
            vocabulary: List of term dictionaries

        Returns:
            Sorted vocabulary list
        """
        if self.sort_method == "quality_score":
            logger.debug("Sorting by Quality Score (highest first)")
            return self._sort_by_quality_score(vocabulary)
        elif self.sort_method == "rarity" and self.frequency_dataset:
            logger.debug("Sorting by rarity (rarest first)")
            return self._sort_by_rarity(vocabulary)
        else:
            logger.debug("No sorting applied")
            return vocabulary

    def _sort_by_quality_score(self, vocabulary: list[dict]) -> list[dict]:
        """
        Sort vocabulary by Quality Score (highest first).

        This puts ML-boosted terms at the top. Terms the model predicts
        the user will approve get higher scores and appear first.

        Args:
            vocabulary: List of term dictionaries

        Returns:
            Sorted vocabulary list (highest Quality Score first)
        """
        return sorted(vocabulary, key=lambda x: float(x.get(VF.QUALITY_SCORE) or 0), reverse=True)

    def _sort_by_rarity(self, vocabulary: list[dict]) -> list[dict]:
        """Sort vocabulary list by rarity (rarest first)."""
        not_in_dataset = []
        in_dataset = []

        for item in vocabulary:
            term = item[VF.TERM].lower()
            if term not in self.frequency_dataset:
                not_in_dataset.append(item)
            else:
                in_dataset.append(item)

        in_dataset.sort(key=lambda x: self.frequency_dataset.get(x[VF.TERM].lower(), float("inf")))
        return not_in_dataset + in_dataset

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _looks_like_person_name(self, term: str) -> bool:
        """Check if term looks like a person name."""
        words = term.split()
        if len(words) < 1 or len(words) > 4:
            return False

        # Check if words are capitalized (typical for names)
        capitalized = all(w[0].isupper() for w in words if w)

        # Check for organization indicators
        for indicator in ORGANIZATION_INDICATORS:
            if indicator.lower() in term.lower():
                return False

        return capitalized

    def _looks_like_organization(self, term: str) -> bool:
        """Check if term looks like an organization."""
        return any(indicator.lower() in term.lower() for indicator in ORGANIZATION_INDICATORS)

    def _load_word_list(self, file_path) -> set[str]:
        """Load a list of words from a file."""
        if file_path is None:
            return set()

        file_path = Path(file_path) if isinstance(file_path, str) else file_path

        if not file_path.exists():
            logger.debug("Word list not found: %s", file_path)
            return set()

        with open(file_path, encoding="utf-8") as f:
            word_list = {line.strip().lower() for line in f if line.strip()}
            logger.debug("Loaded %s words from %s", len(word_list), file_path)
            return word_list

    def _load_frequency_dataset(self) -> tuple[dict[str, int], dict[str, int]]:
        """Load Google word frequency dataset and build rank mapping."""
        from src.core.vocabulary.frequency_data import load_raw_frequency_data

        frequency_dict = load_raw_frequency_data()
        if not frequency_dict:
            return {}, {}

        # Build rank map (most frequent = rank 0)
        sorted_words = sorted(frequency_dict.items(), key=lambda x: x[1], reverse=True)
        rank_map = {word: rank for rank, (word, _) in enumerate(sorted_words)}
        logger.debug("Built rank map for %s words", len(rank_map))

        return frequency_dict, rank_map

    def _ensure_nltk_data(self):
        """Ensure NLTK data is available."""
        try:
            wordnet.synsets("test")
        except LookupError:
            if getattr(sys, "frozen", False):
                hint = "Please reinstall the application to restore data files."
            else:
                hint = "Run: python scripts/download_models.py"
            raise RuntimeError(f"NLTK 'wordnet' corpus not found. {hint}")

    def reload_user_exclusions(self):
        """Reload user exclusions from file."""
        if self.user_exclude_path:
            self.user_exclude_list = self._load_word_list(self.user_exclude_path)
            # Also update NER algorithm's exclusion list
            for alg in self.algorithms:
                if hasattr(alg, "user_exclude_list"):
                    alg.user_exclude_list = self.user_exclude_list

    def _should_enable_bm25(self) -> bool:
        """
        Check if BM25 algorithm should be enabled.

        BM25 requires:
        1. User has enabled BM25 in settings (default: True)
        2. Corpus has at least 5 documents

        Returns:
            True if BM25 should be added to algorithm list
        """
        try:
            from src.config import CORPUS_MIN_DOCUMENTS
            from src.core.vocabulary.corpus_manager import get_corpus_manager
            from src.user_preferences import get_user_preferences

            # Check user preference
            prefs = get_user_preferences()
            if not prefs.get("bm25_enabled", True):
                logger.debug("BM25 disabled by user preference")
                return False

            # Check corpus readiness
            corpus_manager = get_corpus_manager()
            if not corpus_manager.is_corpus_ready(min_docs=CORPUS_MIN_DOCUMENTS):
                doc_count = corpus_manager.get_document_count()
                logger.debug(
                    "BM25 skipped: corpus has %s/%s documents",
                    doc_count,
                    CORPUS_MIN_DOCUMENTS,
                )
                return False

            return True

        except Exception as e:
            logger.debug("BM25 check failed: %s", e)
            return False

    # ========================================================================
    # PER-DOCUMENT PARALLEL EXTRACTION
    # ========================================================================

    def extract_documents(self, documents, progress_callback=None):
        """
        Run full pipeline per document (in parallel), then merge results.

        Args:
            documents: List of {"text", "doc_id", "confidence"}
            progress_callback: Optional callback(current, total, doc_id)

        Returns:
            (vocabulary, filtered_terms) — both are list[dict]
        """
        if not documents:
            return [], []

        # Single doc — no merge needed, just run normally
        if len(documents) == 1:
            doc = documents[0]
            return self.extract(doc["text"], doc_count=1, doc_confidence=doc["confidence"])

        # Multi-doc — parallel extraction
        from concurrent.futures import as_completed

        from src.core.parallel.executor_strategy import ThreadPoolStrategy
        from src.system_resources import get_optimal_workers

        max_workers = min(get_optimal_workers(), len(documents))
        strategy = ThreadPoolStrategy(max_workers=max_workers)

        def _extract_single(doc):
            """Extract vocab from one document (runs in worker thread)."""
            vocab, filtered = self.extract(
                doc["text"], doc_count=1, doc_confidence=doc["confidence"]
            )
            return (doc["doc_id"], doc["confidence"], vocab, filtered)

        # Submit all docs to thread pool
        per_doc_results = []
        futures = {}
        for doc in documents:
            future = strategy.submit(_extract_single, doc)
            futures[future] = doc["doc_id"]

        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            per_doc_results.append(result)
            completed += 1
            if progress_callback:
                progress_callback(completed, len(documents), futures[future])

        strategy.shutdown()

        # Collect filtered terms from all documents (simple dedup by term name)
        all_filtered = []
        seen_filtered = set()
        for _doc_id, _conf, _vocab, filtered in per_doc_results:
            for term_data in filtered:
                lower = term_data.get(VF.TERM, "").lower()
                if lower not in seen_filtered:
                    seen_filtered.add(lower)
                    all_filtered.append(term_data)

        # Merge main vocab results (uses only the vocab portion)
        vocab_results = [(doc_id, conf, vocab) for doc_id, conf, vocab, _f in per_doc_results]
        merged_vocab = self._merge_multi_doc_results(vocab_results, len(documents))

        return merged_vocab, all_filtered

    def _merge_multi_doc_results(self, per_doc_results, total_docs):
        """
        Merge per-document vocab lists into one with proper multi-doc tracking.

        Args:
            per_doc_results: List of (doc_id, confidence, vocab_list) tuples
            total_docs: Total documents in session

        Returns:
            Merged vocabulary list[dict]
        """
        # Index: term_lower -> list of (doc_id, confidence, term_dict)
        term_index = {}

        for doc_id, confidence, vocab_list in per_doc_results:
            for term_dict in vocab_list:
                key = term_dict[VF.TERM].lower()
                if key not in term_index:
                    term_index[key] = []
                term_index[key].append((doc_id, confidence, term_dict))

        # Merge each term
        merged_vocab = []
        total_unique = len(term_index)

        for key, doc_entries in term_index.items():
            merged = self._merge_term_across_docs(doc_entries, total_docs, total_unique)
            merged_vocab.append(merged)

        # Re-apply ML boost with correct multi-doc data
        for term_data in merged_vocab:
            base = term_data["base_quality_score"]
            term_data[VF.QUALITY_SCORE] = self._apply_ml_boost(term_data, base)

        # Sort by quality score descending
        return self._sort_vocabulary(merged_vocab)

    def _merge_term_across_docs(self, doc_entries, total_docs, total_unique):
        """
        Merge one term's data from multiple documents.

        Args:
            doc_entries: [(doc_id, confidence, term_dict), ...]
            total_docs: Total documents in session
            total_unique: Total unique terms across all docs

        Returns:
            Single merged term_dict
        """
        from collections import Counter

        # Build TermSources from actual per-doc data
        doc_ids = []
        confidences = []
        counts_per_doc = []
        for doc_id, confidence, td in doc_entries:
            doc_ids.append(doc_id)
            confidences.append(confidence / 100.0)
            counts_per_doc.append(td.get(VF.OCCURRENCES, td.get(VF.FREQUENCY, 1)))

        sources = TermSources(
            doc_ids=doc_ids,
            confidences=confidences,
            counts_per_doc=counts_per_doc,
        )

        # Term display: most frequent casing
        casing_votes = Counter(td[VF.TERM] for _, _, td in doc_entries)
        best_term = casing_votes.most_common(1)[0][0]

        # Boolean fields: "Yes" if ANY doc says "Yes"
        is_person = any(td.get(VF.IS_PERSON) == VF.YES for _, _, td in doc_entries)
        ner = any(td.get(VF.NER) == VF.YES for _, _, td in doc_entries)
        rake = any(td.get(VF.RAKE) == VF.YES for _, _, td in doc_entries)
        bm25 = any(td.get(VF.BM25) == VF.YES for _, _, td in doc_entries)
        topicrank = any(td.get(VF.TOPICRANK) == VF.YES for _, _, td in doc_entries)
        medical_ner = any(td.get(VF.MEDICALNER) == VF.YES for _, _, td in doc_entries)
        yake = any(td.get(VF.YAKE) == VF.YES for _, _, td in doc_entries)

        # Set fields: union across docs
        all_found_by = set()
        for _, _, td in doc_entries:
            found = td.get(VF.FOUND_BY, td.get(VF.SOURCES, ""))
            for algo in found.replace(",", " ").split():
                algo = algo.strip()
                if algo and algo != "—":
                    all_found_by.add(algo)
        found_by_str = ", ".join(sorted(all_found_by))

        # Numeric: sum / max / first
        total_freq = sum(
            td.get(VF.OCCURRENCES, td.get(VF.FREQUENCY, 0)) for _, _, td in doc_entries
        )
        rarity_rank = doc_entries[0][2].get(VF.GOOGLE_RARITY_RANK, 0)
        topicrank_score = max(
            (td.get("topicrank_score", 0.0) for _, _, td in doc_entries),
            default=0.0,
        )
        # YAKE: lower raw = better, but 0.0 means "no YAKE data", not best score.
        # Filter to non-zero before taking min (best YAKE score across docs).
        yake_values = [
            td.get("yake_score", 0.0) for _, _, td in doc_entries if td.get("yake_score")
        ]
        yake_score = min(yake_values) if yake_values else 0.0
        rake_score = max(
            (td.get("rake_score", 0.0) for _, _, td in doc_entries),
            default=0.0,
        )
        bm25_score = max(
            (td.get("bm25_score", 0.0) for _, _, td in doc_entries),
            default=0.0,
        )
        min_doc_confidence = min(c for _, c, _ in doc_entries)

        # String: longest non-default
        default_role = "Vocabulary term"
        roles = [td.get(VF.ROLE_RELEVANCE, default_role) for _, _, td in doc_entries]
        best_role = max(
            (r for r in roles if r != default_role),
            key=len,
            default=default_role,
        )

        # Definition removed: WordNet definitions unhelpful for legal terms.
        # To reinstate, uncomment below and restore _get_definition() method.
        # definition = ""
        # for _, _, td in doc_entries:
        #     d = td.get("Definition", "")
        #     if d and d != "—":
        #         definition = d
        #         break

        # Algo count: recompute from merged flags
        algo_count = sum([ner, rake, bm25, topicrank, medical_ner, yake])

        # Quality score: recalculate with real TermSources
        quality_score = self._calculate_quality_score(
            is_person=is_person,
            term_count=total_freq,
            frequency_rank=rarity_rank,
            algorithm_count=algo_count,
            term_sources=sources,
            total_docs_in_session=total_docs,
            topicrank_score=topicrank_score,
            yake_score=yake_score,
            rake_score=rake_score,
            bm25_score=bm25_score,
            term=best_term,
        )

        return {
            # Display columns
            VF.TERM: best_term,
            VF.IS_PERSON: VF.YES if is_person else VF.NO,
            VF.FOUND_BY: found_by_str,
            VF.ROLE_RELEVANCE: best_role,
            VF.QUALITY_SCORE: quality_score,
            VF.OCCURRENCES: total_freq,
            VF.GOOGLE_RARITY_RANK: rarity_rank,
            # "Definition": definition or "—",  # Removed: see Phase 4 comment
            VF.SOURCES: found_by_str,
            # Algorithm flags
            VF.NER: VF.YES if ner else VF.NO,
            VF.RAKE: VF.YES if rake else VF.NO,
            VF.BM25: VF.YES if bm25 else VF.NO,
            VF.TOPICRANK: VF.YES if topicrank else VF.NO,
            VF.MEDICALNER: VF.YES if medical_ner else VF.NO,
            VF.YAKE: VF.YES if yake else VF.NO,
            VF.ALGO_COUNT: algo_count,
            # TermSources display
            VF.NUM_DOCS: sources.num_documents,
            VF.OCR_CONFIDENCE: f"{sources.median_confidence:.0%}",
            # TermSources object (for ML/filters)
            "sources": sources,
            "total_docs_in_session": total_docs,
            # ML feature fields
            "base_quality_score": quality_score,
            "occurrences": total_freq,
            "rarity_rank": rarity_rank,
            "algorithms": found_by_str,
            "is_person": 1 if is_person else 0,
            "total_unique_terms": total_unique,
            "source_doc_confidence": min_doc_confidence,
            "topicrank_score": topicrank_score,
            "yake_score": yake_score,
            "rake_score": rake_score,
            "bm25_score": bm25_score,
            "total_word_count": sum(td.get("total_word_count", 0) for _, _, td in doc_entries),
            "corpus_common_term": any(
                td.get("corpus_common_term", False) for _, _, td in doc_entries
            ),
        }

    # ========================================================================
    # PER-DOCUMENT EXTRACTION — Legacy
    # ========================================================================

    def extract_from_document(
        self, text: str, doc_id: str, doc_confidence: float
    ) -> dict[str, int]:
        """
        Extract vocabulary from a single document.

        Per-document extraction for TermSources tracking.
        This method extracts terms from ONE document only. Call this
        for each document, then use merge_document_results() to combine.

        Also populates self._term_casing_variants with frequency-weighted
        casing data so merge_document_results can pick the best variant
        instead of blindly capitalizing.

        Args:
            text: The document text to analyze
            doc_id: Unique identifier for this document (e.g., file hash)
            doc_confidence: OCR/extraction confidence (0-100)

        Returns:
            Dict mapping term (lowercase) → count for this document only.
            Example: {"john smith": 5, "radiculopathy": 3}
        """
        logger.debug("Extracting from document %s... (conf=%.1f%%)", doc_id[:12], doc_confidence)

        # Limit text size (safety net — individual algorithms handle own limits)
        max_chars = VOCABULARY_MAX_TEXT_KB * 1024
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.warning("Truncated to %sKB for processing", VOCABULARY_MAX_TEXT_KB)

        # Run algorithms
        all_results = self._run_algorithms_parallel(text)

        # Merge algorithm results
        merged_terms = self.merger.merge(all_results)

        # Build term → count mapping and track original casing
        term_counts: dict[str, int] = {}
        for merged in merged_terms:
            lower_term = merged.term.lower()
            term_counts[lower_term] = merged.frequency
            # Track casing variants across documents for merge_document_results
            if lower_term not in self._term_casing_variants:
                self._term_casing_variants[lower_term] = {}
            original = merged.term
            self._term_casing_variants[lower_term][original] = (
                self._term_casing_variants[lower_term].get(original, 0) + merged.frequency
            )

        logger.debug("Document %s: %s unique terms", doc_id[:12], len(term_counts))
        return term_counts

    def merge_document_results(
        self, doc_results: list[tuple[str, float, dict[str, int]]], full_text: str | None = None
    ) -> list[dict]:
        """
        Merge per-document extractions into final vocabulary with TermSources.

        Combines results from multiple documents while tracking
        which documents contributed each term occurrence. This enables
        confidence-weighted canonical selection.

        Args:
            doc_results: List of (doc_id, confidence, {term: count}) tuples
                        from extract_from_document() calls
            full_text: Optional combined text for role detection

        Returns:
            Vocabulary list with TermSources attached to each term.
        """
        if not doc_results:
            return []

        total_docs = len(doc_results)
        logger.debug("Merging results from %s documents...", total_docs)

        # Build TermSources for each unique term
        # term_data[lower_term] = {
        #     "original_term": str,  # Best-cased version
        #     "sources": TermSources,
        #     "total_count": int,
        #     "algorithms": set[str],  # Which algorithms found it
        # }
        term_data: dict[str, dict] = {}

        for doc_id, confidence, term_counts in doc_results:
            for term_lower, count in term_counts.items():
                if term_lower not in term_data:
                    term_data[term_lower] = {
                        "original_term": term_lower,  # Will be replaced with best case
                        "doc_ids": [],
                        "confidences": [],
                        "counts_per_doc": [],
                        "total_count": 0,
                    }

                term_data[term_lower]["doc_ids"].append(doc_id)
                term_data[term_lower]["confidences"].append(confidence / 100.0)  # Normalize to 0-1
                term_data[term_lower]["counts_per_doc"].append(count)
                term_data[term_lower]["total_count"] += count

        logger.debug("Found %s unique terms across all documents", len(term_data))

        # Convert to TermSources and build vocabulary
        vocabulary = []
        for term_lower, data in term_data.items():
            # Create TermSources
            sources = TermSources(
                doc_ids=data["doc_ids"],
                confidences=data["confidences"],
                counts_per_doc=data["counts_per_doc"],
            )

            # Use best-cased variant from extraction (frequency-weighted)
            display_term = self._select_best_casing(term_lower)

            # Determine if this is a person (will be refined in _post_process)
            is_person = False  # Default, will be updated below

            # Calculate base quality score (pass TermSources for quality adjustments)
            frequency_rank = self._get_term_frequency_rank(term_lower)
            base_quality_score = self._calculate_quality_score(
                is_person,
                data["total_count"],
                frequency_rank,
                1,
                term_sources=sources,
                total_docs_in_session=total_docs,
                term=display_term,
            )

            term_dict = {
                VF.TERM: display_term,
                VF.IS_PERSON: VF.NO,  # Will be updated by filter chain
                VF.FOUND_BY: "—",
                VF.ROLE_RELEVANCE: "Vocabulary term",
                VF.QUALITY_SCORE: base_quality_score,
                VF.OCCURRENCES: data["total_count"],
                VF.GOOGLE_RARITY_RANK: frequency_rank,
                # "Definition": "—",  # Removed: see Phase 4 comment
                VF.SOURCES: "",
                VF.NER: VF.NO,
                VF.RAKE: VF.NO,
                VF.BM25: VF.NO,
                VF.ALGO_COUNT: 0,
                # Display columns from TermSources
                VF.NUM_DOCS: sources.num_documents,
                VF.OCR_CONFIDENCE: f"{sources.median_confidence:.0%}",
                # TermSources tracking
                "sources": sources,
                "total_docs_in_session": total_docs,
                # ML feature fields
                "base_quality_score": base_quality_score,
                "occurrences": data["total_count"],
                "rarity_rank": frequency_rank,
                "is_person": 0,
                "total_unique_terms": len(term_data),
                "source_doc_confidence": sources.mean_confidence * 100,
                "total_word_count": 0,  # Not available in legacy per-doc path
            }

            vocabulary.append(term_dict)

        # Run filter chain
        logger.debug("Running filter chain on merged results...")
        from src.core.vocabulary.filters import create_optimized_filter_chain

        filter_chain = create_optimized_filter_chain()
        filter_result = filter_chain.run(vocabulary)
        vocabulary = filter_result.vocabulary
        logger.debug("Filter chain complete: %s removed", filter_result.removed_count)

        # Sort and return
        vocabulary = self._sort_vocabulary(vocabulary)
        return vocabulary

    def extract_per_document(self, documents: list[dict], progress_callback=None) -> list[dict]:
        """
        High-level per-document extraction with TermSources tracking.

        Convenience method that handles the full per-document extraction
        workflow. Extracts from each document individually, then merges
        with TermSources for canonical selection.

        Applies preprocessing pipeline to each document before extraction
        to remove transcript artifacts (Q./A. notation, headers, etc.).

        Args:
            documents: List of dicts with keys:
                      - 'text': Document text
                      - 'doc_id': Unique identifier (e.g., file hash)
                      - 'confidence': OCR confidence (0-100)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Final vocabulary list with TermSources attached.
        """
        if not documents:
            return []

        total_docs = len(documents)
        logger.debug("Starting per-document extraction for %s documents", total_docs)

        # Reset casing tracker for this extraction run
        self._term_casing_variants = {}

        # Create preprocessing pipeline to clean transcript artifacts
        # (headers, Q./A. notation, line numbers, etc.) before NER extraction
        from src.core.preprocessing import create_default_pipeline

        preprocessing_pipeline = create_default_pipeline()

        # Extract from each document
        doc_results = []
        combined_text_parts = []

        for i, doc in enumerate(documents):
            text = doc.get("text", "")
            doc_id = doc.get("doc_id", f"doc_{i}")
            confidence = doc.get("confidence", 100.0)

            if not text.strip():
                continue

            # Preprocess text before extraction
            # This removes headers, Q./A. notation, line numbers, etc.
            text = preprocessing_pipeline.process(text)

            # Extract terms from this document
            term_counts = self.extract_from_document(text, doc_id, confidence)
            doc_results.append((doc_id, confidence, term_counts))
            combined_text_parts.append(text)

            if progress_callback:
                progress_callback(i + 1, total_docs)

        # Merge all document results
        combined_text = "\n\n".join(combined_text_parts)
        vocabulary = self.merge_document_results(doc_results, combined_text)

        logger.debug("Per-document extraction complete: %s terms", len(vocabulary))
        return vocabulary
