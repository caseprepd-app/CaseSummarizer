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
The LLM extraction path intentionally uses chunks (different from other algorithms)
for efficiency with Ollama. This is documented and by design.
"""

import logging
import math
import os
import time
from pathlib import Path

import nltk
import spacy
from nltk.corpus import wordnet

from src.config import (
    GOOGLE_WORD_FREQUENCY_FILE,
    VOCABULARY_MAX_TEXT_KB,
    VOCABULARY_MIN_OCCURRENCES,
    VOCABULARY_RARITY_THRESHOLD,
    VOCABULARY_SORT_METHOD,
)
from src.core.utils.tokenizer import STOPWORDS
from src.core.vocabulary.algorithms.base import BaseExtractionAlgorithm
from src.core.vocabulary.preference_learner import get_meta_learner
from src.core.vocabulary.reconciler import VocabularyDeduplicator
from src.core.vocabulary.result_merger import AlgorithmScoreMerger, MergedTerm
from src.core.vocabulary.role_profiles import RoleDetectionProfile, StenographerProfile
from src.core.vocabulary.term_sources import TermSources
from src.user_preferences import get_user_preferences

logger = logging.getLogger(__name__)

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

        # Session 80: Log user exclusions for debugging
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

            # Conditionally add BM25 if enabled and corpus is ready (Session 26)
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

            # Conditionally add TextRank if pytextrank is installed
            try:
                from src.core.vocabulary.algorithms.textrank_algorithm import TextRankAlgorithm

                textrank = TextRankAlgorithm()
                self.algorithms.append(textrank)
                logger.debug("TextRank algorithm enabled")
            except ImportError:
                logger.debug("TextRank unavailable (pytextrank not installed)")

            # Conditionally add MedicalNER if scispacy is installed
            try:
                from src.core.vocabulary.algorithms.scispacy_algorithm import ScispaCyAlgorithm

                medical_ner = ScispaCyAlgorithm()
                self.algorithms.append(medical_ner)
                logger.debug("MedicalNER algorithm enabled")
            except ImportError:
                logger.debug("MedicalNER unavailable (scispacy not installed)")

            # Conditionally add GLiNER if enabled and gliner is installed
            if self._should_enable_gliner():
                try:
                    from src.core.vocabulary.algorithms.gliner_algorithm import GLiNERAlgorithm

                    gliner_labels = prefs.get("gliner_labels", None)
                    gliner = GLiNERAlgorithm(labels=gliner_labels)
                    self.algorithms.append(gliner)
                    logger.debug("GLiNER algorithm enabled with %d labels", len(gliner.labels))
                except ImportError:
                    logger.debug("GLiNER unavailable (gliner not installed)")
                except Exception as e:
                    logger.debug("Failed to initialize GLiNER: %s", e)
        else:
            self.algorithms = algorithms

        # Initialize merger with algorithm weights
        self.merger = AlgorithmScoreMerger(
            algorithm_weights={alg.name: alg.weight for alg in self.algorithms}
        )

        # Ensure NLTK data is available (for definitions)
        self._ensure_nltk_data()

        # Cache spaCy model reference for categorization
        self._nlp = None

        # Initialize meta-learner for ML-boosted quality scores (Session 25)
        self._meta_learner = get_meta_learner()

    @property
    def nlp(self):
        """Get spaCy model (from NER algorithm or load separately)."""
        if self._nlp is None:
            # Try to get from NER algorithm
            for alg in self.algorithms:
                if hasattr(alg, "nlp") and alg.nlp is not None:
                    self._nlp = alg.nlp
                    break
            # Fallback: load minimal model for categorization
            if self._nlp is None:
                try:
                    self._nlp = spacy.load("en_core_web_lg")
                except OSError:
                    self._nlp = spacy.load("en_core_web_sm")
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
            - In-Case Freq: Term occurrence count
            - Freq Rank: Google frequency rank
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

        # 1. Run all enabled algorithms (parallel when possible - Session 69)
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
        vocabulary = self._post_process(merged_terms, text, doc_count, doc_confidence)
        logger.debug("After post-process: %s terms", len(vocabulary))

        # 4. Run vocabulary filter chain (Session 71)
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

        return vocabulary

    def extract_progressive(
        self,
        text: str,
        doc_count: int = 1,
        doc_confidence: float = 100.0,
        partial_callback=None,
        ner_progress_callback=None,
    ) -> list[dict[str, str]]:
        """
        Extract vocabulary with progressive updates for better UX (Session 85).

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

        Returns:
            Final merged vocabulary list (same format as extract())
        """
        from src.core.parallel.executor_strategy import ThreadPoolStrategy
        from src.core.parallel.task_runner import ParallelTaskRunner
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
                result = fast_algorithms[0].extract(text)
                all_results.append(result)
                logger.debug("%s: %s candidates", fast_algorithms[0].name, len(result.candidates))
            else:
                # Parallel execution
                strategy = ThreadPoolStrategy(max_workers=workers)
                try:

                    def run_algorithm(algorithm):
                        result = algorithm.extract(text)
                        return (algorithm.name, result)

                    items = [(alg.name, alg) for alg in fast_algorithms]
                    runner = ParallelTaskRunner(strategy=strategy)
                    task_results = runner.run(run_algorithm, items)

                    for task_result in task_results:
                        if task_result.success:
                            alg_name, result = task_result.result
                            all_results.append(result)
                            logger.debug("%s: %s candidates", alg_name, len(result.candidates))
                finally:
                    strategy.shutdown(wait=True)

            # Send partial results if callback provided
            if partial_callback is not None and all_results:
                logger.debug("Sending partial results (BM25+RAKE)...")
                partial_merged = self.merger.merge(all_results)
                partial_vocab = self._post_process(partial_merged, text, doc_count, doc_confidence)
                # Apply lightweight filter chain (Session 85: skip rarity filter for partial results)
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

        vocabulary = self._post_process(merged_terms, text, doc_count, doc_confidence)

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
        return vocabulary

    def extract_with_llm(
        self,
        text: str,
        doc_count: int = 1,
        include_llm: bool = True,
        progress_callback=None,
    ) -> list[dict]:
        """
        Extract vocabulary using NER and optionally LLM, then reconcile results.

        This method provides a unified extraction pipeline that:
        1. Runs NER extraction (existing algorithm)
        2. Optionally runs LLM extraction (single prompt per chunk)
        3. Reconciles results into unified output with "Found By" column

        Args:
            text: The document text to analyze
            doc_count: Number of documents being processed
            include_llm: Whether to run LLM extraction (default True)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of vocabulary dictionaries with schema:
            - Term: The extracted term
            - Type: Person/Place/Medical/Technical/Unknown
            - Found By: "Both", "NER", or "LLM"
            - Frequency: Occurrence count
            - Quality Score: 0-100 composite score
            - Definition: WordNet definition (optional)
        """
        logger.debug("Starting extract_with_llm (include_llm=%s)", include_llm)
        start_time = time.time()

        # 1. Run NER extraction via existing algorithms
        logger.debug("Phase 1: Running NER extraction...")
        ner_candidates = []
        for algorithm in self.algorithms:
            if not algorithm.enabled:
                continue
            if algorithm.name == "NER":
                result = algorithm.extract(text)
                ner_candidates = result.candidates
                logger.debug("NER found %s candidates", len(ner_candidates))
                break

        # 2. Run LLM extraction if enabled
        llm_terms = []
        if include_llm:
            logger.debug("Phase 2: Running LLM extraction...")
            try:
                from src.core.extraction.llm_extractor import LLMVocabExtractor

                llm_extractor = LLMVocabExtractor()
                llm_result = llm_extractor.extract(
                    text,
                    progress_callback=progress_callback,
                )
                llm_terms = llm_result.terms
                logger.debug(
                    "LLM found %s terms in %.1fms",
                    len(llm_terms),
                    llm_result.processing_time_ms,
                )
            except Exception as e:
                logger.debug("LLM extraction failed: %s", e)
                # Continue with NER-only results

        # 3. Reconcile results
        logger.debug("Phase 3: Reconciling NER and LLM results...")
        reconciler = VocabularyDeduplicator()
        reconciled = reconciler.reconcile(ner_candidates, llm_terms)
        logger.debug("Reconciled to %s unique terms", len(reconciled))

        # 3.5 Filter out common words and apply frequency thresholds (Session 45)
        logger.debug("Phase 3.5: Filtering common words...")
        filtered_reconciled = self._filter_reconciled_terms(reconciled, doc_count, len(text))
        logger.debug(
            "After filtering: %s terms (removed %s)",
            len(filtered_reconciled),
            len(reconciled) - len(filtered_reconciled),
        )
        reconciled = filtered_reconciled

        # 4. Add definitions for Medical/Technical terms
        logger.debug("Phase 4: Adding definitions...")
        for term in reconciled:
            if term.type in ("Medical", "Technical") and not term.definition:
                term.definition = self._get_definition(term.term, term.type == "Person")

        # 5. Convert to CSV format
        csv_data = reconciler.to_csv_data(reconciled, include_definitions=True)

        # 6. Add Role/Relevance using role profile
        for i, row in enumerate(csv_data):
            term = reconciled[i].term
            category = reconciled[i].type
            role = self._get_role_relevance(term, category == "Person", text)
            row["Role/Relevance"] = role

        # 7. Run vocabulary filter chain (Session 71)
        # Consolidates: name dedup, artifact filter, name regularizer,
        # rarity filter, corpus familiarity, and gibberish filter
        logger.debug("Phase 7: Running filter chain...")
        from src.core.vocabulary.filters import create_optimized_filter_chain

        filter_chain = create_optimized_filter_chain()
        filter_result = filter_chain.run(csv_data)
        csv_data = filter_result.vocabulary
        logger.debug(
            "Filter chain complete: %s removed, %s remaining",
            filter_result.removed_count,
            len(csv_data),
        )

        total_time = (time.time() - start_time) * 1000
        logger.debug("extract_with_llm complete in %.1fms, %s terms", total_time, len(csv_data))

        return csv_data

    def _run_algorithms_parallel(self, text: str) -> list:
        """
        Run extraction algorithms in parallel when beneficial (Session 69).

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

        # Log text hash for consistency verification (Session 87)
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
                result = algorithm.extract(text)
                all_results.append(result)
                logger.debug(
                    "%s: %s candidates in %.1fms",
                    algorithm.name,
                    len(result.candidates),
                    result.processing_time_ms,
                )
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
    ) -> list[dict[str, str]]:
        """
        Post-process merged terms: categorize, detect roles, add definitions.

        Args:
            merged_terms: List of MergedTerm from merger
            full_text: Complete document text for role detection
            doc_count: Number of documents being processed
            doc_confidence: Average/min confidence of source documents (0-100)

        Returns:
            Final vocabulary list with all metadata
        """
        vocabulary = []
        seen_terms = set()
        # Estimate total pages across all docs (~3500 chars per transcript page)
        estimated_pages = max(len(full_text) // 3500, 1) * doc_count
        frequency_threshold = estimated_pages * 5
        # Total unique terms for ML occurrence_ratio feature
        total_unique_terms = len(merged_terms)

        # Session 80: Track iteration count for periodic GIL yield
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

            # Frequency filtering (Person names exempt - they're always relevant)
            if not is_person and merged.frequency > frequency_threshold:
                continue

            # Minimum occurrence filtering (Person names exempt)
            # Session 62: Read from user preferences with config fallback
            min_occurrences = get_user_preferences().get(
                "vocab_min_occurrences", VOCABULARY_MIN_OCCURRENCES
            )
            if not is_person and merged.frequency < min_occurrences:
                continue

            # Detect role/relevance using profession-specific profile
            role_relevance = self._get_role_relevance(term, is_person, full_text)

            # Calculate quality score
            frequency_rank = self._get_term_frequency_rank(term)
            textrank_score = float(merged.metadata.get("textrank_score", 0.0))
            base_quality_score = self._calculate_quality_score(
                is_person,
                merged.frequency,
                frequency_rank,
                len(merged.sources),
                textrank_score=textrank_score,
            )

            # Build term data for potential ML boost
            # Create per-algorithm detection flags for detailed tracking
            sources_upper = [s.upper() for s in merged.sources]
            algo_count = len(merged.sources)  # Number of algorithms that found this term

            # Build "Found By" display string from sources
            found_by = ", ".join(merged.sources)  # e.g., "NER, RAKE" or "NER, RAKE, BM25"

            # Session 80: Create legacy TermSources for single-doc extraction path
            # This enables consistent display columns across both extraction methods
            sources_obj = TermSources.create_legacy(merged.frequency, doc_confidence / 100.0)

            term_data = {
                "Term": term,
                "Is Person": "Yes" if is_person else "No",  # Session 52: Replaced Type
                "Found By": found_by,  # Session 52: Show which algorithms found this term
                "Role/Relevance": role_relevance,
                "Quality Score": base_quality_score,
                "In-Case Freq": merged.frequency,
                "Freq Rank": frequency_rank,
                "Definition": self._get_definition(term, is_person),
                "Sources": ",".join(merged.sources),  # Keep for backward compatibility
                # Per-algorithm detection flags (Session 47)
                "NER": "Yes" if "NER" in sources_upper else "No",
                "RAKE": "Yes" if "RAKE" in sources_upper else "No",
                "BM25": "Yes" if "BM25" in sources_upper else "No",
                "TextRank": "Yes" if "TEXTRANK" in sources_upper else "No",
                "MedicalNER": "Yes" if "MEDICALNER" in sources_upper else "No",
                "GLiNER": "Yes" if "GLINER" in sources_upper else "No",
                "Algo Count": algo_count,  # Sum of algorithms that found term
                # Session 80: Display columns from TermSources
                "# Docs": sources_obj.num_documents,
                "Count": sources_obj.total_count,
                "Median Conf": f"{sources_obj.median_confidence:.0%}",
                # Session 78: TermSources object for ML/filters
                "sources": sources_obj,
                "total_docs_in_session": doc_count,
                # ML feature fields (from feedback CSV schema)
                "quality_score": base_quality_score,
                "in_case_freq": merged.frequency,
                "freq_rank": frequency_rank,
                "algorithms": ",".join(merged.sources),
                "is_person": 1 if is_person else 0,  # Session 52: Binary flag for ML
                "total_unique_terms": total_unique_terms,  # For ML occurrence_ratio
                "source_doc_confidence": doc_confidence,  # Session 54: OCR quality for ML
                "textrank_score": float(merged.metadata.get("textrank_score", 0.0)),
            }

            # Apply ML boost if meta-learner is trained (Session 25)
            final_quality_score = self._apply_ml_boost(term_data, base_quality_score)
            term_data["Quality Score"] = final_quality_score

            vocabulary.append(term_data)
            seen_terms.add(lower_term)

            # Session 80: Yield GIL every 50 terms to keep GUI responsive
            iteration_count += 1
            if iteration_count % 50 == 0:
                time.sleep(0)

        return vocabulary

    def _get_role_relevance(self, term: str, is_person: bool, full_text: str) -> str:
        """Get role/relevance description for a term."""
        if is_person:
            return self.role_profile.detect_person_role(term, full_text)
        else:
            # For non-person terms, detect based on term characteristics
            return "Vocabulary term"

    def _filter_reconciled_terms(
        self, reconciled: list, doc_count: int, text_length: int = 0
    ) -> list:
        """
        Filter reconciled terms for single-word noise (Session 45).

        This handles SINGLE-WORD filtering only:
        - Exclude lists (legal/user)
        - Rarity threshold check
        - Frequency bounds (too rare = OCR error, too common = noise)
        - Stopwords

        MULTI-WORD phrase filtering is done CENTRALLY by rarity_filter.py
        in the main pipeline (see extract_with_llm phase 9). This separation
        ensures consistent filtering across all algorithms.

        Args:
            reconciled: List of ReconciledTerm objects
            doc_count: Number of documents being processed

        Returns:
            Filtered list of ReconciledTerm objects
        """
        from src.config import VOCABULARY_MIN_OCCURRENCES

        filtered = []
        seen_terms = set()
        # Estimate total pages across all docs (~3500 chars per transcript page)
        estimated_pages = max(text_length // 3500, 1) * doc_count
        frequency_threshold = estimated_pages * 5

        for term_obj in reconciled:
            term = term_obj.term
            lower_term = term.lower()
            category = term_obj.type

            # Skip duplicates
            if lower_term in seen_terms:
                continue

            # Skip if in exclude lists (common legal words)
            if lower_term in self.exclude_list:
                logger.debug("Skipping excluded term: %s", term)
                continue

            # Skip if in user exclude list
            if lower_term in self.user_exclude_list:
                logger.debug("Skipping user-excluded term: %s", term)
                continue

            # Skip common words based on frequency rank (except Person names)
            if category != "Person" and self.frequency_rank_map:
                rank = self.frequency_rank_map.get(lower_term)
                if rank is not None and rank < self.rarity_threshold:
                    # Word is too common - in top N most common words
                    logger.debug("Skipping common word: %s (rank=%s)", term, rank)
                    continue

            # Session 62: Apply rarity filter to catch common words not in frequency dataset
            # This uses the scaled frequency database which has better coverage for LLM terms
            from src.core.vocabulary.rarity_filter import should_filter_phrase

            is_person = category == "Person"
            if should_filter_phrase(term, is_person):
                logger.debug("Skipping common term via rarity filter: %s", term)
                continue

            # Frequency filtering (PERSON exempt) - skip if too frequent
            if category != "Person" and term_obj.frequency > frequency_threshold:
                logger.debug(
                    "Skipping high-frequency term: %s (freq=%s)",
                    term,
                    term_obj.frequency,
                )
                continue

            # Minimum occurrence filtering (PERSON exempt) - skip if too rare
            # Session 62: Read from user preferences with config fallback
            min_occurrences = get_user_preferences().get(
                "vocab_min_occurrences", VOCABULARY_MIN_OCCURRENCES
            )
            if category != "Person" and term_obj.frequency < min_occurrences:
                logger.debug(
                    "Skipping low-frequency term: %s (freq=%s)",
                    term,
                    term_obj.frequency,
                )
                continue

            # Skip single-character terms
            if len(term) < 2:
                continue

            # Skip single-word terms that are stopwords (uses shared STOPWORDS)
            # Note: Multi-word phrase filtering is done centrally by rarity_filter.py
            if len(term.split()) == 1 and lower_term in STOPWORDS and category != "Person":
                logger.debug("Skipping stopword: %s", term)
                continue

            filtered.append(term_obj)
            seen_terms.add(lower_term)

        return filtered

    def _calculate_quality_score(
        self,
        is_person: bool,
        term_count: int,
        frequency_rank: int,
        algorithm_count: int,
        term_sources: TermSources | None = None,
        total_docs_in_session: int = 1,
        textrank_score: float = 0.0,
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
            textrank_score: TextRank centrality score (0-1), 0 if not available

        Returns:
            Quality score between 0.0 and 100.0
        """
        from src.config import (
            SCORE_ALL_LOW_CONF_PENALTY,
            SCORE_HIGH_CONF_BOOST,
            SCORE_MULTI_DOC_BOOST,
            SCORE_SINGLE_SOURCE_CONF_THRESHOLD,
            SCORE_SINGLE_SOURCE_MIN_DOCS,
            SCORE_SINGLE_SOURCE_PENALTY,
            SCORE_TEXTRANK_CENTRALITY_BOOST,
        )

        score = 50.0  # Base score

        # Boost for multiple occurrences (max +30)
        # Session 131: Log-scaled to better reward high-frequency names (301 occ vs 4 occ)
        # Old: min(term_count * 5, 20) capped at 4 occurrences
        # New: log-scaled to distinguish 10, 100, 300+ occurrences
        # Examples: 1→8, 4→17, 10→25, 100→30, 300→30

        occurrence_boost = min(math.log10(term_count + 1) * 25, 30)
        score += occurrence_boost

        # Boost for rare words (max +20)
        if frequency_rank == 0:
            score += 20  # Not in Google dataset - very rare
        elif frequency_rank > 200000:
            score += 15
        elif frequency_rank > 180000:
            score += 10

        # Boost for person names (NER is reliable for these) (+10)
        if is_person:
            score += 10

        # Boost for multi-algorithm agreement (max +10)
        # Terms found by multiple algorithms are more trustworthy
        if algorithm_count >= 2:
            score += min(algorithm_count * 3, 10)

        # Boost for TextRank centrality (scaled by score, capped at config max)
        # High centrality = term is well-connected in document's word graph
        if textrank_score > 0:
            score += min(textrank_score * 10, SCORE_TEXTRANK_CENTRALITY_BOOST)

        # === TermSources-based adjustments (Session 79) ===
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

        return min(100.0, max(0.0, round(score, 1)))

    def _apply_ml_boost(self, term_data: dict, base_score: float) -> float:
        """
        Apply graduated ML-based scoring if meta-learner is trained.

        Session 55: Uses graduated weight based on user sample count.
        Formula: score = base_score * (1 - ml_weight) + ml_prob * 100 * ml_weight

        The ml_weight increases with training corpus size:
        - < 30 samples: 0% (rules only)
        - 30-50 samples: 45%
        - 51-99 samples: 60%
        - 100-199 samples: 70%
        - 200+ samples: 85%

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
                term = term_data.get("Term", "?")
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

    def _get_definition(self, term: str, is_person: bool) -> str:
        """Get definition for non-person terms only."""
        if is_person:
            return "—"  # Person names don't need definitions

        lower_term = term.lower()
        synsets = wordnet.synsets(lower_term)

        if synsets:
            definition = synsets[0].definition()
            if len(definition) > 100:
                logger.warning(
                    "WordNet definition truncated from %d to 100 chars for '%s'.",
                    len(definition),
                    term,
                )
                definition = definition[:97] + "..."
            return definition

        return "—"

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
        return sorted(vocabulary, key=lambda x: float(x.get("Quality Score") or 0), reverse=True)

    def _sort_by_rarity(self, vocabulary: list[dict]) -> list[dict]:
        """Sort vocabulary list by rarity (rarest first)."""
        not_in_dataset = []
        in_dataset = []

        for item in vocabulary:
            term = item["Term"].lower()
            if term not in self.frequency_dataset:
                not_in_dataset.append(item)
            else:
                in_dataset.append(item)

        in_dataset.sort(key=lambda x: self.frequency_dataset.get(x["Term"].lower(), float("inf")))
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
        frequency_dict = {}
        rank_map = {}

        if not GOOGLE_WORD_FREQUENCY_FILE.exists():
            logger.debug("Frequency dataset not found: %s", GOOGLE_WORD_FREQUENCY_FILE)
            return frequency_dict, rank_map

        try:
            with open(GOOGLE_WORD_FREQUENCY_FILE, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        word, count_str = parts
                        try:
                            count = int(count_str)
                            frequency_dict[word.lower()] = count
                        except ValueError:
                            continue

            logger.debug("Loaded %s words from frequency dataset", len(frequency_dict))

            # Build rank map
            sorted_words = sorted(frequency_dict.items(), key=lambda x: x[1], reverse=True)
            rank_map = {word: rank for rank, (word, _) in enumerate(sorted_words)}
            logger.debug("Built rank map for %s words", len(rank_map))

        except Exception as e:
            logger.debug("Error loading frequency dataset: %s", e)

        return frequency_dict, rank_map

    def _ensure_nltk_data(self):
        """Ensure NLTK data is available."""
        try:
            wordnet.synsets("test")
        except LookupError:
            logger.debug("Downloading NLTK wordnet...")
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)

    def add_user_exclusion(self, term: str) -> bool:
        """Add a term to the user's exclusion list."""
        if not self.user_exclude_path:
            logger.debug("Cannot add exclusion: no user exclude path configured")
            return False

        lower_term = term.lower().strip()
        if not lower_term:
            return False

        self.user_exclude_list.add(lower_term)

        try:
            dirname = os.path.dirname(self.user_exclude_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(self.user_exclude_path, "a", encoding="utf-8") as f:
                f.write(f"{lower_term}\n")
            logger.debug("Added '%s' to user exclusion list", term)
            return True
        except Exception as e:
            logger.debug("Failed to save user exclusion: %s", e)
            return False

    def reload_user_exclusions(self):
        """Reload user exclusions from file."""
        if self.user_exclude_path:
            self.user_exclude_list = self._load_word_list(self.user_exclude_path)
            # Also update NER algorithm's exclusion list
            for alg in self.algorithms:
                if hasattr(alg, "user_exclude_list"):
                    alg.user_exclude_list = self.user_exclude_list

    def _should_enable_gliner(self) -> bool:
        """
        Check if GLiNER algorithm should be enabled.

        GLiNER requires:
        1. User has enabled GLiNER in settings (default: False)
        2. gliner package is installed

        Returns:
            True if GLiNER should be added to algorithm list
        """
        try:
            prefs = get_user_preferences()
            if not prefs.get("gliner_enabled", False):
                logger.debug("GLiNER disabled by user preference")
                return False
            return True
        except Exception as e:
            logger.debug("GLiNER check failed: %s", e)
            return False

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
    # PER-DOCUMENT PARALLEL EXTRACTION (Session 131)
    # ========================================================================

    def extract_documents(self, documents, use_llm=False, progress_callback=None):
        """
        Run full pipeline per document (in parallel), then merge results.

        Args:
            documents: List of {"text", "doc_id", "confidence"}
            use_llm: Whether to use LLM enhancement
            progress_callback: Optional callback(current, total, doc_id)

        Returns:
            Merged vocabulary list[dict] with proper # Docs tracking.
        """
        if not documents:
            return []

        # Single doc — no merge needed, just run normally
        if len(documents) == 1:
            doc = documents[0]
            if use_llm:
                return self.extract_with_llm(doc["text"], doc_count=1)
            return self.extract(doc["text"], doc_count=1, doc_confidence=doc["confidence"])

        # Multi-doc — parallel extraction
        from concurrent.futures import as_completed

        from src.core.parallel.executor_strategy import ThreadPoolStrategy
        from src.system_resources import get_optimal_workers

        max_workers = min(get_optimal_workers(), len(documents))
        strategy = ThreadPoolStrategy(max_workers=max_workers)

        def _extract_single(doc):
            """Extract vocab from one document (runs in worker thread)."""
            if use_llm:
                vocab = self.extract_with_llm(doc["text"], doc_count=1)
            else:
                vocab = self.extract(doc["text"], doc_count=1, doc_confidence=doc["confidence"])
            return (doc["doc_id"], doc["confidence"], vocab)

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

        # Merge all doc results
        return self._merge_multi_doc_results(per_doc_results, len(documents))

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
                key = term_dict["Term"].lower()
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
            base = term_data["quality_score"]
            term_data["Quality Score"] = self._apply_ml_boost(term_data, base)

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
            counts_per_doc.append(td.get("In-Case Freq", td.get("Frequency", 1)))

        sources = TermSources(
            doc_ids=doc_ids,
            confidences=confidences,
            counts_per_doc=counts_per_doc,
        )

        # Term display: most frequent casing
        casing_votes = Counter(td["Term"] for _, _, td in doc_entries)
        best_term = casing_votes.most_common(1)[0][0]

        # Boolean fields: "Yes" if ANY doc says "Yes"
        is_person = any(td.get("Is Person") == "Yes" for _, _, td in doc_entries)
        ner = any(td.get("NER") == "Yes" for _, _, td in doc_entries)
        rake = any(td.get("RAKE") == "Yes" for _, _, td in doc_entries)
        bm25 = any(td.get("BM25") == "Yes" for _, _, td in doc_entries)
        textrank = any(td.get("TextRank") == "Yes" for _, _, td in doc_entries)

        # Set fields: union across docs
        all_found_by = set()
        for _, _, td in doc_entries:
            found = td.get("Found By", td.get("Sources", ""))
            for algo in found.replace(",", " ").split():
                algo = algo.strip()
                if algo and algo != "—":
                    all_found_by.add(algo)
        found_by_str = ", ".join(sorted(all_found_by))

        # Numeric: sum / max / first
        total_freq = sum(td.get("In-Case Freq", td.get("Frequency", 0)) for _, _, td in doc_entries)
        freq_rank = doc_entries[0][2].get("Freq Rank", 0)
        textrank_score = max(
            (td.get("textrank_score", 0.0) for _, _, td in doc_entries),
            default=0.0,
        )
        min_doc_confidence = min(c for _, c, _ in doc_entries)

        # String: longest non-default
        default_role = "Vocabulary term"
        roles = [td.get("Role/Relevance", default_role) for _, _, td in doc_entries]
        best_role = max(
            (r for r in roles if r != default_role),
            key=len,
            default=default_role,
        )

        definition = ""
        for _, _, td in doc_entries:
            d = td.get("Definition", "")
            if d and d != "—":
                definition = d
                break

        # Algo count: recompute from merged flags
        algo_count = sum([ner, rake, bm25, textrank])

        # Quality score: recalculate with real TermSources
        quality_score = self._calculate_quality_score(
            is_person=is_person,
            term_count=total_freq,
            frequency_rank=freq_rank,
            algorithm_count=algo_count,
            term_sources=sources,
            total_docs_in_session=total_docs,
            textrank_score=textrank_score,
        )

        return {
            # Display columns
            "Term": best_term,
            "Is Person": "Yes" if is_person else "No",
            "Found By": found_by_str,
            "Role/Relevance": best_role,
            "Quality Score": quality_score,
            "In-Case Freq": total_freq,
            "Freq Rank": freq_rank,
            "Definition": definition or "—",
            "Sources": found_by_str,
            # Algorithm flags
            "NER": "Yes" if ner else "No",
            "RAKE": "Yes" if rake else "No",
            "BM25": "Yes" if bm25 else "No",
            "TextRank": "Yes" if textrank else "No",
            "Algo Count": algo_count,
            # TermSources display
            "# Docs": sources.num_documents,
            "Count": sources.total_count,
            "Median Conf": f"{sources.median_confidence:.0%}",
            # TermSources object (for ML/filters)
            "sources": sources,
            "total_docs_in_session": total_docs,
            # ML feature fields
            "quality_score": quality_score,
            "in_case_freq": total_freq,
            "freq_rank": freq_rank,
            "algorithms": found_by_str,
            "is_person": 1 if is_person else 0,
            "total_unique_terms": total_unique,
            "source_doc_confidence": min_doc_confidence,
            "textrank_score": textrank_score,
        }

    # ========================================================================
    # PER-DOCUMENT EXTRACTION (Session 78) — Legacy
    # ========================================================================

    def extract_from_document(
        self, text: str, doc_id: str, doc_confidence: float
    ) -> dict[str, int]:
        """
        Extract vocabulary from a single document.

        Session 78: Per-document extraction for TermSources tracking.
        This method extracts terms from ONE document only. Call this
        for each document, then use merge_document_results() to combine.

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

        # Build term → count mapping
        term_counts: dict[str, int] = {}
        for merged in merged_terms:
            lower_term = merged.term.lower()
            term_counts[lower_term] = merged.frequency

        logger.debug("Document %s: %s unique terms", doc_id[:12], len(term_counts))
        return term_counts

    def merge_document_results(
        self, doc_results: list[tuple[str, float, dict[str, int]]], full_text: str | None = None
    ) -> list[dict]:
        """
        Merge per-document extractions into final vocabulary with TermSources.

        Session 78: Combines results from multiple documents while tracking
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

            # Use original term (need to find best-cased version)
            # For now, use title case for multi-word, capitalize for single
            words = term_lower.split()
            if len(words) > 1:
                display_term = " ".join(w.capitalize() for w in words)
            else:
                display_term = term_lower.capitalize()

            # Determine if this is a person (will be refined in _post_process)
            is_person = False  # Default, will be updated below

            # Calculate base quality score (Session 79: pass TermSources for quality adjustments)
            frequency_rank = self._get_term_frequency_rank(term_lower)
            base_quality_score = self._calculate_quality_score(
                is_person,
                data["total_count"],
                frequency_rank,
                1,
                term_sources=sources,
                total_docs_in_session=total_docs,
            )

            term_dict = {
                "Term": display_term,
                "Is Person": "No",  # Will be updated by filter chain
                "Found By": "—",
                "Role/Relevance": "Vocabulary term",
                "Quality Score": base_quality_score,
                "In-Case Freq": data["total_count"],
                "Freq Rank": frequency_rank,
                "Definition": "—",
                "Sources": "",
                "NER": "No",
                "RAKE": "No",
                "BM25": "No",
                "Algo Count": 0,
                # Session 80: Display columns from TermSources
                "# Docs": sources.num_documents,
                "Count": sources.total_count,
                "Median Conf": f"{sources.median_confidence:.0%}",
                # Session 78: TermSources tracking
                "sources": sources,
                "total_docs_in_session": total_docs,
                # ML feature fields
                "quality_score": base_quality_score,
                "in_case_freq": data["total_count"],
                "freq_rank": frequency_rank,
                "is_person": 0,
                "total_unique_terms": len(term_data),
                "source_doc_confidence": sources.mean_confidence * 100,
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

        Session 78: Convenience method that handles the full per-document
        extraction workflow. Extracts from each document individually,
        then merges with TermSources for canonical selection.

        Session 86: Now applies preprocessing pipeline to each document
        before extraction (fixes transcript artifact bug where Q./A. notation
        and headers were appearing in vocabulary results).

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

        # Session 86: Create preprocessing pipeline to clean transcript artifacts
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

            # Session 86: Preprocess text before extraction
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
