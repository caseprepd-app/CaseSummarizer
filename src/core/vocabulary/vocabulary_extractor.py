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
2. Merging and deduplicating results via ResultMerger
3. Post-processing: frequency filtering, ML boost, role detection
4. Name deduplication (fuzzy matching for OCR variants)
5. Artifact filtering (substring containment)
6. Phrase rarity filtering (filter phrases with all-common words)
7. Quality scoring and final output formatting

The extraction algorithms are pluggable via dependency injection.
"""

import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import nltk
import spacy
from nltk.corpus import wordnet

from src.config import (
    GOOGLE_WORD_FREQUENCY_FILE,
    SPACY_DOWNLOAD_TIMEOUT_SEC,
    VOCABULARY_MAX_TEXT_KB,
    VOCABULARY_MIN_OCCURRENCES,
    VOCABULARY_RARITY_THRESHOLD,
    VOCABULARY_SORT_METHOD,
)
from src.logging_config import debug_log
from src.user_preferences import get_user_preferences
from src.core.utils.tokenizer import STOPWORDS
from src.core.vocabulary.algorithms import create_default_algorithms
from src.core.vocabulary.algorithms.base import BaseExtractionAlgorithm
from src.core.vocabulary.meta_learner import VocabularyMetaLearner, get_meta_learner
from src.core.vocabulary.result_merger import MergedTerm, ResultMerger
from src.core.vocabulary.role_profiles import RoleDetectionProfile, StenographerProfile
from src.core.vocabulary.reconciler import VocabularyReconciler

# Organization indicator words for category detection
ORGANIZATION_INDICATORS = {
    'LLP', 'PLLC', 'P.C.', 'LLC', 'Inc', 'Corp', 'Corporation',
    'Law Firm', 'Law Office', 'Firm',
    'Hospital', 'Medical', 'Healthcare', 'Health', 'Clinic',
    'University', 'College', 'School',
    'Bank', 'Insurance', 'Services',
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
        merger: ResultMerger for combining algorithm outputs

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

        # Load common medical/legal words blacklist
        common_blacklist_path = Path(__file__).parent.parent.parent.parent / "config" / "common_medical_legal.txt"
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
                    debug_log(f"[VOCAB] BM25 algorithm enabled (corpus: {corpus_manager.get_document_count()} docs)")
                except Exception as e:
                    debug_log(f"[VOCAB] Failed to initialize BM25: {e}")
        else:
            self.algorithms = algorithms

        # Initialize merger with algorithm weights
        self.merger = ResultMerger(
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
                if hasattr(alg, 'nlp') and alg.nlp is not None:
                    self._nlp = alg.nlp
                    break
            # Fallback: load minimal model for categorization
            if self._nlp is None:
                try:
                    self._nlp = spacy.load("en_core_web_lg")
                except OSError:
                    self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def extract(self, text: str, doc_count: int = 1, doc_confidence: float = 100.0) -> list[dict[str, str]]:
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
        debug_log(f"[VOCAB] Starting multi-algorithm extraction on {original_kb}KB document")

        # Limit text size
        max_chars = VOCABULARY_MAX_TEXT_KB * 1024
        if len(text) > max_chars:
            text = text[:max_chars]
            debug_log(f"[VOCAB] Truncated to {VOCABULARY_MAX_TEXT_KB}KB for processing")

        # 1. Run all enabled algorithms
        all_results = []
        for algorithm in self.algorithms:
            if not algorithm.enabled:
                debug_log(f"[VOCAB] Skipping disabled algorithm: {algorithm.name}")
                continue

            debug_log(f"[VOCAB] Running {algorithm.name} algorithm...")
            start_time = time.time()

            result = algorithm.extract(text)
            all_results.append(result)

            debug_log(
                f"[VOCAB] {algorithm.name}: {len(result.candidates)} candidates "
                f"in {result.processing_time_ms:.1f}ms"
            )

        # 2. Merge results from all algorithms
        debug_log(f"[VOCAB] Merging results from {len(all_results)} algorithms...")
        merged_terms = self.merger.merge(all_results)
        debug_log(f"[VOCAB] After merge: {len(merged_terms)} unique terms")

        # 3. Post-process: categorize, detect roles, add definitions
        debug_log(f"[VOCAB] Post-processing (doc_count={doc_count}, doc_confidence={doc_confidence:.1f}%)...")
        vocabulary = self._post_process(merged_terms, text, doc_count, doc_confidence)
        debug_log(f"[VOCAB] After post-process: {len(vocabulary)} terms")

        # 4. Deduplicate similar Person names (OCR errors, typos)
        debug_log("[VOCAB] Deduplicating similar Person names...")
        from src.core.vocabulary.name_deduplicator import deduplicate_names
        vocabulary = deduplicate_names(vocabulary)
        debug_log(f"[VOCAB] After deduplication: {len(vocabulary)} terms")

        # 5. Filter substring artifacts (e.g., "Ms. Di Leo:" when "Ms. Di Leo" exists)
        debug_log("[VOCAB] Filtering substring artifacts...")
        from src.core.vocabulary.artifact_filter import filter_substring_artifacts
        vocabulary = filter_substring_artifacts(vocabulary)
        debug_log(f"[VOCAB] After artifact filter: {len(vocabulary)} terms")

        # 5b. Regularize names: remove fragments and typos (Session 63)
        # - Fragment filter: "Di Leo" in top quartile → remove "Di", "Leo" from bottom
        # - Typo filter: "Barbra Jenkins" in top → remove "Barbr Jenkins" (1-char diff)
        debug_log("[VOCAB] Regularizing names (fragments + typos)...")
        from src.core.vocabulary.name_regularizer import regularize_names
        vocabulary = regularize_names(vocabulary)
        debug_log(f"[VOCAB] After name regularization: {len(vocabulary)} terms")

        # 6. Filter phrases with overly common component words
        # (e.g., "the same", "left side" - high RAKE scores but no vocab value)
        debug_log("[VOCAB] Filtering common phrase components...")
        from src.core.vocabulary.rarity_filter import filter_common_phrases
        vocabulary = filter_common_phrases(vocabulary)
        debug_log(f"[VOCAB] After phrase filter: {len(vocabulary)} terms")

        # 7. Filter gibberish/nonsense terms (OCR artifacts, random strings)
        # NOTE: Person names are EXEMPT - foreign names may look unusual to English model
        debug_log("[VOCAB] Filtering gibberish terms...")
        from src.core.utils.gibberish_filter import is_gibberish
        gibberish_filtered = []
        for term_data in vocabulary:
            # Person names bypass gibberish check (foreign names look unusual)
            if term_data.get("Is Person") == "Yes":
                gibberish_filtered.append(term_data)
                continue
            # Check if term text is gibberish
            if is_gibberish(term_data["Term"]):
                debug_log(f"[VOCAB] Filtered gibberish: '{term_data['Term']}'")
                continue
            gibberish_filtered.append(term_data)
        vocabulary = gibberish_filtered
        debug_log(f"[VOCAB] Final vocabulary: {len(vocabulary)} terms")

        # 8. Sort vocabulary based on configured method
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
        debug_log(f"[VOCAB] Starting extract_with_llm (include_llm={include_llm})")
        start_time = time.time()

        # 1. Run NER extraction via existing algorithms
        debug_log("[VOCAB] Phase 1: Running NER extraction...")
        ner_candidates = []
        for algorithm in self.algorithms:
            if not algorithm.enabled:
                continue
            if algorithm.name == "NER":
                result = algorithm.extract(text)
                ner_candidates = result.candidates
                debug_log(f"[VOCAB] NER found {len(ner_candidates)} candidates")
                break

        # 2. Run LLM extraction if enabled
        llm_terms = []
        if include_llm:
            debug_log("[VOCAB] Phase 2: Running LLM extraction...")
            try:
                from src.core.extraction.llm_extractor import LLMVocabExtractor

                llm_extractor = LLMVocabExtractor()
                llm_result = llm_extractor.extract(
                    text,
                    progress_callback=progress_callback,
                )
                llm_terms = llm_result.terms
                debug_log(
                    f"[VOCAB] LLM found {len(llm_terms)} terms "
                    f"in {llm_result.processing_time_ms:.1f}ms"
                )
            except Exception as e:
                debug_log(f"[VOCAB] LLM extraction failed: {e}")
                # Continue with NER-only results

        # 3. Reconcile results
        debug_log("[VOCAB] Phase 3: Reconciling NER and LLM results...")
        reconciler = VocabularyReconciler()
        reconciled = reconciler.reconcile(ner_candidates, llm_terms)
        debug_log(f"[VOCAB] Reconciled to {len(reconciled)} unique terms")

        # 3.5 Filter out common words and apply frequency thresholds (Session 45)
        debug_log("[VOCAB] Phase 3.5: Filtering common words...")
        filtered_reconciled = self._filter_reconciled_terms(reconciled, doc_count)
        debug_log(f"[VOCAB] After filtering: {len(filtered_reconciled)} terms (removed {len(reconciled) - len(filtered_reconciled)})")
        reconciled = filtered_reconciled

        # 4. Add definitions for Medical/Technical terms
        debug_log("[VOCAB] Phase 4: Adding definitions...")
        for term in reconciled:
            if term.type in ("Medical", "Technical") and not term.definition:
                term.definition = self._get_definition(term.term, term.type)

        # 5. Convert to CSV format
        csv_data = reconciler.to_csv_data(reconciled, include_definitions=True)

        # 6. Add Role/Relevance using role profile
        for i, row in enumerate(csv_data):
            term = reconciled[i].term
            category = reconciled[i].type
            role = self._get_role_relevance(term, category, text)
            row["Role/Relevance"] = role

        # 7. Deduplicate similar Person names (OCR errors, typos)
        debug_log("[VOCAB] Phase 7: Deduplicating similar Person names...")
        from src.core.vocabulary.name_deduplicator import deduplicate_names
        csv_data = deduplicate_names(csv_data)
        debug_log(f"[VOCAB] After deduplication: {len(csv_data)} terms")

        # 8. Filter substring artifacts (e.g., "Ms. Di Leo:" when "Ms. Di Leo" exists)
        debug_log("[VOCAB] Phase 8: Filtering substring artifacts...")
        from src.core.vocabulary.artifact_filter import filter_substring_artifacts
        csv_data = filter_substring_artifacts(csv_data)

        # 8b. Regularize names: remove fragments and typos (Session 63)
        debug_log("[VOCAB] Phase 8b: Regularizing names (fragments + typos)...")
        from src.core.vocabulary.name_regularizer import regularize_names
        csv_data = regularize_names(csv_data)
        debug_log(f"[VOCAB] After name regularization: {len(csv_data)} terms")

        # 9. Filter phrases with overly common component words
        # (e.g., "the same", "left side" - high RAKE scores but no vocab value)
        debug_log("[VOCAB] Phase 9: Filtering common phrase components...")
        from src.core.vocabulary.rarity_filter import filter_common_phrases
        csv_data = filter_common_phrases(csv_data)
        debug_log(f"[VOCAB] After phrase filter: {len(csv_data)} terms")

        # 10. Filter gibberish/nonsense terms (OCR artifacts, random strings)
        # NOTE: Person names are EXEMPT - foreign names may look unusual to English model
        debug_log("[VOCAB] Phase 10: Filtering gibberish terms...")
        from src.core.utils.gibberish_filter import is_gibberish
        gibberish_filtered = []
        for row in csv_data:
            # Person names bypass gibberish check (foreign names look unusual)
            term_type = row.get("Type", "")
            if term_type == "Person":
                gibberish_filtered.append(row)
                continue
            # Check if term text is gibberish
            if is_gibberish(row["Term"]):
                debug_log(f"[VOCAB] Filtered gibberish: '{row['Term']}'")
                continue
            gibberish_filtered.append(row)
        csv_data = gibberish_filtered
        debug_log(f"[VOCAB] After gibberish filter: {len(csv_data)} terms")

        total_time = (time.time() - start_time) * 1000
        debug_log(f"[VOCAB] extract_with_llm complete in {total_time:.1f}ms, {len(csv_data)} terms")

        return csv_data

    def _post_process(
        self,
        merged_terms: list[MergedTerm],
        full_text: str,
        doc_count: int,
        doc_confidence: float = 100.0
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
        frequency_threshold = doc_count * 4
        # Total unique terms for ML occurrence_ratio feature
        total_unique_terms = len(merged_terms)

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
            min_occurrences = get_user_preferences().get("vocab_min_occurrences", VOCABULARY_MIN_OCCURRENCES)
            if not is_person and merged.frequency < min_occurrences:
                continue

            # Detect role/relevance using profession-specific profile
            role_relevance = self._get_role_relevance(term, is_person, full_text)

            # Calculate quality score
            frequency_rank = self._get_term_frequency_rank(term)
            base_quality_score = self._calculate_quality_score(
                is_person, merged.frequency, frequency_rank, len(merged.sources)
            )

            # Build term data for potential ML boost
            # Create per-algorithm detection flags for detailed tracking
            sources_upper = [s.upper() for s in merged.sources]
            algo_count = len(merged.sources)  # Number of algorithms that found this term

            # Build "Found By" display string from sources
            found_by = ", ".join(merged.sources)  # e.g., "NER, RAKE" or "NER, RAKE, BM25"

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
                "Algo Count": algo_count,  # Sum of algorithms that found term
                # ML feature fields (from feedback CSV schema)
                "quality_score": base_quality_score,
                "in_case_freq": merged.frequency,
                "freq_rank": frequency_rank,
                "algorithms": ",".join(merged.sources),
                "is_person": 1 if is_person else 0,  # Session 52: Binary flag for ML
                "total_unique_terms": total_unique_terms,  # For ML occurrence_ratio
                "source_doc_confidence": doc_confidence,  # Session 54: OCR quality for ML
            }

            # Apply ML boost if meta-learner is trained (Session 25)
            final_quality_score = self._apply_ml_boost(term_data, base_quality_score)
            term_data["Quality Score"] = final_quality_score

            vocabulary.append(term_data)
            seen_terms.add(lower_term)

        return vocabulary

    def _validate_category(self, term: str, suggested_type: str) -> str | None:
        """
        Validate and potentially correct the suggested category.

        Args:
            term: The term text
            suggested_type: Type suggested by merger

        Returns:
            Validated category string, or None if term should be skipped
        """
        lower_term = term.lower()

        # Medical terms take precedence
        if lower_term in self.medical_terms:
            return "Medical"

        # Trust the merged type for most cases
        if suggested_type in ["Person", "Place", "Medical", "Technical"]:
            return suggested_type

        # Unknown needs validation
        if suggested_type == "Unknown":
            # Try to validate with heuristics
            if self._looks_like_person_name(term):
                return "Person"
            if self._looks_like_organization(term):
                return "Place"
            return "Unknown"

        return suggested_type or "Technical"

    def _get_role_relevance(self, term: str, is_person: bool, full_text: str) -> str:
        """Get role/relevance description for a term."""
        if is_person:
            return self.role_profile.detect_person_role(term, full_text)
        else:
            # For non-person terms, detect based on term characteristics
            return "Vocabulary term"

    def _filter_reconciled_terms(self, reconciled: list, doc_count: int) -> list:
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
        frequency_threshold = doc_count * 4  # Same as _post_process

        for term_obj in reconciled:
            term = term_obj.term
            lower_term = term.lower()
            category = term_obj.type

            # Skip duplicates
            if lower_term in seen_terms:
                continue

            # Skip if in exclude lists (common legal words)
            if lower_term in self.exclude_list:
                debug_log(f"[VOCAB FILTER] Skipping excluded term: {term}")
                continue

            # Skip if in user exclude list
            if lower_term in self.user_exclude_list:
                debug_log(f"[VOCAB FILTER] Skipping user-excluded term: {term}")
                continue

            # Skip common words based on frequency rank (except Person names)
            if category != "Person" and self.frequency_rank_map:
                rank = self.frequency_rank_map.get(lower_term)
                if rank is not None and rank < self.rarity_threshold:
                    # Word is too common - in top N most common words
                    debug_log(f"[VOCAB FILTER] Skipping common word: {term} (rank={rank})")
                    continue

            # Session 62: Apply rarity filter to catch common words not in frequency dataset
            # This uses the scaled frequency database which has better coverage for LLM terms
            from src.core.vocabulary.rarity_filter import should_filter_phrase
            is_person = category == "Person"
            if should_filter_phrase(term, is_person):
                debug_log(f"[VOCAB FILTER] Skipping common term via rarity filter: {term}")
                continue

            # Frequency filtering (PERSON exempt) - skip if too frequent
            if category != "Person" and term_obj.frequency > frequency_threshold:
                debug_log(f"[VOCAB FILTER] Skipping high-frequency term: {term} (freq={term_obj.frequency})")
                continue

            # Minimum occurrence filtering (PERSON exempt) - skip if too rare
            # Session 62: Read from user preferences with config fallback
            min_occurrences = get_user_preferences().get("vocab_min_occurrences", VOCABULARY_MIN_OCCURRENCES)
            if category != "Person" and term_obj.frequency < min_occurrences:
                debug_log(f"[VOCAB FILTER] Skipping low-frequency term: {term} (freq={term_obj.frequency})")
                continue

            # Skip single-character terms
            if len(term) < 2:
                continue

            # Skip single-word terms that are stopwords (uses shared STOPWORDS)
            # Note: Multi-word phrase filtering is done centrally by rarity_filter.py
            if len(term.split()) == 1 and lower_term in STOPWORDS and category != "Person":
                debug_log(f"[VOCAB FILTER] Skipping stopword: {term}")
                continue

            filtered.append(term_obj)
            seen_terms.add(lower_term)

        return filtered

    def _calculate_quality_score(
        self, is_person: bool, term_count: int, frequency_rank: int, algorithm_count: int
    ) -> float:
        """
        Calculate composite quality score (0-100).

        Higher score = more likely to be a useful, high-quality term.

        Args:
            is_person: Whether NER detected this as a person name
            term_count: Number of occurrences
            frequency_rank: Google frequency rank
            algorithm_count: Number of algorithms that found this term

        Returns:
            Quality score between 0.0 and 100.0
        """
        score = 50.0  # Base score

        # Boost for multiple occurrences (max +20)
        occurrence_boost = min(term_count * 5, 20)
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
                debug_log(
                    f"[ML] '{term}': prob={preference_prob:.2f}, weight={ml_weight:.0%}, "
                    f"base={base_score:.1f} -> final={final_score:.1f} ({score_diff:+.1f})"
                )

            return final_score

        except Exception as e:
            debug_log(f"[ML] Error applying boost: {e}")
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
            debug_log("[VOCAB] Sorting by Quality Score (highest first)")
            return self._sort_by_quality_score(vocabulary)
        elif self.sort_method == "rarity" and self.frequency_dataset:
            debug_log("[VOCAB] Sorting by rarity (rarest first)")
            return self._sort_by_rarity(vocabulary)
        else:
            debug_log("[VOCAB] No sorting applied")
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
        return sorted(
            vocabulary,
            key=lambda x: float(x.get("Quality Score", 0) or 0),
            reverse=True
        )

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

        in_dataset.sort(key=lambda x: self.frequency_dataset.get(x["Term"].lower(), float('inf')))
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
        for indicator in ORGANIZATION_INDICATORS:
            if indicator.lower() in term.lower():
                return True
        return False

    def _load_word_list(self, file_path) -> set[str]:
        """Load a list of words from a file."""
        if file_path is None:
            return set()

        file_path = Path(file_path) if isinstance(file_path, str) else file_path

        if not file_path.exists():
            debug_log(f"[VOCAB] Word list not found: {file_path}")
            return set()

        with open(file_path, encoding='utf-8') as f:
            word_list = {line.strip().lower() for line in f if line.strip()}
            debug_log(f"[VOCAB] Loaded {len(word_list)} words from {file_path}")
            return word_list

    def _load_frequency_dataset(self) -> tuple[dict[str, int], dict[str, int]]:
        """Load Google word frequency dataset and build rank mapping."""
        frequency_dict = {}
        rank_map = {}

        if not GOOGLE_WORD_FREQUENCY_FILE.exists():
            debug_log(f"[VOCAB] Frequency dataset not found: {GOOGLE_WORD_FREQUENCY_FILE}")
            return frequency_dict, rank_map

        try:
            with open(GOOGLE_WORD_FREQUENCY_FILE, encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        word, count_str = parts
                        try:
                            count = int(count_str)
                            frequency_dict[word.lower()] = count
                        except ValueError:
                            continue

            debug_log(f"[VOCAB] Loaded {len(frequency_dict)} words from frequency dataset")

            # Build rank map
            sorted_words = sorted(frequency_dict.items(), key=lambda x: x[1], reverse=True)
            rank_map = {word: rank for rank, (word, _) in enumerate(sorted_words)}
            debug_log(f"[VOCAB] Built rank map for {len(rank_map)} words")

        except Exception as e:
            debug_log(f"[VOCAB] Error loading frequency dataset: {e}")

        return frequency_dict, rank_map

    def _ensure_nltk_data(self):
        """Ensure NLTK data is available."""
        try:
            wordnet.synsets('test')
        except LookupError:
            debug_log("[VOCAB] Downloading NLTK wordnet...")
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)

    def add_user_exclusion(self, term: str) -> bool:
        """Add a term to the user's exclusion list."""
        if not self.user_exclude_path:
            debug_log("[VOCAB] Cannot add exclusion: no user exclude path configured")
            return False

        lower_term = term.lower().strip()
        if not lower_term:
            return False

        self.user_exclude_list.add(lower_term)

        try:
            os.makedirs(os.path.dirname(self.user_exclude_path), exist_ok=True)
            with open(self.user_exclude_path, 'a', encoding='utf-8') as f:
                f.write(f"{lower_term}\n")
            debug_log(f"[VOCAB] Added '{term}' to user exclusion list")
            return True
        except Exception as e:
            debug_log(f"[VOCAB] Failed to save user exclusion: {e}")
            return False

    def reload_user_exclusions(self):
        """Reload user exclusions from file."""
        if self.user_exclude_path:
            self.user_exclude_list = self._load_word_list(self.user_exclude_path)
            # Also update NER algorithm's exclusion list
            for alg in self.algorithms:
                if hasattr(alg, 'user_exclude_list'):
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
            from src.user_preferences import get_user_preferences
            from src.core.vocabulary.corpus_manager import get_corpus_manager
            from src.config import CORPUS_MIN_DOCUMENTS

            # Check user preference
            prefs = get_user_preferences()
            if not prefs.get("bm25_enabled", True):
                debug_log("[VOCAB] BM25 disabled by user preference")
                return False

            # Check corpus readiness
            corpus_manager = get_corpus_manager()
            if not corpus_manager.is_corpus_ready(min_docs=CORPUS_MIN_DOCUMENTS):
                doc_count = corpus_manager.get_document_count()
                debug_log(f"[VOCAB] BM25 skipped: corpus has {doc_count}/{CORPUS_MIN_DOCUMENTS} documents")
                return False

            return True

        except Exception as e:
            debug_log(f"[VOCAB] BM25 check failed: {e}")
            return False
