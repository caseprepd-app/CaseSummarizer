"""
Feature Extraction for Vocabulary Meta-Learner

Extracts feature vectors from term data for ML model training and prediction.

Features include:
- Count bins (one-hot encoded occurrence frequency)
- freq_per_1k_words: Scale-independent document density (occurrences per 1K words)
- Algorithm source features (NER, RAKE, BM25, TextRank, MedicalNER, GLiNER)
- Character/format features (artifact detection)
- Document quality features
- Name validation features
- TermSources-based per-document confidence features
  - mean_count_per_doc: Average concentration per source document
  - doc_diversity_ratio: Proportion of session docs containing term
"""

from typing import Any

import numpy as np

from src.config import (
    NON_NER_PHRASE_COMMON_WORD_FLOOR,
    get_count_bin_features,
)
from src.core.vocabulary.adjusted_mean import compute_adjusted_mean
from src.core.vocabulary.preference_learner_text_analysis import (
    _get_name_country_data,
    _load_names_datasets,
    _log_rarity_score,
    _max_consonant_run,
)
from src.core.vocabulary.rarity_filter import _load_scaled_frequencies
from src.core.vocabulary.term_sources import TermSources
from src.user_preferences import get_user_preferences

# Medical suffixes for domain-specific feature
MEDICAL_SUFFIXES = (
    "itis",
    "osis",
    "ectomy",
    "pathy",
    "algia",
    "emia",
    "plasty",
    "scopy",
    "otomy",
    "ology",
    "oma",
    "esis",
    "iasis",
    "trophy",
    "megaly",
    "rrhea",
    "rrhage",
    "plegia",
)

# Legal suffixes for domain-specific feature
LEGAL_SUFFIXES = (
    "ant",  # defendant, complainant, claimant
    "ent",  # respondent, decedent
    "ee",  # appellee, mortgagee, payee, lessee
    "or",  # mortgagor, payor, lessor (when legal context)
)

# Title prefixes indicating person names
TITLE_PREFIXES = (
    "dr.",
    "dr ",
    "mr.",
    "mr ",
    "ms.",
    "ms ",
    "mrs.",
    "mrs ",
    "hon.",
    "hon ",
    "judge ",
    "justice ",
    "senator ",
    "rep.",
    "rep ",
)

# Professional suffixes indicating person names
PROFESSIONAL_SUFFIXES = (
    "m.d.",
    "md",
    "esq.",
    "esq",
    "ph.d.",
    "phd",
    "r.n.",
    "rn",
    "d.o.",
    "do",
    "j.d.",
    "jd",
    "d.d.s.",
    "dds",
    "p.a.",
    "pa",
    "n.p.",
    "np",
)

# Feature indices for interpretability
# Only is_person is reliable among type features (NER person detection).
# source_doc_confidence weights terms by OCR quality.
# corpus feature and is_title_case are simplified to binary.
# TermSources features provide per-document confidence tracking.
FEATURE_NAMES = [
    # Count bin features - one-hot encoded
    # Rationale: count=1 could be OCR error, higher counts progressively more reliable
    # Granularity above 7 distinguishes frequent names (119 occ)
    "count_bin_1",  # Exactly 1 occurrence (possible OCR error)
    "count_bin_2",  # Exactly 2 occurrences
    "count_bin_3",  # Exactly 3 occurrences
    "count_bin_4_6",  # 4-6 occurrences
    "count_bin_7_20",  # 7-20 occurrences (mentioned multiple times)
    "count_bin_21_50",  # 21-50 occurrences (appears throughout document)
    "count_bin_51_plus",  # 51+ occurrences (major figure in transcript)
    "log_count",  # Log-scaled count preserving magnitude (log10(count+1))
    "freq_per_1k_words",  # Scale-independent density (count / (total_words / 1000))
    # Algorithm features
    "has_ner",
    "has_rake",
    "has_bm25",  # Per-algorithm tracking
    "has_textrank",  # Binary: TextRank found this term
    "has_medical_ner",  # Binary: MedicalNER (medspacy) found this term
    "has_gliner",  # Binary: GLiNER found this term
    "textrank_score",  # PageRank centrality score (0-1)
    "is_person",  # NER person detection - the only reliable type info
    # Character/format features for artifact detection
    "has_trailing_punctuation",  # "Smith:", "Di Leo." - likely artifacts
    "has_leading_digit",  # "4 Ms. Di Leo", "17 SMITH" - line numbers
    "has_trailing_digit",  # "Smith 17", "Di Leo 2" - page/line number suffixes
    "word_count",  # 1-3 words = good, 4+ = suspicious over-extraction
    "is_all_caps",  # "PLAINTIFF'S EXHIBIT" - headers, not vocabulary
    "is_title_case",  # 1.0 if term.istitle() (proper nouns), 0.0 otherwise
    # Document quality feature
    "source_doc_confidence",  # OCR/extraction confidence (0-100) - lower = more OCR errors
    # Corpus familiarity feature (simplified to binary)
    "corpus_common_term",  # Binary: True if term in >=64% of corpus docs AND >=5 occurrences
    # Word-level features (9 total) + rarity score
    "freq_dict_word_ratio",
    "word_log_rarity_score",  # Mean log-scaled rarity (0=common, 1=rare)
    "term_length",
    "vowel_ratio",
    "is_single_letter",
    "has_internal_digits",
    "has_medical_suffix",
    "has_repeated_chars",
    "contains_hyphen",
    # TermSources-based per-document confidence features (7 total)
    # These provide richer signals about term reliability across source documents
    "mean_count_per_doc",  # Average occurrences per source doc (occurrences / num_docs)
    "doc_diversity_ratio",  # num_docs / total_docs (0-1, spread across corpus)
    "median_doc_confidence",  # Median confidence - robust to single bad scan (0-1)
    "confidence_std_dev",  # Consistency of confidence across docs (0-0.5)
    "high_conf_doc_ratio",  # % of source docs with confidence > 0.80 (0-1)
    "all_low_conf",  # 1 if ALL source docs have conf < 0.60 (red flag)
    # Name validation and domain-specific features (5 total)
    "is_in_names_dataset",  # Any word in international forenames/surnames datasets
    "names_word_ratio",  # Proportion of term's words found in names dataset (0-1)
    "has_forename_and_surname",  # Term has both a forename and surname match (0/1)
    "name_country_spread",  # Geographic diversity of matched names (0-1)
    "has_legal_suffix",  # -ant, -ent, -ee, -or (defendant, appellee, etc.)
    "has_title_prefix",  # Dr., Mr., Ms., Hon., Judge, etc.
    "has_professional_suffix",  # M.D., Esq., Ph.D., R.N., etc.
    "max_consonant_run",  # Longest consonant streak (gibberish detector)
    # Stop word boundary features
    "starts_with_stop_word",  # First word is top-1000 common (e.g., "the same")
    "ends_with_stop_word",  # Last word is top-1000 common (e.g., "Smith the")
    # User-defined indicator pattern features
    "matches_positive_indicator",  # User-defined positive pattern match
    "matches_negative_indicator",  # User-defined negative pattern match
]


def extract_features(term_data: dict[str, Any]) -> np.ndarray:
    """
    Extract feature vector from term data.

    Includes 9 word-level features (frequency, vowel ratio, medical suffix, etc.)
    and 7 TermSources-based per-document features (doc_diversity_ratio,
    median_doc_confidence, confidence_std_dev, high_conf_doc_ratio, all_low_conf).
    Falls back to sensible defaults when TermSources is not available (legacy data).

    Args:
        term_data: Dictionary with term information (from feedback CSV or extractor)
                  May include "sources" (TermSources) and "total_docs_in_session"

    Returns:
        numpy array of 52 features (7 count bins + log_count + 44 other features)

    Raises:
        ValueError: If term_data is not a dict or missing required fields
    """
    # Validate required fields
    if not isinstance(term_data, dict):
        raise ValueError(f"term_data must be dict, got {type(term_data)}")

    # Get the term text
    term = str(term_data.get("Term", "") or term_data.get("term", "") or "")
    if not term:
        raise ValueError("term_data must contain 'Term' or 'term' key with non-empty value")

    term_lower = term.lower()

    # === FREQUENCY FEATURES ===
    occurrences = float(term_data.get("occurrences") or 1)
    count = int(occurrences)

    # Count bins - one-hot encoded via config
    # Rationale: count=1 could be OCR error, higher counts more reliable
    (
        count_bin_1,
        count_bin_2,
        count_bin_3,
        count_bin_4_6,
        count_bin_7_20,
        count_bin_21_50,
        count_bin_51_plus,
    ) = get_count_bin_features(count)

    # Log-scaled count to preserve magnitude within bins
    # bin_51_plus covers 51-∞ with no distinction; log_count provides continuous signal
    # Examples: 1→0, 10→1.0, 100→2.0, 301→2.48
    log_count = np.log10(count + 1)

    # Scale-independent document density: occurrences per 1,000 words
    # 5 occurrences in 500 words = 10/1Kw (prominent)
    # 5 occurrences in 50,000 words = 0.1/1Kw (background noise)
    total_word_count = float(term_data.get("total_word_count") or 0)
    if total_word_count > 0:
        freq_per_1k_words = occurrences / max(total_word_count / 1000.0, 0.1)
    else:
        # Legacy fallback: estimate from total_unique_terms if available
        total_unique_terms = float(term_data.get("total_unique_terms") or 0)
        if total_unique_terms > 0:
            freq_per_1k_words = occurrences / max(total_unique_terms / 1000.0, 0.1)
        else:
            freq_per_1k_words = occurrences / 0.1  # Conservative fallback

    # === ALGORITHM SOURCE FEATURES ===
    algorithms = str(term_data.get("algorithms", "")).lower()
    has_ner = 1.0 if "ner" in algorithms else 0.0
    has_rake = 1.0 if "rake" in algorithms else 0.0
    has_bm25 = 1.0 if "bm25" in algorithms else 0.0
    has_textrank = 1.0 if "textrank" in algorithms else 0.0
    has_medical_ner = 1.0 if "medicalner" in algorithms else 0.0
    has_gliner = 1.0 if "gliner" in algorithms else 0.0
    textrank_score = float(term_data.get("textrank_score", 0.0))

    # === PERSON DETECTION ===
    is_person_val = term_data.get("is_person", 0)
    is_person = (
        float(is_person_val)
        if isinstance(is_person_val, (int, float))
        else (1.0 if str(is_person_val).lower() in ("1", "yes", "true") else 0.0)
    )

    # === CHARACTER/FORMAT FEATURES ===
    # Trailing punctuation (":Smith", "Di Leo.") - likely artifacts
    trailing_punct = ":;.,!?"
    has_trailing_punctuation = (
        1.0 if term and len(term.strip()) > 0 and term[-1] in trailing_punct else 0.0
    )

    # Leading/trailing digits
    has_leading_digit = 1.0 if term and len(term.strip()) > 0 and term[0].isdigit() else 0.0
    has_trailing_digit = 1.0 if term and len(term.strip()) > 0 and term[-1].isdigit() else 0.0

    # Word count
    words = term.split() if term else []
    word_count = float(len(words)) if words else 1.0

    # All caps detection
    alpha_chars = [c for c in term if c.isalpha()]
    is_all_caps = 1.0 if alpha_chars and all(c.isupper() for c in alpha_chars) else 0.0

    # Source document confidence
    source_doc_confidence_raw = float(term_data.get("source_doc_confidence") or 100)
    source_doc_confidence = source_doc_confidence_raw / 100.0

    # Corpus common term feature (simplified to binary)
    # Binary feature: 1.0 if term is common in corpus (>=64% docs AND >=5 occurrences)
    corpus_common_raw = term_data.get("corpus_common_term", False)
    corpus_common_term = 1.0 if corpus_common_raw else 0.0

    # Title case detection
    is_title_case = 1.0 if term and term.istitle() else 0.0

    # === WORD-LEVEL FEATURES ===

    # Word-level frequency dictionary features
    freq_dict = _load_scaled_frequencies()
    words_lower = [w.lower() for w in words] if words else []
    if words_lower:
        words_in_dict = [1.0 if w in freq_dict else 0.0 for w in words_lower]
        freq_dict_word_ratio = sum(words_in_dict) / len(words_in_dict)

        # Log-scaled rarity score (adjusted mean across words)
        # Distinguishes "Comiskey" (rank 96755, rare) from "Clerk" (rank 5435, common)
        # Uses adjusted mean: filters by linear scores, averages log scores
        prefs = get_user_preferences()
        floor = prefs.get("non_ner_phrase_common_word_floor", NON_NER_PHRASE_COMMON_WORD_FLOOR)
        linear_scores = [freq_dict.get(w, 1.0) for w in words_lower]
        log_scores = [_log_rarity_score(s) for s in linear_scores]
        word_log_rarity_score = compute_adjusted_mean(
            log_scores, floor, filter_scores=linear_scores
        )
    else:
        freq_dict_word_ratio = 0.0
        word_log_rarity_score = 0.5  # Default for empty terms

    # Term length (character count)
    term_length = float(len(term))

    # Vowel ratio - gibberish detector
    # Real words ~40% vowels, gibberish often 0% or very low
    vowels = set("aeiouAEIOU")
    if alpha_chars:
        vowel_count = sum(1 for c in alpha_chars if c in vowels)
        vowel_ratio = vowel_count / len(alpha_chars)
    else:
        vowel_ratio = 0.0

    # Single letter detection ("Q", "A" - transcript artifacts)
    is_single_letter = 1.0 if len(term.strip()) == 1 and term.strip().isalpha() else 0.0

    # Internal digits (digits not at start or end)
    # "Smith17" has trailing, "17Smith" has leading, "Sm1th" has internal
    if len(term) > 2:
        internal_chars = term[1:-1]
        has_internal_digits = 1.0 if any(c.isdigit() for c in internal_chars) else 0.0
    else:
        has_internal_digits = 0.0

    # Medical suffix detection - strong signal for legitimate vocabulary
    has_medical_suffix = (
        1.0 if any(term_lower.endswith(suffix) for suffix in MEDICAL_SUFFIXES) else 0.0
    )

    # Repeated characters (3+ in a row) - artifact detection
    # Catches "aaaa", ".....", "---"
    has_repeated_chars = 0.0
    if len(term) >= 3:
        for i in range(len(term) - 2):
            if term[i] == term[i + 1] == term[i + 2]:
                has_repeated_chars = 1.0
                break

    # Contains hyphen - often legitimate compound terms
    contains_hyphen = 1.0 if "-" in term else 0.0

    # === Name validation and domain-specific features ===

    # Check if any word is in the names dataset
    forenames, surnames = _load_names_datasets()
    all_names = forenames | surnames
    is_in_names_dataset = 0.0
    names_word_ratio = 0.0
    has_forename_and_surname = 0.0
    name_country_spread = 0.0
    if words_lower:
        matched_count = 0
        has_forename = False
        has_surname = False
        for w in words_lower:
            if w in all_names:
                is_in_names_dataset = 1.0
                matched_count += 1
            if w in forenames:
                has_forename = True
            if w in surnames:
                has_surname = True
        names_word_ratio = matched_count / len(words_lower)
        has_forename_and_surname = 1.0 if (has_forename and has_surname) else 0.0

        # Geographic spread: max country count among matched words
        name_country_counts, total_countries = _get_name_country_data()
        max_countries = 0
        for w in words_lower:
            if w in name_country_counts:
                max_countries = max(max_countries, name_country_counts[w])
        name_country_spread = max_countries / total_countries if total_countries > 0 else 0.0

    # Legal suffix detection (like medical suffix but for legal terms)
    has_legal_suffix = 1.0 if any(term_lower.endswith(suffix) for suffix in LEGAL_SUFFIXES) else 0.0

    # Title prefix detection (Dr., Mr., Judge, etc.)
    has_title_prefix = (
        1.0 if any(term_lower.startswith(prefix) for prefix in TITLE_PREFIXES) else 0.0
    )

    # Professional suffix detection (M.D., Esq., etc.)
    # Check if term ends with any professional suffix (with optional trailing punct)
    term_lower_stripped = term_lower.rstrip(".,;:")
    has_professional_suffix = (
        1.0
        if any(term_lower_stripped.endswith(suffix) for suffix in PROFESSIONAL_SUFFIXES)
        else 0.0
    )

    # Maximum consonant run (gibberish detector)
    max_consonant_run_val = float(_max_consonant_run(term))

    # === Stop word boundary features ===
    # Catches truncation artifacts where extraction grabbed too much or too little
    from src.config import STOP_WORD_THRESHOLD
    from src.core.vocabulary.rarity_filter import is_common_word

    if words_lower:
        starts_with_stop_word = 1.0 if is_common_word(words_lower[0], STOP_WORD_THRESHOLD) else 0.0
        ends_with_stop_word = 1.0 if is_common_word(words_lower[-1], STOP_WORD_THRESHOLD) else 0.0
    else:
        starts_with_stop_word = 0.0
        ends_with_stop_word = 0.0

    # === User-defined indicator pattern features ===
    from src.core.vocabulary.indicator_patterns import matches_negative, matches_positive

    matches_positive_indicator = 1.0 if matches_positive(term) else 0.0
    matches_negative_indicator = 1.0 if matches_negative(term) else 0.0

    # === TermSources-based per-document features ===
    # These features provide richer signals about term reliability by
    # tracking which source documents contributed each occurrence.
    #
    # When TermSources is available:
    # - Multiple high-confidence documents → more reliable term
    # - Single low-confidence document → potentially OCR error
    # - High confidence_std_dev → inconsistent quality (red flag)
    #
    # NOTE: An ML model trained on user feedback could learn to weight
    # these features based on the user's document types and preferences.

    # Check if TermSources is available
    sources = term_data.get("sources")
    total_docs_in_session = float(term_data.get("total_docs_in_session") or 1)

    if isinstance(sources, TermSources) and sources.num_documents > 0:
        # Extract features from actual TermSources
        num_source_documents = float(sources.num_documents)
        mean_count_per_doc = occurrences / max(num_source_documents, 1.0)
        doc_diversity_ratio = sources.doc_diversity_ratio(int(total_docs_in_session))
        median_doc_confidence = sources.median_confidence
        confidence_std_dev = sources.confidence_std_dev
        high_conf_doc_ratio = sources.high_conf_doc_ratio
        all_low_conf = 1.0 if sources.all_low_conf else 0.0
    else:
        # Legacy fallback: no TermSources available
        # Try to compute from CSV columns if available
        num_docs_raw = float(term_data.get("num_source_documents") or 1)
        mean_count_per_doc = occurrences / max(num_docs_raw, 1.0)
        doc_diversity_ratio = 1.0 / total_docs_in_session
        # Use source_doc_confidence as a single-doc approximation
        median_doc_confidence = source_doc_confidence
        confidence_std_dev = 0.0  # No variance with single source
        high_conf_doc_ratio = 1.0 if source_doc_confidence > 0.80 else 0.0
        all_low_conf = 1.0 if source_doc_confidence < 0.60 else 0.0

    return np.array(
        [
            # Count bin features (7) + log_count (1) + freq_per_1k_words (1)
            count_bin_1,
            count_bin_2,
            count_bin_3,
            count_bin_4_6,
            count_bin_7_20,
            count_bin_21_50,
            count_bin_51_plus,
            log_count,  # Log-scaled count for magnitude within bins
            freq_per_1k_words,  # Scale-independent density
            # Algorithm features (3)
            has_ner,
            has_rake,
            has_bm25,
            has_textrank,
            has_medical_ner,
            has_gliner,
            textrank_score,
            # Type feature (1)
            is_person,
            # Original artifact features (6)
            has_trailing_punctuation,
            has_leading_digit,
            has_trailing_digit,
            word_count,
            is_all_caps,
            is_title_case,
            # Quality features (2)
            source_doc_confidence,
            corpus_common_term,
            # Word-level features (8) + rarity score (1)
            freq_dict_word_ratio,
            word_log_rarity_score,  # Log-scaled word rarity
            term_length,
            vowel_ratio,
            is_single_letter,
            has_internal_digits,
            has_medical_suffix,
            has_repeated_chars,
            contains_hyphen,
            # TermSources features (6)
            mean_count_per_doc,
            doc_diversity_ratio,
            median_doc_confidence,
            confidence_std_dev,
            high_conf_doc_ratio,
            all_low_conf,
            # Name validation and domain features (8)
            is_in_names_dataset,
            names_word_ratio,
            has_forename_and_surname,
            name_country_spread,
            has_legal_suffix,
            has_title_prefix,
            has_professional_suffix,
            max_consonant_run_val,
            # Stop word boundary features (2)
            starts_with_stop_word,
            ends_with_stop_word,
            # User-defined indicator pattern features (2)
            matches_positive_indicator,
            matches_negative_indicator,
        ]
    )
