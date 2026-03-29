"""
Vocabulary Field Name Constants.

Centralizes all dictionary key strings used in vocabulary term dicts
so that typos are caught by the IDE and renames are safe.

Usage:
    from src.core.vocab_schema import VF

    term_data[VF.TERM] = "radiculopathy"
    if term_data.get(VF.IS_PERSON) == VF.YES:
        ...
"""


class VF:
    """Vocabulary Field name constants."""

    # --- Core display columns ---
    TERM = "Term"
    QUALITY_SCORE = "Quality Score"
    IS_PERSON = "Is Person"
    FOUND_BY = "Found By"
    ROLE_RELEVANCE = "Role/Relevance"

    # --- TermSources display columns ---
    OCCURRENCES = "Occurrences"
    NUM_DOCS = "# Docs"
    OCR_CONFIDENCE = "OCR Confidence"

    # --- Algorithm detail columns ---
    NER = "NER"
    RAKE = "RAKE"
    BM25 = "BM25"
    TOPICRANK = "TopicRank"
    MEDICALNER = "MedicalNER"
    YAKE = "YAKE"
    ALGO_COUNT = "Algo Count"

    # --- Additional columns ---
    GOOGLE_RARITY_RANK = "Google Rarity Rank"

    # --- Feedback columns ---
    KEEP = "Keep"
    SKIP = "Skip"

    # --- Legacy / backward-compat keys ---
    SOURCES = "Sources"
    TYPE = "Type"
    FREQUENCY = "Frequency"
    CONFIDENCE = "Confidence"

    # --- Filtered terms ---
    FILTER_REASON = "Filter Reason"

    # --- Standard values ---
    YES = "Yes"
    NO = "No"
