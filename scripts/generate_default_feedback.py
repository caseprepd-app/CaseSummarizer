#!/usr/bin/env python
"""
Generate Default Feedback CSV for Vocabulary ML

Creates universal negative training data for the vocabulary ML system.
These are terms that ALL users would reject regardless of domain:
- Common English phrases with no vocabulary value
- Transcript artifacts (Q., A., page numbers)
- OCR error patterns
- Common words that slip through filters

Key constraints:
- All feedback is NEGATIVE (-1) only
- No person names (is_person = 0)
- No domain-specific terms (no medical/legal terminology)
- Feature values in middle ranges (teaches model these are bad despite okay features)

Usage:
    python scripts/generate_default_feedback.py
"""

import csv
import random
from pathlib import Path

# Output file
OUTPUT_PATH = Path(__file__).parent.parent / "config" / "default_feedback.csv"

# CSV columns (must match FeedbackManager schema)
COLUMNS = [
    "timestamp",
    "document_id",
    "term",
    "feedback",
    "is_person",
    "algorithms",
    "NER_detection",
    "RAKE_detection",
    "BM25_detection",
    "algo_count",
    "quality_score",
    "in_case_freq",
    "freq_rank",
]


# =============================================================================
# UNIVERSAL NEGATIVE TERMS BY CATEGORY
# =============================================================================

# Category A: Common English phrases - no vocabulary prep value
COMMON_PHRASES = [
    "the same",
    "left side",
    "right side",
    "one time",
    "same time",
    "read copy",
    "true copy",
    "next page",
    "first page",
    "last page",
    "long time",
    "short time",
    "few days",
    "several years",
    "present time",
    "that time",
    "at this time",
    "such as",
    "as well",
    "other than",
    "following day",
    "next day",
    "same day",
    "each time",
    "any time",
]

# Category B: Transcript artifacts - universal across legal transcripts
TRANSCRIPT_ARTIFACTS = [
    "Q.",
    "A.",
    "Q:",
    "A:",
    "Q",
    "A",
    "Page 1",
    "Page 2",
    "Page 3",
    "Line 1",
    "Line 2",
    "Exhibit A",
    "Exhibit B",
    "Exhibit 1",
    "Exhibit 2",
    "DIRECT EXAMINATION",
    "CROSS EXAMINATION",
    "RE-DIRECT EXAMINATION",
    "THE WITNESS",
    "THE COURT",
    "THE REPORTER",
    "EXAMINATION BY",
    "QUESTIONS BY",
    "(Whereupon",
    "(WHEREUPON",
    "(Off the record",
    "BY MR.",
    "BY MS.",
    "[sic]",
    "[phonetic]",
    "[indiscernible]",
    "APPEARANCES",
    "CERTIFICATE",
    "PROCEEDINGS",
    "CONTINUATION",
]

# Category C: OCR error patterns - common misreadings
OCR_ARTIFACTS = [
    "1he",  # digit-letter confusion
    "1t",
    "1s",
    "1n",
    "tbe",  # h/b confusion
    "tbat",
    "wbat",
    "wben",
    "0f",  # zero for o
    "0n",
    "0r",
    "rn",  # m misread as rn
    "rnay",
    "sarne",
    "narne",
    "II",  # two l's for H
]

# Category D: Common words that slip through filters
COMMON_WORDS = [
    "age",
    "bill",
    "copy",
    "read",
    "side",
    "time",
    "body",
    "head",
    "foot",
    "hand",
    "back",
    "kind",
    "type",
    "part",
    "form",
    "sort",
    "place",
    "point",
    "area",
    "end",
    "way",
    "amount",
    "number",
    "state",
    "matter",
]


def get_category(term: str) -> str:
    """Determine which category a term belongs to."""
    if term in COMMON_PHRASES:
        return "phrase"
    elif term in TRANSCRIPT_ARTIFACTS:
        return "artifact"
    elif term in OCR_ARTIFACTS:
        return "ocr"
    else:
        return "common_word"


def generate_algorithms(category: str) -> tuple[str, bool, bool, bool]:
    """
    Generate algorithm detection flags based on category.

    Returns:
        Tuple of (algorithms_string, ner_detection, rake_detection, bm25_detection)
    """
    if category == "phrase":
        # Phrases typically come from RAKE
        choice = random.choice([
            ("RAKE", False, True, False),
            ("RAKE, NER", True, True, False),
            ("RAKE, BM25", False, True, True),
        ])
    elif category == "artifact":
        # Artifacts might come from any algorithm
        choice = random.choice([
            ("NER", True, False, False),
            ("RAKE", False, True, False),
            ("NER, RAKE", True, True, False),
        ])
    elif category == "ocr":
        # OCR errors often detected by NER (misidentified entities)
        choice = random.choice([
            ("NER", True, False, False),
            ("NER, RAKE", True, True, False),
        ])
    else:  # common_word
        # Common words often from RAKE or BM25
        choice = random.choice([
            ("RAKE", False, True, False),
            ("BM25", False, False, True),
            ("RAKE, BM25", False, True, True),
        ])
    return choice


def generate_entry(term: str, index: int) -> dict:
    """
    Generate a single feedback entry with realistic feature values.

    Args:
        term: The vocabulary term
        index: Entry index for document ID rotation

    Returns:
        Dictionary with all CSV columns
    """
    category = get_category(term)

    # Fixed timestamp to indicate shipped data
    timestamp = "2025-01-01T00:00:00"

    # Rotate through simulated documents (10 docs)
    doc_id = f"default_training_{(index % 10):03d}"

    # Always negative feedback
    feedback = -1

    # No person names in universal negatives
    is_person = 0

    # Get algorithm detection
    algorithms, ner, rake, bm25 = generate_algorithms(category)
    algo_count = sum([ner, rake, bm25])

    # Quality score: middle range (40-70)
    # These shouldn't look obviously bad - teaches model to reject despite okay scores
    quality_score = random.randint(40, 70)

    # In-case frequency: realistic range (2-15)
    in_case_freq = random.randint(2, 15)

    # Frequency rank: varies by category
    if category in ("common_word", "phrase"):
        # Common words have low rank (more common)
        freq_rank = random.randint(100, 50000)
    elif category == "artifact":
        # Artifacts might not be in frequency list
        freq_rank = random.randint(0, 10000)
    else:  # ocr
        # OCR errors typically not in frequency list
        freq_rank = 0

    return {
        "timestamp": timestamp,
        "document_id": doc_id,
        "term": term,
        "feedback": feedback,
        "is_person": is_person,
        "algorithms": algorithms,
        "NER_detection": ner,
        "RAKE_detection": rake,
        "BM25_detection": bm25,
        "algo_count": algo_count,
        "quality_score": quality_score,
        "in_case_freq": in_case_freq,
        "freq_rank": freq_rank,
    }


def main():
    """Generate the default feedback CSV."""
    # Combine all terms
    all_terms = (
        COMMON_PHRASES
        + TRANSCRIPT_ARTIFACTS
        + OCR_ARTIFACTS
        + COMMON_WORDS
    )

    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in all_terms:
        if term.lower() not in seen:
            seen.add(term.lower())
            unique_terms.append(term)

    print(f"Generating {len(unique_terms)} universal negative entries...")

    # Set seed for reproducibility
    random.seed(42)

    # Generate entries
    entries = []
    for i, term in enumerate(unique_terms):
        entry = generate_entry(term, i)
        entries.append(entry)

    # Write to CSV
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(entries)

    print(f"Written to: {OUTPUT_PATH}")
    print(f"Total entries: {len(entries)}")

    # Summary by category
    categories = {}
    for term in unique_terms:
        cat = get_category(term)
        categories[cat] = categories.get(cat, 0) + 1

    print("\nBreakdown by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
