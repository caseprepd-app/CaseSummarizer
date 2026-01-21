"""
Name Regularization for Vocabulary Extraction

Post-processing filter that removes name fragments and typo variants
using dictionary lookup and confidence-weighted scoring.

Two main filters:

1. Fragment Filter:
   - "Di Leo" (high frequency) → remove "Di", "Leo" (low frequency fragments)
   - Handles spaCy splitting multi-word names into separate entities

2. Typo Filter (1-2 character edit distance):
   - Session 78: Uses CanonicalScorer with branching logic:
     - Exactly ONE variant in dictionary → it wins (100% confidence)
     - ZERO variants known → confidence-weighted score decides
     - MULTIPLE variants known → weighted score as tiebreaker
   - Integrates OCR artifact penalty (10% reduction for suspicious patterns)

NOTE: An ML model trained on user feedback could potentially outperform this
rules-based approach by learning document-specific patterns (medical vs legal
terminology, regional name spellings). This implementation provides a reasonable
baseline that works without training data.

These filters run AFTER artifact_filter.py (which catches longer superstrings)
and work on the opposite direction (catching shorter fragments and typos).

Example:
    Input: ["Di Leo" (50), "Di" (3), "Leo" (2), "Jenkins" (40), "Jenidns" (8)]

    After filtering:
    - "Di" removed (fragment of "Di Leo")
    - "Leo" removed (fragment of "Di Leo")
    - "Jenidns" removed (typo of "Jenkins" - only "Jenkins" is in dictionary)
"""

from src.core.vocabulary.canonical_scorer import create_canonical_scorer
from src.core.vocabulary.string_utils import edit_distance
from src.core.vocabulary.term_sources import TermSources
from src.logging_config import debug_log

# Lazy-loaded known words set for typo resolution
_KNOWN_WORDS: set[str] | None = None


def _load_known_words() -> set[str]:
    """
    Load known words for typo resolution from multiple sources.

    Session 78: Used to decide which of two similar terms (1 char apart) is the typo.
    Logic:
    - One in list, one not → keep the known word, filter the typo
    - Both in list → keep both (user decides)
    - Neither in list → keep both (user decides - may be exotic names)

    Sources:
    1. Google 333k word frequency list (top 50k) - common English words/names
    2. International names dataset (~4.6k names from 106 countries)
       https://github.com/sigpwned/popular-names-by-country-dataset
       CC0 Public Domain license

    Returns:
        Set of lowercase known words/names
    """
    global _KNOWN_WORDS
    if _KNOWN_WORDS is not None:
        return _KNOWN_WORDS

    from pathlib import Path

    from src.config import GOOGLE_WORD_FREQUENCY_FILE

    _KNOWN_WORDS = set()

    # Source 1: Google word frequency list (top 50k)
    if GOOGLE_WORD_FREQUENCY_FILE.exists():
        try:
            max_words = 50000
            with open(GOOGLE_WORD_FREQUENCY_FILE, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= max_words:
                        break
                    parts = line.strip().split("\t")
                    if parts:
                        _KNOWN_WORDS.add(parts[0].lower())
            debug_log(f"[NAME-REG] Loaded {len(_KNOWN_WORDS)} words from Google frequency list")
        except Exception as e:
            debug_log(f"[NAME-REG] Failed to load Google word frequency file: {e}")

    # Source 2: International names dataset
    # https://github.com/sigpwned/popular-names-by-country-dataset
    data_dir = Path(__file__).parent.parent.parent.parent / "data" / "names"
    surnames_file = data_dir / "international_surnames.csv"
    forenames_file = data_dir / "international_forenames.csv"

    names_loaded = 0

    # Load surnames (Romanized Name is column index 5)
    if surnames_file.exists():
        try:
            with open(surnames_file, encoding="utf-8") as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) > 5 and parts[5]:
                        _KNOWN_WORDS.add(parts[5].lower())
                        names_loaded += 1
        except Exception as e:
            debug_log(f"[NAME-REG] Failed to load surnames file: {e}")

    # Load forenames (Romanized Name is column index 11)
    if forenames_file.exists():
        try:
            with open(forenames_file, encoding="utf-8") as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) > 11 and parts[11]:
                        _KNOWN_WORDS.add(parts[11].lower())
                        names_loaded += 1
        except Exception as e:
            debug_log(f"[NAME-REG] Failed to load forenames file: {e}")

    if names_loaded > 0:
        debug_log(f"[NAME-REG] Loaded {names_loaded} international names")

    debug_log(f"[NAME-REG] Total known words for typo resolution: {len(_KNOWN_WORDS)}")

    return _KNOWN_WORDS


def _is_known_term(term: str) -> bool:
    """
    Check if all words in a term are in the known words list.

    Args:
        term: A term like "Leroy Jenkins" or "John Smith"

    Returns:
        True if ALL words in the term are known, False otherwise
    """
    known_words = _load_known_words()
    words = term.lower().split()
    return all(word.strip(".,;:'\"") in known_words for word in words)


def _is_fragment_of(fragment: str, canonical: str) -> bool:
    """
    Check if fragment is a word-level fragment of canonical.

    A fragment is a proper subset of words from the canonical term.
    "Di" is a fragment of "Di Leo" (one word from a two-word name).
    "Hospital" is NOT a fragment of "Memorial Hospital" (partial word match not allowed).

    Args:
        fragment: Potential fragment term (shorter)
        canonical: Canonical term to check against (longer)

    Returns:
        True if fragment is a word-level subset of canonical
    """
    fragment_lower = fragment.lower().strip()
    canonical_lower = canonical.lower().strip()

    # Split into words
    fragment_words = set(fragment_lower.split())
    canonical_words = set(canonical_lower.split())

    # Fragment must have fewer words than canonical
    if len(fragment_words) >= len(canonical_words):
        return False

    # All fragment words must appear in canonical words
    if not fragment_words.issubset(canonical_words):
        return False

    # Must be a PROPER subset (not equal)
    return fragment_words != canonical_words


def filter_name_fragments(
    vocabulary: list[dict],
    top_fraction: float = 0.25,
    count_key: str = "In-Case Freq",
    term_key: str = "Term",
) -> list[dict]:
    """
    Remove terms that are word-level fragments of high-frequency canonical terms.

    Terms in the top fraction (by count) are considered canonical. Terms in
    the bottom portion are checked to see if they're fragments of any canonical term.

    Note: Top-quartile terms are NEVER removed, even if they're fragments of
    each other. This preserves high-frequency terms that users likely want.

    Example:
        "Di Leo" (count=50) in top quartile
        "Di" (count=3) in bottom → removed (fragment of "Di Leo")
        "Leo" (count=2) in bottom → removed (fragment of "Di Leo")

    Args:
        vocabulary: List of term dictionaries
        top_fraction: Fraction of terms to consider canonical (default 0.25 = top quartile)
        count_key: Dictionary key for term frequency count
        term_key: Dictionary key for the term string

    Returns:
        Filtered vocabulary list with fragments removed
    """
    if not vocabulary or len(vocabulary) < 4:
        return vocabulary

    # Sort by count descending
    sorted_vocab = sorted(vocabulary, key=lambda x: int(x.get(count_key) or 0), reverse=True)

    # Split into top quartile and bottom 3/4
    split_index = max(1, int(len(sorted_vocab) * top_fraction))
    top_terms = sorted_vocab[:split_index]
    bottom_terms = sorted_vocab[split_index:]

    # Build set of canonical terms (multi-word only, since fragments are subsets)
    canonical_multiword = []
    for term_dict in top_terms:
        term = term_dict.get(term_key, "")
        if term and len(term.split()) > 1:  # Multi-word terms only
            canonical_multiword.append(term)

    if not canonical_multiword:
        return vocabulary

    debug_log(f"[NAME-REG] Fragment filter: {len(canonical_multiword)} multi-word canonical terms")

    # Filter ONLY bottom terms (top terms are always preserved)
    filtered_bottom = []
    removed_count = 0

    for term_dict in bottom_terms:
        term = term_dict.get(term_key, "")
        is_fragment = False

        for canonical in canonical_multiword:
            if _is_fragment_of(term, canonical):
                debug_log(f"[NAME-REG] Removing fragment '{term}' (subset of '{canonical}')")
                is_fragment = True
                removed_count += 1
                break

        if not is_fragment:
            filtered_bottom.append(term_dict)

    debug_log(f"[NAME-REG] Fragment filter removed {removed_count} terms")

    return top_terms + filtered_bottom


def filter_typo_variants(
    vocabulary: list[dict],
    top_fraction: float = 0.25,
    max_edit_distance: int = 1,
    min_term_length: int = 5,
    count_key: str = "In-Case Freq",
    term_key: str = "Term",
) -> list[dict]:
    """
    Remove terms that are 1-character typos of high-frequency canonical terms.

    DEPRECATED (Session 78): This function uses the old "top 25% frequency"
    approach. Use regularize_names() instead, which uses CanonicalScorer with
    dictionary lookup and confidence-weighted scoring.

    The old approach fails when OCR consistently produces the same error
    (e.g., 99 wrong readings vs 1 correct - the typo becomes "canonical").

    Kept for backward compatibility but no longer called by regularize_names().

    Args:
        vocabulary: List of term dictionaries
        top_fraction: Fraction of terms to consider canonical (default 0.25 = top quartile)
        max_edit_distance: Maximum edit distance to consider a typo (default 1)
        min_term_length: Minimum term length to apply typo filter (default 5)
        count_key: Dictionary key for term frequency count
        term_key: Dictionary key for the term string

    Returns:
        Filtered vocabulary list with typo variants removed
    """
    if not vocabulary or len(vocabulary) < 4:
        return vocabulary

    # Sort by count descending
    sorted_vocab = sorted(vocabulary, key=lambda x: int(x.get(count_key) or 0), reverse=True)

    # Split into top quartile and bottom 3/4
    split_index = max(1, int(len(sorted_vocab) * top_fraction))
    top_terms = sorted_vocab[:split_index]
    bottom_terms = sorted_vocab[split_index:]

    # Build list of canonical terms long enough for typo checking
    canonical_terms = []
    for term_dict in top_terms:
        term = term_dict.get(term_key, "")
        if term and len(term) >= min_term_length:
            canonical_terms.append(term.lower())

    if not canonical_terms:
        return vocabulary

    debug_log(
        f"[NAME-REG] Typo filter: {len(canonical_terms)} canonical terms (len >= {min_term_length})"
    )

    # Filter bottom terms
    filtered_bottom = []
    removed_count = 0

    for term_dict in bottom_terms:
        term = term_dict.get(term_key, "")
        term_lower = term.lower()

        # Skip short terms (too prone to false positives)
        if len(term) < min_term_length:
            filtered_bottom.append(term_dict)
            continue

        is_typo = False

        for canonical in canonical_terms:
            # Skip if exact match (shouldn't happen but safety check)
            if term_lower == canonical:
                continue

            # Skip if lengths differ by more than max_edit_distance
            # (optimization: edit distance can't be less than length difference)
            if abs(len(term_lower) - len(canonical)) > max_edit_distance:
                continue

            distance = edit_distance(term_lower, canonical)

            if distance <= max_edit_distance:
                debug_log(
                    f"[NAME-REG] Removing typo '{term}' (distance={distance} from '{canonical}')"
                )
                is_typo = True
                removed_count += 1
                break

        if not is_typo:
            filtered_bottom.append(term_dict)

    debug_log(f"[NAME-REG] Typo filter removed {removed_count} terms")

    return top_terms + filtered_bottom


def _single_pass_regularize(
    vocabulary: list[dict],
    top_fraction: float,
    min_canonical_count: int,
    count_key: str,
    term_key: str,
) -> tuple[list[dict], int, int]:
    """
    Single pass of name regularization.

    Session 78: Uses CanonicalScorer with branching logic for typo detection:
    - Exactly ONE variant in dictionary → it wins (100% confidence)
    - ZERO variants known → confidence-weighted score decides
    - MULTIPLE variants known → weighted score as tiebreaker

    This replaces the old "top 25% frequency" approach which fails when OCR
    consistently produces the same error (e.g., 99 wrong readings vs 1 correct).

    NOTE: An ML model trained on user feedback could potentially outperform this
    rules-based approach by learning document-specific patterns.

    Args:
        vocabulary: List of term dictionaries
        top_fraction: Fraction of terms to consider canonical (for fragments)
        min_canonical_count: Minimum number of canonical terms
        count_key: Dictionary key for term frequency count
        term_key: Dictionary key for the term string

    Returns:
        Tuple of (filtered vocabulary, fragments_removed, typos_removed)
    """
    if not vocabulary or len(vocabulary) < 4:
        return vocabulary, 0, 0

    # Sort by count descending to determine canonical terms for FRAGMENT filter
    sorted_vocab = sorted(vocabulary, key=lambda x: int(x.get(count_key) or 0), reverse=True)

    # Determine split point for fragment filter only
    fraction_index = int(len(sorted_vocab) * top_fraction)
    max_canonical = int(len(sorted_vocab) * 0.75)
    split_index = min(max(min_canonical_count, fraction_index), max_canonical)
    split_index = min(split_index, len(sorted_vocab))

    top_terms = sorted_vocab[:split_index]
    bottom_terms = sorted_vocab[split_index:]

    # Build canonical multi-word terms for fragment filter
    canonical_multiword = []
    for term_dict in top_terms:
        term = term_dict.get(term_key, "")
        if term and len(term.split()) > 1:
            canonical_multiword.append(term)

    # FRAGMENT FILTER: Only applies to bottom terms vs top terms
    filtered_after_fragments = []
    fragment_removed = 0

    for term_dict in bottom_terms:
        term = term_dict.get(term_key, "")
        is_fragment = False
        for canonical in canonical_multiword:
            if _is_fragment_of(term, canonical):
                is_fragment = True
                fragment_removed += 1
                break
        if not is_fragment:
            filtered_after_fragments.append(term_dict)

    # Combine top + filtered bottom for typo detection
    all_terms = top_terms + filtered_after_fragments

    # TYPO FILTER: Session 78 - Use CanonicalScorer for typo pair resolution
    # Groups similar terms (1-2 edit distance) and picks the canonical variant
    typo_removed = 0
    terms_to_remove: set[str] = set()

    # Build list of terms long enough for typo checking
    long_terms = [(t, t.get(term_key, "")) for t in all_terms if len(t.get(term_key, "")) >= 5]

    # Create scorer once for efficiency
    scorer = create_canonical_scorer()

    # Group similar terms (1-2 edit distance) for canonical selection
    processed: set[str] = set()
    typo_groups: list[list[tuple[dict, str]]] = []

    for i, (dict_a, term_a) in enumerate(long_terms):
        if term_a.lower() in processed:
            continue

        # Find all terms similar to term_a
        similar_group = [(dict_a, term_a)]
        processed.add(term_a.lower())

        for dict_b, term_b in long_terms[i + 1 :]:
            if term_b.lower() in processed:
                continue

            # Skip if lengths differ by more than 2 (optimization)
            if abs(len(term_a) - len(term_b)) > 2:
                continue

            # Check edit distance (1-2 chars)
            if edit_distance(term_a.lower(), term_b.lower()) <= 2:
                similar_group.append((dict_b, term_b))
                processed.add(term_b.lower())

        # If group has multiple entries, use CanonicalScorer to pick winner
        if len(similar_group) > 1:
            typo_groups.append(similar_group)

    # Process each typo group through CanonicalScorer
    for group in typo_groups:
        # Build variants for CanonicalScorer
        variants = []
        for term_dict, term in group:
            # Check for existing TermSources
            sources = term_dict.get("sources")
            if sources is None:
                # Create legacy sources from frequency
                sources = TermSources.create_legacy(
                    term_dict.get(count_key, 1), term_dict.get("source_doc_confidence", 0.85)
                )

            variants.append(
                {
                    "Term": term,
                    "sources": sources,
                    "In-Case Freq": term_dict.get(count_key, 1),
                    "_original_dict": term_dict,
                }
            )

        # Let CanonicalScorer select the winner
        canonical = scorer.select_canonical(variants)
        canonical_term = canonical["Term"].lower()

        # Mark non-canonical terms for removal
        for _term_dict, term in group:
            if term.lower() != canonical_term:
                terms_to_remove.add(term.lower())
                typo_removed += 1
                debug_log(
                    f"[NAME-REG] Removing typo '{term}' in favor of canonical '{canonical['Term']}'"
                )

    # Filter out removed terms
    result = [t for t in all_terms if t.get(term_key, "").lower() not in terms_to_remove]

    return result, fragment_removed, typo_removed


def regularize_names(
    vocabulary: list[dict],
    top_fraction: float = 0.25,
    min_canonical_count: int = 10,
    num_passes: int = 3,
    count_key: str = "In-Case Freq",
    term_key: str = "Term",
) -> list[dict]:
    """
    Apply name regularization filters with multiple passes.

    This function applies fragment and typo filtering multiple times. Each pass:
    1. Removes fragments (e.g., "Di" when "Di Leo" exists in top quartile)
    2. Removes typos using CanonicalScorer (Session 78)

    Session 78: Typo detection uses CanonicalScorer with branching logic:
    - Exactly ONE variant in dictionary → it wins (keeps the known word)
    - ZERO variants known → confidence-weighted score decides (exotic names)
    - MULTIPLE variants known → weighted score as tiebreaker

    This replaces the old "top 25% frequency" approach which failed when OCR
    consistently produced the same error (e.g., 99 wrong readings vs 1 correct).

    NOTE: An ML model trained on user feedback could potentially outperform
    this rules-based approach by learning document-specific patterns.

    Multiple passes allow legitimate terms to "bubble up" as noise is removed.
    For example, if pass 1 removes 5 typos, a term that was #26 might become #21,
    moving into the top quartile for pass 2 where its fragments can be caught.

    Args:
        vocabulary: List of term dictionaries
        top_fraction: Fraction of terms to consider canonical (for fragments, default 0.25)
        min_canonical_count: Minimum number of canonical terms for fragment filter
        num_passes: Number of filtering passes (default 3)
        count_key: Dictionary key for term frequency count
        term_key: Dictionary key for the term string

    Returns:
        Filtered vocabulary list
    """
    if not vocabulary or len(vocabulary) < 4:
        return vocabulary

    debug_log(
        f"[NAME-REG] Starting regularization on {len(vocabulary)} terms ({num_passes} passes)"
    )

    result = vocabulary
    total_fragments = 0
    total_typos = 0

    for pass_num in range(1, num_passes + 1):
        prev_count = len(result)
        result, fragments, typos = _single_pass_regularize(
            result, top_fraction, min_canonical_count, count_key, term_key
        )
        total_fragments += fragments
        total_typos += typos

        removed_this_pass = prev_count - len(result)
        debug_log(
            f"[NAME-REG] Pass {pass_num}: removed {fragments} fragments, {typos} typos ({removed_this_pass} total)"
        )

        # Early exit if no changes this pass
        if removed_this_pass == 0:
            debug_log(f"[NAME-REG] No changes in pass {pass_num}, stopping early")
            break

    debug_log(
        f"[NAME-REG] Regularization complete: removed {total_fragments} fragments, {total_typos} typos total"
    )
    debug_log(f"[NAME-REG] Final: {len(result)} terms remaining")

    return result
