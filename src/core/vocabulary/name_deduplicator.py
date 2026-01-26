"""
Name Deduplication for Vocabulary Extraction.

Handles two types of Person name duplicates:
1. Transcript artifacts: "DI LEO 1 Q", "Di Leo: Objection", "Di Leo 17"
2. OCR/typo variants: "Arthur Jenkins", "Anhur Jenkins", "Arthur Jenidns"

Strategy:
1. Strip transcript artifacts (numbers, Q/A notation, speech attribution)
2. Normalize to canonical form
3. Group by similarity (handles OCR/typos)
4. Keep the cleanest, most frequent variant

Example:
    Input:  ["DI LEO 1 Q", "DI LEO 2", "DI LEO: Objection", "Diana Di Leo"]
    Output: ["Di Leo", "Diana Di Leo"]
"""

import re
from difflib import SequenceMatcher

from src.config import NAME_SIMILARITY_THRESHOLD
from src.core.vocabulary.canonical_scorer import create_canonical_scorer
from src.core.vocabulary.name_regularizer import _load_known_words
from src.core.vocabulary.person_utils import is_person_entry
from src.core.vocabulary.term_sources import TermSources
from src.logging_config import debug_log

# Patterns to strip from person names (transcript artifacts)
TRANSCRIPT_ARTIFACT_PATTERNS = [
    # Q/A notation: "DI LEO 1 Q", "SMITH 2 A", "JONES Q"
    r"\s+\d*\s*[QA](?:\s+|$)",
    # Speech attribution: "DI LEO: Objection", "SMITH: Yes"
    r":\s*\w+.*$",
    # Trailing numbers: "Di Leo 17", "Smith 2"
    r"\s+\d+$",
    # Leading numbers with notation: "1 MR", "2 MS"
    r"^\d+\s+(?:MR|MS|MRS|DR)?\s*",
    # Trailing MR/MS with numbers: "Di Leo 1 MR"
    r"\s+\d+\s*(?:MR|MS|MRS|DR)$",
    # "of" artifacts: "DI LEO 1 of"
    r"\s+\d*\s*of$",
    # "Okay" and similar: "DI LEO 1 Q Okay"
    r"\s+(?:Okay|Yes|No|Right)$",
    # Isolated MR/MS/DR at end (without name): "SMITH MR"
    r"\s+(?:MR|MS|MRS|DR)$",
]

# Compiled patterns for efficiency
_ARTIFACT_REGEXES = [re.compile(p, re.IGNORECASE) for p in TRANSCRIPT_ARTIFACT_PATTERNS]


def deduplicate_names(
    terms: list[dict], similarity_threshold: float = NAME_SIMILARITY_THRESHOLD
) -> list[dict]:
    """
    Merge similar Person names based on artifact removal and fuzzy matching.

    Args:
        terms: List of vocabulary term dicts. Expected keys:
            - Term: The term string
            - Is Person: "Yes" or "No" (NER person detection)
            - In-Case Freq: Occurrence count (int)
        similarity_threshold: Minimum similarity ratio for fuzzy grouping (0.0-1.0)

    Returns:
        Deduplicated list with merged frequencies
    """
    if not terms:
        return terms

    # Separate Person terms from others (Session 70: use centralized check)
    person_terms = [t for t in terms if is_person_entry(t)]
    other_terms = [t for t in terms if not is_person_entry(t)]

    if not person_terms:
        return terms

    debug_log(f"[DEDUP] Processing {len(person_terms)} Person terms for deduplication")

    # Phase 1: Clean transcript artifacts and build canonical mapping
    cleaned_terms = []
    for term in person_terms:
        original = term.get("Term", "")
        cleaned = _strip_transcript_artifacts(original)
        cleaned_terms.append(
            {
                "original": term,
                "cleaned": cleaned,
                "normalized": _normalize_name(cleaned),
            }
        )

    # Phase 2: Group by normalized name (exact match after cleaning)
    groups = _group_by_normalized(cleaned_terms)

    # Phase 3: Fuzzy merge groups that are still similar
    merged_groups = _fuzzy_merge_groups(groups, similarity_threshold)

    # Phase 4: Pick canonical entry from each group
    deduplicated = []
    for group in merged_groups:
        canonical = _select_canonical(group)
        deduplicated.append(canonical)

    merged_count = len(person_terms) - len(deduplicated)
    if merged_count > 0:
        debug_log(
            f"[DEDUP] Merged {merged_count} Person variants into {len(deduplicated)} canonical names"
        )

    return other_terms + deduplicated


def _strip_transcript_artifacts(name: str) -> str:
    """
    Remove common transcript artifacts from a person name.

    Args:
        name: Raw person name possibly containing artifacts

    Returns:
        Cleaned name with artifacts removed
    """
    result = name.strip()

    for regex in _ARTIFACT_REGEXES:
        result = regex.sub("", result).strip()

    # If we stripped everything, return original
    if not result:
        return name.strip()

    return result


def _normalize_name(name: str) -> str:
    """
    Normalize a name for comparison.

    - Title case
    - Single spaces
    - Strip punctuation except hyphens and apostrophes

    Args:
        name: Cleaned name

    Returns:
        Normalized name for grouping
    """
    # Remove punctuation except hyphens and apostrophes
    normalized = re.sub(r"[^\w\s\-']", "", name)
    # Collapse whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()
    # Title case
    normalized = normalized.title()
    return normalized


def _group_by_normalized(cleaned_terms: list[dict]) -> dict[str, list[dict]]:
    """
    Group terms by their normalized form.

    Args:
        cleaned_terms: List of dicts with 'original', 'cleaned', 'normalized' keys

    Returns:
        Dict mapping normalized name to list of term entries
    """
    groups = {}
    for entry in cleaned_terms:
        key = entry["normalized"]
        if key not in groups:
            groups[key] = []
        groups[key].append(entry)
    return groups


def _fuzzy_merge_groups(groups: dict[str, list], threshold: float) -> list[list[dict]]:
    """
    Merge groups whose normalized names are similar.

    Handles OCR variants like "Arthur Jenkins" vs "Anhur Jenkins" that
    survived artifact stripping but are still different keys.

    Session 70 Optimization: Uses first-letter blocking to reduce O(n²) to ~O(n²/26).
    Only compares names that share the same first letter, since OCR rarely
    corrupts the first character. This gives 5-10x speedup for large name lists.

    Args:
        groups: Dict mapping normalized name to entries
        threshold: Similarity threshold

    Returns:
        List of merged groups
    """
    keys = list(groups.keys())

    # Build candidate pairs using first-letter blocking (Session 70)
    # Only compare names that start with the same letter
    candidate_pairs = _build_candidate_pairs(keys)

    merged_indices = set()
    result_groups = []
    for i, key1 in enumerate(keys):
        if i in merged_indices:
            continue

        # Start a new merged group
        current_group = list(groups[key1])
        merged_indices.add(i)

        # Only check pairs that were identified as candidates
        for j in range(i + 1, len(keys)):
            if j in merged_indices:
                continue

            key2 = keys[j]

            # Skip if not a candidate pair (different first letters)
            if (i, j) not in candidate_pairs:
                continue

            similarity = SequenceMatcher(None, key1.lower(), key2.lower()).ratio()
            if similarity >= threshold:
                current_group.extend(groups[key2])
                merged_indices.add(j)
                debug_log(
                    f"[DEDUP] Fuzzy merged '{key2}' into '{key1}' (similarity: {similarity:.2f})"
                )

        result_groups.append(current_group)

    return result_groups


def _build_candidate_pairs(keys: list[str]) -> set[tuple[int, int]]:
    """
    Build candidate pairs using first-letter blocking.

    Only names starting with the same letter are compared.
    This reduces O(n²) comparisons to ~O(n²/26) for uniformly distributed names.

    Session 70 optimization for large vocabularies (500+ names).

    Args:
        keys: List of normalized name strings

    Returns:
        Set of (i, j) index pairs that should be compared
    """
    from collections import defaultdict

    # Group indices by first letter (case-insensitive)
    by_first_letter: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        if key:
            first_char = key[0].lower()
            by_first_letter[first_char].append(i)

    # Build pairs only within same first-letter groups
    pairs: set[tuple[int, int]] = set()
    for indices in by_first_letter.values():
        for a, idx1 in enumerate(indices):
            for idx2 in indices[a + 1 :]:
                pairs.add((idx1, idx2))

    return pairs


def _word_validity_score(name: str) -> float:
    """
    Score a name based on how many of its words are in the known word lists.

    Session 78: Used by _select_canonical to prefer valid dictionary words
    over OCR typos when merging similar names.

    Example:
        "Arthur Jenkins" → 1.0 (both words known)
        "Arthur Jenidns" → 0.5 (only "Arthur" known)
        "Xyzabc Qwerty" → 0.0 (neither word known)

    Args:
        name: A person's name to score

    Returns:
        Float between 0.0 and 1.0 indicating proportion of known words
    """
    if not name:
        return 0.0

    known_words = _load_known_words()
    words = name.lower().split()

    if not words:
        return 0.0

    known_count = sum(1 for word in words if word.strip(".,;:'\"") in known_words)
    return known_count / len(words)


def _term_validity_score(term: str) -> float:
    """
    Alias for _word_validity_score for external imports.

    This is exported for use in tests (test_name_regularizer.py).
    """
    return _word_validity_score(term)


def _select_canonical(group: list[dict]) -> dict:
    """
    Select the canonical entry from a group of similar names.

    Session 78: Now uses CanonicalScorer with branching logic:
    - If exactly ONE variant is fully known (in dictionary) → it wins
    - If ZERO variants are known → weighted score decides (exotic name)
    - If MULTIPLE variants are known → weighted score tiebreaker

    The CanonicalScorer uses confidence-weighted scoring with TermSources:
        score = (sum(confidence * count))^1.1 * ocr_penalty

    Falls back to legacy scoring if no TermSources are present (backward compat).

    NOTE: An ML model trained on user feedback could potentially outperform
    this rules-based approach by learning user-specific preferences and
    document type patterns (medical vs legal terminology, regional names).

    Args:
        group: List of similar name entries, each with:
            - 'original': The vocabulary dict (Term, In-Case Freq, sources?)
            - 'cleaned': Artifact-stripped name string
            - 'normalized': Normalized name for display

    Returns:
        Single canonical term dict with merged frequency and sources
    """
    if len(group) == 1:
        return group[0]["original"]

    freq_key = "In-Case Freq"

    # Check if any entry has TermSources (Session 78 infrastructure)
    has_sources = any(
        "sources" in e["original"] and isinstance(e["original"].get("sources"), TermSources)
        for e in group
    )

    if has_sources:
        # Use new CanonicalScorer with branching logic
        return _select_canonical_with_scorer(group, freq_key)
    else:
        # Legacy path: Use heuristic scoring (backward compatibility)
        return _select_canonical_legacy(group, freq_key)


def _select_canonical_with_scorer(group: list[dict], freq_key: str) -> dict:
    """
    Select canonical using CanonicalScorer (Session 78).

    Uses dictionary presence and confidence-weighted scoring to select
    the most likely correct spelling from a group of similar variants.

    Args:
        group: List of similar name entries
        freq_key: Key for frequency field

    Returns:
        Canonical term dict with merged frequency and sources
    """
    # Build variants list for CanonicalScorer
    # Use the cleaned/normalized name as Term for comparison
    variants = []
    for entry in group:
        original = entry["original"]
        variant = {
            "Term": entry["normalized"],  # Use normalized for canonical selection
            "sources": original.get("sources"),
            "In-Case Freq": original.get(freq_key, 1),
            "_original_entry": entry,  # Keep reference to full entry
        }
        # Create legacy sources if missing
        if variant["sources"] is None:
            variant["sources"] = TermSources.create_legacy(
                variant["In-Case Freq"], original.get("source_doc_confidence", 0.85)
            )
        variants.append(variant)

    # Use CanonicalScorer for selection
    scorer = create_canonical_scorer()
    canonical_variant = scorer.select_canonical(variants)

    # Build the output dict
    best_entry = canonical_variant.get("_original_entry", group[0])
    canonical = best_entry["original"].copy()

    # Use the canonical term (what the scorer decided)
    canonical["Term"] = canonical_variant["Term"]
    canonical[freq_key] = canonical_variant["In-Case Freq"]

    # Include merged sources
    if "sources" in canonical_variant:
        canonical["sources"] = canonical_variant["sources"]

    # Update internal key if present
    if "in_case_freq" in canonical:
        canonical["in_case_freq"] = canonical[freq_key]

    # Log what we merged
    if len(group) > 1:
        merged_terms = [
            e["original"].get("Term", "") for e in group if e["normalized"] != canonical["Term"]
        ]
        if merged_terms:
            debug_log(
                f"[DEDUP] Merged {len(merged_terms)} variants into "
                f"'{canonical['Term']}': {merged_terms[:5]}"
                f"{'...' if len(merged_terms) > 5 else ''}"
            )

    return canonical


def _select_canonical_legacy(group: list[dict], freq_key: str) -> dict:
    """
    Legacy canonical selection using heuristic scoring.

    Kept for backward compatibility when TermSources aren't available.
    Uses word validity, casing, length, and frequency to score candidates.

    Session 78: This will be phased out once per-document tracking is
    fully integrated into the pipeline.

    Args:
        group: List of similar name entries
        freq_key: Key for frequency field

    Returns:
        Canonical term dict with merged frequency
    """
    total_freq = sum(e["original"].get(freq_key, 1) for e in group)

    # Score each candidate
    def score_candidate(entry):
        cleaned = entry["cleaned"]
        original = entry["original"]

        score = 0

        # Strongly prefer valid dictionary words over typos
        validity_score = _word_validity_score(cleaned)
        score += validity_score * 50

        # Prefer names that aren't all caps (more readable)
        if cleaned != cleaned.upper():
            score += 10

        # Prefer names with proper casing (mixed case)
        if any(c.isupper() for c in cleaned) and any(c.islower() for c in cleaned):
            score += 5

        # Prefer shorter cleaned names (less artifact residue)
        if len(cleaned) >= 3:
            score += max(0, 20 - len(cleaned))

        # Prefer higher frequency original
        score += min(original.get(freq_key, 0), 10)

        return score

    # Sort by score descending
    sorted_group = sorted(group, key=score_candidate, reverse=True)

    # Build canonical entry
    best = sorted_group[0]
    canonical = best["original"].copy()

    canonical["Term"] = best["normalized"]
    canonical[freq_key] = total_freq

    if "in_case_freq" in canonical:
        canonical["in_case_freq"] = total_freq

    # Log what we merged
    if len(group) > 1:
        variants = [e["original"].get("Term", "") for e in sorted_group[1:]]
        debug_log(
            f"[DEDUP] Merged {len(variants)} variants into "
            f"'{canonical['Term']}': {variants[:5]}"
            f"{'...' if len(variants) > 5 else ''}"
        )

    return canonical


def find_potential_duplicates(terms: list[dict]) -> dict[str, str]:
    """
    Find Person names where one is a word-subset of another.

    Detects cases like "Antonio Vargas" being a subset of "Antonio Fernandez Vargas"
    where first and last names match but middle name(s) differ.

    These are NOT auto-merged (could be different people) but flagged for user review.

    Args:
        terms: List of vocabulary term dicts with "Term" and "Is Person" keys

    Returns:
        Dict mapping shorter_term -> longer_term for potential duplicates
        e.g., {"Antonio Vargas": "Antonio Fernandez Vargas"}
    """
    # Get Person terms only
    person_terms = [t for t in terms if is_person_entry(t)]

    if len(person_terms) < 2:
        return {}

    potential_duplicates: dict[str, str] = {}

    # Build word sets for each person name
    name_data = []
    for term in person_terms:
        name = term.get("Term", "").strip()
        if not name:
            continue
        words = name.lower().split()
        if len(words) < 2:
            continue  # Skip single-word names
        name_data.append(
            {
                "term": name,
                "words": set(words),
                "first": words[0],
                "last": words[-1],
                "word_count": len(words),
            }
        )

    # Compare pairs looking for subset relationships
    for i, a in enumerate(name_data):
        for b in name_data[i + 1 :]:
            # Check if first and last names match
            if a["first"] != b["first"] or a["last"] != b["last"]:
                continue

            # Check if one is a proper subset of the other
            if a["words"] < b["words"]:
                # a is subset of b (a is shorter)
                potential_duplicates[a["term"]] = b["term"]
                debug_log(
                    f"[DEDUP] Potential duplicate: '{a['term']}' may be same as '{b['term']}'"
                )
            elif b["words"] < a["words"]:
                # b is subset of a (b is shorter)
                potential_duplicates[b["term"]] = a["term"]
                debug_log(
                    f"[DEDUP] Potential duplicate: '{b['term']}' may be same as '{a['term']}'"
                )

    if potential_duplicates:
        debug_log(f"[DEDUP] Found {len(potential_duplicates)} potential duplicate name(s)")

    return potential_duplicates
