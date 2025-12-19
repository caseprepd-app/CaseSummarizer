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
from src.logging_config import debug as debug_log


# Similarity threshold for fuzzy matching (after artifact removal)
NAME_SIMILARITY_THRESHOLD = 0.85

# Patterns to strip from person names (transcript artifacts)
TRANSCRIPT_ARTIFACT_PATTERNS = [
    # Q/A notation: "DI LEO 1 Q", "SMITH 2 A", "JONES Q"
    r'\s+\d*\s*[QA](?:\s+|$)',
    # Speech attribution: "DI LEO: Objection", "SMITH: Yes"
    r':\s*\w+.*$',
    # Trailing numbers: "Di Leo 17", "Smith 2"
    r'\s+\d+$',
    # Leading numbers with notation: "1 MR", "2 MS"
    r'^\d+\s+(?:MR|MS|MRS|DR)?\s*',
    # Trailing MR/MS with numbers: "Di Leo 1 MR"
    r'\s+\d+\s*(?:MR|MS|MRS|DR)$',
    # "of" artifacts: "DI LEO 1 of"
    r'\s+\d*\s*of$',
    # "Okay" and similar: "DI LEO 1 Q Okay"
    r'\s+(?:Okay|Yes|No|Right)$',
    # Isolated MR/MS/DR at end (without name): "SMITH MR"
    r'\s+(?:MR|MS|MRS|DR)$',
]

# Compiled patterns for efficiency
_ARTIFACT_REGEXES = [re.compile(p, re.IGNORECASE) for p in TRANSCRIPT_ARTIFACT_PATTERNS]


def deduplicate_names(terms: list[dict], similarity_threshold: float = NAME_SIMILARITY_THRESHOLD) -> list[dict]:
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

    # Separate Person terms from others (check "Is Person" field)
    person_terms = [t for t in terms if t.get("Is Person") == "Yes"]
    other_terms = [t for t in terms if t.get("Is Person") != "Yes"]

    if not person_terms:
        return terms

    debug_log(f"[DEDUP] Processing {len(person_terms)} Person terms for deduplication")

    # Phase 1: Clean transcript artifacts and build canonical mapping
    cleaned_terms = []
    for term in person_terms:
        original = term.get("Term", "")
        cleaned = _strip_transcript_artifacts(original)
        cleaned_terms.append({
            "original": term,
            "cleaned": cleaned,
            "normalized": _normalize_name(cleaned),
        })

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
        debug_log(f"[DEDUP] Merged {merged_count} Person variants into {len(deduplicated)} canonical names")

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
        result = regex.sub('', result).strip()

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
    normalized = re.sub(r'\s+', ' ', normalized).strip()
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

    Args:
        groups: Dict mapping normalized name to entries
        threshold: Similarity threshold

    Returns:
        List of merged groups
    """
    keys = list(groups.keys())
    merged_indices = set()
    result_groups = []

    for i, key1 in enumerate(keys):
        if i in merged_indices:
            continue

        # Start a new merged group
        current_group = list(groups[key1])
        merged_indices.add(i)

        # Find similar groups to merge
        for j, key2 in enumerate(keys):
            if j <= i or j in merged_indices:
                continue

            similarity = SequenceMatcher(None, key1.lower(), key2.lower()).ratio()
            if similarity >= threshold:
                current_group.extend(groups[key2])
                merged_indices.add(j)
                debug_log(f"[DEDUP] Fuzzy merged '{key2}' into '{key1}' (similarity: {similarity:.2f})")

        result_groups.append(current_group)

    return result_groups


def _select_canonical(group: list[dict]) -> dict:
    """
    Select the canonical entry from a group of similar names.

    Priority:
    1. Cleanest form (shortest cleaned name that isn't just initials)
    2. Highest frequency
    3. Most "proper" casing (has both upper and lower)

    Args:
        group: List of similar name entries

    Returns:
        Single canonical term dict with merged frequency
    """
    if len(group) == 1:
        return group[0]["original"]

    # Calculate total frequency
    freq_key = "In-Case Freq"
    total_freq = sum(e["original"].get(freq_key, 1) for e in group)

    # Score each candidate
    def score_candidate(entry):
        cleaned = entry["cleaned"]
        original = entry["original"]

        score = 0

        # Prefer names that aren't all caps (more readable)
        if cleaned != cleaned.upper():
            score += 10

        # Prefer names with proper casing (mixed case)
        if any(c.isupper() for c in cleaned) and any(c.islower() for c in cleaned):
            score += 5

        # Prefer shorter cleaned names (less artifact residue)
        # But not TOO short (avoid just initials)
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

    # Use the cleaned name as the Term (but title-cased)
    canonical["Term"] = best["normalized"]
    canonical[freq_key] = total_freq

    # Update internal key if present
    if "in_case_freq" in canonical:
        canonical["in_case_freq"] = total_freq

    # Log what we merged
    if len(group) > 1:
        variants = [e["original"].get("Term", "") for e in sorted_group[1:]]
        debug_log(f"[DEDUP] Merged {len(variants)} variants into '{canonical['Term']}': {variants[:5]}{'...' if len(variants) > 5 else ''}")

    return canonical
