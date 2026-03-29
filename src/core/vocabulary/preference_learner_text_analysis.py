"""
Text Analysis Helpers for Vocabulary Meta-Learner

Pure helper functions for text analysis used by feature extraction.
These have no dependencies on other meta_learner modules.

Functions:
- _load_names_datasets(): Load international forenames/surnames
- _max_consonant_run(): Calculate consecutive consonants (gibberish detector)
- _log_rarity_score(): Convert linear rank to log-scaled rarity
"""

import csv
import logging
import math
import threading

logger = logging.getLogger(__name__)

# Module-level cache for names datasets
_forenames_set: set[str] | None = None
_surnames_set: set[str] | None = None
_name_country_counts: dict[str, int] | None = None
_total_countries: int = 0
_names_lock = threading.Lock()


def _load_names_datasets() -> tuple[set[str], set[str]]:
    """
    Load international forenames and surnames datasets.

    Thread-safe with double-check locking pattern.
    Also builds country-count data for geographic spread features.

    Returns:
        Tuple of (forenames_set, surnames_set) - lowercase name sets

    Data files are in data/names/ directory:
    - international_forenames.csv (2,480 names)
    - international_surnames.csv (2,576 names)
    """
    global _forenames_set, _surnames_set, _name_country_counts, _total_countries

    # Fast path: already loaded
    if _forenames_set is not None and _surnames_set is not None:
        return _forenames_set, _surnames_set

    # Slow path: need to load (with lock)
    with _names_lock:
        # Double-check after acquiring lock
        if _forenames_set is not None and _surnames_set is not None:
            return _forenames_set, _surnames_set

        from src.core.paths import get_data_dir

        data_dir = get_data_dir() / "names"
        all_countries: set[str] = set()
        country_sets: dict[str, set[str]] = {}  # name -> set of countries

        # Load forenames
        _forenames_set = set()
        forenames_file = data_dir / "international_forenames.csv"
        if forenames_file.exists():
            try:
                with open(forenames_file, encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        name = row.get("Romanized Name", "").strip().lower()
                        country = row.get("Country", "").strip()
                        if name:
                            _forenames_set.add(name)
                            if country:
                                all_countries.add(country)
                                if name not in country_sets:
                                    country_sets[name] = set()
                                country_sets[name].add(country)
                logger.debug("Loaded %d forenames", len(_forenames_set))
            except Exception as e:
                logger.warning("Error loading forenames: %s", e, exc_info=True)

        # Load surnames
        _surnames_set = set()
        surnames_file = data_dir / "international_surnames.csv"
        if surnames_file.exists():
            try:
                with open(surnames_file, encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        name = row.get("Romanized Name", "").strip().lower()
                        country = row.get("Country", "").strip()
                        if name:
                            _surnames_set.add(name)
                            if country:
                                all_countries.add(country)
                                if name not in country_sets:
                                    country_sets[name] = set()
                                country_sets[name].add(country)
                logger.debug("Loaded %d surnames", len(_surnames_set))
            except Exception as e:
                logger.warning("Error loading surnames: %s", e, exc_info=True)

        # Build country count cache
        _total_countries = len(all_countries) if all_countries else 1
        _name_country_counts = {name: len(countries) for name, countries in country_sets.items()}
        logger.debug(
            "Built country data: %d countries, %d names with country info",
            _total_countries,
            len(_name_country_counts),
        )

        return _forenames_set, _surnames_set


def _get_name_country_data() -> tuple[dict[str, int], int]:
    """
    Get cached name-to-country-count mapping and total country count.

    Must be called after _load_names_datasets() (which builds the cache).

    Returns:
        Tuple of (name_country_counts dict, total_countries int)
        - name_country_counts: lowercase name -> number of countries it appears in
        - total_countries: total distinct countries across all datasets
    """
    # Ensure data is loaded
    _load_names_datasets()
    return _name_country_counts or {}, _total_countries or 1


def _max_consonant_run(text: str) -> int:
    """
    Calculate the maximum run of consecutive consonants in text.

    Real English words rarely have more than 3 consecutive consonants
    (e.g., "strengths" = 3). Gibberish often has longer runs.

    Args:
        text: Text to analyze

    Returns:
        Length of longest consonant run
    """
    consonants = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")
    max_run = 0
    current_run = 0

    for char in text:
        if char in consonants:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    return max_run


def _log_rarity_score(linear_score: float, total_words: int = 333000) -> float:
    """
    Convert linear rank percentile to log-scaled rarity score.

    The frequency dictionary stores rank/total values (linear 0-1 scale where
    0 = most common word). Logarithmic scaling compresses the common end,
    making differences between rare words more meaningful.

    Distinguishes "Comiskey" (rank 96755, rare) from "Clerk" (rank 5435,
    common) -- a binary "in frequency dict" flag loses this information.

    Args:
        linear_score: rank / total_words (0.0 = most common)
        total_words: Size of frequency dataset (default 333000 for Google data)

    Returns:
        Log-scaled score (0.0 = most common, 1.0 = rarest)
        Examples:
            rank=1 (the) → 0.0
            rank=100 → 0.36
            rank=1000 → 0.54
            rank=5435 (Clerk) → 0.68
            rank=96755 (Comiskey) → 0.90
    """
    if linear_score <= 0:
        return 0.0

    rank = int(linear_score * total_words)
    if rank <= 0:
        return 0.0

    return math.log10(rank + 1) / math.log10(total_words + 1)
