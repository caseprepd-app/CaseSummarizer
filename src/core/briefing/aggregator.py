"""
Data Aggregator for Case Briefing Generator.

Merges and deduplicates extracted data from multiple chunks into a
unified structure. This is the REDUCE phase of the Map-Reduce pattern.

Key Features:
- Fuzzy name matching to deduplicate names across chunks
- Text similarity for allegation/defense deduplication
- Case type determination from multiple hints
- Party consolidation (plaintiffs, defendants)
- Source tracking for provenance

The output AggregatedBriefingData is then passed to the NarrativeSynthesizer.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from src.logging_config import debug_log

from .extractor import ChunkExtraction

# Name category priority for merging (higher = preferred)
CATEGORY_PRIORITY = {
    "PARTY": 4,
    "MEDICAL": 3,
    "WITNESS": 2,
    "OTHER": 1,
}

# Transcript artifacts to filter out (these are headers, not people)
TRANSCRIPT_ARTIFACTS = {
    "CROSS",
    "DIRECT",
    "RECROSS",
    "REDIRECT",
    "RE-DIRECT",
    "RE-CROSS",
    "EXAMINATION",
    "CROSS EXAMINATION",
    "DIRECT EXAMINATION",
    "SIR",
    "MA'AM",
    "MADAM",
    "THE COURT",
    "THE WITNESS",
    "THE CLERK",
    "Q",
    "A",
    "BY",
    "MR",
    "MS",
    "MRS",
    "DR",  # Single-letter/title artifacts
}

# Minimum name length to accept (filters OCR garbage)
MIN_NAME_LENGTH = 3


@dataclass
class PersonEntry:
    """
    A person extracted from documents with consolidated information.

    Attributes:
        canonical_name: Best/longest version of the name
        aliases: Other name variations found
        role: Primary role description
        category: Category (PARTY, MEDICAL, WITNESS, OTHER)
        sources: List of source documents where name appeared
    """

    canonical_name: str
    aliases: set = field(default_factory=set)
    role: str = ""
    category: str = "OTHER"
    sources: set = field(default_factory=set)

    def __hash__(self):
        return hash(self.canonical_name)

    def __eq__(self, other):
        if not isinstance(other, PersonEntry):
            return NotImplemented
        return self.canonical_name == other.canonical_name


@dataclass
class AggregatedBriefingData:
    """
    Consolidated data from all document chunks.

    This is the output of the REDUCE phase, ready for narrative synthesis.

    Attributes:
        case_type: Determined case type (e.g., "medical malpractice")
        plaintiffs: List of plaintiff names
        defendants: List of defendant names
        allegations: List of unique allegations
        defenses: List of unique defenses
        people_by_category: Dict mapping category to list of PersonEntry
        key_facts: List of unique key facts
        dates: List of unique dates mentioned
        vocabulary: List of unusual/technical terms for layperson reference
        source_documents: List of all source document names
        extraction_stats: Dict with processing statistics
    """

    case_type: str = ""
    plaintiffs: list[str] = field(default_factory=list)
    defendants: list[str] = field(default_factory=list)
    allegations: list[str] = field(default_factory=list)
    defenses: list[str] = field(default_factory=list)
    people_by_category: dict[str, list[PersonEntry]] = field(default_factory=dict)
    key_facts: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    vocabulary: list[str] = field(default_factory=list)
    source_documents: list[str] = field(default_factory=list)
    extraction_stats: dict = field(default_factory=dict)


class DataAggregator:
    """
    Aggregates extracted data from multiple chunks.

    Handles deduplication and normalization across all chunks:
    - Names: Fuzzy matching with configurable threshold
    - Text: Similarity-based deduplication
    - Case type: Voting from multiple hints

    Example:
        aggregator = DataAggregator(name_similarity_threshold=0.85)
        aggregated = aggregator.aggregate(extractions)
        print(aggregated.case_type)
        print(aggregated.people_by_category["MEDICAL"])
    """

    def __init__(
        self,
        name_similarity_threshold: float | None = None,
        text_similarity_threshold: float | None = None,
    ):
        """
        Initialize the aggregator.

        Args:
            name_similarity_threshold: Min similarity (0-1) to consider names as same person.
                                       Defaults to NAME_SIMILARITY_THRESHOLD from config.
            text_similarity_threshold: Min similarity (0-1) to consider text as duplicate.
                                       Defaults to TEXT_SIMILARITY_THRESHOLD from config.
        """
        from src.config import NAME_SIMILARITY_THRESHOLD, TEXT_SIMILARITY_THRESHOLD

        self.name_similarity_threshold = (
            name_similarity_threshold
            if name_similarity_threshold is not None
            else NAME_SIMILARITY_THRESHOLD
        )
        self.text_similarity_threshold = (
            text_similarity_threshold
            if text_similarity_threshold is not None
            else TEXT_SIMILARITY_THRESHOLD
        )

        debug_log(
            f"[DataAggregator] Initialized: name_sim={self.name_similarity_threshold}, "
            f"text_sim={self.text_similarity_threshold}"
        )

    def aggregate(self, extractions: list[ChunkExtraction]) -> AggregatedBriefingData:
        """
        Aggregate multiple chunk extractions into unified data.

        Args:
            extractions: List of ChunkExtraction objects from the MAP phase

        Returns:
            AggregatedBriefingData with deduplicated, merged information
        """
        if not extractions:
            debug_log("[DataAggregator] No extractions to aggregate")
            return AggregatedBriefingData()

        debug_log(f"[DataAggregator] Aggregating {len(extractions)} chunk extractions")

        # Track sources
        source_docs = list({e.source_document for e in extractions})
        successful_extractions = [e for e in extractions if e.extraction_success]

        # Aggregate each field type
        plaintiffs = self._aggregate_parties(extractions, "plaintiffs")
        defendants = self._aggregate_parties(extractions, "defendants")
        allegations = self._aggregate_text_list(extractions, "allegations")
        defenses = self._aggregate_text_list(extractions, "defenses")
        key_facts = self._aggregate_text_list(extractions, "key_facts")
        dates = self._aggregate_dates(extractions)
        vocabulary = self._aggregate_vocabulary(extractions)
        case_type = self._determine_case_type(extractions)
        people_by_category = self._aggregate_names(extractions)

        # Build stats
        stats = {
            "total_chunks": len(extractions),
            "successful_chunks": len(successful_extractions),
            "source_documents": len(source_docs),
            "total_names_found": sum(len(v) for v in people_by_category.values()),
            "total_allegations": len(allegations),
            "total_defenses": len(defenses),
            "total_vocabulary_terms": len(vocabulary),
        }

        debug_log(f"[DataAggregator] Aggregation complete: {stats}")

        return AggregatedBriefingData(
            case_type=case_type,
            plaintiffs=plaintiffs,
            defendants=defendants,
            allegations=allegations,
            defenses=defenses,
            people_by_category=people_by_category,
            key_facts=key_facts,
            dates=dates,
            vocabulary=vocabulary,
            source_documents=source_docs,
            extraction_stats=stats,
        )

    def _aggregate_parties(
        self,
        extractions: list[ChunkExtraction],
        party_type: str,
    ) -> list[str]:
        """
        Aggregate party names (plaintiffs or defendants).

        Args:
            extractions: All chunk extractions
            party_type: "plaintiffs" or "defendants"

        Returns:
            Deduplicated list of party names
        """
        all_names = []
        for extraction in extractions:
            names = extraction.parties.get(party_type, [])
            all_names.extend(names)

        # Deduplicate with fuzzy matching
        return self._deduplicate_names_simple(all_names)

    def _aggregate_text_list(
        self,
        extractions: list[ChunkExtraction],
        field_name: str,
    ) -> list[str]:
        """
        Aggregate a text list field with deduplication.

        Args:
            extractions: All chunk extractions
            field_name: Field to aggregate (e.g., "allegations", "defenses")

        Returns:
            Deduplicated list of text items
        """
        all_items = []
        for extraction in extractions:
            items = getattr(extraction, field_name, [])
            all_items.extend(items)

        return self._deduplicate_text(all_items)

    def _aggregate_dates(self, extractions: list[ChunkExtraction]) -> list[str]:
        """
        Aggregate and deduplicate dates.

        Args:
            extractions: All chunk extractions

        Returns:
            List of unique date strings
        """
        all_dates = []
        for extraction in extractions:
            all_dates.extend(extraction.dates_mentioned)

        # Simple deduplication (dates are more structured)
        seen = set()
        unique_dates = []
        for date in all_dates:
            normalized = date.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_dates.append(date.strip())

        return unique_dates

    def _aggregate_vocabulary(self, extractions: list[ChunkExtraction]) -> list[str]:
        """
        Aggregate and deduplicate vocabulary terms.

        Collects unusual/technical terms from all chunks for layperson reference.

        Args:
            extractions: All chunk extractions

        Returns:
            List of unique vocabulary terms, sorted alphabetically
        """
        all_terms = []
        for extraction in extractions:
            all_terms.extend(extraction.vocabulary)

        # Case-insensitive deduplication, keeping original capitalization
        seen = set()
        unique_terms = []
        for term in all_terms:
            normalized = term.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_terms.append(term.strip())

        # Sort alphabetically for consistent output
        unique_terms.sort(key=str.lower)

        debug_log(
            f"[DataAggregator] Vocabulary: {len(all_terms)} raw → {len(unique_terms)} unique terms"
        )
        return unique_terms

    def _determine_case_type(self, extractions: list[ChunkExtraction]) -> str:
        """
        Determine case type from all hints using voting.

        Args:
            extractions: All chunk extractions

        Returns:
            Most likely case type string
        """
        all_hints = []
        for extraction in extractions:
            all_hints.extend(extraction.case_type_hints)

        if not all_hints:
            return "unknown"

        # Normalize hints
        normalized = [self._normalize_case_type(h) for h in all_hints]
        normalized = [h for h in normalized if h]

        if not normalized:
            return "unknown"

        # Vote on most common
        counts = Counter(normalized)
        most_common = counts.most_common(1)[0][0]

        debug_log(f"[DataAggregator] Case type voting: {counts.most_common(3)}")
        return most_common

    def _normalize_case_type(self, hint: str) -> str:
        """
        Normalize case type hint to standard form.

        Args:
            hint: Raw case type hint

        Returns:
            Normalized case type string
        """
        hint_lower = hint.lower().strip()

        # Map common variations to standard types
        type_mappings = {
            "medical malpractice": ["med mal", "medical negligence", "medical malpractice"],
            "personal injury": ["pi", "personal injury", "injury"],
            "negligence": ["negligence", "negligent"],
            "wrongful death": ["wrongful death", "death"],
            "breach of contract": ["breach of contract", "contract breach", "breach"],
            "premises liability": ["premises liability", "slip and fall", "premises"],
            "product liability": ["product liability", "defective product", "product defect"],
            "motor vehicle accident": ["mva", "car accident", "motor vehicle", "auto accident"],
        }

        for standard_type, variations in type_mappings.items():
            for variation in variations:
                if variation in hint_lower:
                    return standard_type

        # Return cleaned original if no match
        return hint_lower

    def _aggregate_names(
        self,
        extractions: list[ChunkExtraction],
    ) -> dict[str, list[PersonEntry]]:
        """
        Aggregate and deduplicate names across all chunks.

        Uses fuzzy matching to merge name variations and preserves
        the most informative version as the canonical name.

        Args:
            extractions: All chunk extractions

        Returns:
            Dict mapping category to list of PersonEntry objects
        """
        # Collect all name entries
        all_entries = []
        for extraction in extractions:
            for name_dict in extraction.names_mentioned:
                all_entries.append(
                    {
                        **name_dict,
                        "source": extraction.source_document,
                    }
                )

        if not all_entries:
            return {}

        # Group by fuzzy-matched canonical name
        merged_people: dict[str, PersonEntry] = {}

        filtered_count = 0
        for entry in all_entries:
            name = entry.get("name", "").strip()
            role = entry.get("role", "")
            category = entry.get("category", "OTHER").upper()
            source = entry.get("source", "")

            if not name:
                continue

            # Filter out transcript artifacts and short names
            if self._is_artifact(name):
                filtered_count += 1
                continue

            # Find existing match
            match_key = self._find_name_match(name, merged_people)

            if match_key:
                # Merge with existing
                person = merged_people[match_key]
                person.aliases.add(name)
                person.sources.add(source)

                # Update canonical name if this one is longer/better
                if len(name) > len(person.canonical_name):
                    person.aliases.add(person.canonical_name)
                    person.canonical_name = name

                # Update category if higher priority
                if CATEGORY_PRIORITY.get(category, 0) > CATEGORY_PRIORITY.get(person.category, 0):
                    person.category = category

                # Update role if more informative
                if len(role) > len(person.role):
                    person.role = role
            else:
                # Create new entry
                normalized_key = self._normalize_name(name)
                merged_people[normalized_key] = PersonEntry(
                    canonical_name=name,
                    aliases=set(),
                    role=role,
                    category=category,
                    sources={source} if source else set(),
                )

        # Group by category
        by_category: dict[str, list[PersonEntry]] = {}
        for person in merged_people.values():
            if person.category not in by_category:
                by_category[person.category] = []
            by_category[person.category].append(person)

        # Sort each category by name
        for category in by_category:
            by_category[category].sort(key=lambda p: p.canonical_name)

        debug_log(
            f"[DataAggregator] Aggregated {len(all_entries)} name entries "
            f"into {sum(len(v) for v in by_category.values())} unique people "
            f"(filtered {filtered_count} artifacts)"
        )

        return by_category

    def _is_artifact(self, name: str) -> bool:
        """
        Check if a name is a transcript artifact or invalid.

        Filters out:
        - Transcript headers (CROSS, DIRECT, etc.)
        - Very short names (likely OCR errors)
        - Names that are just titles

        Args:
            name: Name string to check

        Returns:
            True if this should be filtered out
        """
        # Check minimum length
        if len(name) < MIN_NAME_LENGTH:
            return True

        # Check against artifact list (case-insensitive)
        name_upper = name.upper().strip()
        if name_upper in TRANSCRIPT_ARTIFACTS:
            return True

        # Check if name starts with common artifacts
        for artifact in TRANSCRIPT_ARTIFACTS:
            if name_upper.startswith(artifact + " "):
                return True

        # Filter names that are just numbers or punctuation
        alphanumeric = "".join(c for c in name if c.isalnum())
        return bool(not alphanumeric or alphanumeric.isdigit())

    def _find_name_match(
        self,
        name: str,
        existing: dict[str, PersonEntry],
    ) -> str | None:
        """
        Find existing name that matches this one.

        Uses normalized comparison and fuzzy matching.

        Args:
            name: Name to match
            existing: Dict of normalized_name -> PersonEntry

        Returns:
            Matching key or None
        """
        normalized = self._normalize_name(name)

        # Exact normalized match
        if normalized in existing:
            return normalized

        # Fuzzy match against existing
        for key, person in existing.items():
            # Check canonical name
            if self._names_match(name, person.canonical_name):
                return key

            # Check aliases
            for alias in person.aliases:
                if self._names_match(name, alias):
                    return key

        return None

    def _names_match(self, name1: str, name2: str) -> bool:
        """
        Check if two names refer to the same person.

        Uses multiple strategies:
        1. Normalized exact match
        2. Sequence similarity
        3. Name component overlap

        Args:
            name1: First name
            name2: Second name

        Returns:
            True if names likely match
        """
        norm1 = self._normalize_name(name1)
        norm2 = self._normalize_name(name2)

        # Exact normalized match
        if norm1 == norm2:
            return True

        # One is substring of other (e.g., "Smith" vs "Dr. John Smith")
        if norm1 in norm2 or norm2 in norm1:
            # Only if the shorter one is a significant portion
            shorter = min(len(norm1), len(norm2))
            longer = max(len(norm1), len(norm2))
            if shorter > 3 and shorter / longer > 0.4:
                return True

        # Sequence similarity
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        if similarity >= self.name_similarity_threshold:
            return True

        # Check name component overlap
        parts1 = set(norm1.split())
        parts2 = set(norm2.split())
        if parts1 and parts2:
            overlap = len(parts1 & parts2)
            min_parts = min(len(parts1), len(parts2))
            if min_parts > 0 and overlap / min_parts >= 0.5:
                return True

        return False

    def _normalize_name(self, name: str) -> str:
        """
        Normalize name for comparison.

        Removes titles, punctuation, extra spaces, and lowercases.

        Args:
            name: Raw name string

        Returns:
            Normalized name for matching
        """
        # Lowercase
        result = name.lower()

        # Remove common titles
        titles = [
            r"\bdr\.?\b",
            r"\bmr\.?\b",
            r"\bmrs\.?\b",
            r"\bms\.?\b",
            r"\bmd\b",
            r"\bjr\.?\b",
            r"\bsr\.?\b",
            r"\besq\.?\b",
            r"\bii\b",
            r"\biii\b",
        ]
        for title in titles:
            result = re.sub(title, "", result, flags=re.IGNORECASE)

        # Remove punctuation
        result = re.sub(r"[,.'\"()]", "", result)

        # Normalize whitespace
        result = " ".join(result.split())

        return result.strip()

    def _deduplicate_names_simple(self, names: list[str]) -> list[str]:
        """
        Simple name deduplication without role/category tracking.

        Used for plaintiffs/defendants lists where we just need unique names.

        Args:
            names: List of name strings

        Returns:
            Deduplicated list preserving longest version
        """
        if not names:
            return []

        # Group by normalized form
        by_normalized: dict[str, list[str]] = {}
        for name in names:
            if not name or not name.strip():
                continue
            normalized = self._normalize_name(name)
            if normalized:
                if normalized not in by_normalized:
                    by_normalized[normalized] = []
                by_normalized[normalized].append(name.strip())

        # Take longest version of each
        unique = []
        for variants in by_normalized.values():
            # Pick the longest, most complete version
            best = max(variants, key=len)
            unique.append(best)

        return sorted(unique)

    def _deduplicate_text(self, items: list[str]) -> list[str]:
        """
        Deduplicate text items by similarity.

        Removes near-duplicates while preserving the more complete version.

        Args:
            items: List of text strings

        Returns:
            Deduplicated list
        """
        if not items:
            return []

        # Clean and filter
        cleaned = []
        for item in items:
            text = item.strip()
            if text and len(text) > 10:  # Skip very short items
                cleaned.append(text)

        if not cleaned:
            return []

        # Similarity-based deduplication (O(n²) but avoids list.remove() overhead)
        # Use index tracking instead of list.remove() for better performance
        unique_indices = []  # Track indices of items to keep
        unique_lower = []  # Lowercase versions for comparison

        for i, item in enumerate(cleaned):
            item_lower = item.lower()
            is_duplicate = False
            replace_idx = None

            for j, (existing_idx, existing_lower) in enumerate(
                zip(unique_indices, unique_lower, strict=False)
            ):
                existing = cleaned[existing_idx]

                # Check substring relationship
                if item_lower in existing_lower or existing_lower in item_lower:
                    # Keep the longer one
                    if len(item) > len(existing):
                        replace_idx = j
                    is_duplicate = True
                    break

                # Check similarity
                similarity = SequenceMatcher(None, item_lower, existing_lower).ratio()
                if similarity >= self.text_similarity_threshold:
                    # Keep the longer one
                    if len(item) > len(existing):
                        replace_idx = j
                    is_duplicate = True
                    break

            if replace_idx is not None:
                # Replace existing with longer item
                unique_indices[replace_idx] = i
                unique_lower[replace_idx] = item_lower
            elif not is_duplicate:
                unique_indices.append(i)
                unique_lower.append(item_lower)

        return [cleaned[i] for i in unique_indices]
