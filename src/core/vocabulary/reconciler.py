"""
Vocabulary Reconciler: Merges NER and LLM extraction results (Session 45 Update).

This module combines results from multiple extraction methods (NER and LLM)
into a unified output with a "Found By" column indicating which methods
found each term.

Key features:
- Fuzzy matching for term deduplication (handles slight variations)
- Type conflict resolution (NER wins as it's more reliable for entities)
- Sorted output: "Both" first (highest confidence), then alphabetically
- Session 45: Separate reconciliation for People and Vocabulary

Output formats:
    People: Name | Role | Found By | Confidence
    Vocabulary: Term | Type | Found By | Frequency | Definition

Usage:
    from src.core.vocabulary.reconciler import VocabularyReconciler

    reconciler = VocabularyReconciler()

    # Reconcile people (names with roles)
    people = reconciler.reconcile_people(ner_people, llm_people)

    # Reconcile vocabulary (technical terms)
    terms = reconciler.reconcile(ner_terms, llm_terms)

    for person in people:
        print(f"{person.name} ({person.role}) - {person.found_by}")

    for term in terms:
        print(f"{term.term} - {term.found_by}")
"""

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher

from src.categories import normalize_category

logger = logging.getLogger(__name__)

# Type alias for found_by values
# Session 61: Changed from Literal to str to support comma-separated algorithms
FoundBy = str  # e.g., "NER", "LLM", "NER, LLM", "NER, RAKE, BM25"


@dataclass
class ReconciledPerson:
    """
    A person/organization after reconciling NER and LLM results (Session 45).

    Attributes:
        name: Full name of the person or organization
        role: Role in the case (plaintiff, defendant, attorney, witness, etc.)
        found_by: Which method(s) found this person ("Both", "NER", "LLM")
        ner_confidence: Confidence from NER (0 if not found by NER)
        llm_confidence: Confidence from LLM (0 if not found by LLM)
    """

    name: str
    role: str
    found_by: FoundBy
    ner_confidence: float = 0.0
    llm_confidence: float = 0.0

    @property
    def combined_confidence(self) -> float:
        """Calculate combined confidence from methods that found this item."""
        # Session 61: Check for multiple algorithms via comma
        if "," in self.found_by:
            # Average of both, with bonus for agreement
            return min(1.0, (self.ner_confidence + self.llm_confidence) / 2 + 0.1)
        elif "NER" in self.found_by:
            return self.ner_confidence
        else:
            return self.llm_confidence


@dataclass
class ReconciledTerm:
    """
    A term after reconciling NER and LLM results.

    Attributes:
        term: The canonical term text
        type: Category (Person, Place, Medical, Technical, Unknown)
        found_by: Which method(s) found this term ("Both", "NER", "LLM")
        frequency: Total occurrences across all sources
        definition: WordNet definition if available
        ner_confidence: Confidence from NER (0 if not found by NER)
        llm_confidence: Confidence from LLM (0 if not found by LLM)
    """

    term: str
    type: str
    found_by: FoundBy
    frequency: int = 1
    definition: str = ""
    ner_confidence: float = 0.0
    llm_confidence: float = 0.0

    @property
    def combined_confidence(self) -> float:
        """Calculate combined confidence from methods that found this item."""
        # Session 61: Check for multiple algorithms via comma
        if "," in self.found_by:
            # Average of both, with bonus for agreement
            return min(1.0, (self.ner_confidence + self.llm_confidence) / 2 + 0.1)
        elif "NER" in self.found_by:
            return self.ner_confidence
        else:
            return self.llm_confidence


class VocabularyReconciler:
    """
    Reconciles NER and LLM extraction results into unified output.

    The reconciler performs these steps:
    1. Normalize all terms (case-insensitive matching)
    2. Find matches using fuzzy matching (handles variations like "Dr. Smith" vs "Dr Smith")
    3. Merge matched terms, marking them as "Both"
    4. Resolve type conflicts (NER type wins for entity types like Person/Place)
    5. Sort results: "Both" first, then alphabetically by term

    Example:
        reconciler = VocabularyReconciler()

        # ner_terms: list of CandidateTerm from NER
        # llm_terms: list of LLMTerm from LLM extractor
        results = reconciler.reconcile(ner_terms, llm_terms)

        # Results sorted: "Both" terms first (highest confidence)
        for term in results:
            print(f"{term.term}: {term.type} ({term.found_by})")
    """

    def __init__(self, similarity_threshold: float | None = None):
        """
        Initialize the reconciler.

        Args:
            similarity_threshold: Minimum similarity ratio (0-1) for fuzzy matching.
                                 Defaults to NAME_SIMILARITY_THRESHOLD from config.
                                 0.85 catches variations like "Dr. John Smith"
                                 vs "Dr John Smith" or "LLC" vs "L.L.C."
        """
        from src.config import NAME_SIMILARITY_THRESHOLD

        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else NAME_SIMILARITY_THRESHOLD
        )

    def reconcile_people(
        self,
        ner_people: list,  # list[CandidateTerm] with type=Person from NER
        llm_people: list,  # list[LLMPerson] from LLM extractor
    ) -> list[ReconciledPerson]:
        """
        Reconcile NER and LLM people extraction results (Session 45).

        Args:
            ner_people: PERSON entities from NER (CandidateTerm with suggested_type="Person")
            llm_people: People from LLM extractor (LLMPerson objects with role)

        Returns:
            Sorted list of ReconciledPerson objects.
            Sorted by: Found By ("Both" first), then Name alphabetically.
        """
        logger.debug(
            "Starting people reconciliation: %s NER, %s LLM",
            len(ner_people),
            len(llm_people),
        )

        # Build lookup dictionaries
        ner_dict = self._build_people_dict(ner_people, source="NER")
        llm_dict = self._build_people_dict(llm_people, source="LLM")

        results: list[ReconciledPerson] = []
        matched_llm_keys: set[str] = set()

        # Process NER people first
        for ner_key, ner_data in ner_dict.items():
            llm_match_key = self._find_match(ner_key, llm_dict)

            if llm_match_key:
                # Found in both - merge
                llm_data = llm_dict[llm_match_key]
                matched_llm_keys.add(llm_match_key)

                # Use LLM's role if available (more contextual)
                role = llm_data.get("role", "other")
                if role == "other" and ner_data.get("role"):
                    role = ner_data["role"]

                results.append(
                    ReconciledPerson(
                        name=ner_data["name"],  # Use NER's canonical form
                        role=role,
                        found_by="NER, LLM",  # Session 61: List algorithms instead of "Both"
                        ner_confidence=ner_data["confidence"],
                        llm_confidence=llm_data["confidence"],
                    )
                )
            else:
                # NER only
                results.append(
                    ReconciledPerson(
                        name=ner_data["name"],
                        role=ner_data.get("role", "other"),
                        found_by="NER",
                        ner_confidence=ner_data["confidence"],
                        llm_confidence=0.0,
                    )
                )

        # Add LLM-only people
        for llm_key, llm_data in llm_dict.items():
            if llm_key not in matched_llm_keys:
                results.append(
                    ReconciledPerson(
                        name=llm_data["name"],
                        role=llm_data.get("role", "other"),
                        found_by="LLM",
                        ner_confidence=0.0,
                        llm_confidence=llm_data["confidence"],
                    )
                )

        # Sort results: multiple algorithms first, then alphabetically by name
        # Session 61: Sort by comma count (more commas = more algorithms = higher priority)
        sorted_results = sorted(results, key=lambda p: (-p.found_by.count(","), p.name.lower()))

        # Log summary - Session 61: Updated to handle comma-separated found_by
        multi_count = sum(1 for r in sorted_results if "," in r.found_by)
        ner_only = sum(1 for r in sorted_results if r.found_by == "NER")
        llm_only = sum(1 for r in sorted_results if r.found_by == "LLM")

        logger.debug(
            "People complete: %s unique people (Multiple: %s, NER only: %s, LLM only: %s)",
            len(sorted_results),
            multi_count,
            ner_only,
            llm_only,
        )

        return sorted_results

    def _build_people_dict(self, people: list, source: str) -> dict[str, dict]:
        """
        Build lookup dictionary from people list (Session 45).

        Args:
            people: List of person objects (CandidateTerm or LLMPerson)
            source: Source identifier ("NER" or "LLM")

        Returns:
            Dict mapping normalized name key to person data
        """
        result = {}

        for person_obj in people:
            # Handle both CandidateTerm and LLMPerson
            if hasattr(person_obj, "name"):
                # LLMPerson
                name = person_obj.name
                role = getattr(person_obj, "role", "other")
                confidence = getattr(person_obj, "confidence", 0.8)
            elif hasattr(person_obj, "term"):
                # CandidateTerm from NER (PERSON entity)
                name = person_obj.term
                role = "other"  # NER doesn't provide roles
                confidence = getattr(person_obj, "confidence", 0.9)
            else:
                continue

            key = self._normalize_term(name)
            if not key:
                continue

            if key in result:
                # Keep higher confidence version
                if confidence > result[key]["confidence"]:
                    result[key]["confidence"] = confidence
                    result[key]["name"] = name
                    if role != "other":
                        result[key]["role"] = role
            else:
                result[key] = {
                    "name": name,
                    "role": role,
                    "confidence": confidence,
                }

        return result

    def reconcile(
        self,
        ner_terms: list,  # list[CandidateTerm] - avoiding circular import
        llm_terms: list,  # list[LLMTerm] - avoiding circular import
    ) -> list[ReconciledTerm]:
        """
        Reconcile NER and LLM results into unified output.

        Args:
            ner_terms: Terms from NER algorithm (CandidateTerm objects)
            llm_terms: Terms from LLM extractor (LLMTerm objects)

        Returns:
            Sorted list of ReconciledTerm objects.
            Sorted by: Found By ("Both" first), then Term alphabetically.
        """
        logger.debug(
            "Starting reconciliation: %s NER terms, %s LLM terms",
            len(ner_terms),
            len(llm_terms),
        )

        # Build lookup dictionaries
        ner_dict = self._build_term_dict(ner_terms, source="NER")
        llm_dict = self._build_term_dict(llm_terms, source="LLM")

        results: list[ReconciledTerm] = []
        matched_llm_keys: set[str] = set()

        # Process NER terms first (they have priority for type resolution)
        for ner_key, ner_data in ner_dict.items():
            llm_match_key = self._find_match(ner_key, llm_dict)

            if llm_match_key:
                # Found in both - merge
                llm_data = llm_dict[llm_match_key]
                matched_llm_keys.add(llm_match_key)

                results.append(
                    ReconciledTerm(
                        term=ner_data["term"],  # Use NER's canonical form
                        type=self._resolve_type(ner_data["type"], llm_data["type"]),
                        found_by="NER, LLM",  # Session 61: List algorithms instead of "Both"
                        frequency=ner_data["frequency"] + llm_data["frequency"],
                        ner_confidence=ner_data["confidence"],
                        llm_confidence=llm_data["confidence"],
                    )
                )
            else:
                # NER only
                results.append(
                    ReconciledTerm(
                        term=ner_data["term"],
                        type=normalize_category(ner_data["type"]),
                        found_by="NER",
                        frequency=ner_data["frequency"],
                        ner_confidence=ner_data["confidence"],
                        llm_confidence=0.0,
                    )
                )

        # Add LLM-only terms
        for llm_key, llm_data in llm_dict.items():
            if llm_key not in matched_llm_keys:
                results.append(
                    ReconciledTerm(
                        term=llm_data["term"],
                        type=normalize_category(llm_data["type"]),
                        found_by="LLM",
                        frequency=llm_data["frequency"],
                        ner_confidence=0.0,
                        llm_confidence=llm_data["confidence"],
                    )
                )

        # Sort results
        sorted_results = self._sort_results(results)

        # Log summary - Session 61: Updated to handle comma-separated found_by
        multi_count = sum(1 for r in sorted_results if "," in r.found_by)
        ner_only = sum(1 for r in sorted_results if r.found_by == "NER")
        llm_only = sum(1 for r in sorted_results if r.found_by == "LLM")

        logger.debug(
            "Complete: %s unique terms (Multiple: %s, NER only: %s, LLM only: %s)",
            len(sorted_results),
            multi_count,
            ner_only,
            llm_only,
        )

        return sorted_results

    def _build_term_dict(self, terms: list, source: str) -> dict[str, dict]:
        """
        Build lookup dictionary from term list.

        Groups terms by normalized key and aggregates frequency.

        Args:
            terms: List of term objects (CandidateTerm or LLMTerm)
            source: Source identifier ("NER" or "LLM")

        Returns:
            Dict mapping normalized key to term data
        """
        result = {}

        for term_obj in terms:
            # Handle both CandidateTerm and LLMTerm
            if hasattr(term_obj, "suggested_type"):
                # CandidateTerm from NER
                term = term_obj.term
                term_type = term_obj.suggested_type or "Unknown"
                confidence = term_obj.confidence
                frequency = term_obj.frequency
            else:
                # LLMTerm
                term = term_obj.term
                term_type = term_obj.type
                confidence = getattr(term_obj, "confidence", 0.8)
                frequency = 1

            key = self._normalize_term(term)

            if key in result:
                # Aggregate frequency
                result[key]["frequency"] += frequency
                # Keep higher confidence
                if confidence > result[key]["confidence"]:
                    result[key]["confidence"] = confidence
                    result[key]["term"] = term  # Use higher-confidence form
            else:
                result[key] = {
                    "term": term,
                    "type": term_type,
                    "confidence": confidence,
                    "frequency": frequency,
                }

        return result

    def _normalize_term(self, term: str) -> str:
        """
        Normalize term for matching.

        Handles case, punctuation variations, and common abbreviation differences.

        Args:
            term: Raw term text

        Returns:
            Normalized key for matching
        """
        if not term:
            return ""

        # Lowercase
        normalized = term.lower()

        # Remove common punctuation variations
        normalized = normalized.replace(".", "").replace(",", "")

        # Normalize whitespace
        normalized = " ".join(normalized.split())

        return normalized

    def _find_match(self, term_key: str, term_dict: dict[str, dict]) -> str | None:
        """
        Find matching term using fuzzy matching.

        First tries exact match, then fuzzy match above threshold.

        Args:
            term_key: Normalized term key to search for
            term_dict: Dictionary to search in

        Returns:
            Matching key from term_dict, or None if no match
        """
        # Exact match first
        if term_key in term_dict:
            return term_key

        # Fuzzy match
        best_match = None
        best_ratio = 0.0

        for candidate_key in term_dict:
            ratio = SequenceMatcher(None, term_key, candidate_key).ratio()
            if ratio > best_ratio and ratio >= self.similarity_threshold:
                best_ratio = ratio
                best_match = candidate_key

        return best_match

    def _resolve_type(self, ner_type: str | None, llm_type: str) -> str:
        """
        Resolve type conflict between NER and LLM.

        NER wins for Person/Place (entity types) since spaCy is reliable for these.
        LLM wins for Medical/Technical since it has better context understanding.

        Args:
            ner_type: Type from NER (may be None)
            llm_type: Type from LLM

        Returns:
            Resolved type string
        """
        # Normalize both
        ner_type = normalize_category(ner_type) if ner_type else "Unknown"
        llm_type = normalize_category(llm_type)

        # NER is authoritative for entity types
        if ner_type in ("Person", "Place"):
            return ner_type

        # LLM is better for semantic types
        if llm_type in ("Medical", "Technical"):
            return llm_type

        # Default to NER if it has a non-Unknown type
        if ner_type != "Unknown":
            return ner_type

        return llm_type

    def _sort_results(self, terms: list[ReconciledTerm]) -> list[ReconciledTerm]:
        """
        Sort results by algorithm count (most first), then Term alphabetically.

        Session 61: Updated to sort by comma count - more commas means more
        algorithms found the term, indicating higher confidence.

        Args:
            terms: Unsorted list of reconciled terms

        Returns:
            Sorted list with multi-algorithm terms first
        """
        return sorted(terms, key=lambda t: (-t.found_by.count(","), t.term.lower()))

    def to_csv_data(
        self,
        terms: list[ReconciledTerm],
        include_definitions: bool = True,
    ) -> list[dict]:
        """
        Convert reconciled terms to format expected by DynamicOutputWidget.

        Args:
            terms: List of ReconciledTerm objects
            include_definitions: Whether to include Definition column

        Returns:
            List of dicts with CSV column data
        """
        csv_data = []

        for term in terms:
            row = {
                "Term": term.term,
                "Type": term.type,
                "Found By": term.found_by,
                "Frequency": term.frequency,
                "Quality Score": round(term.combined_confidence * 100, 1),
            }

            if include_definitions:
                row["Definition"] = term.definition or ""

            csv_data.append(row)

        return csv_data

    def people_to_csv_data(self, people: list[ReconciledPerson]) -> list[dict]:
        """
        Convert reconciled people to format expected by DynamicOutputWidget (Session 45).

        Args:
            people: List of ReconciledPerson objects

        Returns:
            List of dicts with CSV column data for people table
        """
        csv_data = []

        for person in people:
            row = {
                "Name": person.name,
                "Role": person.role.replace("_", " ").title(),
                "Found By": person.found_by,
                "Confidence": round(person.combined_confidence * 100, 1),
            }
            csv_data.append(row)

        return csv_data

    def combined_to_csv_data(
        self,
        people: list[ReconciledPerson],
        terms: list[ReconciledTerm],
    ) -> list[dict]:
        """
        Combine people and vocabulary into unified table format (Session 45).

        Session 61: Updated to match GUI_DISPLAY_COLUMNS expected keys:
        - "Term" (was "Name/Term")
        - "Is Person" Yes/No (was "Type"/"Category")
        - "Quality Score" (was "Confidence")
        - "Found By" (unchanged)

        Args:
            people: List of ReconciledPerson objects
            terms: List of ReconciledTerm objects

        Returns:
            List of dicts with unified CSV column data matching GUI expectations
        """
        csv_data = []

        # Add people first (they're the primary output)
        for person in people:
            row = {
                "Term": person.name,
                "Is Person": "Yes",  # People are always persons
                "Found By": person.found_by,
                "Quality Score": round(person.combined_confidence * 100, 1),
                # Keep legacy fields for backward compatibility
                "Role/Relevance": person.role.replace("_", " ").title(),
            }
            csv_data.append(row)

        # Add vocabulary terms
        for term in terms:
            # Determine if this term is a person based on type
            is_person = term.type.lower() in ("person", "people", "name")
            row = {
                "Term": term.term,
                "Is Person": "Yes" if is_person else "No",
                "Found By": term.found_by,
                "Quality Score": round(term.combined_confidence * 100, 1),
                # Keep legacy fields for backward compatibility
                "Type": term.type,
            }
            csv_data.append(row)

        return csv_data
