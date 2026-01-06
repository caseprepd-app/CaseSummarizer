"""
Role Detection Profiles for Vocabulary Extraction

This module provides profession-specific role/relevance detection for the
vocabulary extractor. Different professions need different contextual information:

- Stenographers: Party roles (plaintiff, defendant), witness types, medical roles
- Lawyers: Opposing counsel, expert witness credentials, judicial contacts
- Paralegals: Case management contacts, filing locations, service addresses

Usage:
    >>> from src.core.vocabulary.role_profiles import StenographerProfile
    >>> profile = StenographerProfile()
    >>> role = profile.detect_person_role("Dr. Martinez", document_text)
    >>> print(role)  # "Treating physician"

To add a new profile:
    1. Create a new class inheriting from RoleDetectionProfile
    2. Define profession-specific pattern constants
    3. Implement detect_person_role() and detect_place_relevance()
    4. Import in vocabulary_extractor.py
"""

import re


class RoleDetectionProfile:
    """
    Base class for profession-specific role detection.

    This allows the vocabulary extractor to be retrofitted for different professions
    without modifying core extraction logic. Each profession subclass defines its own
    patterns and detection priorities.

    Attributes:
        person_patterns: List of (regex_pattern, role_string) tuples for people
        place_patterns: List of (regex_pattern, relevance_string) tuples for places
    """

    def __init__(self):
        """Initialize with empty pattern lists (override in subclass)."""
        self.person_patterns: list[tuple[str, str]] = []
        self.place_patterns: list[tuple[str, str]] = []

    def detect_person_role(self, person_name: str, text: str) -> str:
        """
        Detect a person's role in the case by searching document text.

        Args:
            person_name: The person's name to search for
            text: Full document text to search within

        Returns:
            Role string (e.g., "Plaintiff", "Witness", "Treating physician")
            Returns fallback string if no pattern matches

        Note:
            Override in subclass for profession-specific detection logic.
        """
        raise NotImplementedError("Subclass must implement detect_person_role()")

    def detect_place_relevance(self, place_name: str, text: str) -> str:
        """
        Detect a place/organization's relevance to the case.

        Args:
            place_name: The place/organization name to search for
            text: Full document text to search within

        Returns:
            Relevance string (e.g., "Accident location", "Medical facility")
            Returns fallback string if no pattern matches

        Note:
            Override in subclass for profession-specific detection logic.
        """
        raise NotImplementedError("Subclass must implement detect_place_relevance()")


# ============================================================================
# Stenographer Profile - Court Reporter Deposition Preparation
# ============================================================================

# Person role patterns for stenographers (ordered by specificity)
# Format: (regex_pattern, role_description)
# UPDATED: Changed [a-z]+ to [a-zA-Z]+ to match ALL CAPS names like "ANDY CHOY"
STENOGRAPHER_PERSON_PATTERNS: list[tuple[str, str]] = [
    # Party roles (most specific first)
    (
        r"plaintiff[\'s]?\s+(?:attorney|counsel|lawyer)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
        "Plaintiff attorney",
    ),
    (
        r"defendant[\'s]?\s+(?:attorney|counsel|lawyer)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
        "Defendant attorney",
    ),
    (r"plaintiff\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)", "Plaintiff"),
    (r"defendant\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)", "Defendant"),
    # Medical professionals (specific roles first)
    (
        r"treating\s+(?:physician|doctor)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
        "Treating physician",
    ),
    (r"(?:Dr\.|Doctor)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)", "Medical professional"),
    (
        r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*),?\s+(?:a|an)\s+(?:nurse|physician|surgeon|doctor|therapist)",
        "Medical professional",
    ),
    # Witnesses
    (r"witness\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)", "Witness"),
    (r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+testified", "Witness"),
    (r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:was called as a|called to testify)", "Witness"),
]

# Place/organization relevance patterns for stenographers
# Format: (regex_pattern, relevance_description)
# UPDATED: Stricter patterns to prevent false positives (e.g., "CHOY" matching "CHOY Medical Center")
STENOGRAPHER_PLACE_PATTERNS: list[tuple[str, str]] = [
    # Accident/incident locations (unchanged - already have context)
    (r"accident\s+(?:at|on|near)\s+([A-Z][a-zA-Z\s]+)", "Accident location"),
    (
        r"incident\s+(?:occurred|happened)\s+(?:at|on|near)\s+([A-Z][a-zA-Z\s]+)",
        "Incident location",
    ),
    (r"collision\s+(?:at|on|near)\s+([A-Z][a-zA-Z\s]+)", "Collision location"),
    # Medical facilities - REQUIRE preposition OR 2+ word facility name to avoid false positives
    # This prevents "CHOY" from matching "CHOY Medical Center" (person name vs. place)
    (
        r"(?:at|to|from|near)\s+([A-Z][a-zA-Z\s]+(?:Hospital|Medical Center|Clinic))",
        "Medical facility",
    ),
    (
        r"([A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+)\s+(?:Hospital|Medical Center|Clinic)",
        "Medical facility",
    ),  # 2+ words
    (r"surgery\s+(?:at|performed at)\s+([A-Z][a-zA-Z\s]+)", "Surgery location"),
    (r"treatment\s+(?:at|received at)\s+([A-Z][a-zA-Z\s]+)", "Treatment location"),
    # Law firms - detect before general ORG categorization
    (r"([A-Z][a-zA-Z]+(?:\s+[&,]\s+[A-Z][a-zA-Z]+)+)\s+(?:LAW FIRM|LLP|PC|PLLC)", "Law firm"),
    (r"([A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+)\s+&\s+([A-Z][a-zA-Z]+)", "Law firm"),  # "Smith & Jones"
    (r"THE\s+([A-Z][a-zA-Z\s]+)\s+LAW FIRM", "Law firm"),  # "THE JACOB D. FUCHSBERG LAW FIRM"
    # Employment (unchanged - already have context)
    (r"employed\s+(?:at|by)\s+([A-Z][a-zA-Z\s]+)", "Workplace"),
    (r"works\s+(?:at|for)\s+([A-Z][a-zA-Z\s]+)", "Workplace"),
]


class StenographerProfile(RoleDetectionProfile):
    """
    Stenographer-focused role detection profile.

    Optimized for court reporters preparing for depositions and trials.
    Prioritizes:
    - Party identification (plaintiff, defendant)
    - Attorney/counsel roles
    - Witness classification
    - Medical professional roles (treating physicians, nurses, therapists)
    - Accident/incident locations
    - Medical facility names

    Example Output:
        Person: "Dr. Sarah Martinez" → "Treating physician"
        Person: "John Smith" → "Plaintiff"
        Place: "Lenox Hill Hospital" → "Medical facility"
        Place: "Brooklyn Bridge" → "Accident location"
    """

    def __init__(self):
        """Initialize with stenographer-specific patterns."""
        super().__init__()
        self.person_patterns = STENOGRAPHER_PERSON_PATTERNS
        self.place_patterns = STENOGRAPHER_PLACE_PATTERNS

    def detect_person_role(self, person_name: str, text: str) -> str:
        """
        Detect a person's role in the case using stenographer-relevant patterns.

        Searches document text for contextual clues around the person's name.
        Returns the most specific role found (e.g., "Plaintiff attorney" is more
        specific than "Attorney").

        Args:
            person_name: The person's name (e.g., "Dr. Sarah Martinez")
            text: Full document text to search

        Returns:
            Role string like "Plaintiff", "Treating physician", "Witness"
            Falls back to "Person in case" if no pattern matches
        """
        # Try each pattern in order (most specific first)
        for pattern, role in self.person_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract the name from the pattern match
                matched_name = match.group(1) if match.lastindex >= 1 else None
                if matched_name and self._names_match(person_name, matched_name):
                    return role

        # Fallback: check for title in name itself
        if person_name.startswith("Dr.") or person_name.startswith("Doctor"):
            return "Medical professional"

        return "Person in case"

    def detect_place_relevance(self, place_name: str, text: str) -> str:
        """
        Detect a place/organization's relevance using stenographer patterns.

        Searches for contextual mentions of the place to determine why it matters
        to the case (accident location, medical facility, workplace, etc.).

        Args:
            place_name: The place/organization name (e.g., "Lenox Hill Hospital")
            text: Full document text to search

        Returns:
            Relevance string like "Medical facility", "Accident location"
            Falls back to "Location mentioned in case" if no pattern matches
        """
        for pattern, relevance in self.place_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract the place name from the pattern match
                matched_place = match.group(1) if match.lastindex >= 1 else None
                if matched_place and self._places_match(place_name, matched_place):
                    return relevance

        return "Location mentioned in case"

    def _names_match(self, name1: str, name2: str) -> bool:
        """
        Check if two names match (case-insensitive, partial matching allowed).

        Args:
            name1: First name to compare
            name2: Second name to compare

        Returns:
            True if names substantially overlap
        """
        # Normalize: lowercase, strip titles/punctuation
        norm1 = name1.lower().replace("dr.", "").replace("doctor", "").strip()
        norm2 = name2.lower().replace("dr.", "").replace("doctor", "").strip()

        # Check for substantial overlap (handles "John Smith" vs "Smith, John")
        return norm1 in norm2 or norm2 in norm1

    def _places_match(self, place1: str, place2: str) -> bool:
        """
        Check if two place names match (require 50% token overlap).

        Prevents false positives like:
        - "CHOY" matching "CHOY Medical Center" (person name vs. place)
        - "Smith" matching "Smith Hospital" (surname vs. facility)

        Args:
            place1: First place name
            place2: Second place name

        Returns:
            True if at least 50% of tokens overlap
        """
        # Normalize: lowercase
        norm1 = place1.lower().strip()
        norm2 = place2.lower().strip()

        # Tokenize and compare
        tokens1 = set(norm1.split())
        tokens2 = set(norm2.split())

        if not tokens1 or not tokens2:
            return False

        # Calculate overlap ratio
        overlap = len(tokens1 & tokens2)
        min_tokens = min(len(tokens1), len(tokens2))

        # Require 50% overlap (prevents substring false positives)
        return overlap / min_tokens >= 0.5


# ============================================================================
# Future Profiles (Placeholders for Documentation)
# ============================================================================

# Uncomment and implement when needed:
#
# class LawyerProfile(RoleDetectionProfile):
#     """
#     Lawyer-focused role detection.
#
#     Priorities:
#     - Opposing counsel identification
#     - Expert witness credentials
#     - Judge/mediator names
#     - Key decision-makers in litigation
#     """
#     pass
#
# class ParalegalProfile(RoleDetectionProfile):
#     """
#     Paralegal-focused role detection.
#
#     Priorities:
#     - Case management contacts
#     - Filing locations (courts, clerk offices)
#     - Service addresses
#     - Document sources (who provided what)
#     """
#     pass
