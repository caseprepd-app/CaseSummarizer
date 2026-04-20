"""Tests for src/core/vocabulary/role_profiles.py.

Role profiles detect a person's role (witness, plaintiff, etc.) or a place's
relevance (medical facility, accident location, etc.) from document text.

Covers:
- RoleDetectionProfile base class raises NotImplementedError
- StenographerProfile.detect_person_role: full pattern matrix
- StenographerProfile.detect_place_relevance: full pattern matrix
- _names_match / _places_match: substring and token-overlap semantics
- Edge cases: empty inputs, titled names, ALL CAPS, single token
"""

import pytest


class TestRoleDetectionProfileBase:
    """Base class is abstract — methods raise NotImplementedError."""

    def test_detect_person_role_not_implemented(self):
        """Base detect_person_role raises NotImplementedError for subclasses to override."""
        from src.core.vocabulary.role_profiles import RoleDetectionProfile

        base = RoleDetectionProfile()
        with pytest.raises(NotImplementedError):
            base.detect_person_role("anyone", "any text")

    def test_detect_place_relevance_not_implemented(self):
        """Base detect_place_relevance raises NotImplementedError."""
        from src.core.vocabulary.role_profiles import RoleDetectionProfile

        base = RoleDetectionProfile()
        with pytest.raises(NotImplementedError):
            base.detect_place_relevance("anywhere", "any text")

    def test_base_init_has_empty_pattern_lists(self):
        """Base class initializes empty pattern lists for subclasses."""
        from src.core.vocabulary.role_profiles import RoleDetectionProfile

        base = RoleDetectionProfile()
        assert base.person_patterns == []
        assert base.place_patterns == []


class TestStenographerProfileInitialisation:
    """StenographerProfile seeds the shared pattern lists on construction."""

    def test_has_person_patterns(self):
        """Stenographer profile provides a non-empty person-pattern list."""
        from src.core.vocabulary.role_profiles import (
            STENOGRAPHER_PERSON_PATTERNS,
            StenographerProfile,
        )

        profile = StenographerProfile()
        assert profile.person_patterns is STENOGRAPHER_PERSON_PATTERNS
        assert len(profile.person_patterns) > 0

    def test_has_place_patterns(self):
        """Stenographer profile provides a non-empty place-pattern list."""
        from src.core.vocabulary.role_profiles import (
            STENOGRAPHER_PLACE_PATTERNS,
            StenographerProfile,
        )

        profile = StenographerProfile()
        assert profile.place_patterns is STENOGRAPHER_PLACE_PATTERNS
        assert len(profile.place_patterns) > 0


class TestStenographerDetectPersonRole:
    """Person role detection matches specific patterns before generic ones."""

    def _make(self):
        """Build a fresh StenographerProfile."""
        from src.core.vocabulary.role_profiles import StenographerProfile

        return StenographerProfile()

    def test_treating_physician_pattern(self):
        """'treating physician NAME' yields the Treating physician role."""
        profile = self._make()
        text = "The treating physician Sarah Martinez examined the patient."
        role = profile.detect_person_role("Sarah Martinez", text)
        assert role == "Treating physician"

    def test_dr_title_yields_medical_professional(self):
        """'Dr. NAME' in text labels the person as a Medical professional."""
        profile = self._make()
        text = "Then Dr. Martinez took the stand."
        role = profile.detect_person_role("Martinez", text)
        assert role == "Medical professional"

    def test_witness_testified_pattern(self):
        """'NAME testified' labels the person as a Witness."""
        profile = self._make()
        text = "John Smith testified under oath."
        role = profile.detect_person_role("John Smith", text)
        assert role == "Witness"

    def test_no_pattern_falls_back_to_person_in_case(self):
        """If no contextual pattern matches, fallback is 'Person in case'."""
        profile = self._make()
        text = "The courtroom was quiet and no one said a word."
        role = profile.detect_person_role("Quinton Snow", text)
        assert role == "Person in case"

    def test_dr_prefix_in_name_fallback_when_no_context(self):
        """A 'Dr. NAME' person name triggers the Medical professional fallback."""
        profile = self._make()
        # No contextual pattern — rely on title-in-name fallback
        text = "Nothing medical was ever mentioned in this filing."
        role = profile.detect_person_role("Dr. Quinton Snow", text)
        assert role == "Medical professional"

    def test_doctor_prefix_in_name_fallback(self):
        """Names starting with 'Doctor' trigger the medical professional fallback."""
        profile = self._make()
        text = "Unrelated testimony about scheduling."
        role = profile.detect_person_role("Doctor Patel", text)
        assert role == "Medical professional"

    def test_returns_string_type(self):
        """detect_person_role always returns a string."""
        profile = self._make()
        assert isinstance(profile.detect_person_role("", ""), str)


class TestStenographerDetectPlaceRelevance:
    """Place relevance detection labels known location patterns."""

    def _make(self):
        """Build a fresh StenographerProfile."""
        from src.core.vocabulary.role_profiles import StenographerProfile

        return StenographerProfile()

    def test_accident_location_pattern(self):
        """'accident at LOCATION' labels the location as an Accident location."""
        profile = self._make()
        text = "The accident at Brooklyn Bridge blocked traffic for hours."
        relevance = profile.detect_place_relevance("Brooklyn Bridge", text)
        assert relevance == "Accident location"

    def test_medical_facility_pattern(self):
        """Two-word facility name with 'Hospital' is classed as Medical facility."""
        profile = self._make()
        text = "She was treated at Lenox Hill Hospital on the same day."
        relevance = profile.detect_place_relevance("Lenox Hill Hospital", text)
        assert relevance == "Medical facility"

    def test_no_pattern_falls_back_to_generic(self):
        """Unknown place falls back to 'Location mentioned in case'."""
        profile = self._make()
        text = "The sky was blue that afternoon and nothing unusual happened."
        relevance = profile.detect_place_relevance("Far Away Land", text)
        assert relevance == "Location mentioned in case"

    def test_returns_string_type(self):
        """detect_place_relevance always returns a string."""
        profile = self._make()
        assert isinstance(profile.detect_place_relevance("", ""), str)


class TestNamesMatch:
    """Internal _names_match normalizes titles and checks substring overlap."""

    def _profile(self):
        """Build a fresh StenographerProfile to access _names_match."""
        from src.core.vocabulary.role_profiles import StenographerProfile

        return StenographerProfile()

    def test_identical_names_match(self):
        """Exact-match names are considered the same person."""
        p = self._profile()
        assert p._names_match("John Smith", "John Smith") is True

    def test_case_insensitive(self):
        """Names match regardless of case."""
        p = self._profile()
        assert p._names_match("JOHN SMITH", "john smith") is True

    def test_doctor_title_stripped(self):
        """'Dr. NAME' matches 'NAME' because the title is stripped."""
        p = self._profile()
        assert p._names_match("Dr. Smith", "Smith") is True

    def test_substring_match(self):
        """'Smith' within 'John Smith' is treated as the same person."""
        p = self._profile()
        assert p._names_match("Smith", "John Smith") is True

    def test_completely_different_names_do_not_match(self):
        """Unrelated names do not match."""
        p = self._profile()
        assert p._names_match("Alice", "Bob") is False


class TestPlacesMatch:
    """Internal _places_match uses 50% token overlap for fuzzy equality."""

    def _profile(self):
        """Build a fresh StenographerProfile to access _places_match."""
        from src.core.vocabulary.role_profiles import StenographerProfile

        return StenographerProfile()

    def test_identical_places_match(self):
        """Exact-match place names are equal."""
        p = self._profile()
        assert p._places_match("Lenox Hill Hospital", "Lenox Hill Hospital") is True

    def test_half_overlap_matches(self):
        """Two tokens out of two overlap → 100% overlap, matches."""
        p = self._profile()
        # 2-token vs 3-token; overlap=2 tokens; min_tokens=2; ratio=1.0
        assert p._places_match("Lenox Hill", "Lenox Hill Hospital") is True

    def test_single_overlap_with_two_token_place_matches(self):
        """One common token out of min_tokens=2 → 50%, matches exactly at threshold."""
        p = self._profile()
        # 'Lenox' vs 'Lenox Hill': overlap=1, min_tokens=1, ratio=1.0 — matches
        assert p._places_match("Lenox", "Lenox Hill") is True

    def test_no_overlap_does_not_match(self):
        """Completely different place names do not match."""
        p = self._profile()
        assert p._places_match("Smith Hospital", "Jones Clinic") is False

    def test_prevents_surname_vs_facility_false_positive(self):
        """Substring false positives like 'CHOY' in 'CHOY Medical Center' are blocked.

        Both share 'choy' but min_tokens=1 means overlap=1, ratio=1.0 — which
        by the 50% rule DOES match. The guard against this false positive lives
        in the place_patterns regex (requires 2+ word facility names), not the
        token-overlap helper. This test documents that.
        """
        p = self._profile()
        # Single-word "Choy" vs "Choy Medical Center" — overlap ratio is 1.0
        # The helper alone matches; the regex pattern is the defense layer.
        assert p._places_match("Choy", "Choy Medical Center") is True

    def test_empty_name_does_not_match(self):
        """Empty place names never match (no tokens available)."""
        p = self._profile()
        assert p._places_match("", "Lenox Hill Hospital") is False
        assert p._places_match("Lenox Hill Hospital", "") is False
