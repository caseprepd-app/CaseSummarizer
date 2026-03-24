"""
Tests for pattern_filter.py — PatternFilter class and pre-built filters.

Covers MatchMethod modes, case sensitivity, and all public filter functions.
"""

from src.core.utils.pattern_filter import (
    ADDRESS_FILTER,
    CASE_CITATION_FILTER,
    DOCUMENT_FRAGMENT_FILTER,
    GEOGRAPHIC_CODE_FILTER,
    OCR_ERROR_FILTER,
    VARIATION_FILTER,
    MatchMethod,
    PatternFilter,
    is_valid_acronym,
    matches_entity_filter,
    matches_token_filter,
)

# =========================================================================
# PatternFilter core behavior
# =========================================================================


class TestPatternFilter:
    """Tests for the PatternFilter dataclass."""

    def test_search_mode_finds_anywhere(self):
        """SEARCH matches pattern anywhere in the string."""
        pf = PatternFilter(patterns=(r"hello",), method=MatchMethod.SEARCH)
        assert pf.matches("say hello world")
        assert not pf.matches("goodbye world")

    def test_match_mode_only_at_start(self):
        """MATCH only matches at the start of the string."""
        pf = PatternFilter(patterns=(r"hello",), method=MatchMethod.MATCH)
        assert pf.matches("hello world")
        assert not pf.matches("say hello")

    def test_fullmatch_requires_entire_string(self):
        """FULLMATCH requires the entire string to match."""
        pf = PatternFilter(patterns=(r"hello",), method=MatchMethod.FULLMATCH)
        assert pf.matches("hello")
        assert not pf.matches("hello world")

    def test_case_insensitive_default(self):
        """Matching is case-insensitive by default."""
        pf = PatternFilter(patterns=(r"hello",))
        assert pf.matches("HELLO")
        assert pf.matches("Hello")

    def test_case_sensitive_when_set(self):
        """Case-sensitive matching rejects wrong case."""
        pf = PatternFilter(patterns=(r"Hello",), case_sensitive=True)
        assert pf.matches("Hello World")
        assert not pf.matches("hello world")

    def test_multiple_patterns_any_match(self):
        """Returns True if ANY pattern matches."""
        pf = PatternFilter(patterns=(r"cat", r"dog"))
        assert pf.matches("I have a cat")
        assert pf.matches("I have a dog")
        assert not pf.matches("I have a fish")

    def test_empty_patterns_never_match(self):
        """Empty pattern tuple never matches anything."""
        pf = PatternFilter(patterns=())
        assert not pf.matches("anything")

    def test_regex_special_chars(self):
        """Regex patterns work with special characters."""
        pf = PatternFilter(patterns=(r"\d{3}-\d{4}",))
        assert pf.matches("Call 555-1234")
        assert not pf.matches("no numbers")


# =========================================================================
# Pre-built NER filters
# =========================================================================


class TestVariationFilter:
    """Tests for VARIATION_FILTER (word variations)."""

    def test_possessives(self):
        """Catches possessives like plaintiff's."""
        assert VARIATION_FILTER.matches("plaintiff's")

    def test_parenthetical_plurals(self):
        """Catches plaintiff(s) format."""
        assert VARIATION_FILTER.matches("plaintiff(s)")

    def test_hyphenated(self):
        """Catches hyphenated like pre-trial."""
        assert VARIATION_FILTER.matches("pre-trial")

    def test_normal_word_not_matched(self):
        """Normal words don't trigger variation filter."""
        assert not VARIATION_FILTER.matches("plaintiff")
        assert not VARIATION_FILTER.matches("John Smith")


class TestOCRErrorFilter:
    """Tests for OCR_ERROR_FILTER."""

    def test_digit_letter_mix(self):
        """Catches OCR artifacts with digits mixed into words."""
        assert OCR_ERROR_FILTER.matches("3ohn5mith")
        assert OCR_ERROR_FILTER.matches("Joh3n")

    def test_line_break_artifacts(self):
        """Catches line-break hyphenation artifacts."""
        assert OCR_ERROR_FILTER.matches("Hos-Pital")

    def test_leading_digit(self):
        """Catches leading-digit artifacts."""
        assert OCR_ERROR_FILTER.matches("1earn")

    def test_clean_word_passes(self):
        """Normal words are not flagged as OCR errors."""
        assert not OCR_ERROR_FILTER.matches("Hospital")
        assert not OCR_ERROR_FILTER.matches("plaintiff")


class TestAddressFilter:
    """Tests for ADDRESS_FILTER."""

    def test_street_address(self):
        """Catches street names."""
        assert ADDRESS_FILTER.matches("123 Main Street")
        assert ADDRESS_FILTER.matches("Oak Avenue")

    def test_floor_notation(self):
        """Catches floor notations."""
        assert ADDRESS_FILTER.matches("3rd Floor")

    def test_not_an_address(self):
        """Non-address text passes through."""
        assert not ADDRESS_FILTER.matches("radiculopathy")


class TestDocumentFragmentFilter:
    """Tests for DOCUMENT_FRAGMENT_FILTER."""

    def test_court_headers(self):
        """Catches court header fragments."""
        assert DOCUMENT_FRAGMENT_FILTER.matches("SUPREME COURT")
        assert DOCUMENT_FRAGMENT_FILTER.matches("CIVIL COURT")

    def test_attorney_for(self):
        """Catches attorney designations."""
        assert DOCUMENT_FRAGMENT_FILTER.matches("Attorney for Plaintiff")

    def test_page_numbers(self):
        """Catches page number formats."""
        assert DOCUMENT_FRAGMENT_FILTER.matches("3 of 10")

    def test_normal_text(self):
        """Normal legal text passes through."""
        assert not DOCUMENT_FRAGMENT_FILTER.matches("radiculopathy")


class TestCaseCitationFilter:
    """Tests for CASE_CITATION_FILTER."""

    def test_standard_citation(self):
        """Catches Smith v. Jones format."""
        assert CASE_CITATION_FILTER.matches("Smith v. Jones")
        assert CASE_CITATION_FILTER.matches("Smith v Jones")

    def test_not_a_citation(self):
        """Non-citation text passes."""
        assert not CASE_CITATION_FILTER.matches("John Smith")


class TestGeographicCodeFilter:
    """Tests for GEOGRAPHIC_CODE_FILTER."""

    def test_zip_code(self):
        """Catches ZIP codes."""
        assert GEOGRAPHIC_CODE_FILTER.matches("10001")
        assert GEOGRAPHIC_CODE_FILTER.matches("10001-1234")

    def test_state_zip(self):
        """Catches state + ZIP format."""
        assert GEOGRAPHIC_CODE_FILTER.matches("NY 10001")

    def test_not_a_code(self):
        """Non-code text passes."""
        assert not GEOGRAPHIC_CODE_FILTER.matches("hello")


# =========================================================================
# Composite filter functions
# =========================================================================


class TestMatchesEntityFilter:
    """Tests for matches_entity_filter()."""

    def test_too_short(self):
        """Entities under 3 chars are filtered."""
        assert matches_entity_filter("Hi")
        assert matches_entity_filter("")

    def test_too_long(self):
        """Entities over 60 chars are filtered."""
        assert matches_entity_filter("a" * 61)

    def test_valid_entity_passes(self):
        """Normal entities are not filtered."""
        assert not matches_entity_filter("John Smith")
        assert not matches_entity_filter("radiculopathy")

    def test_address_filtered(self):
        """Address fragments are filtered."""
        assert matches_entity_filter("123 Main Street")

    def test_case_citation_filtered(self):
        """Case citations are filtered."""
        assert matches_entity_filter("Smith v. Jones")

    def test_boilerplate_filtered(self):
        """Legal boilerplate is filtered."""
        assert matches_entity_filter("Verified Complaint")


class TestMatchesTokenFilter:
    """Tests for matches_token_filter()."""

    def test_zip_code_filtered(self):
        """ZIP codes are filtered as tokens."""
        assert matches_token_filter("10001")

    def test_ocr_error_filtered(self):
        """OCR errors are filtered."""
        assert matches_token_filter("3ohn5mith")

    def test_normal_word_passes(self):
        """Normal words pass the token filter."""
        assert not matches_token_filter("plaintiff")
        assert not matches_token_filter("radiculopathy")


class TestIsValidAcronym:
    """Tests for is_valid_acronym()."""

    def test_valid_acronym(self):
        """Standard acronyms are valid."""
        assert is_valid_acronym("FBI")
        assert is_valid_acronym("CEO")
        assert is_valid_acronym("MRI")

    def test_title_abbreviations_excluded(self):
        """Title abbreviations like DR, MR, MD are not valid acronyms."""
        assert not is_valid_acronym("DR")
        assert not is_valid_acronym("MD")
        assert not is_valid_acronym("ESQ")

    def test_lowercase_not_acronym(self):
        """Lowercase strings are not acronyms."""
        assert not is_valid_acronym("hello")
        assert not is_valid_acronym("mr")

    def test_single_letter_not_acronym(self):
        """Single uppercase letter is not an acronym (needs 2+)."""
        assert not is_valid_acronym("A")

    def test_mixed_case_not_acronym(self):
        """Mixed case is not a valid acronym."""
        assert not is_valid_acronym("Fbi")
