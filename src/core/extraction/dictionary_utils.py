"""
Dictionary Utilities for Text Extraction.

Provides dictionary-based text validation and confidence scoring for the
document extraction pipeline. Used to assess extraction quality and decide
whether OCR fallback is needed.

Example usage:
    >>> utils = TermExtractionHelpers()
    >>> confidence = utils.calculate_confidence("The plaintiff filed a motion.")
    >>> print(f"Text quality: {confidence:.1f}%")
    Text quality: 85.0%

    >>> utils.is_valid_word("plaintiff")
    True
    >>> utils.is_valid_word("xyzabc")
    False
"""

import logging
import re

import nltk
from nltk.corpus import words

logger = logging.getLogger(__name__)


class TermExtractionHelpers:
    """
    Dictionary-based utilities for text validation and quality assessment.

    Loads the NLTK English words corpus and provides methods to:
    - Calculate what percentage of text consists of valid English words
    - Check if individual words are in the dictionary
    - Tokenize text for word-level comparison

    The dictionary confidence score is used to decide whether digital PDF
    extraction succeeded or if OCR fallback is needed (threshold: 60%).

    Attributes:
        english_words: Set of lowercase English words from NLTK corpus
        legal_keywords: Set of common legal document headers (COURT, PLAINTIFF, etc.)
    """

    def __init__(self):
        """
        Initialize TermExtractionHelpers by loading the NLTK words corpus.

        Downloads the corpus if not already available.
        """
        self.english_words: set[str] = set()
        self.legal_keywords: set[str] = set()
        self._load_dictionary()
        self._load_legal_keywords()

    def _load_dictionary(self) -> None:
        """
        Load NLTK English words corpus.

        Downloads automatically if not present. The corpus contains ~235,000
        English words used for dictionary confidence calculation.
        """
        logger.debug("Loading NLTK English words corpus")

        try:
            self.english_words = {word.lower() for word in words.words()}
            logger.debug("Loaded %d English words", len(self.english_words))
        except LookupError:
            logger.warning("NLTK words corpus not found. Downloading...")
            nltk.download("words", quiet=True)
            self.english_words = {word.lower() for word in words.words()}
            logger.debug("Downloaded and loaded %d English words", len(self.english_words))

    def _load_legal_keywords(self) -> None:
        """
        Load legal document keywords for header detection.

        These keywords help identify legal document headers that should be
        preserved during text normalization even if they're short or all-caps.
        """
        # Hardcoded legal keywords: works offline, no network dependency, covers
        # common terms across all US jurisdictions. Extensible via config if needed.
        self.legal_keywords = {
            "COURT",
            "PLAINTIFF",
            "DEFENDANT",
            "APPEARANCES",
            "SUPREME",
            "MOTION",
            "AFFIDAVIT",
            "EXHIBIT",
            "DEPOSITION",
            "TESTIMONY",
            "COMPLAINT",
            "ANSWER",
            "SUMMONS",
            "NOTICE",
            "ORDER",
            "JUDGE",
            "ATTORNEY",
            "COUNSEL",
            "PARTY",
            "ACTION",
        }
        logger.debug("Loaded %d legal keywords", len(self.legal_keywords))

    def calculate_confidence(self, text: str) -> float:
        """
        Calculate what percentage of words are valid English words.

        This is the primary metric for deciding if PDF text extraction succeeded
        or if OCR fallback is needed. A score below 60% typically indicates
        the PDF contains scanned images rather than selectable text.

        Uses a two-tier scoring system:
        - Words in NLTK dictionary: 1.0 (known valid English)
        - Words in Google frequency dataset but not NLTK: 0.5 (uncommon but real)
        - Words in neither: 0.0 (likely garbage/OCR error)

        Args:
            text: Text to analyze

        Returns:
            Confidence percentage (0-100). Higher = more valid English words.

        Example:
            >>> utils = TermExtractionHelpers()
            >>> utils.calculate_confidence("The quick brown fox")
            100.0
            >>> utils.calculate_confidence("xkcd asdf qwerty")
            0.0
        """
        if not text:
            return 0.0

        # Tokenize (simple split on whitespace and punctuation)
        tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())

        if len(tokens) == 0:
            return 0.0

        # Load Google word frequency data (cached after first load)
        from src.core.vocabulary.rarity_filter import _load_scaled_frequencies

        google_words = _load_scaled_frequencies()

        # Score each token:
        # - 1.0 if in NLTK dictionary (known valid English word)
        # - 0.5 if not in NLTK but in Google (uncommon but real word)
        # - 0.0 if in neither (likely garbage/OCR error)
        total_score = 0.0
        for token in tokens:
            if token in self.english_words:
                total_score += 1.0
            elif token in google_words:
                total_score += 0.5
            # else: 0.0 (no addition)

        confidence = (total_score / len(tokens)) * 100
        return confidence

    def is_valid_word(self, word: str) -> bool:
        """
        Check if a word is in the English dictionary.

        Used by word-level voting to decide between PDF extractor outputs
        when PyMuPDF and pdfplumber disagree on a word.

        Args:
            word: Word to check (will be lowercased and stripped of punctuation)

        Returns:
            True if word is in dictionary, False otherwise

        Example:
            >>> utils = TermExtractionHelpers()
            >>> utils.is_valid_word("plaintiff")
            True
            >>> utils.is_valid_word("Plaintiff")  # case-insensitive
            True
            >>> utils.is_valid_word("plaintiff,")  # strips punctuation
            True
            >>> utils.is_valid_word("xyzabc")
            False
        """
        return word.lower().strip(".,;:'\"()[]{}") in self.english_words

    def tokenize_for_voting(self, text: str) -> list[str]:
        """
        Tokenize text into words for voting alignment.

        Preserves punctuation attached to words so text can be reconstructed
        after word-level voting between PDF extractors.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens (words with attached punctuation)

        Example:
            >>> utils = TermExtractionHelpers()
            >>> utils.tokenize_for_voting("Hello, world!")
            ['Hello,', 'world!']
        """
        return text.split()
