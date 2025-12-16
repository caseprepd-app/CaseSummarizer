"""
Shared Tokenizer Module

Provides unified tokenization for all BM25/vocabulary operations.
Ensures consistent text processing across vocabulary extraction and retrieval.

Usage:
    from src.utils.tokenizer import tokenize, TokenizerConfig, STOPWORDS

    # Default tokenization (min_length=3, filter stopwords)
    tokens = tokenize("This is some text with medical terminology.")

    # Custom config for retrieval (no stopword filtering)
    config = TokenizerConfig(filter_stopwords=False, min_length=1)
    tokens = tokenize("What are the plaintiff's claims?", config)
"""

import re
from dataclasses import dataclass, field

# Standard regex pattern preserving hyphens and apostrophes
# Matches words that:
# - Start with a letter
# - Can contain letters, digits, apostrophes, and hyphens
# - End with a letter or digit
# Also matches single letters
TOKEN_PATTERN = re.compile(
    r"\b[a-zA-Z][a-zA-Z0-9'-]*[a-zA-Z0-9]\b|\b[a-zA-Z]\b"
)

# Unified stopwords (merged from all BM25 implementations)
# These are common words that don't carry meaning for search/extraction
STOPWORDS: frozenset[str] = frozenset({
    # Articles
    'a', 'an', 'the',
    # Conjunctions
    'and', 'or', 'but', 'nor', 'so', 'yet',
    # Prepositions
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
    'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under',
    # Be verbs
    'is', 'was', 'are', 'were', 'been', 'be', 'being', 'am',
    # Have verbs
    'have', 'has', 'had', 'having',
    # Do verbs
    'do', 'does', 'did', 'doing',
    # Modal verbs
    'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
    'need', 'dare', 'ought', 'used',
    # Pronouns
    'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'its', 'our', 'their',
    'mine', 'yours', 'hers', 'ours', 'theirs',
    # Demonstratives
    'this', 'that', 'these', 'those',
    # Interrogatives
    'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
    # Quantifiers
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
    'such', 'any', 'many', 'much',
    # Negation/affirmation
    'no', 'not', 'yes',
    # Adverbs
    'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also',
    'now', 'here', 'there', 'then', 'once', 'again', 'further',
    # Common verbs
    'said', 'says', 'say',
    # Other common words
    'if', 'because', 'until', 'while',
})


@dataclass
class TokenizerConfig:
    """
    Configuration for tokenization behavior.

    Attributes:
        min_length: Minimum token length (default 3)
        filter_stopwords: Whether to remove stopwords (default True)
        stopwords: Set of stopwords to filter (default STOPWORDS)
    """
    min_length: int = 3
    filter_stopwords: bool = True
    stopwords: frozenset[str] = field(default_factory=lambda: STOPWORDS)


# Default config singleton for common use case
_DEFAULT_CONFIG = TokenizerConfig()

# Retrieval config (no stopword filtering, min length 1)
RETRIEVAL_CONFIG = TokenizerConfig(
    min_length=1,
    filter_stopwords=False,
)


def tokenize(text: str, config: TokenizerConfig | None = None) -> list[str]:
    """
    Tokenize text into lowercase words.

    Uses consistent regex pattern across all BM25/vocabulary operations.
    Optionally filters stopwords and short tokens.

    Args:
        text: Input text to tokenize
        config: Optional TokenizerConfig. If None, uses default settings
                (min_length=3, filter_stopwords=True)

    Returns:
        List of lowercase word tokens

    Example:
        >>> tokenize("The plaintiff's medical records show...")
        ['plaintiff', 'medical', 'records', 'show']

        >>> tokenize("Who?", RETRIEVAL_CONFIG)
        ['who']
    """
    if config is None:
        config = _DEFAULT_CONFIG

    # Extract words using standard pattern
    words = TOKEN_PATTERN.findall(text.lower())

    # Apply filters
    result = []
    for word in words:
        # Length filter
        if len(word) < config.min_length:
            continue

        # Stopword filter
        if config.filter_stopwords and word in config.stopwords:
            continue

        result.append(word)

    return result


def tokenize_simple(text: str) -> list[str]:
    """
    Simple tokenization for retrieval (no filtering).

    Preserves all words including short ones and stopwords.
    Use this for query tokenization where every word matters.

    Args:
        text: Input text to tokenize

    Returns:
        List of lowercase word tokens
    """
    return tokenize(text, RETRIEVAL_CONFIG)
