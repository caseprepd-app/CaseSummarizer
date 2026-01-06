"""
Shared Categories Configuration for CaseSummarizer.

Provides unified category definitions used by both NER and LLM extraction.
This ensures consistent categorization across all extraction methods.

Usage:
    from src.categories import get_category_list, get_ner_mapping, is_valid_category

    categories = get_category_list()  # ['Person', 'Place', 'Medical', 'Technical', 'Unknown']
    mapping = get_ner_mapping()       # {'PERSON': 'Person', 'ORG': 'Place', ...}
    is_valid_category('Person')       # True
"""

import json
from functools import lru_cache
from pathlib import Path

from src.logging_config import debug_log

# Path to categories configuration file
CATEGORIES_FILE = Path(__file__).parent.parent / "config" / "categories.json"


class CategoriesError(Exception):
    """Exception raised when categories configuration cannot be loaded."""

    pass


@lru_cache(maxsize=1)
def load_categories() -> dict:
    """
    Load categories from config file.

    Returns:
        dict: Full categories configuration including categories list and mappings.

    Raises:
        CategoriesError: If config file cannot be loaded or is invalid.
    """
    if not CATEGORIES_FILE.exists():
        debug_log(f"[CATEGORIES] Config file not found: {CATEGORIES_FILE}")
        # Return default configuration
        return _get_default_config()

    try:
        with open(CATEGORIES_FILE, encoding="utf-8") as f:
            config = json.load(f)

        # Validate required keys
        if "categories" not in config:
            raise CategoriesError("Missing 'categories' key in config")
        if "ner_mapping" not in config:
            raise CategoriesError("Missing 'ner_mapping' key in config")

        debug_log(f"[CATEGORIES] Loaded {len(config['categories'])} categories from config")
        return config

    except json.JSONDecodeError as e:
        debug_log(f"[CATEGORIES] Invalid JSON in config file: {e}")
        raise CategoriesError(f"Invalid JSON in categories config: {e}") from e
    except Exception as e:
        debug_log(f"[CATEGORIES] Error loading config: {e}")
        raise CategoriesError(f"Error loading categories config: {e}") from e


def _get_default_config() -> dict:
    """Return default configuration if config file is missing."""
    return {
        "categories": [
            {"id": "Person", "description": "Named individuals"},
            {"id": "Place", "description": "Organizations and locations"},
            {"id": "Medical", "description": "Medical/healthcare terms"},
            {"id": "Technical", "description": "Legal and technical terms"},
            {"id": "Unknown", "description": "Unclassified terms"},
        ],
        "ner_mapping": {"PERSON": "Person", "ORG": "Place", "GPE": "Place", "LOC": "Place"},
        "llm_prompt_categories": "Person, Place, Medical, Technical, Unknown",
    }


def get_category_list() -> list[str]:
    """
    Return list of valid category IDs.

    Returns:
        list[str]: List of category ID strings, e.g., ['Person', 'Place', 'Medical', ...]
    """
    config = load_categories()
    return [cat["id"] for cat in config["categories"]]


def get_category_descriptions() -> dict[str, str]:
    """
    Return mapping of category IDs to their descriptions.

    Returns:
        dict[str, str]: Mapping like {'Person': 'Named individuals', ...}
    """
    config = load_categories()
    return {cat["id"]: cat["description"] for cat in config["categories"]}


def get_ner_mapping() -> dict[str, str]:
    """
    Return mapping from spaCy entity types to categories.

    Returns:
        dict[str, str]: Mapping like {'PERSON': 'Person', 'ORG': 'Place', ...}
    """
    config = load_categories()
    return config.get("ner_mapping", {})


def get_llm_prompt_categories() -> str:
    """
    Return comma-separated category list for LLM prompts.

    Returns:
        str: String like "Person, Place, Medical, Technical, Unknown"
    """
    config = load_categories()
    return config.get("llm_prompt_categories", ", ".join(get_category_list()))


def is_valid_category(category: str) -> bool:
    """
    Check if category ID is valid.

    Args:
        category: Category ID to validate.

    Returns:
        bool: True if category is valid, False otherwise.
    """
    return category in get_category_list()


def normalize_category(category: str) -> str:
    """
    Normalize a category string to a valid category ID.

    Handles case-insensitive matching and common variations.

    Args:
        category: Category string to normalize.

    Returns:
        str: Valid category ID, or 'Unknown' if no match found.
    """
    if not category:
        return "Unknown"

    # Direct match
    if is_valid_category(category):
        return category

    # Case-insensitive match
    category_lower = category.lower()
    for valid_cat in get_category_list():
        if valid_cat.lower() == category_lower:
            return valid_cat

    # Common variations
    variations = {
        "name": "Person",
        "names": "Person",
        "people": "Person",
        "individual": "Person",
        "org": "Place",
        "organization": "Place",
        "location": "Place",
        "med": "Medical",
        "health": "Medical",
        "healthcare": "Medical",
        "tech": "Technical",
        "legal": "Technical",
        "other": "Unknown",
    }

    if category_lower in variations:
        return variations[category_lower]

    return "Unknown"


def clear_cache():
    """Clear the cached configuration. Useful for testing or config reloading."""
    load_categories.cache_clear()
