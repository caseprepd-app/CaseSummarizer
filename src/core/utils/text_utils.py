"""
Text Utility Functions

Common text processing functions used across the application.

Includes preprocessing integration for AI summary preparation.
"""

import logging

logger = logging.getLogger(__name__)


def combine_document_texts(
    documents: list[dict],
    include_headers: bool = False,
    separator: str = "\n\n",
    preprocess: bool = True,
) -> str:
    """
    Combine extracted text from multiple documents into a single string.

    Prefers 'preprocessed_text' (already cleaned) over 'extracted_text' (raw).
    Falls back to preprocessing if preprocessed_text is not available.

    Args:
        documents: List of document result dictionaries. Each dict should have
                  'preprocessed_text' or 'extracted_text' key (and optionally
                  'filename' for headers).
        include_headers: If True, prefix each document's text with its filename
                        formatted as "--- filename ---"
        separator: String to use between documents (default: double newline)
        preprocess: If True and 'preprocessed_text' not available, apply
                   preprocessing pipeline as fallback. Default True.

    Returns:
        Combined text from all documents. Documents without text are skipped.

    Example:
        >>> docs = [
        ...     {'filename': 'a.pdf', 'extracted_text': 'Hello'},
        ...     {'filename': 'b.pdf', 'extracted_text': 'World'}
        ... ]
        >>> combine_document_texts(docs)
        'Hello\\n\\nWorld'
        >>> combine_document_texts(docs, include_headers=True)
        '--- a.pdf ---\\nHello\\n\\n--- b.pdf ---\\nWorld'
    """
    combined_parts = []

    for doc in documents:
        # Prefer preprocessed_text (already cleaned) over extracted_text (raw)
        text = doc.get("preprocessed_text") or doc.get("extracted_text", "")
        if not text:
            continue

        if include_headers:
            filename = doc.get("filename", "Unknown")
            combined_parts.append(f"--- {filename} ---\n{text}")
        else:
            combined_parts.append(text)

    combined_text = separator.join(combined_parts)

    # Check if any documents had preprocessed_text (already cleaned)
    has_preprocessed = any(doc.get("preprocessed_text") for doc in documents)

    # Only apply preprocessing as fallback if no preprocessed_text was available
    if preprocess and combined_text and not has_preprocessed:
        try:
            from src.core.preprocessing import create_default_pipeline

            pipeline = create_default_pipeline()
            combined_text = pipeline.process(combined_text)
            logger.debug("Preprocessing applied (fallback): %d changes", pipeline.total_changes)
        except ImportError as e:
            logger.debug("Preprocessing not available: %s", e)
        except Exception as e:
            logger.error("Preprocessing error (using raw text): %s", e, exc_info=True)
    elif has_preprocessed:
        logger.debug("Using pre-cleaned text (no additional preprocessing needed)")

    return combined_text


def get_documents_folder() -> str:
    """
    Get the user's Documents folder path (Windows).

    Uses Windows registry to find the actual Documents folder location,
    which may differ from the default if the user has moved it.

    Returns:
        Path to Documents folder, or ~/Documents as fallback
    """
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
        ) as key:
            return winreg.QueryValueEx(key, "Personal")[0]
    except Exception:
        from pathlib import Path

        return str(Path.home() / "Documents")
