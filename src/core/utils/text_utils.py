"""
Text Utility Functions

Common text processing functions used across the application.

Includes preprocessing integration for AI summary preparation.
"""

from src.logging_config import debug_log


def combine_document_texts(
    documents: list[dict],
    include_headers: bool = False,
    separator: str = "\n\n",
    preprocess: bool = True,
) -> str:
    """
    Combine extracted text from multiple documents into a single string.

    Optionally applies smart preprocessing to clean text before AI summarization.

    Args:
        documents: List of document result dictionaries. Each dict should have
                  'extracted_text' key (and optionally 'filename' for headers).
        include_headers: If True, prefix each document's text with its filename
                        formatted as "--- filename ---"
        separator: String to use between documents (default: double newline)
        preprocess: If True, apply preprocessing pipeline (line number removal,
                   header/footer removal, Q/A conversion). Default True.

    Returns:
        Combined text from all documents. Documents without 'extracted_text'
        or with empty text are skipped.

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
        text = doc.get("extracted_text", "")
        if not text:
            continue

        if include_headers:
            filename = doc.get("filename", "Unknown")
            combined_parts.append(f"--- {filename} ---\n{text}")
        else:
            combined_parts.append(text)

    combined_text = separator.join(combined_parts)

    # Apply preprocessing if enabled
    if preprocess and combined_text:
        try:
            from src.core.preprocessing import create_default_pipeline

            pipeline = create_default_pipeline()
            combined_text = pipeline.process(combined_text)
            debug_log(f"[TEXT UTILS] Preprocessing applied: {pipeline.total_changes} changes")
        except ImportError as e:
            debug_log(f"[TEXT UTILS] Preprocessing not available: {e}")
        except Exception as e:
            debug_log(f"[TEXT UTILS] Preprocessing error (using raw text): {e}")

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
