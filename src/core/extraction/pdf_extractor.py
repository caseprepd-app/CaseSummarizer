"""
PDF Text Extraction with Hybrid Voting.

Extracts text from PDF files using a dual-extractor pipeline with word-level
voting to maximize accuracy:

    1. PyMuPDF (primary) - Fast, accurate for most PDFs
    2. pdfplumber (secondary) - Good fallback, different parsing approach
    3. Word-level voting - When extractors disagree, dictionary words win

If text quality is below threshold (60%), falls back to OCR.

Example usage:
    >>> from src.core.extraction.dictionary_utils import DictionaryUtils
    >>> dictionary = DictionaryUtils()
    >>> extractor = PDFExtractor(dictionary)
    >>> result = extractor.extract(Path("document.pdf"))
    >>> print(f"Method: {result['method']}, Confidence: {result['confidence']}%")
    Method: hybrid_voting, Confidence: 85%
"""

from difflib import SequenceMatcher
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from src.config import DEBUG_MODE, MIN_DICTIONARY_CONFIDENCE
from src.logging_config import Timer, debug, error

from .dictionary_utils import DictionaryUtils


class PDFExtractor:
    """
    Extracts text from PDF files using hybrid dual-extractor voting.

    Uses PyMuPDF as primary extractor and pdfplumber as secondary. When both
    succeed, word-level voting reconciles differences by preferring dictionary
    words over OCR errors.

    Attributes:
        dictionary: DictionaryUtils instance for word validation
    """

    def __init__(self, dictionary: DictionaryUtils):
        """
        Initialize the PDF extractor.

        Args:
            dictionary: DictionaryUtils instance for word validation during voting
        """
        self.dictionary = dictionary

    def extract(self, file_path: Path) -> dict:
        """
        Extract text from a PDF file using hybrid extraction.

        Pipeline:
            1. Try PyMuPDF extraction (primary)
            2. Try pdfplumber extraction (secondary)
            3. Reconcile with word-level voting if both succeed
            4. Return result with method and confidence

        Args:
            file_path: Path to the PDF file

        Returns:
            Dict with keys:
                - text: Extracted text (or None if failed)
                - page_count: Number of pages
                - method: 'hybrid_voting', 'pymupdf_only', 'pdfplumber_only'
                - confidence: Dictionary confidence percentage
                - needs_ocr: True if quality too low for digital extraction
                - error: Error type if extraction failed

        Example:
            >>> extractor = PDFExtractor(DictionaryUtils())
            >>> result = extractor.extract(Path("scan.pdf"))
            >>> if result['needs_ocr']:
            ...     print("Falling back to OCR")
        """
        debug(f"Processing PDF: {file_path.name}")

        # Step 1: Try PyMuPDF extraction (primary)
        with Timer("PyMuPDF text extraction"):
            primary_text, page_count, primary_error = self._extract_pymupdf(file_path)

        # Step 2: Try pdfplumber extraction (secondary)
        with Timer("pdfplumber text extraction"):
            secondary_text, secondary_page_count, secondary_error = self._extract_pdfplumber(
                file_path
            )

        # Use whichever page count we got
        page_count = page_count or secondary_page_count

        # Step 3: Determine extraction method based on what succeeded
        text = None
        method = None

        if primary_text and secondary_text:
            # Both succeeded - reconcile with word-level voting
            with Timer("Word-level voting reconciliation"):
                text = self.reconcile_extractions(primary_text, secondary_text)
            method = "hybrid_voting"
            debug("Using hybrid extraction with word-level voting")

        elif primary_text:
            # Only PyMuPDF succeeded
            text = primary_text
            method = "pymupdf_only"
            debug(f"Using PyMuPDF only (pdfplumber failed: {secondary_error})")

        elif secondary_text:
            # Only pdfplumber succeeded
            text = secondary_text
            method = "pdfplumber_only"
            debug(f"Using pdfplumber only (PyMuPDF failed: {primary_error})")

        else:
            # Both failed
            error_type = primary_error or secondary_error or "unknown"
            return {
                "text": None,
                "page_count": page_count,
                "method": None,
                "confidence": 0,
                "needs_ocr": False,
                "error": error_type,
            }

        # Step 4: Check text quality
        with Timer("Dictionary confidence check"):
            confidence = self.dictionary.calculate_confidence(text)
        debug(f"Dictionary confidence: {confidence:.1f}% (method: {method})")

        # Determine if OCR fallback is needed
        needs_ocr = confidence <= MIN_DICTIONARY_CONFIDENCE or len(text) <= 1000

        if not needs_ocr:
            debug(f"Using {method} extraction")

        return {
            "text": text,
            "page_count": page_count,
            "method": method,
            "confidence": int(confidence),
            "needs_ocr": needs_ocr,
            "error": None,
        }

    def _extract_pymupdf(self, file_path: Path) -> tuple[str | None, int, str | None]:
        """
        Extract text from PDF using PyMuPDF (fitz).

        PyMuPDF is faster and more accurate than pdfplumber for most PDFs.
        Used as the primary extractor in hybrid mode.

        Args:
            file_path: Path to the PDF file

        Returns:
            Tuple of (text, page_count, error_type) where error_type is None
            on success, or one of: 'password', 'corrupted', 'empty', 'unknown'
        """
        try:
            text = ""
            page_count = 0

            doc = fitz.open(file_path)
            page_count = len(doc)
            debug(f"PyMuPDF: PDF has {page_count} pages")

            if page_count == 0:
                doc.close()
                error("PyMuPDF: PDF has no pages")
                return None, 0, "empty"

            for i, page in enumerate(doc, 1):
                if DEBUG_MODE and i % 10 == 0:
                    debug(f"PyMuPDF: Extracting page {i}/{page_count}")

                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"

            doc.close()
            return text, page_count, None

        except fitz.FileDataError as e:
            error_msg = str(e).lower()
            if "password" in error_msg or "encrypted" in error_msg:
                error("PyMuPDF: PDF is password-protected or encrypted")
                return None, 0, "password"
            else:
                error("PyMuPDF: PDF file appears to be corrupted")
                return None, 0, "corrupted"

        except Exception as e:
            error_msg = str(e).lower()
            if "permission" in error_msg:
                error("PyMuPDF: Permission denied when accessing PDF")
                return None, 0, "permission"
            else:
                error(f"PyMuPDF: Failed to extract PDF text: {e!s}", exc_info=True)
                return None, 0, "unknown"

    def _extract_pdfplumber(self, file_path: Path) -> tuple[str | None, int, str | None]:
        """
        Extract text from PDF using pdfplumber.

        pdfplumber uses a different parsing approach than PyMuPDF and can
        succeed where PyMuPDF fails (and vice versa).

        Args:
            file_path: Path to the PDF file

        Returns:
            Tuple of (text, page_count, error_type) where error_type is None
            on success, or one of: 'password', 'corrupted', 'empty', 'unknown'
        """
        try:
            text = ""
            page_count = 0

            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                debug(f"pdfplumber: PDF has {page_count} pages")

                if page_count == 0:
                    error("pdfplumber: PDF has no pages")
                    return None, 0, "empty"

                for i, page in enumerate(pdf.pages, 1):
                    if DEBUG_MODE and i % 10 == 0:
                        debug(f"pdfplumber: Extracting page {i}/{page_count}")

                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            return text, page_count, None

        except Exception as e:
            error_msg = str(e).lower()

            if "password" in error_msg or "encrypted" in error_msg:
                error("pdfplumber: PDF is password-protected or encrypted")
                return None, 0, "password"
            elif "damaged" in error_msg or "corrupt" in error_msg or "invalid" in error_msg:
                error("pdfplumber: PDF file appears to be corrupted or damaged")
                return None, 0, "corrupted"
            elif "permission" in error_msg:
                error("pdfplumber: Permission denied when accessing PDF")
                return None, 0, "permission"
            else:
                error(f"pdfplumber: Failed to extract PDF text: {e!s}", exc_info=True)
                return None, 0, "unknown"

    def reconcile_extractions(self, primary_text: str, secondary_text: str) -> str:
        """
        Reconcile two PDF extractions using word-level voting.

        When both extractors succeed, this method aligns their outputs word-by-word
        and picks the better word at each position:
            - If words match: keep them
            - If words differ and one is in dictionary: use dictionary word
            - If both valid or both invalid: use primary (PyMuPDF)

        This catches OCR-like errors where one extractor misreads a character
        (e.g., "tbe" vs "the").

        Args:
            primary_text: Text from PyMuPDF (preferred when tied)
            secondary_text: Text from pdfplumber (fallback)

        Returns:
            Reconciled text with best words from each extractor

        Example:
            >>> extractor = PDFExtractor(DictionaryUtils())
            >>> # PyMuPDF got "tbe", pdfplumber got "the"
            >>> extractor.reconcile_extractions("tbe quick fox", "the quick fox")
            'the quick fox'
        """
        # Tokenize both texts
        primary_tokens = self.dictionary.tokenize_for_voting(primary_text)
        secondary_tokens = self.dictionary.tokenize_for_voting(secondary_text)

        if not primary_tokens:
            return secondary_text
        if not secondary_tokens:
            return primary_text

        # Use SequenceMatcher to align the two token sequences
        matcher = SequenceMatcher(None, primary_tokens, secondary_tokens)
        result_tokens = []
        corrections_made = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                # Words match - keep primary
                result_tokens.extend(primary_tokens[i1:i2])

            elif tag == "replace":
                # Words differ - vote on each pair
                primary_chunk = primary_tokens[i1:i2]
                secondary_chunk = secondary_tokens[j1:j2]

                # Align chunks (may be different lengths)
                max_len = max(len(primary_chunk), len(secondary_chunk))
                for k in range(max_len):
                    p_word = primary_chunk[k] if k < len(primary_chunk) else ""
                    s_word = secondary_chunk[k] if k < len(secondary_chunk) else ""

                    if not p_word:
                        result_tokens.append(s_word)
                    elif not s_word:
                        result_tokens.append(p_word)
                    else:
                        # Both have words - vote
                        p_valid = self.dictionary.is_valid_word(p_word)
                        s_valid = self.dictionary.is_valid_word(s_word)

                        if p_valid and not s_valid:
                            result_tokens.append(p_word)
                        elif s_valid and not p_valid:
                            result_tokens.append(s_word)
                            corrections_made += 1
                            debug(f"[VOTING] '{p_word}' -> '{s_word}' (dictionary correction)")
                        else:
                            # Both valid or both invalid - use primary
                            result_tokens.append(p_word)

            elif tag == "delete":
                # Words only in primary - keep them
                result_tokens.extend(primary_tokens[i1:i2])

            elif tag == "insert":
                # Words only in secondary - only add if they're real words
                for token in secondary_tokens[j1:j2]:
                    if len(token) > 1 and self.dictionary.is_valid_word(token):
                        result_tokens.append(token)

        if corrections_made > 0:
            debug(f"[VOTING] Made {corrections_made} dictionary corrections")

        return " ".join(result_tokens)
