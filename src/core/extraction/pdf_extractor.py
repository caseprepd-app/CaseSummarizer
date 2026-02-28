"""
PDF Text Extraction with Hybrid Voting.

Extracts text from PDF files using a dual-extractor pipeline with word-level
voting to maximize accuracy:

    1. PyMuPDF (primary) - Fast, accurate for most PDFs
    2. pdfplumber (secondary) - Good fallback, different parsing approach
    3. Word-level voting - When extractors disagree, dictionary words win

If text quality is below threshold (60%), falls back to OCR.

Example usage:
    >>> from src.core.extraction.dictionary_utils import DictionaryTextValidator
    >>> dictionary = DictionaryTextValidator()
    >>> extractor = PDFExtractor(dictionary)
    >>> result = extractor.extract(Path("document.pdf"))
    >>> print(f"Method: {result['method']}, Confidence: {result['confidence']}%")
    Method: hybrid_voting, Confidence: 85%
"""

import logging
from difflib import SequenceMatcher
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from src.config import MIN_DICTIONARY_CONFIDENCE
from src.logging_config import Timer

from .dictionary_utils import DictionaryTextValidator
from .layout_analyzer import LayoutAnalyzer

logger = logging.getLogger(__name__)


def extract_portfolio_pdf(file_path: Path) -> bytes | None:
    """
    Detect PDF Portfolio/Bundle files and extract the embedded PDF.

    PDF Portfolios are wrapper PDFs (often a 1-page cover sheet) with the
    real document embedded as an attachment. This function finds and returns
    the first embedded .pdf attachment.

    Args:
        file_path: Path to the PDF file

    Returns:
        Raw bytes of the embedded PDF, or None if not a portfolio
    """
    try:
        with fitz.open(file_path) as doc:
            if doc.embfile_count() == 0:
                return None

            logger.info(
                "PDF Portfolio detected: %d embedded file(s) in %s",
                doc.embfile_count(),
                file_path.name,
            )

            # Find first .pdf attachment
            for name in doc.embfile_names():
                info = doc.embfile_info(name)
                embedded_filename = info.get("filename", "")
                if embedded_filename.lower().endswith(".pdf"):
                    logger.info(
                        "Extracting embedded PDF: %s (%d bytes)",
                        embedded_filename,
                        info.get("size", 0),
                    )
                    return doc.embfile_get(name)

            logger.warning(
                "PDF Portfolio has attachments but none are PDFs: %s",
                doc.embfile_names(),
            )
            return None

    except Exception as e:
        logger.debug("Portfolio detection skipped: %s", e)
        return None


class PDFExtractor:
    """
    Extracts text from PDF files using hybrid dual-extractor voting.

    Uses PyMuPDF as primary extractor and pdfplumber as secondary. When both
    succeed, word-level voting reconciles differences by preferring dictionary
    words over OCR errors.

    Attributes:
        dictionary: DictionaryTextValidator instance for word validation
    """

    def __init__(self, dictionary: DictionaryTextValidator):
        """
        Initialize the PDF extractor.

        Args:
            dictionary: DictionaryTextValidator instance for word validation during voting
        """
        self.dictionary = dictionary
        self._layout_analyzer = LayoutAnalyzer()

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
            >>> extractor = PDFExtractor(DictionaryTextValidator())
            >>> result = extractor.extract(Path("scan.pdf"))
            >>> if result['needs_ocr']:
            ...     print("Falling back to OCR")
        """
        logger.debug("Processing PDF: %s", file_path.name)

        # Step 1: Try layout-aware PyMuPDF extraction, fall back to flat
        with Timer("PyMuPDF text extraction"):
            primary_text, page_count, primary_error = self._extract_pymupdf_layout(file_path)
            if primary_text is None and primary_error is None:
                # Layout detection failed gracefully; try flat extraction
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
            logger.debug("Using hybrid extraction with word-level voting")

        elif primary_text:
            # Only PyMuPDF succeeded
            text = primary_text
            method = "pymupdf_only"
            logger.debug("Using PyMuPDF only (pdfplumber failed: %s)", secondary_error)

        elif secondary_text:
            # Only pdfplumber succeeded
            text = secondary_text
            method = "pdfplumber_only"
            logger.debug("Using pdfplumber only (PyMuPDF failed: %s)", primary_error)

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
        logger.debug("Dictionary confidence: %.1f%% (method: %s)", confidence, method)

        # Determine if OCR fallback is needed
        needs_ocr = confidence <= MIN_DICTIONARY_CONFIDENCE or len(text) <= 1000

        if not needs_ocr:
            logger.debug("Using %s extraction", method)

        return {
            "text": text,
            "page_count": page_count,
            "method": method,
            "confidence": int(confidence),
            "needs_ocr": needs_ocr,
            "error": None,
        }

    def _extract_pymupdf_layout(self, file_path: Path) -> tuple[str | None, int, str | None]:
        """
        Extract text using PyMuPDF with layout-aware zone clipping.

        Uses LayoutAnalyzer to detect header/footer/line-number zones, then
        clips each page to the content zone before extracting text. This
        prevents boilerplate from entering the extracted text.

        Returns (None, 0, None) — no error — when layout detection fails
        gracefully (too few pages, no repeating blocks). The caller should
        fall back to flat extraction in that case.

        Args:
            file_path: Path to the PDF file

        Returns:
            Tuple of (text, page_count, error_type)
        """
        try:
            with fitz.open(file_path) as doc:
                page_count = len(doc)
                if page_count == 0:
                    return None, 0, "empty"

                # Detect content zone from sample pages
                zone = self._layout_analyzer.detect_zones(doc)
                if zone is None:
                    logger.debug("Layout extraction: zone detection failed, will use flat")
                    return None, page_count, None

                # Build clip rectangle
                clip = fitz.Rect(zone.left, zone.top, zone.right, zone.bottom)
                logger.debug(
                    "Layout extraction: clipping to (%.0f, %.0f, %.0f, %.0f)",
                    clip.x0,
                    clip.y0,
                    clip.x1,
                    clip.y1,
                )

                pages_text = []
                for i, page in enumerate(doc, 1):
                    if i % 10 == 0:
                        logger.debug("PyMuPDF layout: Extracting page %d/%d", i, page_count)
                    page_text = page.get_text(clip=clip)
                    if page_text:
                        pages_text.append(page_text)

                text = "\f".join(pages_text)

                # Safety check: if clipping removed too much text, reject
                flat_text = "".join(p.get_text() for p in doc)
                flat_words = len(flat_text.split())
                clip_words = len(text.split())

                if flat_words > 0:
                    ratio = clip_words / flat_words
                    if ratio < 0.70:
                        logger.warning(
                            "Layout extraction kept only %.0f%% of words "
                            "(%.0f vs %.0f); falling back to flat",
                            ratio * 100,
                            clip_words,
                            flat_words,
                        )
                        return None, page_count, None

                    logger.debug(
                        "Layout extraction kept %.0f%% of words (removed boilerplate)",
                        ratio * 100,
                    )

                return text, page_count, None

        except fitz.FileDataError:
            # Let the flat extractor handle the error reporting
            return None, 0, None

        except Exception as e:
            logger.debug("Layout extraction failed: %s", e)
            return None, 0, None

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
            page_count = 0

            with fitz.open(file_path) as doc:
                page_count = len(doc)
                logger.debug("PyMuPDF: PDF has %d pages", page_count)

                if page_count == 0:
                    logger.error("PyMuPDF: PDF has no pages")
                    return None, 0, "empty"

                pages_text = []
                for i, page in enumerate(doc, 1):
                    if i % 10 == 0:
                        logger.debug("PyMuPDF: Extracting page %d/%d", i, page_count)

                    page_text = page.get_text()
                    if page_text:
                        pages_text.append(page_text)

                text = "\f".join(pages_text)

            return text, page_count, None

        except fitz.FileDataError as e:
            error_msg = str(e).lower()
            if "password" in error_msg or "encrypted" in error_msg:
                logger.error("PyMuPDF: PDF is password-protected or encrypted")
                return None, 0, "password"
            else:
                logger.error("PyMuPDF: PDF file appears to be corrupted")
                return None, 0, "corrupted"

        except Exception as e:
            error_msg = str(e).lower()
            if "permission" in error_msg:
                logger.error("PyMuPDF: Permission denied when accessing PDF")
                return None, 0, "permission"
            else:
                logger.error("PyMuPDF: Failed to extract PDF text: %s", e, exc_info=True)
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
            page_count = 0

            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                logger.debug("pdfplumber: PDF has %d pages", page_count)

                if page_count == 0:
                    logger.error("pdfplumber: PDF has no pages")
                    return None, 0, "empty"

                pages_text = []
                for i, page in enumerate(pdf.pages, 1):
                    if i % 10 == 0:
                        logger.debug("pdfplumber: Extracting page %d/%d", i, page_count)

                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)

                text = "\f".join(pages_text)

            return text, page_count, None

        except Exception as e:
            error_msg = str(e).lower()

            if "password" in error_msg or "encrypted" in error_msg:
                logger.error("pdfplumber: PDF is password-protected or encrypted")
                return None, 0, "password"
            elif "damaged" in error_msg or "corrupt" in error_msg or "invalid" in error_msg:
                logger.error("pdfplumber: PDF file appears to be corrupted or damaged")
                return None, 0, "corrupted"
            elif "permission" in error_msg:
                logger.error("pdfplumber: Permission denied when accessing PDF")
                return None, 0, "permission"
            else:
                logger.error("pdfplumber: Failed to extract PDF text: %s", e, exc_info=True)
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
            >>> extractor = PDFExtractor(DictionaryTextValidator())
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
                            logger.debug("'%s' -> '%s' (dictionary correction)", p_word, s_word)
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
            logger.debug("Made %d dictionary corrections", corrections_made)

        return " ".join(result_tokens)
