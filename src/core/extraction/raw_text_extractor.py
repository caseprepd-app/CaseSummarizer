"""
Raw Text Extraction Facade.

Provides a unified interface for extracting text from legal documents.
Delegates to specialized modules for each file format and processing step.

Supported formats: PDF, DOCX, TXT, RTF, PNG, JPG

Pipeline Steps:
    1. EXTRACTION - Read text from files (digital PDF, OCR, or file readers)
    2. NORMALIZATION - De-hyphenation, page numbers, line filtering
    2.5. SANITIZATION - Fix mojibake, control chars, redaction handling

Example usage:
    >>> extractor = RawTextExtractor()
    >>> result = extractor.process_document("complaint.pdf")
    >>> print(f"Status: {result['status']}, Confidence: {result['confidence']}%")
    Status: success, Confidence: 85%

    >>> # Access extracted text
    >>> text = result['extracted_text']

This module can be run standalone via command line:
    python -m src.core.extraction.raw_text_extractor --input document.pdf
"""

import contextlib
import logging
from pathlib import Path

from src.config import (
    LARGE_FILE_WARNING_MB,
    MAX_FILE_SIZE_MB,
    OCR_CONFIDENCE_THRESHOLD,
)

# Character sanitization (external module)
from src.core.sanitization import CharacterSanitizer
from src.logging_config import Timer

logger = logging.getLogger(__name__)

# Internal modules (the actual implementation)
from .case_number_extractor import CaseNumberExtractor
from .dictionary_utils import DictionaryTextValidator
from .file_readers import FileReaders
from .ocr_processor import OCRProcessor
from .pdf_extractor import PDFExtractor
from .text_normalizer import TextNormalizer


class RawTextExtractor:
    """
    Facade for document text extraction.

    Coordinates specialized modules to extract and normalize text from
    legal documents. Handles PDF (digital/OCR), DOCX, TXT, RTF, and images.

    This class maintains backward compatibility with existing code while
    delegating work to focused, testable modules.

    Attributes:
        dictionary: DictionaryTextValidator for word validation and confidence
        normalizer: TextNormalizer for text cleanup pipeline
        pdf_extractor: PDFExtractor for hybrid PDF extraction
        ocr_processor: OCRProcessor for scanned documents
        file_readers: FileReaders for TXT/RTF/DOCX/image files
        case_extractor: CaseNumberExtractor for legal case numbers
        character_sanitizer: CharacterSanitizer for encoding cleanup

    Example:
        >>> extractor = RawTextExtractor()
        >>> result = extractor.process_document("motion.pdf")
        >>> if result['status'] == 'success':
        ...     print(f"Extracted {len(result['extracted_text'])} chars")
    """

    def __init__(self, jurisdiction: str = "ny", ocr_allowed: bool = True):
        """
        Initialize the RawTextExtractor with all component modules.

        Args:
            jurisdiction: Legal jurisdiction (ny, ca, federal). Currently
                         only affects logging; keywords are universal.
            ocr_allowed: Whether OCR processing is permitted. When False,
                        scanned PDFs return low-quality digital text instead
                        of attempting OCR (used when Tesseract is missing).
        """
        self.jurisdiction = jurisdiction
        self.ocr_allowed = ocr_allowed

        with Timer("RawTextExtractor initialization"):
            # Initialize component modules
            self.dictionary = DictionaryTextValidator()
            self.normalizer = TextNormalizer(self.dictionary.legal_keywords)
            self.pdf_extractor = PDFExtractor(self.dictionary)
            self.ocr_processor = OCRProcessor(self.dictionary)
            self.file_readers = FileReaders(self.dictionary, self.ocr_processor)
            self.case_extractor = CaseNumberExtractor()
            self.character_sanitizer = CharacterSanitizer()

        logger.debug("RawTextExtractor initialized for jurisdiction: %s", jurisdiction)

    def process_document(self, file_path: str, progress_callback=None) -> dict:
        """
        Process a single document through the full extraction pipeline.

        Pipeline:
            1. Validate file (exists, size limits)
            2. Extract text based on file type
            3. Extract case numbers from raw text
            4. Normalize text (de-hyphenation, page numbers, filtering)
            5. Sanitize characters (mojibake, control chars, redactions)
            6. Calculate quality confidence

        Args:
            file_path: Path to the document file
            progress_callback: Optional callback(message: str, percent: int)

        Returns:
            Dict with keys:
                - filename: Name of the file
                - file_path: Full path to file
                - status: 'success', 'warning', or 'error'
                - method: Extraction method used
                - confidence: Quality score (0-100)
                - extracted_text: Processed text content
                - page_count: Number of pages (for PDFs)
                - file_size: File size in bytes
                - case_numbers: List of detected case numbers
                - error_message: Error description (if failed)

        Example:
            >>> extractor = RawTextExtractor()
            >>> result = extractor.process_document("brief.pdf")
            >>> print(f"{result['filename']}: {result['status']}")
        """
        file_path = Path(file_path)
        filename = file_path.name

        logger.info("Processing document: %s", filename)

        def report_progress(message: str, percent: int):
            if progress_callback:
                with contextlib.suppress(Exception):
                    progress_callback(message, percent)

        # Initialize result
        result = {
            "filename": filename,
            "file_path": str(file_path),
            "status": "success",
            "method": None,
            "confidence": 0,
            "extracted_text": "",
            "page_count": None,
            "file_size": 0,
            "case_numbers": [],
            "error_message": None,
        }

        try:
            with Timer(f"Processing {filename}"):
                report_progress(f"Starting {filename}", 0)

                # Step 1: Validate file
                validation = self._validate_file(file_path)
                if validation["error"]:
                    result["status"] = "error"
                    result["error_message"] = validation["error"]
                    logger.error("%s", validation["error"])
                    return result

                result["file_size"] = validation["file_size"]

                # Step 2: Extract text based on file type
                report_progress("Extracting text", 20)
                extraction = self._extract_by_type(file_path)

                if extraction["status"] in ("error", "ocr_skipped"):
                    result["status"] = "error"
                    result["error_message"] = extraction.get(
                        "error_message", "File contains no extractable text."
                    )
                    logger.error("%s", result["error_message"])
                    return result

                result["method"] = extraction["method"]
                result["confidence"] = extraction["confidence"]
                result["extracted_text"] = extraction["text"]
                result["page_count"] = extraction.get("page_count")

                # Step 3: Extract case numbers (before normalization)
                report_progress("Extracting case numbers", 60)
                if result["extracted_text"]:
                    result["case_numbers"] = self.case_extractor.extract(result["extracted_text"])
                    if result["case_numbers"]:
                        logger.debug("Found case numbers: %s", result["case_numbers"])

                # Step 4: Normalize text
                report_progress("Normalizing text", 70)
                if result["extracted_text"]:
                    with Timer("Text normalization"):
                        result["extracted_text"] = self.normalizer.normalize(
                            result["extracted_text"]
                        )

                    if not result["extracted_text"].strip():
                        result["status"] = "error"
                        result["error_message"] = (
                            "Unable to extract readable text. "
                            "File may be corrupted or contain only images."
                        )
                        logger.error("%s", result["error_message"])
                        return result
                else:
                    result["status"] = "error"
                    result["error_message"] = "File contains no extractable text."
                    logger.error("Empty text extracted from: %s", result.get("filename", "unknown"))
                    return result

                # Step 5: Sanitize characters
                report_progress("Fixing character encoding...", 80)
                if result["extracted_text"]:
                    with Timer("Character sanitization"):
                        sanitized, stats = self.character_sanitizer.sanitize(
                            result["extracted_text"]
                        )
                        result["extracted_text"] = sanitized

                        if any(stats.values()):
                            logger.debug("Sanitization stats: %s", stats)

                # Step 6: Set final status based on confidence
                if (
                    result["status"] == "success"
                    and result["confidence"] < OCR_CONFIDENCE_THRESHOLD
                ):
                    result["status"] = "warning"

                report_progress("Complete", 100)

        except Exception as e:
            result["status"] = "error"
            result["error_message"] = f"Unexpected error: {e!s}"
            logger.error("Error processing %s: %s", filename, e, exc_info=True)
            report_progress("Error", 0)

        return result

    def _validate_file(self, file_path: Path) -> dict:
        """Validate file exists and is within size limits."""
        if not file_path.exists():
            return {"error": f"File not found: {file_path}", "file_size": 0}

        file_size = file_path.stat().st_size
        size_mb = file_size / (1024 * 1024)

        if size_mb > MAX_FILE_SIZE_MB:
            return {
                "error": f"File exceeds maximum size ({MAX_FILE_SIZE_MB}MB). "
                f"File size: {size_mb:.1f}MB",
                "file_size": file_size,
            }

        if size_mb > LARGE_FILE_WARNING_MB:
            logger.warning("Large file detected (%.1fMB). Processing may take longer.", size_mb)

        return {"error": None, "file_size": file_size}

    def _extract_by_type(self, file_path: Path) -> dict:
        """Route extraction to appropriate handler based on file type."""
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return self._process_pdf(file_path)
        elif ext == ".txt":
            return self.file_readers.read_text_file(file_path)
        elif ext == ".rtf":
            return self.file_readers.read_rtf_file(file_path)
        elif ext == ".docx":
            return self.file_readers.read_docx_file(file_path)
        elif ext in [".png", ".jpg", ".jpeg"]:
            return self.file_readers.read_image_file(file_path)
        else:
            return {
                "status": "error",
                "error_message": f"Unsupported file type: {ext}. "
                f"Supported formats: PDF, DOCX, TXT, RTF, PNG, JPG",
                "text": None,
                "method": None,
                "confidence": 0,
            }

    def _process_pdf(self, file_path: Path) -> dict:
        """Process PDF with hybrid extraction, falling back to OCR if needed."""
        import tempfile

        from .pdf_extractor import extract_portfolio_pdf

        logger.debug("Processing PDF: %s", file_path.name)

        # Step 0: Check for PDF Portfolio (bundle with embedded PDF)
        temp_path = None
        try:
            embedded_bytes = extract_portfolio_pdf(file_path)
            if embedded_bytes:
                temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
                temp_file.write(embedded_bytes)
                temp_file.close()
                temp_path = Path(temp_file.name)
                logger.info("Using embedded PDF from portfolio: %s", file_path.name)
                file_path = temp_path

            return self._process_pdf_inner(file_path)
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()

    def _process_pdf_inner(self, file_path: Path) -> dict:
        """Core PDF processing logic (extracted for portfolio support)."""
        from src.config import MIN_DICTIONARY_CONFIDENCE

        # Step 0: Pre-scan for scanned pages (fast, avoids unnecessary extraction)
        with Timer("Scanned page pre-detection"):
            scanned_pages, all_scanned = self.pdf_extractor.detect_scanned_pages(file_path)

        if all_scanned:
            logger.debug("All pages are scanned — skipping digital extraction, going to OCR")
            if not self.ocr_allowed:
                return {
                    "status": "ocr_skipped",
                    "error_message": "OCR skipped — Tesseract not installed.",
                    "text": None,
                    "method": None,
                    "confidence": 0,
                    "page_count": None,
                }
            with Timer("Full OCR (all scanned)"):
                ocr_result = self.ocr_processor.process_pdf(file_path)
            return {
                "status": ocr_result["status"],
                "error_message": ocr_result.get("error_message"),
                "text": ocr_result["text"],
                "method": ocr_result["method"],
                "confidence": ocr_result["confidence"],
                "page_count": ocr_result["page_count"],
            }

        from src.config import PDFPLUMBER_SKIP_CONFIDENCE

        # Step 1: Try PyMuPDF extraction (primary)
        # Uses facade methods so tests can mock them
        with Timer("PyMuPDF text extraction"):
            primary_text, page_count, primary_error = self._extract_text_pymupdf(file_path)

        # Step 2: Check if PyMuPDF is good enough to skip pdfplumber
        text = None
        method = None
        confidence = 0.0
        has_cid = False
        secondary_text = None
        secondary_page_count = 0

        if primary_text:
            with Timer("PyMuPDF confidence check"):
                primary_conf = self._calculate_dictionary_confidence(primary_text)
            has_cid = PDFExtractor.has_cid_problem(primary_text)

            if primary_conf >= PDFPLUMBER_SKIP_CONFIDENCE and not has_cid:
                # PyMuPDF is good enough — skip pdfplumber entirely
                text = primary_text
                method = "pymupdf_only"
                confidence = primary_conf
                logger.info(
                    "PyMuPDF confidence %.1f%% >= %d%% — skipping pdfplumber",
                    primary_conf,
                    PDFPLUMBER_SKIP_CONFIDENCE,
                )
            else:
                # PyMuPDF quality insufficient — run pdfplumber as backup
                reason = "CID markers" if has_cid else f"confidence {primary_conf:.1f}%"
                logger.info(
                    "PyMuPDF %s below threshold — running pdfplumber for comparison",
                    reason,
                )
                with Timer("pdfplumber text extraction"):
                    secondary_text, secondary_page_count, secondary_error = self._extract_pdf_text(
                        file_path
                    )

                if secondary_text:
                    text, method = self._pick_best_extraction(primary_text, secondary_text)
                else:
                    text = primary_text
                    method = "pymupdf_only"
                    logger.debug("Using PyMuPDF only (pdfplumber failed: %s)", secondary_error)
        else:
            # PyMuPDF failed entirely — try pdfplumber
            logger.info("PyMuPDF extraction failed — trying pdfplumber")
            with Timer("pdfplumber text extraction"):
                secondary_text, secondary_page_count, secondary_error = self._extract_pdf_text(
                    file_path
                )

            if secondary_text:
                text = secondary_text
                method = "pdfplumber_only"
            else:
                error_type = primary_error or secondary_error or "unknown"
                error_messages = {
                    "password": "PDF is password-protected or encrypted",
                    "corrupted": "PDF file appears to be corrupted or damaged",
                    "permission": "Permission denied when accessing PDF",
                    "empty": "PDF has no pages or content",
                }
                return {
                    "status": "error",
                    "error_message": error_messages.get(
                        error_type, f"Failed to extract PDF text: {error_type}"
                    ),
                    "text": None,
                    "method": None,
                    "confidence": 0,
                    "page_count": page_count or secondary_page_count,
                }

        # Use whichever page count we got
        page_count = page_count or secondary_page_count

        # Step 3: Final confidence and CID check
        # In the fast path (pymupdf_only with high confidence), these are already set.
        # For all other paths, calculate now.
        if not (method == "pymupdf_only" and text is primary_text):
            with Timer("Dictionary confidence check"):
                confidence = self._calculate_dictionary_confidence(text)
            has_cid = PDFExtractor.has_cid_problem(text)
            if has_cid:
                logger.debug("CID markers detected — will trigger OCR fallback")
        logger.debug("Dictionary confidence: %.1f%% (method: %s)", confidence, method)

        # Decision: Use digital text or fall back to OCR
        needs_ocr = confidence <= MIN_DICTIONARY_CONFIDENCE or len(text) <= 1000 or has_cid

        if needs_ocr and not self.ocr_allowed:
            logger.debug(
                "Digital text quality insufficient but OCR is disabled. "
                "Returning low-quality digital text as fallback."
            )
            return {
                "status": "ocr_skipped",
                "error_message": "OCR skipped — Tesseract not installed.",
                "text": text,
                "method": method,
                "confidence": int(confidence),
                "page_count": page_count,
            }

        if needs_ocr:
            logger.debug("Digital text quality insufficient. Performing OCR...")
            with Timer("OCR Processing"):
                ocr_result = self.ocr_processor.process_pdf(file_path, page_count)

            return {
                "status": ocr_result["status"],
                "error_message": ocr_result.get("error_message"),
                "text": ocr_result["text"],
                "method": ocr_result["method"],
                "confidence": ocr_result["confidence"],
                "page_count": ocr_result["page_count"],
            }

        # Step 5: Splice OCR text for scanned pages in mixed documents
        if scanned_pages and self.ocr_allowed:
            # Convert 0-indexed scanned_pages to 1-indexed for OCR
            ocr_page_nums = sorted(p + 1 for p in scanned_pages)
            logger.debug("Mixed document: OCR-ing scanned pages %s", ocr_page_nums)

            with Timer("Per-page OCR for mixed document"):
                ocr_result = self.ocr_processor.process_pages(file_path, ocr_page_nums, page_count)

            if ocr_result["status"] == "success" and ocr_result["pages"]:
                text = self._splice_ocr_pages(text, ocr_result["pages"])
                method = f"{method}+partial_ocr"
                # Recalculate confidence with spliced text
                confidence = self._calculate_dictionary_confidence(text)

        # Digital extraction succeeded
        return {
            "status": "success",
            "error_message": None,
            "text": text,
            "method": method,
            "confidence": int(confidence),
            "page_count": page_count,
        }

    @staticmethod
    def _splice_ocr_pages(digital_text: str, ocr_pages: dict[int, str]) -> str:
        """
        Replace scanned pages in digital text with OCR results.

        Splits digital text on form-feed (\\f) page separators, replaces
        pages that were OCR'd, and rejoins.

        Args:
            digital_text: Full digital extraction with \\f page separators
            ocr_pages: Dict mapping 1-indexed page numbers to OCR text

        Returns:
            Text with scanned pages replaced by OCR text.
        """
        pages = digital_text.split("\f")
        for page_num, ocr_text in ocr_pages.items():
            idx = page_num - 1  # Convert 1-indexed to 0-indexed
            if 0 <= idx < len(pages):
                pages[idx] = ocr_text
        return "\f".join(pages)

    # =========================================================================
    # BACKWARD COMPATIBILITY - Delegation wrappers for tests
    # =========================================================================
    # These methods delegate to component modules but maintain the original
    # method signatures for test compatibility.

    @property
    def english_words(self) -> set[str]:
        """Access dictionary words (for test mocking)."""
        return self.dictionary.english_words

    @english_words.setter
    def english_words(self, value: set[str]):
        """Set dictionary words (for test mocking)."""
        self.dictionary.english_words = value

    @property
    def legal_keywords(self) -> set[str]:
        """Access legal keywords."""
        return self.dictionary.legal_keywords

    def _calculate_dictionary_confidence(self, text: str) -> float:
        """Delegate to dictionary module (for test compatibility)."""
        return self.dictionary.calculate_confidence(text)

    def _is_valid_word(self, word: str) -> bool:
        """Delegate to dictionary module (for test compatibility)."""
        return self.dictionary.is_valid_word(word)

    def _tokenize_for_voting(self, text: str) -> list[str]:
        """Delegate to dictionary module (for test compatibility)."""
        return self.dictionary.tokenize_for_voting(text)

    def _normalize_text(self, text: str) -> str:
        """Delegate to normalizer module (for test compatibility)."""
        return self.normalizer.normalize(text)

    def _is_page_number(self, line: str) -> bool:
        """Delegate to normalizer module (for test compatibility)."""
        return self.normalizer._is_page_number(line)

    def _extract_case_numbers(self, text: str) -> list[str]:
        """Delegate to case extractor module (for test compatibility)."""
        return self.case_extractor.extract(text)

    def _pick_best_extraction(self, primary_text: str, secondary_text: str) -> tuple[str, str]:
        """Pick the extractor output with higher dictionary confidence.

        Args:
            primary_text: Text from PyMuPDF
            secondary_text: Text from pdfplumber

        Returns:
            Tuple of (best_text, method_name)
        """
        primary_conf = self._calculate_dictionary_confidence(primary_text)
        secondary_conf = self._calculate_dictionary_confidence(secondary_text)

        if primary_conf >= secondary_conf:
            method = "pymupdf_best"
        else:
            method = "pdfplumber_best"

        logger.debug(
            "Hybrid extraction: PyMuPDF=%.1f%% vs pdfplumber=%.1f%% → %s",
            primary_conf,
            secondary_conf,
            method,
        )

        best_text = primary_text if primary_conf >= secondary_conf else secondary_text
        return best_text, method

    def _extract_text_pymupdf(self, file_path) -> tuple:
        """Delegate to PDF extractor — try layout-aware first, then flat."""
        text, page_count, error = self.pdf_extractor._extract_pymupdf_layout(file_path)
        if text is None and error is None:
            # Layout detection failed gracefully; fall back to flat extraction
            text, page_count, error = self.pdf_extractor._extract_pymupdf(file_path)
        return text, page_count, error

    def _extract_pdf_text(self, file_path) -> tuple:
        """Delegate to PDF extractor module (for test compatibility)."""
        return self.pdf_extractor._extract_pdfplumber(file_path)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================


def main():
    """Command-line interface for the raw text extractor."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CasePrepd Raw Text Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.core.extraction.raw_text_extractor --input complaint.pdf
  python -m src.core.extraction.raw_text_extractor --input *.pdf --output-dir ./extracted
        """,
    )

    parser.add_argument("--input", nargs="+", required=True, help="Input file(s) to process")
    parser.add_argument(
        "--output-dir", default="./extracted", help="Output directory (default: ./extracted)"
    )
    parser.add_argument(
        "--jurisdiction",
        default="ny",
        choices=["ny", "ca", "federal"],
        help="Legal jurisdiction (default: ny)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extractor
    logger.info("Initializing extractor (jurisdiction: %s)", args.jurisdiction)
    extractor = RawTextExtractor(jurisdiction=args.jurisdiction)

    # Process files
    results = []
    for file_path in args.input:
        result = extractor.process_document(file_path)
        results.append(result)

        # Save extracted text if successful
        if result["status"] in ["success", "warning"] and result["extracted_text"]:
            output_filename = f"{Path(result['filename']).stem}_extracted.txt"
            output_path = output_dir / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result["extracted_text"])

            logger.info("Saved: %s", output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)

    for result in results:
        status_symbol = {"success": "[OK]", "warning": "[WARN]", "error": "[ERROR]"}.get(
            result["status"], "[?]"
        )
        print(f"\n{status_symbol} {result['filename']}")
        print(f"  Method: {result['method'] or 'N/A'}")
        print(f"  Confidence: {result['confidence']}%")

        if result.get("page_count"):
            print(f"  Pages: {result['page_count']}")

        if result["error_message"]:
            print(f"  Error: {result['error_message']}")

    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    warnings = sum(1 for r in results if r["status"] == "warning")
    errors = sum(1 for r in results if r["status"] == "error")

    print("\n" + "=" * 60)
    print(f"Total: {total} | Success: {success} | Warnings: {warnings} | Errors: {errors}")


if __name__ == "__main__":
    main()
