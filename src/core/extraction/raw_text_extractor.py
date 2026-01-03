"""
Raw Text Extraction Module

Extracts raw text from legal documents (PDF, DOCX, TXT, RTF, PNG, JPG) and applies
basic normalization (de-hyphenation, page number removal, whitespace normalization).

This module implements Steps 1-2 of the document processing pipeline:
- Step 1: Extract text from files (PDF digital/OCR, DOCX, TXT, RTF, PNG/JPG via OCR)
- Step 2: Apply basic text normalization (OCR error fixing, structural normalization)
- Step 2.5: Character sanitization (fix mojibake, remove control chars, redaction handling)

Session 73: Added DOCX and direct image (PNG/JPG) import support.

This module can be used standalone via command line or as part of the larger document processing pipeline.
"""

import argparse
import os
import re
import time
from pathlib import Path

# NLP
import nltk

# PDF processing
import pdfplumber
import pytesseract
from nltk.corpus import words

# OCR
from pdf2image import convert_from_path

# Local imports
from src.config import (
    DEBUG_DEFAULT_FILE,
    DEBUG_MODE,
    LARGE_FILE_WARNING_MB,
    MAX_FILE_SIZE_MB,
    MIN_DICTIONARY_CONFIDENCE,
    MIN_LINE_LENGTH,
    OCR_CONFIDENCE_THRESHOLD,
    OCR_DPI,
    OCR_PREPROCESSING_ENABLED,
    OCR_DENOISE_STRENGTH,
    OCR_ENABLE_CLAHE,
)

# Image preprocessing for OCR
from src.core.extraction.image_preprocessor import ImagePreprocessor, PreprocessingStats

# Character sanitization
from src.core.sanitization import CharacterSanitizer

# Logging (use canonical location for new code)
from src.logging_config import Timer, debug, error, info, warning


class RawTextExtractor:
    """
    Extracts and normalizes raw text from legal documents.

    Handles PDF (digital and scanned), TXT, and RTF files.
    Returns extracted text with confidence scores and processing metadata.

    This class implements Steps 1-2.5 of the document pipeline:
    1. Text Extraction: Reads PDF/TXT/RTF and applies OCR if needed
    2. Basic Normalization: De-hyphenation, page number removal, whitespace normalization
    2.5: Character Sanitization: Fix mojibake, remove control chars, handle redactions
    """

    def __init__(self, jurisdiction: str = "ny"):
        """
        Initialize the RawTextExtractor.

        Args:
            jurisdiction: Legal jurisdiction for keyword loading (ny, ca, federal)
        """
        self.jurisdiction = jurisdiction
        self.legal_keywords: set[str] = set()
        self.english_words: set[str] = set()
        self.character_sanitizer = CharacterSanitizer()

        with Timer("RawTextExtractor initialization"):
            self._load_keywords()
            self._load_dictionary()

    def _load_keywords(self):
        """Load legal keywords for the jurisdiction."""
        debug(f"Loading legal keywords for jurisdiction: {self.jurisdiction}")

        # For now, use a default set since keyword files don't exist yet
        # TODO: Download from Dropbox and cache locally
        self.legal_keywords = {
            "COURT", "PLAINTIFF", "DEFENDANT", "APPEARANCES", "SUPREME",
            "MOTION", "AFFIDAVIT", "EXHIBIT", "DEPOSITION", "TESTIMONY",
            "COMPLAINT", "ANSWER", "SUMMONS", "NOTICE", "ORDER",
            "JUDGE", "ATTORNEY", "COUNSEL", "PARTY", "ACTION"
        }

        debug(f"Loaded {len(self.legal_keywords)} legal keywords")

    def _load_dictionary(self):
        """Load NLTK English words corpus."""
        debug("Loading NLTK English words corpus")

        try:
            # Try to load the words corpus
            self.english_words = {word.lower() for word in words.words()}
            debug(f"Loaded {len(self.english_words)} English words")
        except LookupError:
            # Download if not available
            warning("NLTK words corpus not found. Downloading...")
            nltk.download('words', quiet=not DEBUG_MODE)
            self.english_words = {word.lower() for word in words.words()}
            debug(f"Downloaded and loaded {len(self.english_words)} English words")

    def process_document(self, file_path: str, progress_callback=None) -> dict:
        """
        Process a single document (extraction + basic normalization).

        Args:
            file_path: Path to the document file
            progress_callback: Optional callback function(message: str, percent: int)
                             for progress updates

        Returns:
            Dictionary with keys:
                - filename: Name of the file
                - file_path: Full path to file
                - status: 'success', 'warning', or 'error'
                - method: 'direct_read', 'digital_text', 'ocr', 'rtf_extraction'
                - confidence: OCR confidence score (0-100)
                - extracted_text: Extracted and normalized text content
                - page_count: Number of pages (for PDFs)
                - file_size: File size in bytes
                - case_numbers: List of detected case numbers
                - error_message: Error description (if status is 'error')
        """
        file_path = Path(file_path)
        filename = file_path.name

        info(f"Processing document: {filename}")

        # Helper function to report progress
        def report_progress(message: str, percent: int):
            if progress_callback:
                try:
                    progress_callback(message, percent)
                except Exception:
                    pass  # Ignore callback errors

        result = {
            'filename': filename,
            'file_path': str(file_path),
            'status': 'success',
            'method': None,
            'confidence': 0,
            'extracted_text': '',
            'page_count': None,  # Changed from 'pages' to match FileReviewTable
            'file_size': 0,      # Changed from 'size_mb' to store bytes (not MB)
            'case_numbers': [],
            'error_message': None
        }

        try:
            with Timer(f"Processing {filename}"):
                report_progress(f"Starting {filename}", 0)

                # Check file exists
                if not file_path.exists():
                    result['status'] = 'error'
                    result['error_message'] = f"File not found: {file_path}"
                    error(result['error_message'])
                    return result

                # Get file size in bytes (for display in FileReviewTable)
                result['file_size'] = file_path.stat().st_size
                size_mb = result['file_size'] / (1024 * 1024)

                # Check file size limits
                if size_mb > MAX_FILE_SIZE_MB:
                    result['status'] = 'error'
                    result['error_message'] = f"File exceeds maximum size ({MAX_FILE_SIZE_MB}MB). File size: {size_mb:.1f}MB"
                    error(result['error_message'])
                    return result

                if size_mb > LARGE_FILE_WARNING_MB:
                    warning(f"Large file detected ({size_mb:.1f}MB). Processing may take longer.")

                # Determine file type and process
                file_extension = file_path.suffix.lower()

                report_progress("Extracting text", 20)

                if file_extension in ['.txt', '.rtf']:
                    result.update(self._process_text_file(file_path))
                elif file_extension == '.pdf':
                    result.update(self._process_pdf(file_path))
                elif file_extension == '.docx':
                    # Session 73: DOCX import support
                    result.update(self._process_docx(file_path))
                elif file_extension in ['.png', '.jpg', '.jpeg']:
                    # Session 73: Direct image import support (OCR)
                    result.update(self._process_image(file_path))
                else:
                    result['status'] = 'error'
                    result['error_message'] = f"Unsupported file type: {file_extension}. Supported formats: PDF, DOCX, TXT, RTF, PNG, JPG"
                    error(result['error_message'])
                    return result

                report_progress("Extracting case numbers", 60)

                # Extract case numbers from raw text (before normalization removes short lines)
                if result['status'] != 'error' and result['extracted_text']:
                    result['case_numbers'] = self._extract_case_numbers(result['extracted_text'])
                    if result['case_numbers']:
                        debug(f"Found case numbers: {result['case_numbers']}")

                report_progress("Normalizing text", 70)

                # Apply basic text normalization
                if result['status'] != 'error' and result['extracted_text']:
                    with Timer("Text normalization"):
                        result['extracted_text'] = self._normalize_text(result['extracted_text'])

                    # Check if normalization resulted in empty text
                    if len(result['extracted_text'].strip()) == 0:
                        result['status'] = 'error'
                        result['error_message'] = "Unable to extract readable text. File may be corrupted or contain only images."
                        error(result['error_message'])

                report_progress("Sanitizing characters", 80)

                # Apply character sanitization (Step 2.5)
                # Fixes mojibake, removes control chars, handles redactions, transliterates accents
                if result['status'] != 'error' and result['extracted_text']:
                    with Timer("Character sanitization"):
                        sanitized_text, sanitization_stats = self.character_sanitizer.sanitize(result['extracted_text'])
                        result['extracted_text'] = sanitized_text

                        # Log sanitization details
                        if any(sanitization_stats.values()):
                            debug(f"Character sanitization stats: {sanitization_stats}")
                            for log_entry in self.character_sanitizer.get_log():
                                debug(f"  - {log_entry}")

                # Set status based on confidence
                if result['status'] == 'success':
                    if result['confidence'] < OCR_CONFIDENCE_THRESHOLD:
                        result['status'] = 'warning'

                report_progress("Complete", 100)
                debug(f"DEBUG_EXTRACTOR: Final result for {filename} - file_size: {result['file_size']}, page_count: {result['page_count']}")

        except Exception as e:
            result['status'] = 'error'
            result['error_message'] = f"Unexpected error: {str(e)}"
            error(f"Error processing {filename}: {str(e)}", exc_info=True)
            report_progress("Error", 0)

        return result

    def _process_text_file(self, file_path: Path) -> dict:
        """Process TXT or RTF file."""
        debug(f"Processing as text file: {file_path.name}")

        try:
            # Check if RTF or plain text
            if file_path.suffix.lower() == '.rtf':
                # RTF file - use striprtf to extract plain text
                from striprtf.striprtf import rtf_to_text

                with open(file_path, encoding='utf-8', errors='ignore') as f:
                    rtf_content = f.read()

                text = rtf_to_text(rtf_content)
                method = 'rtf_extraction'
                debug(f"Extracted {len(text)} characters from RTF")
            else:
                # Plain text file
                with open(file_path, encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                method = 'direct_read'

            # Calculate actual text quality confidence
            confidence = self._calculate_dictionary_confidence(text)
            debug(f"Text file dictionary confidence: {confidence:.1f}%")

            return {
                'method': method,
                'confidence': int(confidence),
                'extracted_text': text,
                'status': 'success'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error_message': f"Failed to read text file: {str(e)}"
            }

    def _process_docx(self, file_path: Path) -> dict:
        """
        Process Word document (.docx).

        Session 73: DOCX import support using python-docx (already installed for export).

        Args:
            file_path: Path to the DOCX file

        Returns:
            Dict with extracted text, method, status, confidence
        """
        debug(f"Processing as Word document: {file_path.name}")

        try:
            from docx import Document

            doc = Document(file_path)

            # Extract text from all paragraphs
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n".join(paragraphs)

            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_text:
                        text += "\n" + " | ".join(row_text)

            if not text.strip():
                return {
                    'status': 'error',
                    'error_message': "Word document contains no readable text."
                }

            # Calculate text quality confidence
            confidence = self._calculate_dictionary_confidence(text)
            debug(f"DOCX dictionary confidence: {confidence:.1f}%")

            return {
                'method': 'docx_extraction',
                'confidence': int(confidence),
                'extracted_text': text,
                'page_count': len(doc.sections) or 1,  # Approximate page count
                'status': 'success'
            }

        except Exception as e:
            return {
                'status': 'error',
                'error_message': f"Failed to read Word document: {str(e)}"
            }

    def _process_image(self, file_path: Path) -> dict:
        """
        Process image file (.png, .jpg, .jpeg) using OCR.

        Session 73: Direct image import support. Reuses existing OCR pipeline
        with image preprocessing.

        Args:
            file_path: Path to the image file

        Returns:
            Dict with extracted text, method, status, confidence
        """
        debug(f"Processing as image: {file_path.name}")

        try:
            from PIL import Image

            # Load image
            img = Image.open(file_path)

            # Preprocess image using existing pipeline
            if OCR_PREPROCESSING_ENABLED:
                preprocessor = ImagePreprocessor(
                    denoise_strength=OCR_DENOISE_STRENGTH,
                    enable_clahe=OCR_ENABLE_CLAHE
                )
                processed_img, preprocessing_stats = preprocessor.preprocess(img)
                debug(f"Image preprocessing applied: {preprocessing_stats}")
            else:
                processed_img = img.convert("L")  # At minimum convert to grayscale

            # Perform OCR
            text = pytesseract.image_to_string(processed_img)

            if not text.strip():
                return {
                    'status': 'error',
                    'error_message': "Could not extract text from image. Image may not contain readable text."
                }

            # Calculate OCR quality confidence
            confidence = self._calculate_dictionary_confidence(text)
            debug(f"Image OCR dictionary confidence: {confidence:.1f}%")

            return {
                'method': 'image_ocr',
                'confidence': int(confidence),
                'extracted_text': text,
                'page_count': 1,
                'status': 'success' if confidence >= OCR_CONFIDENCE_THRESHOLD else 'warning'
            }

        except Exception as e:
            return {
                'status': 'error',
                'error_message': f"Failed to process image: {str(e)}"
            }

    def _process_pdf(self, file_path: Path) -> dict:
        """Process PDF file (digital or scanned)."""
        debug(f"Processing as PDF: {file_path.name}")

        # Step 1: Try digital text extraction
        with Timer("Digital PDF text extraction"):
            text, page_count, error_type = self._extract_pdf_text(file_path)

        if text is None:
            # Error occurred
            # ... (error handling as before)
            return {
                'status': 'error',
                'error_message': '...',
                'page_count': page_count
            }

        # Step 2: Heuristic check
        with Timer("Dictionary confidence check"):
            dictionary_confidence = self._calculate_dictionary_confidence(text)
        debug(f"Dictionary confidence: {dictionary_confidence:.1f}%")

        # Decision
        if dictionary_confidence > MIN_DICTIONARY_CONFIDENCE and len(text) > 1000:
            debug("Using digital text extraction")
            return {
                'method': 'digital_text',
                'confidence': int(dictionary_confidence),
                'extracted_text': text,
                'page_count': page_count,
                'status': 'success'
            }
        else:
            debug("Digital text quality insufficient. Performing OCR...")
            with Timer("OCR Processing"):
                return self._perform_ocr(file_path, page_count)

    def _extract_pdf_text(self, file_path: Path) -> tuple[str | None, int, str | None]:
        """
        Extract text from PDF using pdfplumber.

        Returns:
            (text, page_count, error_type) where error_type is None on success,
            or one of: 'password', 'corrupted', 'empty', 'unknown'
        """
        try:
            text = ""
            page_count = 0

            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                debug(f"PDF has {page_count} pages")

                if page_count == 0:
                    error("PDF has no pages")
                    return None, 0, 'empty'

                for i, page in enumerate(pdf.pages, 1):
                    if DEBUG_MODE and i % 10 == 0:
                        debug(f"Extracting page {i}/{page_count}")

                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            return text, page_count, None

        except Exception as e:
            error_msg = str(e).lower()

            # Categorize error types
            if "password" in error_msg or "encrypted" in error_msg:
                error("PDF is password-protected or encrypted")
                return None, 0, 'password'
            elif "damaged" in error_msg or "corrupt" in error_msg or "invalid" in error_msg:
                error("PDF file appears to be corrupted or damaged")
                return None, 0, 'corrupted'
            elif "permission" in error_msg:
                error("Permission denied when accessing PDF")
                return None, 0, 'permission'
            else:
                error(f"Failed to extract PDF text: {str(e)}", exc_info=True)
                return None, 0, 'unknown'

    def _calculate_dictionary_confidence(self, text: str) -> float:
        """
        Calculate what percentage of words are valid English words.

        Args:
            text: Text to analyze

        Returns:
            Confidence percentage (0-100)
        """
        if not text:
            return 0.0

        # Tokenize (simple split on whitespace and punctuation)
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())

        if len(tokens) == 0:
            return 0.0

        # Count valid English words
        valid_words = sum(1 for token in tokens if token in self.english_words)

        confidence = (valid_words / len(tokens)) * 100
        return confidence

    def _perform_ocr(self, file_path: Path, page_count: int) -> dict:
        """
        Perform OCR on PDF using Tesseract with optional image preprocessing.

        The preprocessing pipeline (when enabled) applies:
        1. Grayscale conversion
        2. Noise removal (denoising)
        3. Contrast enhancement (CLAHE)
        4. Adaptive thresholding (binarization)
        5. Deskewing (rotation correction)
        6. Border padding

        Research shows preprocessing can improve OCR accuracy by 20-50%.

        Args:
            file_path: Path to PDF
            page_count: Number of pages (from pdfplumber)

        Returns:
            Result dictionary
        """
        debug(f"Starting OCR on {file_path.name}")

        try:
            # Convert PDF to images
            with Timer("PDF to images conversion"):
                images = convert_from_path(str(file_path), dpi=OCR_DPI)

            # Initialize preprocessor if enabled
            preprocessor = None
            if OCR_PREPROCESSING_ENABLED:
                preprocessor = ImagePreprocessor(
                    denoise_strength=OCR_DENOISE_STRENGTH,
                    enable_clahe=OCR_ENABLE_CLAHE,
                )
                debug(f"OCR preprocessing enabled (denoise={OCR_DENOISE_STRENGTH}, clahe={OCR_ENABLE_CLAHE})")

            # Track preprocessing stats across all pages
            total_preprocessing_time = 0.0
            total_skew_corrections = 0

            # OCR each page
            ocr_text = ""
            for i, image in enumerate(images, 1):
                if DEBUG_MODE:
                    debug(f"OCR processing page {i}/{len(images)}")

                # Apply preprocessing if enabled
                if preprocessor is not None:
                    with Timer(f"Preprocessing page {i}", auto_log=DEBUG_MODE):
                        image, stats = preprocessor.preprocess(image)
                        total_preprocessing_time += stats.total_time_ms
                        if stats.skew_corrected:
                            total_skew_corrections += 1
                            debug(f"  Page {i}: corrected {stats.skew_angle:.1f}° skew")

                with Timer(f"OCR page {i}", auto_log=DEBUG_MODE):
                    page_text = pytesseract.image_to_string(image)
                    ocr_text += page_text + "\n"

            # Log preprocessing summary
            if preprocessor is not None:
                debug(f"Preprocessing summary: {total_preprocessing_time:.1f}ms total, {total_skew_corrections}/{len(images)} pages deskewed")

            # Calculate confidence
            confidence = self._calculate_dictionary_confidence(ocr_text)
            debug(f"OCR confidence: {confidence:.1f}%")

            return {
                'method': 'ocr' if preprocessor is None else 'ocr_enhanced',
                'confidence': int(confidence),
                'extracted_text': ocr_text,
                'page_count': page_count or len(images),
                'status': 'success'
            }

        except Exception as e:
            return {
                'status': 'error',
                'error_message': f"OCR processing failed: {str(e)}",
                'page_count': page_count
            }

    def _is_page_number(self, line: str) -> bool:
        """
        Check if a line is a page number.

        Common patterns:
        - "Page 1", "Page 1 of 10"
        - "- 1 -", "- 2 -"
        - Just a number: "1", "2"
        - "P. 1", "Pg. 1"
        """
        line = line.strip()

        # Pattern 1: "Page X" or "Page X of Y"
        if re.match(r'^Page\s+\d+(\s+of\s+\d+)?$', line, re.IGNORECASE):
            return True

        # Pattern 2: "- X -" or "– X –"
        if re.match(r'^[-–]\s*\d+\s*[-–]$', line):
            return True

        # Pattern 3: Just a number (but not if it's part of a list like "1.")
        if re.match(r'^\d+$', line) and len(line) <= 4:
            return True

        # Pattern 4: "P. X" or "Pg. X" or "p. X"
        if re.match(r'^P(g)?\.?\s*\d+$', line, re.IGNORECASE):
            return True

        # Pattern 5: "X/Y" (page X of Y)
        if re.match(r'^\d+/\d+$', line):
            return True

        return False

    def _extract_case_numbers(self, text: str) -> list:
        """
        Extract case numbers from text.

        Common patterns:
        - "Case No. 1:23-cv-12345"
        - "Index No. 123456/2024"
        - "Docket No. 2024-12345"
        - "Case No.: 12345"
        """
        case_numbers = []

        # Federal court pattern: "Case No. 1:23-cv-12345"
        federal = re.findall(r'Case\s+No\.?\s*:?\s*\d+:\d+-\w+-\d+', text, re.IGNORECASE)
        case_numbers.extend(federal)

        # NY Index Number: "Index No. 123456/2024"
        index = re.findall(r'Index\s+No\.?\s*:?\s*\d+/\d{4}', text, re.IGNORECASE)
        case_numbers.extend(index)

        # Generic docket: "Docket No. 2024-12345"
        docket = re.findall(r'Docket\s+No\.?\s*:?\s*\d+-\d+', text, re.IGNORECASE)
        case_numbers.extend(docket)

        # Generic case number: "Case No.: 12345"
        generic = re.findall(r'Case\s+No\.?\s*:?\s*\d+', text, re.IGNORECASE)
        case_numbers.extend(generic)

        return list(set(case_numbers))  # Remove duplicates

    def _normalize_text(self, text: str) -> str:
        """
        Apply basic text normalization rules (Step 2 of pipeline).

        4-stage normalization pipeline with comprehensive logging:
        1. De-hyphenation
        2. Page number removal
        3. Line filtering
        4. Whitespace normalization

        Normalization Rules:
        1. De-hyphenation (rejoin words split across lines) - FIRST to preserve content
        2. Page number removal
        3. Line filtering (remove short lines, require lowercase or legal headers)
        4. Whitespace normalization

        Args:
            text: Raw extracted text

        Returns:
            Fully normalized text
        """
        debug("Applying text normalization rules")

        # Stage 1: De-hyphenation (do this FIRST before line filtering)
        debug("  Stage 1: De-hyphenation")
        start = time.time()
        original_len = len(text)

        try:
            text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
            duration = time.time() - start
            debug(f"    ✅ SUCCESS ({duration:.3f}s) - Rejoin hyphenated words")
            debug(f"       Input: {original_len} | Output: {len(text)} | Delta: {len(text) - original_len:+d}")
        except Exception as e:
            duration = time.time() - start
            debug(f"    ❌ FAILED ({duration:.3f}s) - {type(e).__name__}: {str(e)}")
            raise

        # Stage 2: Page Number Removal
        debug("  Stage 2: Page number removal")
        start = time.time()
        original_len = len(text)

        try:
            lines = text.split('\n')
            lines_filtered = []
            for line in lines:
                if not self._is_page_number(line):
                    lines_filtered.append(line)
                else:
                    debug(f"    Removed page number: {line}")
            removed_count = len(lines) - len(lines_filtered)
            text = '\n'.join(lines_filtered)
            duration = time.time() - start
            debug(f"    ✅ SUCCESS ({duration:.3f}s) - Removed {removed_count} page markers")
            debug(f"       Input: {original_len} | Output: {len(text)} | Delta: {len(text) - original_len:+d}")
        except Exception as e:
            duration = time.time() - start
            debug(f"    ❌ FAILED ({duration:.3f}s) - {type(e).__name__}: {str(e)}")
            raise

        # Stage 3: Line Filtering
        debug("  Stage 3: Line filtering")
        start = time.time()
        original_len = len(text)

        try:
            normalized_lines = []
            for line in text.split('\n'):
                # Minimum length check
                if len(line) <= MIN_LINE_LENGTH:
                    # Exception: Allow short legal headers even if under minimum length
                    is_legal_header = (
                        line.isupper() and
                        len(line) < 50 and
                        any(keyword in line for keyword in self.legal_keywords)
                    )
                    if not is_legal_header:
                        continue

                # Check if line has lowercase letters
                has_lowercase = any(c.islower() for c in line)

                # Check if it's a legal header (all caps + contains legal keyword)
                is_legal_header = (
                    line.isupper() and
                    len(line) < 50 and
                    any(keyword in line for keyword in self.legal_keywords)
                )

                # Count character types
                alpha_count = sum(c.isalpha() for c in line)
                other_count = sum(not c.isalpha() and not c.isspace() for c in line)

                # Keep line if it passes all tests
                if (has_lowercase or is_legal_header) and alpha_count > other_count:
                    normalized_lines.append(line)

            removed_count = len(text.split('\n')) - len(normalized_lines)
            text = '\n'.join(normalized_lines)
            duration = time.time() - start
            debug(f"    ✅ SUCCESS ({duration:.3f}s) - Filtered {removed_count} lines")
            debug(f"       Input: {original_len} | Output: {len(text)} | Delta: {len(text) - original_len:+d}")
        except Exception as e:
            duration = time.time() - start
            debug(f"    ❌ FAILED ({duration:.3f}s) - {type(e).__name__}: {str(e)}")
            raise

        # Stage 4: Whitespace Normalization
        debug("  Stage 4: Whitespace normalization")
        start = time.time()
        original_len = len(text)

        try:
            # Remove excess blank lines (max 1 between paragraphs)
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
            text = text.strip()
            duration = time.time() - start
            debug(f"    ✅ SUCCESS ({duration:.3f}s) - Normalize whitespace")
            debug(f"       Input: {original_len} | Output: {len(text)} | Delta: {len(text) - original_len:+d}")
        except Exception as e:
            duration = time.time() - start
            debug(f"    ❌ FAILED ({duration:.3f}s) - {type(e).__name__}: {str(e)}")
            raise

        return text


def main():
    """Command-line interface for the raw text extractor."""
    parser = argparse.ArgumentParser(
        description="LocalScribe Raw Text Extractor - Extract and normalize text from legal documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python raw_text_extractor.py --input complaint.pdf

  # Process multiple files
  python raw_text_extractor.py --input file1.pdf file2.pdf file3.pdf

  # Specify output directory and jurisdiction
  python raw_text_extractor.py --input *.pdf --output-dir ./extracted --jurisdiction ny

  # Debug mode (verbose logging)
  DEBUG=true python raw_text_extractor.py --input test.pdf
        """
    )

    parser.add_argument(
        '--input',
        nargs='+',
        required=True,
        help='Input file(s) to process (PDF, TXT, RTF)'
    )

    parser.add_argument(
        '--output-dir',
        default='./extracted',
        help='Output directory for extracted text files (default: ./extracted)'
    )

    parser.add_argument(
        '--jurisdiction',
        default='ny',
        choices=['ny', 'ca', 'federal'],
        help='Legal jurisdiction for keyword filtering (default: ny)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extractor
    info(f"Initializing LocalScribe Raw Text Extractor (jurisdiction: {args.jurisdiction})")
    extractor = RawTextExtractor(jurisdiction=args.jurisdiction)

    # Process files
    results = []
    for file_path in args.input:
        result = extractor.process_document(file_path)
        results.append(result)

        # Save extracted text if successful
        if result['status'] in ['success', 'warning'] and result['extracted_text']:
            # Create output filename
            output_filename = f"{Path(result['filename']).stem}_extracted.txt"
            output_path = output_dir / output_filename

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['extracted_text'])

            info(f"Saved extracted text to: {output_path}")

    # Print summary report
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)

    for result in results:
        status_symbol = {
            'success': '[OK]',
            'warning': '[WARN]',
            'error': '[ERROR]'
        }.get(result['status'], '[?]')

        print(f"\n{status_symbol} {result['filename']}")
        print(f"  Status: {result['status'].upper()}")
        print(f"  Method: {result['method'] or 'N/A'}")
        print(f"  Confidence: {result['confidence']}%")
        size_mb = result['file_size'] / (1024 * 1024) if result['file_size'] else 0
        print(f"  Size: {size_mb:.2f} MB")

        if result.get('page_count'):
            print(f"  Pages: {result['page_count']}")

        if result['error_message']:
            print(f"  Error: {result['error_message']}")

    # Summary statistics
    total = len(results)
    success = sum(1 for r in results if r['status'] == 'success')
    warnings = sum(1 for r in results if r['status'] == 'warning')
    errors = sum(1 for r in results if r['status'] == 'error')

    print("\n" + "=" * 60)
    print(f"Total: {total} | Success: {success} | Warnings: {warnings} | Errors: {errors}")
    print("=" * 60)


if __name__ == "__main__":
    # Use debug default file if in debug mode and no args provided
    if DEBUG_MODE and len(os.sys.argv) == 1 and DEBUG_DEFAULT_FILE.exists():
        info("DEBUG MODE: Using default test file")
        os.sys.argv.extend(['--input', str(DEBUG_DEFAULT_FILE)])

    main()
