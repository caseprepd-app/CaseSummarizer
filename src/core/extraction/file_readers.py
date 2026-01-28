"""
File Readers for Non-PDF Document Formats.

Provides text extraction for TXT, RTF, DOCX, and image files (PNG/JPG).
Each reader returns a standardized result dict with text, confidence, and status.

Supported formats:
    - TXT: Direct file read (UTF-8)
    - RTF: striprtf library conversion
    - DOCX: python-docx paragraph and table extraction
    - PNG/JPG: OCR via Tesseract (delegates to OCRProcessor)

Example usage:
    >>> from src.core.extraction.dictionary_utils import TermExtractionHelpers
    >>> from src.core.extraction.ocr_processor import OCRProcessor
    >>> readers = FileReaders(TermExtractionHelpers(), OCRProcessor(TermExtractionHelpers()))
    >>> result = readers.read_text_file(Path("document.txt"))
    >>> print(result['method'])  # 'direct_read'
"""

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

from .dictionary_utils import TermExtractionHelpers


class FileReaders:
    """
    Reads text from non-PDF document formats.

    Provides a unified interface for extracting text from TXT, RTF, DOCX,
    and image files. Each method returns a standardized result dict.

    Attributes:
        dictionary: TermExtractionHelpers for confidence calculation
        ocr_processor: Optional OCRProcessor for image files
    """

    def __init__(self, dictionary: TermExtractionHelpers, ocr_processor=None):
        """
        Initialize the file readers.

        Args:
            dictionary: TermExtractionHelpers instance for confidence calculation
            ocr_processor: Optional OCRProcessor for image files
        """
        self.dictionary = dictionary
        self.ocr_processor = ocr_processor

    def read_text_file(self, file_path: Path) -> dict:
        """
        Read a plain text (.txt) file.

        Args:
            file_path: Path to the text file

        Returns:
            Dict with keys:
                - text: File contents
                - method: 'direct_read'
                - confidence: Dictionary confidence percentage
                - status: 'success' or 'error'
                - error_message: Error description if failed

        Example:
            >>> readers = FileReaders(TermExtractionHelpers())
            >>> result = readers.read_text_file(Path("notes.txt"))
            >>> print(f"Read {len(result['text'])} characters")
        """
        logger.debug("Reading text file: %s", file_path.name)

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                text = f.read()

            confidence = self.dictionary.calculate_confidence(text)
            logger.debug("Text file dictionary confidence: %.1f%%", confidence)

            return {
                "text": text,
                "method": "direct_read",
                "confidence": int(confidence),
                "status": "success",
                "error_message": None,
            }

        except Exception as e:
            return {
                "text": None,
                "method": None,
                "confidence": 0,
                "status": "error",
                "error_message": f"Failed to read text file: {e!s}",
            }

    def read_rtf_file(self, file_path: Path) -> dict:
        """
        Read a Rich Text Format (.rtf) file.

        Uses striprtf library to convert RTF to plain text.

        Args:
            file_path: Path to the RTF file

        Returns:
            Dict with keys:
                - text: Converted plain text
                - method: 'rtf_extraction'
                - confidence: Dictionary confidence percentage
                - status: 'success' or 'error'
                - error_message: Error description if failed

        Example:
            >>> readers = FileReaders(TermExtractionHelpers())
            >>> result = readers.read_rtf_file(Path("document.rtf"))
        """
        logger.debug("Reading RTF file: %s", file_path.name)

        try:
            from striprtf.striprtf import rtf_to_text

            with open(file_path, encoding="utf-8", errors="ignore") as f:
                rtf_content = f.read()

            text = rtf_to_text(rtf_content)
            logger.debug("Extracted %d characters from RTF", len(text))

            confidence = self.dictionary.calculate_confidence(text)
            logger.debug("RTF dictionary confidence: %.1f%%", confidence)

            return {
                "text": text,
                "method": "rtf_extraction",
                "confidence": int(confidence),
                "status": "success",
                "error_message": None,
            }

        except Exception as e:
            return {
                "text": None,
                "method": None,
                "confidence": 0,
                "status": "error",
                "error_message": f"Failed to read RTF file: {e!s}",
            }

    def read_docx_file(self, file_path: Path) -> dict:
        """
        Read a Word document (.docx) file.

        Extracts text from paragraphs and tables using python-docx.

        Args:
            file_path: Path to the DOCX file

        Returns:
            Dict with keys:
                - text: Extracted text from paragraphs and tables
                - method: 'docx_extraction'
                - confidence: Dictionary confidence percentage
                - page_count: Approximate page count (from sections)
                - status: 'success' or 'error'
                - error_message: Error description if failed

        Example:
            >>> readers = FileReaders(TermExtractionHelpers())
            >>> result = readers.read_docx_file(Path("report.docx"))
            >>> print(f"Pages: ~{result['page_count']}")
        """
        logger.debug("Reading Word document: %s", file_path.name)

        try:
            from docx import Document

            doc = Document(file_path)

            # Extract text from paragraphs
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
                    "text": None,
                    "method": None,
                    "confidence": 0,
                    "page_count": 0,
                    "status": "error",
                    "error_message": "Word document contains no readable text.",
                }

            confidence = self.dictionary.calculate_confidence(text)
            logger.debug("DOCX dictionary confidence: %.1f%%", confidence)

            return {
                "text": text,
                "method": "docx_extraction",
                "confidence": int(confidence),
                "page_count": len(doc.sections) or 1,  # Approximate
                "status": "success",
                "error_message": None,
            }

        except Exception as e:
            return {
                "text": None,
                "method": None,
                "confidence": 0,
                "page_count": 0,
                "status": "error",
                "error_message": f"Failed to read Word document: {e!s}",
            }

    def read_image_file(self, file_path: Path) -> dict:
        """
        Read an image file (.png, .jpg, .jpeg) using OCR.

        Delegates to OCRProcessor for Tesseract-based text extraction.

        Args:
            file_path: Path to the image file

        Returns:
            Dict with keys:
                - text: OCR-extracted text
                - method: 'image_ocr'
                - confidence: Dictionary confidence percentage
                - page_count: Always 1 for images
                - status: 'success' or 'error'
                - error_message: Error description if failed

        Example:
            >>> readers = FileReaders(TermExtractionHelpers(), OCRProcessor(TermExtractionHelpers()))
            >>> result = readers.read_image_file(Path("scan.png"))
        """
        logger.debug("Reading image file: %s", file_path.name)

        if self.ocr_processor is None:
            return {
                "text": None,
                "method": None,
                "confidence": 0,
                "page_count": 1,
                "status": "error",
                "error_message": "OCR processor not available for image files.",
            }

        try:
            img = Image.open(file_path)
            result = self.ocr_processor.process_image(img)

            # Add page_count to result
            result["page_count"] = 1

            return result

        except Exception as e:
            return {
                "text": None,
                "method": None,
                "confidence": 0,
                "page_count": 1,
                "status": "error",
                "error_message": f"Failed to process image: {e!s}",
            }
