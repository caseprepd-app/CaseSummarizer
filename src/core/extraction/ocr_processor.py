"""
OCR Processing for Scanned Documents.

Performs Optical Character Recognition (OCR) on PDF pages or images using
Tesseract with optional image preprocessing to improve accuracy.

The preprocessing pipeline can improve OCR accuracy by 20-50% on scanned
documents by:
    - Correcting orientation (90/180/270 degree rotation)
    - Cropping document from background (phone photos)
    - Converting to grayscale
    - Removing noise
    - Enhancing contrast
    - Binarizing (adaptive thresholding)
    - Deskewing (fixing slight rotation)

Example usage:
    >>> from src.core.extraction.dictionary_utils import DictionaryUtils
    >>> processor = OCRProcessor(DictionaryUtils())
    >>> result = processor.process_pdf(Path("scanned.pdf"), page_count=10)
    >>> print(f"OCR confidence: {result['confidence']}%")
"""

import logging
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path

from src.config import (
    OCR_DENOISE_STRENGTH,
    OCR_DPI,
    OCR_ENABLE_CLAHE,
    OCR_PREPROCESSING_ENABLED,
)
from src.logging_config import Timer

logger = logging.getLogger(__name__)

from .dictionary_utils import DictionaryUtils
from .image_preprocessor import ImagePreprocessor


class OCRProcessor:
    """
    Performs OCR on scanned PDFs and images with optional preprocessing.

    Uses Tesseract for text recognition and an optional preprocessing pipeline
    to improve accuracy on poor-quality scans.

    Attributes:
        dictionary: DictionaryUtils for confidence calculation
        preprocessor: Optional ImagePreprocessor for image enhancement
    """

    def __init__(self, dictionary: DictionaryUtils):
        """
        Initialize the OCR processor.

        Args:
            dictionary: DictionaryUtils instance for confidence calculation
        """
        self.dictionary = dictionary
        self.preprocessor = None

        if OCR_PREPROCESSING_ENABLED:
            self.preprocessor = ImagePreprocessor(
                denoise_strength=OCR_DENOISE_STRENGTH,
                enable_clahe=OCR_ENABLE_CLAHE,
            )
            logger.debug(
                "OCR preprocessing enabled (denoise=%d, clahe=%s)",
                OCR_DENOISE_STRENGTH,
                OCR_ENABLE_CLAHE,
            )

    def process_pdf(self, file_path: Path, page_count: int | None = None) -> dict:
        """
        Perform OCR on a PDF file.

        Converts each PDF page to an image, optionally preprocesses it,
        then runs Tesseract OCR.

        Args:
            file_path: Path to the PDF file
            page_count: Known page count (for result, if already determined)

        Returns:
            Dict with keys:
                - text: OCR-extracted text
                - page_count: Number of pages processed
                - method: 'ocr' or 'ocr_enhanced' (with preprocessing)
                - confidence: Dictionary confidence percentage
                - status: 'success' or 'error'
                - error_message: Error description if failed

        Example:
            >>> processor = OCRProcessor(DictionaryUtils())
            >>> result = processor.process_pdf(Path("scan.pdf"))
            >>> print(result['method'])  # 'ocr_enhanced' if preprocessing enabled
        """
        logger.debug("Starting OCR on %s", file_path.name)

        try:
            # Convert PDF to images
            with Timer("PDF to images conversion"):
                images = convert_from_path(str(file_path), dpi=OCR_DPI)

            # Track preprocessing stats
            total_preprocessing_time = 0.0
            total_skew_corrections = 0

            # OCR each page
            ocr_text = ""
            for i, image in enumerate(images, 1):
                logger.debug("OCR processing page %d/%d", i, len(images))

                # Apply preprocessing if enabled
                if self.preprocessor is not None:
                    with Timer(f"Preprocessing page {i}", auto_log=True):
                        image, stats = self.preprocessor.preprocess(image)
                        total_preprocessing_time += stats.total_time_ms
                        if stats.skew_corrected:
                            total_skew_corrections += 1
                            logger.debug("Page %d: corrected %.1f deg skew", i, stats.skew_angle)

                # Run Tesseract
                with Timer(f"OCR page {i}", auto_log=True):
                    page_text = pytesseract.image_to_string(image)
                    ocr_text += page_text + "\n"

            # Log preprocessing summary
            if self.preprocessor is not None:
                logger.debug(
                    "Preprocessing summary: %.1fms total, %d/%d pages deskewed",
                    total_preprocessing_time,
                    total_skew_corrections,
                    len(images),
                )

            # Calculate confidence
            confidence = self.dictionary.calculate_confidence(ocr_text)
            logger.debug("OCR confidence: %.1f%%", confidence)

            return {
                "text": ocr_text,
                "page_count": page_count or len(images),
                "method": "ocr_enhanced" if self.preprocessor else "ocr",
                "confidence": int(confidence),
                "status": "success",
                "error_message": None,
            }

        except Exception as e:
            return {
                "text": None,
                "page_count": page_count,
                "method": None,
                "confidence": 0,
                "status": "error",
                "error_message": f"OCR processing failed: {e!s}",
            }

    def process_image(self, image) -> dict:
        """
        Perform OCR on a single image (PIL Image object).

        Args:
            image: PIL Image object to process

        Returns:
            Dict with keys:
                - text: OCR-extracted text
                - method: 'image_ocr'
                - confidence: Dictionary confidence percentage
                - status: 'success' or 'error'
                - error_message: Error description if failed

        Example:
            >>> from PIL import Image
            >>> processor = OCRProcessor(DictionaryUtils())
            >>> img = Image.open("document.png")
            >>> result = processor.process_image(img)
        """
        try:
            # Apply preprocessing if enabled
            if self.preprocessor is not None:
                processed_img, stats = self.preprocessor.preprocess(image)
                logger.debug("Image preprocessing applied: %s", stats)
            else:
                # At minimum convert to grayscale
                processed_img = image.convert("L")

            # Perform OCR
            text = pytesseract.image_to_string(processed_img)

            if not text.strip():
                return {
                    "text": None,
                    "method": "image_ocr",
                    "confidence": 0,
                    "status": "error",
                    "error_message": "Could not extract text from image.",
                }

            # Calculate confidence
            confidence = self.dictionary.calculate_confidence(text)
            logger.debug("Image OCR confidence: %.1f%%", confidence)

            return {
                "text": text,
                "method": "image_ocr",
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
                "error_message": f"Failed to process image: {e!s}",
            }
