"""
OCR Availability Detection.

Checks whether Tesseract and Poppler executables are available on the system.
Used by the UI layer to decide whether to show an install prompt before
processing scanned documents.

This module lives in the extraction layer and has NO UI imports.

Example:
    >>> status = check_ocr_availability()
    >>> if status == OCRStatus.AVAILABLE:
    ...     print("OCR ready")
"""

import logging
import shutil
from enum import Enum

logger = logging.getLogger(__name__)


class OCRStatus(Enum):
    """
    OCR system availability status.

    Attributes:
        AVAILABLE: Both Tesseract and Poppler are installed.
        TESSERACT_MISSING: Tesseract not found (Poppler may or may not be present).
        POPPLER_MISSING: Poppler not found (Tesseract is present).
        BOTH_MISSING: Neither Tesseract nor Poppler found.
    """

    AVAILABLE = "available"
    TESSERACT_MISSING = "tesseract_missing"
    POPPLER_MISSING = "poppler_missing"
    BOTH_MISSING = "both_missing"


def check_ocr_availability() -> OCRStatus:
    """
    Check whether OCR executables are installed.

    Looks for 'tesseract' and 'pdftoppm' (Poppler) on the system PATH.

    Returns:
        OCRStatus indicating which components are available.
    """
    has_tesseract = shutil.which("tesseract") is not None
    has_poppler = shutil.which("pdftoppm") is not None

    if has_tesseract and has_poppler:
        logger.debug("OCR available: Tesseract and Poppler found")
        return OCRStatus.AVAILABLE
    elif not has_tesseract and not has_poppler:
        logger.debug("OCR unavailable: Tesseract and Poppler both missing")
        return OCRStatus.BOTH_MISSING
    elif not has_tesseract:
        logger.debug("OCR partially available: Tesseract missing")
        return OCRStatus.TESSERACT_MISSING
    else:
        logger.debug("OCR partially available: Poppler missing")
        return OCRStatus.POPPLER_MISSING
