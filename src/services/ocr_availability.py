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


def _find_tesseract() -> bool:
    """
    Check if Tesseract is available: bundled, on PATH, or in standard locations.

    Returns:
        True if tesseract executable is found.
    """
    from src.config import TESSERACT_BUNDLED_EXE

    if TESSERACT_BUNDLED_EXE.exists():
        logger.debug("Tesseract found at bundled path %s", TESSERACT_BUNDLED_EXE)
        return True

    if shutil.which("tesseract") is not None:
        return True

    # Check standard Windows install locations
    import os
    from pathlib import Path

    standard_paths = [
        Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
        Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
        Path(os.environ.get("LOCALAPPDATA", ""), "Tesseract-OCR/tesseract.exe"),
    ]
    for path in standard_paths:
        if path.exists():
            logger.debug("Tesseract found at %s (not on PATH)", path)
            return True

    return False


def _find_poppler() -> bool:
    """
    Check if Poppler (pdftoppm) is available: bundled or on PATH.

    Returns:
        True if pdftoppm executable is found.
    """
    from src.config import POPPLER_BUNDLED_DIR

    if (POPPLER_BUNDLED_DIR / "pdftoppm.exe").exists():
        logger.debug("Poppler found at bundled path %s", POPPLER_BUNDLED_DIR)
        return True

    if shutil.which("pdftoppm") is not None:
        return True

    return False


def check_ocr_availability() -> OCRStatus:
    """
    Check whether OCR executables are installed.

    Looks for Tesseract and Poppler (pdftoppm) in bundled paths first,
    then on PATH and standard Windows install locations.

    Returns:
        OCRStatus indicating which components are available.
    """
    has_tesseract = _find_tesseract()
    has_poppler = _find_poppler()

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
