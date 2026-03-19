"""
OCR Processing for Scanned Documents.

Performs Optical Character Recognition (OCR) on PDF pages or images using
Tesseract with optional image preprocessing to improve accuracy.

Each public method returns an ExtractionResult.

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
    >>> from src.core.extraction.dictionary_utils import DictionaryTextValidator
    >>> processor = OCRProcessor(DictionaryTextValidator())
    >>> result = processor.process_pdf(Path("scanned.pdf"), page_count=10)
    >>> print(f"OCR confidence: {result['confidence']}%")
"""

import logging
import os
import shutil
import sys
from pathlib import Path

from src.config import (
    OCR_DENOISE_STRENGTH,
    OCR_DPI,
    OCR_ENABLE_CLAHE,
    OCR_PREPROCESSING_ENABLED,
)
from src.logging_config import Timer

logger = logging.getLogger(__name__)

# Standard Windows install locations for Tesseract
_TESSERACT_STANDARD_PATHS = [
    Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
    Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
    Path(os.environ.get("LOCALAPPDATA", ""), "Tesseract-OCR/tesseract.exe"),
]

_tesseract_patched = False


def _suppress_tesseract_console():
    """
    Patch pytesseract on Windows so tesseract.exe runs without a console window.

    Replaces pytesseract's subprocess.Popen reference with a wrapper that injects
    CREATE_NO_WINDOW into every call, preventing a terminal from briefly flashing
    on screen each time OCR is performed. No-op on non-Windows platforms.
    """
    import subprocess
    import types

    import pytesseract.pytesseract as _pyt

    _orig_popen = subprocess.Popen

    def _no_window_popen(*args, creationflags=0, **kwargs):
        """Popen wrapper that adds CREATE_NO_WINDOW on Windows."""
        return _orig_popen(
            *args, creationflags=creationflags | subprocess.CREATE_NO_WINDOW, **kwargs
        )

    proxy = types.ModuleType("subprocess")
    proxy.__dict__.update(vars(subprocess))
    proxy.Popen = _no_window_popen
    _pyt.subprocess = proxy
    logger.debug("Patched pytesseract to suppress console window")


def _configure_tesseract():
    """
    Set pytesseract.tesseract_cmd to the best available Tesseract binary.

    Checks bundled binary first, then PATH, then standard Windows install
    locations so users don't need to manually install or configure Tesseract.
    Also patches pytesseract on Windows (once) to hide the tesseract console.
    """
    global _tesseract_patched

    import pytesseract

    from src.config import TESSERACT_BUNDLED_EXE

    if TESSERACT_BUNDLED_EXE.exists():
        pytesseract.tesseract_cmd = str(TESSERACT_BUNDLED_EXE)
        logger.debug("Using bundled Tesseract at %s", TESSERACT_BUNDLED_EXE)
    elif shutil.which("tesseract") is not None:
        logger.debug("Using Tesseract from system PATH")
    else:
        found = False
        for path in _TESSERACT_STANDARD_PATHS:
            if path.exists():
                pytesseract.tesseract_cmd = str(path)
                logger.debug("Configured pytesseract to use %s", path)
                found = True
                break
        if not found:
            logger.warning(
                "Tesseract not found (bundled, PATH, or standard locations). "
                "OCR will not be available."
            )

    if sys.platform == "win32" and not _tesseract_patched:
        _suppress_tesseract_console()
        _tesseract_patched = True


from .dictionary_utils import DictionaryTextValidator
from .image_preprocessor import ImagePreprocessor


class OCRProcessor:
    """
    Performs OCR on scanned PDFs and images with optional preprocessing.

    Uses Tesseract for text recognition and an optional preprocessing pipeline
    to improve accuracy on poor-quality scans.

    Attributes:
        dictionary: DictionaryTextValidator for confidence calculation
        preprocessor: Optional ImagePreprocessor for image enhancement
    """

    def __init__(self, dictionary: DictionaryTextValidator):
        """
        Initialize the OCR processor.

        Args:
            dictionary: DictionaryTextValidator instance for confidence calculation
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

    def process_pdf(self, file_path: Path, page_count: int | None = None):
        """
        Perform OCR on a PDF file.

        Converts each PDF page to an image, optionally preprocesses it,
        then runs Tesseract OCR.

        Args:
            file_path: Path to the PDF file
            page_count: Known page count (for result, if already determined)

        Returns:
            ExtractionResult with text, method='ocr'/'ocr_enhanced', confidence.

        Example:
            >>> processor = OCRProcessor(DictionaryTextValidator())
            >>> result = processor.process_pdf(Path("scan.pdf"))
            >>> print(result['method'])  # 'ocr_enhanced' if preprocessing enabled
        """
        logger.debug("Starting OCR on %s", file_path.name)

        try:
            import pytesseract
            from pdf2image import convert_from_path

            _configure_tesseract()

            # Convert PDF to images (use bundled Poppler if available)
            from src.config import POPPLER_BUNDLED_DIR

            poppler_kwargs = {}
            if POPPLER_BUNDLED_DIR.exists():
                poppler_kwargs["poppler_path"] = str(POPPLER_BUNDLED_DIR)

            with Timer("PDF to images conversion"):
                images = convert_from_path(str(file_path), dpi=OCR_DPI, **poppler_kwargs)

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
            from .extraction_result import ExtractionResult

            confidence = self.dictionary.calculate_confidence(ocr_text)
            logger.debug("OCR confidence: %.1f%%", confidence)

            return ExtractionResult.success(
                ocr_text,
                "ocr_enhanced" if self.preprocessor else "ocr",
                confidence,
                page_count=page_count or len(images),
            )

        except Exception as e:
            from .extraction_result import ExtractionResult

            error_msg = str(e)
            if "poppler" in error_msg.lower() or "pdftoppm" in error_msg.lower():
                error_msg = (
                    "OCR unavailable -- Poppler binaries are missing or damaged. "
                    "Try reinstalling the application to restore them."
                )
            else:
                error_msg = f"OCR processing failed: {error_msg}"

            return ExtractionResult.error(error_msg, page_count=page_count)

    def process_pages(self, file_path: Path, page_numbers: list[int], page_count: int):
        """
        Perform OCR on specific pages of a PDF file.

        Converts only the requested pages to images and runs OCR on them.
        Page numbers are 1-indexed (matching PDF convention).

        Args:
            file_path: Path to the PDF file
            page_numbers: List of 1-indexed page numbers to OCR
            page_count: Total page count (for result metadata)

        Returns:
            ExtractionResult with pages dict, method, confidence.
        """
        logger.debug(
            "Starting per-page OCR on %s: pages %s",
            file_path.name,
            page_numbers,
        )

        try:
            import pytesseract
            from pdf2image import convert_from_path

            _configure_tesseract()

            from src.config import POPPLER_BUNDLED_DIR

            poppler_kwargs = {}
            if POPPLER_BUNDLED_DIR.exists():
                poppler_kwargs["poppler_path"] = str(POPPLER_BUNDLED_DIR)

            pages_text = {}
            all_ocr_text = []

            for page_num in sorted(page_numbers):
                logger.debug("OCR processing page %d/%d", page_num, page_count)

                with Timer(f"PDF to image page {page_num}"):
                    images = convert_from_path(
                        str(file_path),
                        dpi=OCR_DPI,
                        first_page=page_num,
                        last_page=page_num,
                        **poppler_kwargs,
                    )

                if not images:
                    logger.warning("No image produced for page %d", page_num)
                    continue

                image = images[0]

                if self.preprocessor is not None:
                    image, _stats = self.preprocessor.preprocess(image)

                with Timer(f"OCR page {page_num}"):
                    page_text = pytesseract.image_to_string(image)

                pages_text[page_num] = page_text
                all_ocr_text.append(page_text)

            from .extraction_result import ExtractionResult

            combined = "\n".join(all_ocr_text)
            confidence = self.dictionary.calculate_confidence(combined) if combined else 0

            return ExtractionResult.success(
                combined,
                "ocr_partial_enhanced" if self.preprocessor else "ocr_partial",
                confidence,
                pages=pages_text,
            )

        except Exception as e:
            from .extraction_result import ExtractionResult

            error_msg = str(e)
            if "poppler" in error_msg.lower() or "pdftoppm" in error_msg.lower():
                error_msg = (
                    "OCR unavailable -- Poppler binaries are missing or damaged. "
                    "Try reinstalling the application to restore them."
                )
            else:
                error_msg = f"Per-page OCR failed: {error_msg}"

            return ExtractionResult.error(error_msg)

    def process_image(self, image):
        """
        Perform OCR on a single image (PIL Image object).

        Args:
            image: PIL Image object to process

        Returns:
            ExtractionResult with text, method='image_ocr', confidence.

        Example:
            >>> from PIL import Image
            >>> processor = OCRProcessor(DictionaryTextValidator())
            >>> img = Image.open("document.png")
            >>> result = processor.process_image(img)
        """
        from .extraction_result import ExtractionResult

        try:
            import pytesseract

            _configure_tesseract()

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
                return ExtractionResult.error(
                    "Could not extract text from image.",
                    method="image_ocr",
                )

            # Calculate confidence
            confidence = self.dictionary.calculate_confidence(text)
            logger.debug("Image OCR confidence: %.1f%%", confidence)

            return ExtractionResult.success(text, "image_ocr", confidence)

        except Exception as e:
            return ExtractionResult.error(f"Failed to process image: {e!s}")
