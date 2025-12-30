"""
Image Preprocessing Module for OCR Enhancement

Provides a preprocessing pipeline for scanned document images to improve OCR accuracy.
Based on research showing preprocessing can improve OCR accuracy by 20-50%.

Pipeline stages:
1. Grayscale conversion
2. Noise removal (denoising)
3. Contrast enhancement (CLAHE)
4. Adaptive thresholding (binarization)
5. Deskewing (rotation correction)
6. Border padding

References:
- Tesseract documentation: https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
- OpenCV documentation: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src.logging_config import debug, warning

# Lazy import deskew to avoid import errors if not installed
_deskew_module = None


def _get_deskew():
    """Lazy load deskew module."""
    global _deskew_module
    if _deskew_module is None:
        try:
            from deskew import determine_skew
            _deskew_module = determine_skew
        except ImportError:
            warning("deskew library not installed. Skew correction disabled.")
            _deskew_module = False
    return _deskew_module


@dataclass
class PreprocessingStats:
    """Statistics from image preprocessing."""
    original_size: tuple[int, int] = (0, 0)
    processed_size: tuple[int, int] = (0, 0)
    skew_angle: float = 0.0
    skew_corrected: bool = False
    denoised: bool = False
    contrast_enhanced: bool = False
    binarized: bool = False
    border_added: bool = False
    total_time_ms: float = 0.0
    stage_times: dict = field(default_factory=dict)


class ImagePreprocessor:
    """
    Preprocesses scanned document images for improved OCR accuracy.

    This class implements a multi-stage preprocessing pipeline:
    1. Grayscale conversion - Simplifies the image for processing
    2. Noise removal - Reduces scanner artifacts and noise
    3. Contrast enhancement - Improves text/background separation
    4. Adaptive thresholding - Converts to black/white with local adaptation
    5. Deskewing - Corrects rotated scans
    6. Border padding - Ensures text isn't at edge (Tesseract requirement)

    Usage:
        preprocessor = ImagePreprocessor()
        processed_image, stats = preprocessor.preprocess(pil_image)
        # processed_image is a PIL Image ready for pytesseract
    """

    def __init__(
        self,
        denoise_strength: int = 10,
        adaptive_block_size: int = 11,
        adaptive_constant: int = 2,
        border_size: int = 10,
        max_skew_angle: float = 10.0,
        enable_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: tuple[int, int] = (8, 8),
    ):
        """
        Initialize the preprocessor with configurable parameters.

        Args:
            denoise_strength: Strength of denoising filter (higher = more smoothing, 10 is default)
            adaptive_block_size: Size of pixel neighborhood for adaptive thresholding (must be odd)
            adaptive_constant: Constant subtracted from mean in adaptive threshold
            border_size: Size of white border to add (pixels)
            max_skew_angle: Maximum angle to attempt deskewing (degrees)
            enable_clahe: Whether to apply CLAHE contrast enhancement
            clahe_clip_limit: Contrast limit for CLAHE (2.0 is typical)
            clahe_grid_size: Grid size for CLAHE histogram equalization
        """
        self.denoise_strength = denoise_strength
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_constant = adaptive_constant
        self.border_size = border_size
        self.max_skew_angle = max_skew_angle
        self.enable_clahe = enable_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size

        # Ensure block size is odd
        if self.adaptive_block_size % 2 == 0:
            self.adaptive_block_size += 1

    def preprocess(self, image: Image.Image) -> tuple[Image.Image, PreprocessingStats]:
        """
        Apply full preprocessing pipeline to an image.

        Args:
            image: PIL Image to preprocess

        Returns:
            Tuple of (processed PIL Image, PreprocessingStats)
        """
        start_time = time.time()
        stats = PreprocessingStats()
        stats.original_size = image.size

        debug(f"Starting image preprocessing: {image.size[0]}x{image.size[1]}")

        # Convert PIL Image to OpenCV format (numpy array)
        cv_image = self._pil_to_cv2(image)

        # Stage 1: Grayscale conversion
        stage_start = time.time()
        if len(cv_image.shape) == 3:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
            debug("  [1] Grayscale conversion: applied")
        else:
            gray = cv_image
            debug("  [1] Grayscale conversion: already grayscale")
        stats.stage_times['grayscale'] = (time.time() - stage_start) * 1000

        # Stage 2: Noise removal (denoising)
        stage_start = time.time()
        try:
            denoised = cv2.fastNlMeansDenoising(gray, None, self.denoise_strength, 7, 21)
            stats.denoised = True
            debug(f"  [2] Denoising: applied (strength={self.denoise_strength})")
        except Exception as e:
            denoised = gray
            warning(f"  [2] Denoising failed: {e}")
        stats.stage_times['denoise'] = (time.time() - stage_start) * 1000

        # Stage 3: Contrast enhancement (CLAHE)
        stage_start = time.time()
        if self.enable_clahe:
            try:
                clahe = cv2.createCLAHE(
                    clipLimit=self.clahe_clip_limit,
                    tileGridSize=self.clahe_grid_size
                )
                enhanced = clahe.apply(denoised)
                stats.contrast_enhanced = True
                debug(f"  [3] CLAHE contrast enhancement: applied (clip={self.clahe_clip_limit})")
            except Exception as e:
                enhanced = denoised
                warning(f"  [3] CLAHE enhancement failed: {e}")
        else:
            enhanced = denoised
            debug("  [3] CLAHE contrast enhancement: skipped (disabled)")
        stats.stage_times['clahe'] = (time.time() - stage_start) * 1000

        # Stage 4: Adaptive thresholding (binarization)
        stage_start = time.time()
        try:
            binary = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.adaptive_block_size,
                self.adaptive_constant
            )
            stats.binarized = True
            debug(f"  [4] Adaptive thresholding: applied (block={self.adaptive_block_size}, C={self.adaptive_constant})")
        except Exception as e:
            # Fallback to Otsu's method
            try:
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                stats.binarized = True
                debug(f"  [4] Adaptive thresholding failed, Otsu fallback: applied")
            except Exception as e2:
                binary = enhanced
                warning(f"  [4] All binarization failed: {e2}")
        stats.stage_times['binarize'] = (time.time() - stage_start) * 1000

        # Stage 5: Deskewing
        stage_start = time.time()
        deskewed, skew_angle = self._deskew(binary)
        stats.skew_angle = skew_angle
        stats.skew_corrected = abs(skew_angle) > 0.5  # Only count if significant correction
        if stats.skew_corrected:
            debug(f"  [5] Deskewing: corrected {skew_angle:.2f}° rotation")
        else:
            debug(f"  [5] Deskewing: no significant skew detected ({skew_angle:.2f}°)")
        stats.stage_times['deskew'] = (time.time() - stage_start) * 1000

        # Stage 6: Border padding
        stage_start = time.time()
        if self.border_size > 0:
            padded = cv2.copyMakeBorder(
                deskewed,
                self.border_size, self.border_size,
                self.border_size, self.border_size,
                cv2.BORDER_CONSTANT,
                value=255  # White border
            )
            stats.border_added = True
            debug(f"  [6] Border padding: added {self.border_size}px white border")
        else:
            padded = deskewed
            debug("  [6] Border padding: skipped (disabled)")
        stats.stage_times['border'] = (time.time() - stage_start) * 1000

        # Convert back to PIL Image
        result = self._cv2_to_pil(padded)
        stats.processed_size = result.size
        stats.total_time_ms = (time.time() - start_time) * 1000

        debug(f"Image preprocessing complete: {stats.processed_size[0]}x{stats.processed_size[1]} in {stats.total_time_ms:.1f}ms")

        return result, stats

    def _deskew(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Detect and correct skew in an image.

        Args:
            image: Grayscale/binary image as numpy array

        Returns:
            Tuple of (deskewed image, detected angle in degrees)
        """
        determine_skew = _get_deskew()

        if determine_skew is False:
            # deskew library not available
            return image, 0.0

        try:
            # Determine skew angle
            angle = determine_skew(image)

            # Handle None result (no skew detected)
            if angle is None:
                return image, 0.0

            # Limit correction to reasonable angles
            if abs(angle) > self.max_skew_angle:
                warning(f"Detected skew angle {angle:.2f}° exceeds max {self.max_skew_angle}°, skipping correction")
                return image, angle

            # Skip very small corrections (< 0.5 degrees)
            if abs(angle) < 0.5:
                return image, angle

            # Rotate to correct skew
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Calculate new image bounds after rotation
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            # Adjust the rotation matrix for the new bounds
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]

            rotated = cv2.warpAffine(
                image,
                rotation_matrix,
                (new_w, new_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255  # White background
            )

            return rotated, angle

        except Exception as e:
            warning(f"Deskew failed: {e}")
            return image, 0.0

    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format (numpy array)."""
        # Convert to RGB if necessary (PIL might be RGBA, L, etc.)
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode == 'L':
            # Already grayscale
            return np.array(pil_image)
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        return np.array(pil_image)

    def _cv2_to_pil(self, cv_image: np.ndarray) -> Image.Image:
        """Convert OpenCV format (numpy array) to PIL Image."""
        if len(cv_image.shape) == 2:
            # Grayscale
            return Image.fromarray(cv_image, mode='L')
        elif len(cv_image.shape) == 3:
            # Color - convert BGR to RGB if needed
            if cv_image.shape[2] == 3:
                return Image.fromarray(cv_image)
            elif cv_image.shape[2] == 4:
                return Image.fromarray(cv_image, mode='RGBA')

        return Image.fromarray(cv_image)


def preprocess_for_ocr(
    image: Image.Image,
    denoise_strength: int = 10,
    enable_clahe: bool = True,
) -> tuple[Image.Image, PreprocessingStats]:
    """
    Convenience function to preprocess an image for OCR.

    Args:
        image: PIL Image to preprocess
        denoise_strength: Strength of denoising (10 is default, higher = more smoothing)
        enable_clahe: Whether to apply contrast enhancement

    Returns:
        Tuple of (processed PIL Image, PreprocessingStats)
    """
    preprocessor = ImagePreprocessor(
        denoise_strength=denoise_strength,
        enable_clahe=enable_clahe,
    )
    return preprocessor.preprocess(image)


# Quick test if run directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python image_preprocessor.py <image_path>")
        print("Example: python image_preprocessor.py scanned_doc.png")
        sys.exit(1)

    input_path = sys.argv[1]

    # Load image
    image = Image.open(input_path)
    print(f"Loaded: {input_path} ({image.size[0]}x{image.size[1]})")

    # Preprocess
    preprocessor = ImagePreprocessor()
    processed, stats = preprocessor.preprocess(image)

    # Save output
    output_path = input_path.rsplit('.', 1)[0] + '_preprocessed.png'
    processed.save(output_path)

    print(f"\nPreprocessing Stats:")
    print(f"  Original size: {stats.original_size}")
    print(f"  Processed size: {stats.processed_size}")
    print(f"  Skew angle: {stats.skew_angle:.2f}°")
    print(f"  Skew corrected: {stats.skew_corrected}")
    print(f"  Denoised: {stats.denoised}")
    print(f"  Contrast enhanced: {stats.contrast_enhanced}")
    print(f"  Binarized: {stats.binarized}")
    print(f"  Border added: {stats.border_added}")
    print(f"  Total time: {stats.total_time_ms:.1f}ms")
    print(f"\nSaved to: {output_path}")
