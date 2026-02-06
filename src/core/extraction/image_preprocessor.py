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

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

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
            logger.warning("deskew library not installed. Skew correction disabled.")
            _deskew_module = False
    return _deskew_module


@dataclass
class PreprocessingStats:
    """Statistics from image preprocessing."""

    original_size: tuple[int, int] = (0, 0)
    processed_size: tuple[int, int] = (0, 0)
    # Orientation correction (90°/180°/270°)
    orientation_angle: int = 0
    orientation_confidence: float = 0.0
    orientation_corrected: bool = False
    # Document detection and cropping
    document_detected: bool = False
    document_corners: list | None = None
    document_cropped: bool = False
    # Fine deskewing (small angles)
    skew_angle: float = 0.0
    skew_corrected: bool = False
    # Other stages
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
        # New: Orientation and document detection
        enable_orientation_correction: bool = True,
        orientation_confidence_threshold: float = 2.0,
        enable_document_detection: bool = True,
        min_document_area_ratio: float = 0.1,
        # Existing parameters
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
            enable_orientation_correction: Detect and fix 90°/180°/270° rotation using Tesseract OSD
            orientation_confidence_threshold: Minimum confidence to apply orientation fix (0-10 scale)
            enable_document_detection: Detect document edges and crop/perspective-correct
            min_document_area_ratio: Minimum document area as ratio of image (0.1 = 10%)
            denoise_strength: Strength of denoising filter (higher = more smoothing, 10 is default)
            adaptive_block_size: Size of pixel neighborhood for adaptive thresholding (must be odd)
            adaptive_constant: Constant subtracted from mean in adaptive threshold
            border_size: Size of white border to add (pixels)
            max_skew_angle: Maximum angle to attempt deskewing (degrees)
            enable_clahe: Whether to apply CLAHE contrast enhancement
            clahe_clip_limit: Contrast limit for CLAHE (2.0 is typical)
            clahe_grid_size: Grid size for CLAHE histogram equalization
        """
        # New features
        self.enable_orientation_correction = enable_orientation_correction
        self.orientation_confidence_threshold = orientation_confidence_threshold
        self.enable_document_detection = enable_document_detection
        self.min_document_area_ratio = min_document_area_ratio
        # Existing
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

        logger.debug("Starting image preprocessing: %dx%d", image.size[0], image.size[1])

        # Stage 0a: Orientation correction
        image = self._stage_orientation(image, stats)

        # Stage 0b: Document detection and cropping
        image = self._stage_document_detection(image, stats)

        # Convert PIL Image to OpenCV format
        cv_image = self._pil_to_cv2(image)

        # Stage 1: Grayscale conversion
        gray = self._stage_grayscale(cv_image, stats)

        # Stage 2: Noise removal
        denoised = self._stage_denoise(gray, stats)

        # Stage 3: Contrast enhancement
        enhanced = self._stage_contrast(denoised, stats)

        # Stage 4: Binarization
        binary = self._stage_binarize(enhanced, stats)

        # Stage 5: Deskewing
        deskewed = self._stage_deskew(binary, stats)

        # Stage 6: Border padding
        padded = self._stage_border(deskewed, stats)

        # Convert back to PIL Image
        result = self._cv2_to_pil(padded)
        stats.processed_size = result.size
        stats.total_time_ms = (time.time() - start_time) * 1000

        logger.debug(
            "Image preprocessing complete: %dx%d in %.1fms",
            stats.processed_size[0],
            stats.processed_size[1],
            stats.total_time_ms,
        )

        return result, stats

    def _stage_orientation(self, image: Image.Image, stats: PreprocessingStats) -> Image.Image:
        """Stage 0a: Detect and correct 90°/180°/270° rotation."""
        stage_start = time.time()
        if self.enable_orientation_correction:
            image = self._detect_and_correct_orientation(image, stats)
        else:
            logger.debug("Orientation correction: skipped (disabled)")
        stats.stage_times["orientation"] = (time.time() - stage_start) * 1000
        return image

    def _stage_document_detection(
        self, image: Image.Image, stats: PreprocessingStats
    ) -> Image.Image:
        """Stage 0b: Detect document edges and crop."""
        stage_start = time.time()
        if self.enable_document_detection:
            image = self._detect_and_crop_document(image, stats)
        else:
            logger.debug("Document detection: skipped (disabled)")
        stats.stage_times["document_crop"] = (time.time() - stage_start) * 1000
        return image

    def _stage_grayscale(self, cv_image: np.ndarray, stats: PreprocessingStats) -> np.ndarray:
        """Stage 1: Convert to grayscale."""
        stage_start = time.time()
        if len(cv_image.shape) == 3:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
            logger.debug("Grayscale conversion: applied")
        else:
            gray = cv_image
            logger.debug("Grayscale conversion: already grayscale")
        stats.stage_times["grayscale"] = (time.time() - stage_start) * 1000
        return gray

    def _stage_denoise(self, gray: np.ndarray, stats: PreprocessingStats) -> np.ndarray:
        """Stage 2: Apply noise removal."""
        stage_start = time.time()
        try:
            denoised = cv2.fastNlMeansDenoising(gray, None, self.denoise_strength, 7, 21)
            stats.denoised = True
            logger.debug("Denoising: applied (strength=%d)", self.denoise_strength)
        except Exception as e:
            denoised = gray
            logger.warning("Denoising failed: %s", e)
        stats.stage_times["denoise"] = (time.time() - stage_start) * 1000
        return denoised

    def _stage_contrast(self, denoised: np.ndarray, stats: PreprocessingStats) -> np.ndarray:
        """Stage 3: Apply CLAHE contrast enhancement."""
        stage_start = time.time()
        if self.enable_clahe:
            try:
                clahe = cv2.createCLAHE(
                    clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size
                )
                enhanced = clahe.apply(denoised)
                stats.contrast_enhanced = True
                logger.debug(
                    "CLAHE contrast enhancement: applied (clip=%.1f)", self.clahe_clip_limit
                )
            except Exception as e:
                enhanced = denoised
                logger.warning("CLAHE enhancement failed: %s", e)
        else:
            enhanced = denoised
            logger.debug("CLAHE contrast enhancement: skipped (disabled)")
        stats.stage_times["clahe"] = (time.time() - stage_start) * 1000
        return enhanced

    def _stage_binarize(self, enhanced: np.ndarray, stats: PreprocessingStats) -> np.ndarray:
        """Stage 4: Apply adaptive thresholding (binarization)."""
        stage_start = time.time()
        try:
            binary = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.adaptive_block_size,
                self.adaptive_constant,
            )
            stats.binarized = True
            logger.debug(
                "Adaptive thresholding: applied (block=%d, C=%d)",
                self.adaptive_block_size,
                self.adaptive_constant,
            )
        except Exception as e1:
            logger.debug("Adaptive thresholding failed: %s, trying Otsu fallback", e1)
            try:
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                stats.binarized = True
                logger.debug("Adaptive thresholding failed, Otsu fallback: applied")
            except Exception as e2:
                binary = enhanced
                logger.warning("All binarization failed: %s", e2)
        stats.stage_times["binarize"] = (time.time() - stage_start) * 1000
        return binary

    def _stage_deskew(self, binary: np.ndarray, stats: PreprocessingStats) -> np.ndarray:
        """Stage 5: Detect and correct skew."""
        stage_start = time.time()
        deskewed, skew_angle = self._deskew(binary)
        stats.skew_angle = skew_angle
        stats.skew_corrected = abs(skew_angle) > 0.5
        if stats.skew_corrected:
            logger.debug("Deskewing: corrected %.2f deg rotation", skew_angle)
        else:
            logger.debug("Deskewing: no significant skew detected (%.2f deg)", skew_angle)
        stats.stage_times["deskew"] = (time.time() - stage_start) * 1000
        return deskewed

    def _stage_border(self, deskewed: np.ndarray, stats: PreprocessingStats) -> np.ndarray:
        """Stage 6: Add white border padding."""
        stage_start = time.time()
        if self.border_size > 0:
            padded = cv2.copyMakeBorder(
                deskewed,
                self.border_size,
                self.border_size,
                self.border_size,
                self.border_size,
                cv2.BORDER_CONSTANT,
                value=255,
            )
            stats.border_added = True
            logger.debug("Border padding: added %dpx white border", self.border_size)
        else:
            padded = deskewed
            logger.debug("Border padding: skipped (disabled)")
        stats.stage_times["border"] = (time.time() - stage_start) * 1000
        return padded

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
                logger.warning(
                    "Detected skew angle %.2f deg exceeds max %.2f deg, skipping correction",
                    angle,
                    self.max_skew_angle,
                )
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
                borderValue=255,  # White background
            )

            return rotated, angle

        except Exception as e:
            logger.warning("Deskew failed: %s", e)
            return image, 0.0

    def _detect_and_correct_orientation(
        self, image: Image.Image, stats: PreprocessingStats
    ) -> Image.Image:
        """
        Detect and correct large rotations (90°, 180°, 270°) using Tesseract OSD.

        Args:
            image: PIL Image to check
            stats: PreprocessingStats to update

        Returns:
            Rotated PIL Image (or original if no rotation needed)
        """
        try:
            import pytesseract
        except ImportError:
            logger.warning("pytesseract not available for orientation detection")
            return image

        try:
            # Use Tesseract's Orientation and Script Detection
            osd_output = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)

            orientation = osd_output.get("orientation", 0)
            confidence = osd_output.get("orientation_conf", 0.0)

            stats.orientation_angle = orientation
            stats.orientation_confidence = confidence

            # Only correct if confidence is high enough and rotation is significant
            if confidence >= self.orientation_confidence_threshold and orientation != 0:
                # Rotate in opposite direction to correct
                # PIL rotates counter-clockwise, so we negate
                corrected = image.rotate(-orientation, expand=True, fillcolor="white")
                stats.orientation_corrected = True
                logger.debug(
                    "Orientation correction: rotated %d deg (confidence=%.1f)",
                    orientation,
                    confidence,
                )
                return corrected
            else:
                if orientation != 0:
                    logger.debug(
                        "Orientation detection: %d deg detected but confidence too low (%.1f < %.1f)",
                        orientation,
                        confidence,
                        self.orientation_confidence_threshold,
                    )
                else:
                    logger.debug("Orientation detection: no rotation needed")
                return image

        except Exception as e:
            # OSD can fail on images with little/no text
            logger.debug("Orientation detection skipped: %s", e)
            return image

    def _detect_and_crop_document(
        self, image: Image.Image, stats: PreprocessingStats
    ) -> Image.Image:
        """
        Detect document edges and apply perspective correction to crop out background.

        Uses Canny edge detection and contour finding to locate a 4-sided document,
        then applies perspective transform for a top-down view.

        Args:
            image: PIL Image (possibly a phone photo with background)
            stats: PreprocessingStats to update

        Returns:
            Cropped and perspective-corrected PIL Image (or original if no document found)
        """
        cv_image = self._pil_to_cv2(image)
        original_h, original_w = cv_image.shape[:2]
        min_area = original_h * original_w * self.min_document_area_ratio

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY) if len(cv_image.shape) == 3 else cv_image

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate edges to close gaps
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.debug("Document detection: no contours found")
            return image

        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        document_contour = None

        for contour in contours[:5]:  # Check top 5 largest contours
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Approximate the contour to reduce points
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # We're looking for a quadrilateral (4 points)
            if len(approx) == 4:
                document_contour = approx
                break

        if document_contour is None:
            logger.debug("Document detection: no 4-sided document found")
            return image

        # Found a document - extract the 4 corners
        corners = document_contour.reshape(4, 2)
        stats.document_detected = True
        stats.document_corners = corners.tolist()

        # Order corners: top-left, top-right, bottom-right, bottom-left
        ordered_corners = self._order_corners(corners)

        # Calculate target dimensions
        width_top = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
        width_bottom = np.linalg.norm(ordered_corners[2] - ordered_corners[3])
        height_left = np.linalg.norm(ordered_corners[3] - ordered_corners[0])
        height_right = np.linalg.norm(ordered_corners[2] - ordered_corners[1])

        target_width = int(max(width_top, width_bottom))
        target_height = int(max(height_left, height_right))

        # Define destination points for perspective transform
        dst_corners = np.array(
            [
                [0, 0],
                [target_width - 1, 0],
                [target_width - 1, target_height - 1],
                [0, target_height - 1],
            ],
            dtype=np.float32,
        )

        # Apply perspective transform
        src_corners = ordered_corners.astype(np.float32)
        matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
        warped = cv2.warpPerspective(cv_image, matrix, (target_width, target_height))

        stats.document_cropped = True
        logger.debug(
            "Document detection: cropped %dx%d to %dx%d",
            original_w,
            original_h,
            target_width,
            target_height,
        )

        return self._cv2_to_pil(warped)

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order 4 corners as: top-left, top-right, bottom-right, bottom-left.

        Args:
            corners: Array of 4 corner points

        Returns:
            Ordered corners array
        """
        # Sort by sum of coordinates (top-left has smallest sum, bottom-right has largest)
        sorted_by_sum = corners[np.argsort(corners.sum(axis=1))]
        top_left = sorted_by_sum[0]
        bottom_right = sorted_by_sum[3]

        # Sort by difference (top-right has smallest diff, bottom-left has largest)
        sorted_by_diff = corners[np.argsort(np.diff(corners, axis=1).ravel())]
        top_right = sorted_by_diff[0]
        bottom_left = sorted_by_diff[3]

        return np.array([top_left, top_right, bottom_right, bottom_left])

    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format (numpy array)."""
        # Convert to RGB if necessary (PIL might be RGBA, L, etc.)
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")
        elif pil_image.mode == "L":
            # Already grayscale
            return np.array(pil_image)
        elif pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        return np.array(pil_image)

    def _cv2_to_pil(self, cv_image: np.ndarray) -> Image.Image:
        """Convert OpenCV format (numpy array) to PIL Image."""
        if len(cv_image.shape) == 2:
            # Grayscale
            return Image.fromarray(cv_image, mode="L")
        elif len(cv_image.shape) == 3:
            # Color - convert BGR to RGB if needed
            if cv_image.shape[2] == 3:
                return Image.fromarray(cv_image)
            elif cv_image.shape[2] == 4:
                return Image.fromarray(cv_image, mode="RGBA")

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
    output_path = input_path.rsplit(".", 1)[0] + "_preprocessed.png"
    processed.save(output_path)

    print("\nPreprocessing Stats:")
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
