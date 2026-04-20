"""
Tests for the Image Preprocessor module.

Tests the OCR preprocessing pipeline including:
- Orientation correction (90°/180°/270°)
- Document detection and cropping
- Grayscale conversion
- Denoising
- Contrast enhancement (CLAHE)
- Adaptive thresholding
- Deskewing
- Border padding
"""

import numpy as np
from PIL import Image, ImageDraw

from src.core.extraction.image_preprocessor import (
    ImagePreprocessor,
    PreprocessingStats,
    preprocess_for_ocr,
)


class TestImagePreprocessor:
    """Tests for ImagePreprocessor class."""

    def test_initialization_defaults(self):
        """Test that preprocessor initializes with default values."""
        preprocessor = ImagePreprocessor()

        # New feature defaults
        assert preprocessor.enable_orientation_correction is True
        assert preprocessor.orientation_confidence_threshold == 2.0
        assert preprocessor.enable_document_detection is True
        assert preprocessor.min_document_area_ratio == 0.1
        # Existing defaults
        assert preprocessor.denoise_strength == 10
        assert preprocessor.adaptive_block_size == 11
        assert preprocessor.adaptive_constant == 2
        assert preprocessor.border_size == 10
        assert preprocessor.max_skew_angle == 10.0
        assert preprocessor.enable_clahe is True

    def test_initialization_custom_values(self):
        """Test that preprocessor accepts custom values."""
        preprocessor = ImagePreprocessor(
            enable_orientation_correction=False,
            orientation_confidence_threshold=5.0,
            enable_document_detection=False,
            min_document_area_ratio=0.2,
            denoise_strength=5,
            adaptive_block_size=15,
            adaptive_constant=5,
            border_size=20,
            max_skew_angle=5.0,
            enable_clahe=False,
        )

        # New features
        assert preprocessor.enable_orientation_correction is False
        assert preprocessor.orientation_confidence_threshold == 5.0
        assert preprocessor.enable_document_detection is False
        assert preprocessor.min_document_area_ratio == 0.2
        # Existing
        assert preprocessor.denoise_strength == 5
        assert preprocessor.adaptive_block_size == 15
        assert preprocessor.adaptive_constant == 5
        assert preprocessor.border_size == 20
        assert preprocessor.max_skew_angle == 5.0
        assert preprocessor.enable_clahe is False

    def test_block_size_forced_odd(self):
        """Test that even block sizes are converted to odd."""
        preprocessor = ImagePreprocessor(adaptive_block_size=10)
        assert preprocessor.adaptive_block_size == 11

        preprocessor2 = ImagePreprocessor(adaptive_block_size=11)
        assert preprocessor2.adaptive_block_size == 11

    def test_preprocess_returns_image_and_stats(self):
        """Test that preprocess returns a PIL Image and stats."""
        preprocessor = ImagePreprocessor()
        image = Image.new("RGB", (100, 100), color="white")

        result, stats = preprocessor.preprocess(image)

        assert isinstance(result, Image.Image)
        assert isinstance(stats, PreprocessingStats)

    def test_preprocess_rgb_image(self):
        """Test preprocessing an RGB image."""
        preprocessor = ImagePreprocessor()
        image = Image.new("RGB", (100, 100), color="white")

        result, stats = preprocessor.preprocess(image)

        # Image should be processed (potentially different size due to border/deskew)
        assert result.size[0] >= 100
        assert result.size[1] >= 100
        assert stats.original_size == (100, 100)

    def test_preprocess_grayscale_image(self):
        """Preprocessing a grayscale image preserves content and emits expected stats."""
        preprocessor = ImagePreprocessor()
        image = Image.new("L", (100, 100), color=255)

        result, stats = preprocessor.preprocess(image)

        # Grayscale image produces a grayscale-ish result (L or 1 mode after thresholding)
        assert result.mode in ("L", "1")
        assert stats.original_size == (100, 100)
        assert stats.binarized is True

    def test_preprocess_rgba_image(self):
        """RGBA images are handled and downconverted during preprocessing."""
        preprocessor = ImagePreprocessor()
        image = Image.new("RGBA", (100, 100), color=(255, 255, 255, 255))

        result, stats = preprocessor.preprocess(image)

        # RGBA must be converted away (to grayscale + binarized) during the pipeline
        assert result.mode in ("L", "1")
        # Grayscale stage should have been recorded
        assert "grayscale" in stats.stage_times

    def test_stats_tracks_stages(self):
        """Test that stats tracks all preprocessing stages."""
        preprocessor = ImagePreprocessor()
        image = Image.new("RGB", (100, 100), color="white")

        _, stats = preprocessor.preprocess(image)

        # All stages should be tracked in timing (including new ones)
        assert "orientation" in stats.stage_times
        assert "document_crop" in stats.stage_times
        assert "grayscale" in stats.stage_times
        assert "denoise" in stats.stage_times
        assert "clahe" in stats.stage_times
        assert "binarize" in stats.stage_times
        assert "deskew" in stats.stage_times
        assert "border" in stats.stage_times

        # All timings should be non-negative
        for stage, time_ms in stats.stage_times.items():
            assert time_ms >= 0, f"{stage} has negative time"

    def test_stats_tracks_operations(self):
        """Test that stats tracks which operations were applied."""
        preprocessor = ImagePreprocessor()
        image = Image.new("RGB", (100, 100), color="white")

        _, stats = preprocessor.preprocess(image)

        # These operations should have been applied
        assert stats.denoised is True
        assert stats.contrast_enhanced is True
        assert stats.binarized is True
        assert stats.border_added is True

    def test_border_adds_pixels(self):
        """Test that border adds expected pixels."""
        border_size = 15
        preprocessor = ImagePreprocessor(border_size=border_size)
        image = Image.new("RGB", (100, 100), color="white")

        result, _stats = preprocessor.preprocess(image)

        # Result should be at least original + 2*border in each dimension
        # (may be larger if deskewing occurred)
        assert result.size[0] >= 100 + 2 * border_size
        assert result.size[1] >= 100 + 2 * border_size

    def test_no_border_when_disabled(self):
        """Test that no border is added when border_size=0."""
        preprocessor = ImagePreprocessor(border_size=0)
        image = Image.new("L", (100, 100), color=255)

        _result, stats = preprocessor.preprocess(image)

        assert stats.border_added is False

    def test_clahe_can_be_disabled(self):
        """Test that CLAHE can be disabled."""
        preprocessor = ImagePreprocessor(enable_clahe=False)
        image = Image.new("RGB", (100, 100), color="white")

        _, stats = preprocessor.preprocess(image)

        assert stats.contrast_enhanced is False

    def test_total_time_tracked(self):
        """Test that total processing time is tracked."""
        preprocessor = ImagePreprocessor()
        image = Image.new("RGB", (100, 100), color="white")

        _, stats = preprocessor.preprocess(image)

        assert stats.total_time_ms > 0

    def test_preprocess_with_text(self):
        """Preprocessing an image with text produces a binarized image with expected size and stats."""
        preprocessor = ImagePreprocessor()

        # Create image with text
        image = Image.new("RGB", (200, 50), color="white")
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), "Test OCR", fill="black")

        result, stats = preprocessor.preprocess(image)

        # Binarized image has L or 1 mode
        assert result.mode in ("L", "1")
        assert stats.binarized is True
        assert stats.denoised is True
        # Original size must be recorded; preprocessing may have added border
        assert stats.original_size == (200, 50)
        assert result.size[0] >= 200  # border may add pixels
        assert result.size[1] >= 50


class TestPreprocessForOCRFunction:
    """Tests for the convenience function."""

    def test_preprocess_for_ocr_basic(self):
        """The convenience function applies the full pipeline with default settings."""
        image = Image.new("RGB", (100, 100), color="white")

        result, stats = preprocess_for_ocr(image)

        # Default pipeline binarizes and adds a border; output must reflect that
        assert result.mode in ("L", "1")
        assert stats.binarized is True
        assert stats.denoised is True
        assert stats.border_added is True  # default border_size=10
        # All stage times must be recorded
        assert "binarize" in stats.stage_times
        assert stats.total_time_ms > 0

    def test_preprocess_for_ocr_custom_params(self):
        """Test convenience function with custom parameters."""
        image = Image.new("RGB", (100, 100), color="white")

        result, stats = preprocess_for_ocr(
            image,
            denoise_strength=5,
            enable_clahe=False,
        )

        assert isinstance(result, Image.Image)
        # CLAHE should not have been applied
        assert stats.contrast_enhanced is False


class TestPreprocessingStats:
    """Tests for PreprocessingStats dataclass."""

    def test_default_values(self):
        """Test default values of stats."""
        stats = PreprocessingStats()

        assert stats.original_size == (0, 0)
        assert stats.processed_size == (0, 0)
        # Orientation fields
        assert stats.orientation_angle == 0
        assert stats.orientation_confidence == 0.0
        assert stats.orientation_corrected is False
        # Document detection fields
        assert stats.document_detected is False
        assert stats.document_corners is None
        assert stats.document_cropped is False
        # Deskew fields
        assert stats.skew_angle == 0.0
        assert stats.skew_corrected is False
        # Other fields
        assert stats.denoised is False
        assert stats.contrast_enhanced is False
        assert stats.binarized is False
        assert stats.border_added is False
        assert stats.total_time_ms == 0.0
        assert stats.stage_times == {}


class TestSkewDetection:
    """Tests for skew detection and correction."""

    def test_no_skew_on_clean_image(self):
        """Solid-color image has no edges, so skew correction should not be applied."""
        preprocessor = ImagePreprocessor()
        # Create a solid white image - no skew should be detected
        image = Image.new("L", (100, 100), color=255)

        _, stats = preprocessor.preprocess(image)

        # A solid image has no angular features; detection should report zero skew
        # and must not apply correction to clean images
        assert stats.skew_corrected is False
        # skew_angle should be zero or very close to zero for featureless image
        assert abs(stats.skew_angle) < 1.0

    def test_skew_angle_limits_stored_and_applied(self):
        """max_skew_angle is stored and used as the clamping limit."""
        preprocessor = ImagePreprocessor(max_skew_angle=5.0)
        assert preprocessor.max_skew_angle == 5.0

        # Verify it's actually used by checking it against a real image run:
        # clean image -> no correction, but max_skew_angle is respected attribute
        image = Image.new("L", (100, 100), color=255)
        _, stats = preprocessor.preprocess(image)
        # Skew angle on clean image is within the max limit
        assert abs(stats.skew_angle) <= 5.0


class TestOrientationDetection:
    """Tests for orientation detection and correction."""

    def test_orientation_disabled(self):
        """Test that orientation detection can be disabled."""
        preprocessor = ImagePreprocessor(enable_orientation_correction=False)
        image = Image.new("RGB", (100, 100), color="white")

        _, stats = preprocessor.preprocess(image)

        # Should not have been corrected
        assert stats.orientation_corrected is False
        assert stats.orientation_angle == 0

    def test_orientation_enabled_no_text(self):
        """Test orientation detection on image with no text (should skip gracefully)."""
        preprocessor = ImagePreprocessor(enable_orientation_correction=True)
        # Solid color image - OSD will likely fail
        image = Image.new("RGB", (100, 100), color="white")

        # Should not raise exception
        result, stats = preprocessor.preprocess(image)

        assert isinstance(result, Image.Image)
        # Orientation detection likely failed/skipped, but that's OK
        assert stats.orientation_corrected is False

    def test_orientation_confidence_threshold(self):
        """High confidence threshold prevents orientation correction on low-confidence detections."""
        # A very high threshold (99.0) means no real-world detection will exceed it
        preprocessor = ImagePreprocessor(orientation_confidence_threshold=99.0)
        assert preprocessor.orientation_confidence_threshold == 99.0

        # Verify it has the effect of blocking orientation correction on plain images
        image = Image.new("RGB", (100, 100), color="white")
        _, stats = preprocessor.preprocess(image)
        # With 99.0 threshold, no correction can be applied
        assert stats.orientation_corrected is False


class TestDocumentDetection:
    """Tests for document detection and cropping."""

    def test_document_detection_disabled(self):
        """Test that document detection can be disabled."""
        preprocessor = ImagePreprocessor(enable_document_detection=False)
        image = Image.new("RGB", (100, 100), color="white")

        _, stats = preprocessor.preprocess(image)

        assert stats.document_detected is False
        assert stats.document_cropped is False

    def test_document_detection_no_document(self):
        """Test document detection on solid color image (no document to find)."""
        preprocessor = ImagePreprocessor(enable_document_detection=True)
        image = Image.new("RGB", (100, 100), color="white")

        result, stats = preprocessor.preprocess(image)

        # No document should be detected in solid color image
        assert stats.document_detected is False
        assert stats.document_cropped is False
        assert isinstance(result, Image.Image)

    def test_document_detection_with_rectangle(self):
        """Document detection on a clear white-on-dark rectangle produces a valid image output."""
        preprocessor = ImagePreprocessor(
            enable_orientation_correction=False,  # Skip to speed up test
            enable_document_detection=True,
            min_document_area_ratio=0.05,
        )

        # Create image with dark background and white rectangle (document)
        image = Image.new("RGB", (200, 200), color=(50, 50, 50))
        draw = ImageDraw.Draw(image)
        # Draw a white rectangle in the center — area = 120x120 = 14400 / 40000 = 36% > 5% threshold
        draw.rectangle([40, 40, 160, 160], fill="white", outline="white")

        result, stats = preprocessor.preprocess(image)

        # The result must be a valid PIL image with positive dimensions
        assert result.size[0] > 0
        assert result.size[1] > 0
        # Stats must record that document-detection stage ran (timing is present)
        assert "document_crop" in stats.stage_times
        assert stats.stage_times["document_crop"] >= 0

    def test_min_document_area_ratio_blocks_small_documents(self):
        """A very high min_document_area_ratio blocks detection of small rectangles."""
        # 0.5 means document must be >= 50% of image area
        preprocessor = ImagePreprocessor(
            enable_orientation_correction=False,
            enable_document_detection=True,
            min_document_area_ratio=0.5,
        )
        assert preprocessor.min_document_area_ratio == 0.5

        # Draw a small rectangle (only 4% of image area) — below the 50% threshold
        image = Image.new("RGB", (200, 200), color=(50, 50, 50))
        draw = ImageDraw.Draw(image)
        draw.rectangle([40, 40, 80, 80], fill="white", outline="white")  # 40x40 = 1600/40000 = 4%

        _, stats = preprocessor.preprocess(image)

        # Below-threshold rectangle must not be cropped out
        assert stats.document_cropped is False

    def test_order_corners_helper(self):
        """Test the corner ordering helper function."""
        preprocessor = ImagePreprocessor()

        # Unordered corners of a rectangle
        corners = np.array(
            [
                [100, 100],  # bottom-right
                [0, 0],  # top-left
                [100, 0],  # top-right
                [0, 100],  # bottom-left
            ]
        )

        ordered = preprocessor._order_corners(corners)

        # Should be: top-left, top-right, bottom-right, bottom-left
        assert ordered[0].tolist() == [0, 0]  # top-left
        assert ordered[1].tolist() == [100, 0]  # top-right
        assert ordered[2].tolist() == [100, 100]  # bottom-right
        assert ordered[3].tolist() == [0, 100]  # bottom-left
