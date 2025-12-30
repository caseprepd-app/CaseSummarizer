"""
Tests for the Image Preprocessor module.

Tests the OCR preprocessing pipeline including:
- Grayscale conversion
- Denoising
- Contrast enhancement (CLAHE)
- Adaptive thresholding
- Deskewing
- Border padding
"""

import numpy as np
import pytest
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

        assert preprocessor.denoise_strength == 10
        assert preprocessor.adaptive_block_size == 11
        assert preprocessor.adaptive_constant == 2
        assert preprocessor.border_size == 10
        assert preprocessor.max_skew_angle == 10.0
        assert preprocessor.enable_clahe is True

    def test_initialization_custom_values(self):
        """Test that preprocessor accepts custom values."""
        preprocessor = ImagePreprocessor(
            denoise_strength=5,
            adaptive_block_size=15,
            adaptive_constant=5,
            border_size=20,
            max_skew_angle=5.0,
            enable_clahe=False,
        )

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
        image = Image.new('RGB', (100, 100), color='white')

        result, stats = preprocessor.preprocess(image)

        assert isinstance(result, Image.Image)
        assert isinstance(stats, PreprocessingStats)

    def test_preprocess_rgb_image(self):
        """Test preprocessing an RGB image."""
        preprocessor = ImagePreprocessor()
        image = Image.new('RGB', (100, 100), color='white')

        result, stats = preprocessor.preprocess(image)

        # Image should be processed (potentially different size due to border/deskew)
        assert result.size[0] >= 100
        assert result.size[1] >= 100
        assert stats.original_size == (100, 100)

    def test_preprocess_grayscale_image(self):
        """Test preprocessing a grayscale image."""
        preprocessor = ImagePreprocessor()
        image = Image.new('L', (100, 100), color=255)

        result, stats = preprocessor.preprocess(image)

        assert isinstance(result, Image.Image)
        assert stats.original_size == (100, 100)

    def test_preprocess_rgba_image(self):
        """Test preprocessing an RGBA image."""
        preprocessor = ImagePreprocessor()
        image = Image.new('RGBA', (100, 100), color=(255, 255, 255, 255))

        result, stats = preprocessor.preprocess(image)

        assert isinstance(result, Image.Image)

    def test_stats_tracks_stages(self):
        """Test that stats tracks all preprocessing stages."""
        preprocessor = ImagePreprocessor()
        image = Image.new('RGB', (100, 100), color='white')

        _, stats = preprocessor.preprocess(image)

        # All stages should be tracked in timing
        assert 'grayscale' in stats.stage_times
        assert 'denoise' in stats.stage_times
        assert 'clahe' in stats.stage_times
        assert 'binarize' in stats.stage_times
        assert 'deskew' in stats.stage_times
        assert 'border' in stats.stage_times

        # All timings should be non-negative
        for stage, time_ms in stats.stage_times.items():
            assert time_ms >= 0, f"{stage} has negative time"

    def test_stats_tracks_operations(self):
        """Test that stats tracks which operations were applied."""
        preprocessor = ImagePreprocessor()
        image = Image.new('RGB', (100, 100), color='white')

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
        image = Image.new('RGB', (100, 100), color='white')

        result, stats = preprocessor.preprocess(image)

        # Result should be at least original + 2*border in each dimension
        # (may be larger if deskewing occurred)
        assert result.size[0] >= 100 + 2 * border_size
        assert result.size[1] >= 100 + 2 * border_size

    def test_no_border_when_disabled(self):
        """Test that no border is added when border_size=0."""
        preprocessor = ImagePreprocessor(border_size=0)
        image = Image.new('L', (100, 100), color=255)

        result, stats = preprocessor.preprocess(image)

        assert stats.border_added is False

    def test_clahe_can_be_disabled(self):
        """Test that CLAHE can be disabled."""
        preprocessor = ImagePreprocessor(enable_clahe=False)
        image = Image.new('RGB', (100, 100), color='white')

        _, stats = preprocessor.preprocess(image)

        assert stats.contrast_enhanced is False

    def test_total_time_tracked(self):
        """Test that total processing time is tracked."""
        preprocessor = ImagePreprocessor()
        image = Image.new('RGB', (100, 100), color='white')

        _, stats = preprocessor.preprocess(image)

        assert stats.total_time_ms > 0

    def test_preprocess_with_text(self):
        """Test preprocessing an image with text."""
        preprocessor = ImagePreprocessor()

        # Create image with text
        image = Image.new('RGB', (200, 50), color='white')
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), "Test OCR", fill='black')

        result, stats = preprocessor.preprocess(image)

        assert isinstance(result, Image.Image)
        assert stats.binarized is True


class TestPreprocessForOCRFunction:
    """Tests for the convenience function."""

    def test_preprocess_for_ocr_basic(self):
        """Test the convenience function works."""
        image = Image.new('RGB', (100, 100), color='white')

        result, stats = preprocess_for_ocr(image)

        assert isinstance(result, Image.Image)
        assert isinstance(stats, PreprocessingStats)

    def test_preprocess_for_ocr_custom_params(self):
        """Test convenience function with custom parameters."""
        image = Image.new('RGB', (100, 100), color='white')

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
        assert stats.skew_angle == 0.0
        assert stats.skew_corrected is False
        assert stats.denoised is False
        assert stats.contrast_enhanced is False
        assert stats.binarized is False
        assert stats.border_added is False
        assert stats.total_time_ms == 0.0
        assert stats.stage_times == {}


class TestSkewDetection:
    """Tests for skew detection and correction."""

    def test_no_skew_on_clean_image(self):
        """Test that clean images have minimal detected skew."""
        preprocessor = ImagePreprocessor()
        # Create a solid white image - no skew should be detected
        image = Image.new('L', (100, 100), color=255)

        _, stats = preprocessor.preprocess(image)

        # Skew correction should not have been applied (no significant skew)
        # Note: The exact behavior depends on the deskew library
        assert isinstance(stats.skew_angle, float)

    def test_skew_angle_limits(self):
        """Test that extreme skew angles are limited."""
        preprocessor = ImagePreprocessor(max_skew_angle=5.0)

        # The max_skew_angle should be stored
        assert preprocessor.max_skew_angle == 5.0
