"""
Tests for GPU detection and dynamic context window sizing (Session 64).

Tests:
1. GPU keyword detection (_is_dedicated_gpu)
2. VRAM-based context size calculation (get_optimal_context_size)
3. VRAM conversion (get_vram_gb)
4. GPU info aggregation (get_gpu_info)
5. User preferences context size methods
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestIsDedicatedGpu:
    """Test GPU keyword detection logic."""

    def test_nvidia_geforce_detected(self):
        """NVIDIA GeForce cards should be detected as dedicated."""
        from src.core.utils.gpu_detector import _is_dedicated_gpu

        is_dedicated, vendor = _is_dedicated_gpu("NVIDIA GeForce RTX 3080")
        assert is_dedicated is True
        assert vendor == "nvidia"

    def test_nvidia_rtx_detected(self):
        """NVIDIA RTX cards should be detected as dedicated."""
        from src.core.utils.gpu_detector import _is_dedicated_gpu

        is_dedicated, vendor = _is_dedicated_gpu("NVIDIA RTX 4090")
        assert is_dedicated is True
        assert vendor == "nvidia"

    def test_nvidia_gtx_detected(self):
        """NVIDIA GTX cards should be detected as dedicated."""
        from src.core.utils.gpu_detector import _is_dedicated_gpu

        is_dedicated, vendor = _is_dedicated_gpu("NVIDIA GeForce GTX 1080 Ti")
        assert is_dedicated is True
        assert vendor == "nvidia"

    def test_nvidia_quadro_detected(self):
        """NVIDIA Quadro workstation cards should be detected."""
        from src.core.utils.gpu_detector import _is_dedicated_gpu

        is_dedicated, vendor = _is_dedicated_gpu("NVIDIA Quadro P6000")
        assert is_dedicated is True
        assert vendor == "nvidia"

    def test_amd_radeon_rx_detected(self):
        """AMD Radeon RX cards should be detected as dedicated."""
        from src.core.utils.gpu_detector import _is_dedicated_gpu

        is_dedicated, vendor = _is_dedicated_gpu("AMD Radeon RX 7900 XTX")
        assert is_dedicated is True
        assert vendor == "amd"

    def test_amd_radeon_pro_detected(self):
        """AMD Radeon Pro workstation cards should be detected."""
        from src.core.utils.gpu_detector import _is_dedicated_gpu

        is_dedicated, vendor = _is_dedicated_gpu("AMD Radeon Pro W6800")
        assert is_dedicated is True
        assert vendor == "amd"

    def test_intel_integrated_excluded(self):
        """Intel integrated GPUs should NOT be detected as dedicated."""
        from src.core.utils.gpu_detector import _is_dedicated_gpu

        is_dedicated, vendor = _is_dedicated_gpu("Intel UHD Graphics 630")
        assert is_dedicated is False
        assert vendor == "integrated"

    def test_intel_iris_excluded(self):
        """Intel Iris graphics should NOT be detected as dedicated."""
        from src.core.utils.gpu_detector import _is_dedicated_gpu

        is_dedicated, vendor = _is_dedicated_gpu("Intel Iris Xe Graphics")
        assert is_dedicated is False
        assert vendor == "integrated"

    def test_amd_integrated_radeon_graphics_excluded(self):
        """AMD integrated Radeon Graphics (APU) should NOT be detected."""
        from src.core.utils.gpu_detector import _is_dedicated_gpu

        is_dedicated, vendor = _is_dedicated_gpu("AMD Radeon Graphics")
        assert is_dedicated is False
        assert vendor == "integrated"

    def test_unknown_gpu_conservative(self):
        """Unknown GPU types should return False (conservative)."""
        from src.core.utils.gpu_detector import _is_dedicated_gpu

        is_dedicated, vendor = _is_dedicated_gpu("Some Unknown GPU")
        assert is_dedicated is False
        assert vendor == "unknown"

    def test_case_insensitive_matching(self):
        """GPU detection should be case-insensitive."""
        from src.core.utils.gpu_detector import _is_dedicated_gpu

        # Lowercase
        is_dedicated, vendor = _is_dedicated_gpu("nvidia geforce rtx 3080")
        assert is_dedicated is True
        assert vendor == "nvidia"

        # Mixed case
        is_dedicated, vendor = _is_dedicated_gpu("AMD RADEON RX 6800")
        assert is_dedicated is True
        assert vendor == "amd"


class TestOptimalContextSize:
    """Test VRAM-based context size calculation."""

    def test_24gb_vram_gets_64k_context(self):
        """24GB+ VRAM should get 64K context."""
        from src.core.utils.gpu_detector import get_optimal_context_size

        # Mock 24GB VRAM
        with patch('src.core.utils.gpu_detector.get_gpu_info') as mock_info:
            mock_info.return_value = {
                "has_gpu": True,
                "vram_bytes": 24 * 1024**3,  # 24GB
                "vendor": "nvidia",
            }
            context = get_optimal_context_size()
            assert context == 64000

    def test_16gb_vram_gets_48k_context(self):
        """16-24GB VRAM should get 48K context."""
        from src.core.utils.gpu_detector import get_optimal_context_size

        with patch('src.core.utils.gpu_detector.get_gpu_info') as mock_info:
            mock_info.return_value = {
                "has_gpu": True,
                "vram_bytes": 16 * 1024**3,  # 16GB
                "vendor": "nvidia",
            }
            context = get_optimal_context_size()
            assert context == 48000

    def test_12gb_vram_gets_32k_context(self):
        """12-16GB VRAM should get 32K context."""
        from src.core.utils.gpu_detector import get_optimal_context_size

        with patch('src.core.utils.gpu_detector.get_gpu_info') as mock_info:
            mock_info.return_value = {
                "has_gpu": True,
                "vram_bytes": 12 * 1024**3,  # 12GB
                "vendor": "nvidia",
            }
            context = get_optimal_context_size()
            assert context == 32000

    def test_8gb_vram_gets_16k_context(self):
        """8-12GB VRAM should get 16K context."""
        from src.core.utils.gpu_detector import get_optimal_context_size

        with patch('src.core.utils.gpu_detector.get_gpu_info') as mock_info:
            mock_info.return_value = {
                "has_gpu": True,
                "vram_bytes": 8 * 1024**3,  # 8GB
                "vendor": "nvidia",
            }
            context = get_optimal_context_size()
            assert context == 16000

    def test_6gb_vram_gets_8k_context(self):
        """6-8GB VRAM should get 8K context."""
        from src.core.utils.gpu_detector import get_optimal_context_size

        with patch('src.core.utils.gpu_detector.get_gpu_info') as mock_info:
            mock_info.return_value = {
                "has_gpu": True,
                "vram_bytes": 6 * 1024**3,  # 6GB
                "vendor": "nvidia",
            }
            context = get_optimal_context_size()
            assert context == 8000

    def test_4gb_vram_gets_4k_context(self):
        """<6GB VRAM should get 4K context (safe default)."""
        from src.core.utils.gpu_detector import get_optimal_context_size

        with patch('src.core.utils.gpu_detector.get_gpu_info') as mock_info:
            mock_info.return_value = {
                "has_gpu": True,
                "vram_bytes": 4 * 1024**3,  # 4GB
                "vendor": "nvidia",
            }
            context = get_optimal_context_size()
            assert context == 4000

    def test_no_gpu_gets_4k_context(self):
        """No GPU should get 4K context (safe default)."""
        from src.core.utils.gpu_detector import get_optimal_context_size

        with patch('src.core.utils.gpu_detector.get_gpu_info') as mock_info:
            mock_info.return_value = {
                "has_gpu": False,
                "vram_bytes": 0,
                "vendor": None,
            }
            context = get_optimal_context_size()
            assert context == 4000


class TestVramGb:
    """Test VRAM byte to GB conversion."""

    def test_vram_conversion_8gb(self):
        """8GB VRAM should convert correctly."""
        from src.core.utils.gpu_detector import get_vram_gb

        with patch('src.core.utils.gpu_detector.get_gpu_info') as mock_info:
            mock_info.return_value = {
                "has_gpu": True,
                "vram_bytes": 8 * 1024**3,
                "vendor": "nvidia",
            }
            vram_gb = get_vram_gb()
            assert vram_gb == 8.0

    def test_vram_conversion_no_gpu(self):
        """No GPU should return 0.0 GB."""
        from src.core.utils.gpu_detector import get_vram_gb

        with patch('src.core.utils.gpu_detector.get_gpu_info') as mock_info:
            mock_info.return_value = {
                "has_gpu": False,
                "vram_bytes": 0,
                "vendor": None,
            }
            vram_gb = get_vram_gb()
            assert vram_gb == 0.0


class TestGpuStatusText:
    """Test human-readable GPU status text."""

    def test_status_text_with_gpu(self):
        """GPU status should show GPU name when detected."""
        from src.core.utils.gpu_detector import get_gpu_status_text

        with patch('src.core.utils.gpu_detector.get_gpu_info') as mock_info:
            mock_info.return_value = {
                "has_gpu": True,
                "gpu_name": "NVIDIA GeForce RTX 3080",
                "vendor": "nvidia",
                "vram_bytes": 10 * 1024**3,
            }
            status = get_gpu_status_text()
            assert "RTX 3080" in status
            assert "detected" in status.lower()

    def test_status_text_no_gpu(self):
        """No GPU should show appropriate message."""
        from src.core.utils.gpu_detector import get_gpu_status_text

        with patch('src.core.utils.gpu_detector.get_gpu_info') as mock_info:
            mock_info.return_value = {
                "has_gpu": False,
                "gpu_name": None,
                "vendor": None,
                "vram_bytes": 0,
            }
            status = get_gpu_status_text()
            assert "no dedicated gpu" in status.lower()


class TestUserPreferencesContextSize:
    """Test user preferences context size methods."""

    def test_context_size_mode_default_auto(self):
        """Default context size mode should be 'auto'."""
        from src.user_preferences import UserPreferencesManager
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            prefs_file = Path(tmpdir) / "test_prefs.json"
            manager = UserPreferencesManager(prefs_file)

            mode = manager.get_context_size_mode()
            assert mode == "auto"

    def test_context_size_mode_manual_int(self):
        """Manual context size should return the int value."""
        from src.user_preferences import UserPreferencesManager
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            prefs_file = Path(tmpdir) / "test_prefs.json"
            manager = UserPreferencesManager(prefs_file)

            manager.set_context_size_mode(16000)
            mode = manager.get_context_size_mode()
            assert mode == 16000

    def test_context_size_mode_validation(self):
        """Invalid context sizes should raise ValueError."""
        from src.user_preferences import UserPreferencesManager
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            prefs_file = Path(tmpdir) / "test_prefs.json"
            manager = UserPreferencesManager(prefs_file)

            with pytest.raises(ValueError):
                manager.set_context_size_mode(12345)  # Invalid size

    def test_effective_context_size_manual(self):
        """Manual mode should return the set value."""
        from src.user_preferences import UserPreferencesManager
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            prefs_file = Path(tmpdir) / "test_prefs.json"
            manager = UserPreferencesManager(prefs_file)

            manager.set_context_size_mode(32000)
            effective = manager.get_effective_context_size()
            assert effective == 32000

    def test_effective_context_size_auto_uses_gpu(self):
        """Auto mode should use GPU-detected context size."""
        from src.user_preferences import UserPreferencesManager
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            prefs_file = Path(tmpdir) / "test_prefs.json"
            manager = UserPreferencesManager(prefs_file)

            # Set to auto (default)
            manager.set_context_size_mode("auto")

            # Mock GPU detection
            with patch('src.core.utils.gpu_detector.get_optimal_context_size') as mock_ctx:
                mock_ctx.return_value = 48000
                effective = manager.get_effective_context_size()
                assert effective == 48000
                mock_ctx.assert_called_once()


class TestVramContextTiers:
    """Test the VRAM_CONTEXT_TIERS configuration."""

    def test_tiers_are_ordered(self):
        """VRAM tiers should be in descending order."""
        from src.core.utils.gpu_detector import VRAM_CONTEXT_TIERS

        vram_values = [tier[0] for tier in VRAM_CONTEXT_TIERS]
        assert vram_values == sorted(vram_values, reverse=True), (
            "VRAM tiers should be in descending order for correct matching"
        )

    def test_tiers_have_zero_fallback(self):
        """Last tier should have 0 VRAM (fallback)."""
        from src.core.utils.gpu_detector import VRAM_CONTEXT_TIERS

        last_tier = VRAM_CONTEXT_TIERS[-1]
        assert last_tier[0] == 0, "Last tier should have 0 VRAM as fallback"
        assert last_tier[1] == 4000, "Fallback context should be 4000"

    def test_all_context_sizes_valid(self):
        """All context sizes should be reasonable values."""
        from src.core.utils.gpu_detector import VRAM_CONTEXT_TIERS

        valid_sizes = [4000, 8000, 16000, 32000, 48000, 64000]
        for min_vram, context_size in VRAM_CONTEXT_TIERS:
            assert context_size in valid_sizes, (
                f"Context size {context_size} is not in valid sizes"
            )
