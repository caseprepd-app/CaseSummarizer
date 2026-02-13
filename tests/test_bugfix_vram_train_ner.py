"""
Tests for bugfixes: VRAM detection, train() call, NER progress logging.

Fix 1: WMI AdapterRAM is uint32 (maxes at 4GB). For NVIDIA GPUs detected
       via WMI, always cross-check nvidia-smi for accurate VRAM.
Fix 2: settings_registry called learner.train(force=True) but train() only
       accepts feedback_manager. Removed the invalid parameter.
Fix 3: NER progress callback fired only every 10% of chunks. Added 30-second
       time-based floor so large documents don't create 5+ minute silent gaps.
"""

from unittest.mock import MagicMock, patch

# ============================================================================
# Fix 1: VRAM detection -- WMI uint32 overflow + nvidia-smi cross-check
# ============================================================================


class TestWmiVramOverflow:
    """Test that NVIDIA GPUs detected via WMI get nvidia-smi VRAM cross-check."""

    def _clear_gpu_cache(self):
        """Clear the module-level GPU info cache so get_gpu_info() re-detects."""
        import src.core.utils.gpu_detector as gd

        gd._gpu_info_cache = None
        gd.has_dedicated_gpu.cache_clear()

    def test_nvidia_wmi_4gb_corrected_by_nvidia_smi(self):
        """WMI reports ~4GB for 8GB 3070, nvidia-smi should correct it."""
        self._clear_gpu_cache()
        from src.core.utils.gpu_detector import get_gpu_info

        wmi_result = {
            "gpu_name": "NVIDIA GeForce RTX 3070",
            "vendor": "nvidia",
            "vram_bytes": 4293918720,  # ~4GB (WMI uint32 overflow)
        }
        cli_result = {
            "gpu_name": "NVIDIA GeForce RTX 3070",
            "vendor": "nvidia",
            "vram_bytes": 8 * 1024**3,  # 8GB (correct)
        }

        with (
            patch("src.core.utils.gpu_detector._detect_gpu_pytorch", return_value=None),
            patch("src.core.utils.gpu_detector._detect_gpu_wmi", return_value=wmi_result),
            patch("src.core.utils.gpu_detector._detect_gpu_cli", return_value=cli_result),
        ):
            info = get_gpu_info()
            assert info["vram_bytes"] == 8 * 1024**3, "Should use nvidia-smi's 8GB, not WMI's 4GB"
            assert info["detection_method"] == "wmi+cli"

    def test_nvidia_wmi_zero_vram_corrected_by_nvidia_smi(self):
        """WMI reports 0 VRAM (complete overflow), nvidia-smi should correct."""
        self._clear_gpu_cache()
        from src.core.utils.gpu_detector import get_gpu_info

        wmi_result = {
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "vendor": "nvidia",
            "vram_bytes": 0,  # Complete overflow
        }
        cli_result = {
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "vendor": "nvidia",
            "vram_bytes": 24 * 1024**3,  # 24GB
        }

        with (
            patch("src.core.utils.gpu_detector._detect_gpu_pytorch", return_value=None),
            patch("src.core.utils.gpu_detector._detect_gpu_wmi", return_value=wmi_result),
            patch("src.core.utils.gpu_detector._detect_gpu_cli", return_value=cli_result),
        ):
            info = get_gpu_info()
            assert info["vram_bytes"] == 24 * 1024**3
            assert info["detection_method"] == "wmi+cli"

    def test_nvidia_wmi_correct_vram_still_uses_nvidia_smi(self):
        """Even if WMI shows correct-ish VRAM, nvidia-smi should override for NVIDIA."""
        self._clear_gpu_cache()
        from src.core.utils.gpu_detector import get_gpu_info

        wmi_result = {
            "gpu_name": "NVIDIA GeForce GTX 1650",
            "vendor": "nvidia",
            "vram_bytes": 4 * 1024**3,  # 4GB (correct for this card)
        }
        cli_result = {
            "gpu_name": "NVIDIA GeForce GTX 1650",
            "vendor": "nvidia",
            "vram_bytes": 4 * 1024**3,  # 4GB (confirms)
        }

        with (
            patch("src.core.utils.gpu_detector._detect_gpu_pytorch", return_value=None),
            patch("src.core.utils.gpu_detector._detect_gpu_wmi", return_value=wmi_result),
            patch("src.core.utils.gpu_detector._detect_gpu_cli", return_value=cli_result),
        ):
            info = get_gpu_info()
            assert info["vram_bytes"] == 4 * 1024**3
            assert info["detection_method"] == "wmi+cli"

    def test_nvidia_wmi_no_nvidia_smi_keeps_wmi_vram(self):
        """If nvidia-smi not available, fall back to WMI VRAM (imperfect but non-zero)."""
        self._clear_gpu_cache()
        from src.core.utils.gpu_detector import get_gpu_info

        wmi_result = {
            "gpu_name": "NVIDIA GeForce RTX 3070",
            "vendor": "nvidia",
            "vram_bytes": 4293918720,  # ~4GB
        }

        with (
            patch("src.core.utils.gpu_detector._detect_gpu_pytorch", return_value=None),
            patch("src.core.utils.gpu_detector._detect_gpu_wmi", return_value=wmi_result),
            patch("src.core.utils.gpu_detector._detect_gpu_cli", return_value=None),
        ):
            info = get_gpu_info()
            assert info["vram_bytes"] == 4293918720, (
                "Should keep WMI value when nvidia-smi unavailable"
            )
            assert info["detection_method"] == "wmi"

    def test_amd_wmi_zero_vram_falls_back_to_cli(self):
        """AMD GPU with 0 VRAM should still fall back to CLI."""
        self._clear_gpu_cache()
        from src.core.utils.gpu_detector import get_gpu_info

        wmi_result = {
            "gpu_name": "AMD Radeon RX 7900 XTX",
            "vendor": "amd",
            "vram_bytes": 0,
        }
        cli_result = {
            "gpu_name": "AMD Radeon RX 7900 XTX",
            "vendor": "amd",
            "vram_bytes": 24 * 1024**3,
        }

        with (
            patch("src.core.utils.gpu_detector._detect_gpu_pytorch", return_value=None),
            patch("src.core.utils.gpu_detector._detect_gpu_wmi", return_value=wmi_result),
            patch("src.core.utils.gpu_detector._detect_gpu_cli", return_value=cli_result),
        ):
            info = get_gpu_info()
            assert info["vram_bytes"] == 24 * 1024**3
            assert info["detection_method"] == "wmi+cli"

    def test_amd_wmi_nonzero_vram_kept_as_is(self):
        """AMD GPU with non-zero WMI VRAM should not call CLI (no overflow issue)."""
        self._clear_gpu_cache()
        from src.core.utils.gpu_detector import get_gpu_info

        wmi_result = {
            "gpu_name": "AMD Radeon RX 6600",
            "vendor": "amd",
            "vram_bytes": 4 * 1024**3,  # 4GB (could be correct for this card)
        }

        with (
            patch("src.core.utils.gpu_detector._detect_gpu_pytorch", return_value=None),
            patch("src.core.utils.gpu_detector._detect_gpu_wmi", return_value=wmi_result),
            patch("src.core.utils.gpu_detector._detect_gpu_cli") as mock_cli,
        ):
            info = get_gpu_info()
            assert info["vram_bytes"] == 4 * 1024**3
            assert info["detection_method"] == "wmi"
            mock_cli.assert_not_called()

    def test_pytorch_takes_priority_over_wmi(self):
        """PyTorch CUDA detection should bypass WMI entirely."""
        self._clear_gpu_cache()
        from src.core.utils.gpu_detector import get_gpu_info

        pytorch_result = {
            "gpu_name": "NVIDIA GeForce RTX 3070",
            "vendor": "nvidia",
            "vram_bytes": 8 * 1024**3,
        }

        with (
            patch("src.core.utils.gpu_detector._detect_gpu_pytorch", return_value=pytorch_result),
            patch("src.core.utils.gpu_detector._detect_gpu_wmi") as mock_wmi,
            patch("src.core.utils.gpu_detector._detect_gpu_cli") as mock_cli,
        ):
            info = get_gpu_info()
            assert info["vram_bytes"] == 8 * 1024**3
            assert info["detection_method"] == "pytorch"
            mock_wmi.assert_not_called()
            mock_cli.assert_not_called()

    def test_8gb_vram_maps_to_16k_context(self):
        """8GB VRAM (corrected by nvidia-smi) should get 16K context, not 2K."""
        self._clear_gpu_cache()
        from src.core.utils.gpu_detector import get_gpu_info, get_optimal_context_size

        wmi_result = {
            "gpu_name": "NVIDIA GeForce RTX 3070",
            "vendor": "nvidia",
            "vram_bytes": 4293918720,  # ~4GB (WMI overflow)
        }
        cli_result = {
            "gpu_name": "NVIDIA GeForce RTX 3070",
            "vendor": "nvidia",
            "vram_bytes": 8 * 1024**3,  # 8GB (correct)
        }

        with (
            patch("src.core.utils.gpu_detector._detect_gpu_pytorch", return_value=None),
            patch("src.core.utils.gpu_detector._detect_gpu_wmi", return_value=wmi_result),
            patch("src.core.utils.gpu_detector._detect_gpu_cli", return_value=cli_result),
        ):
            info = get_gpu_info()
            # Now test context size uses the corrected VRAM
            with patch("src.core.utils.gpu_detector.get_gpu_info", return_value=info):
                context = get_optimal_context_size()
                assert context == 16000, "8GB VRAM should get 16K context, not 2K"


# ============================================================================
# Fix 2: train(force=True) TypeError
# ============================================================================


class TestTrainCallSignature:
    """Test that settings_registry calls train() without invalid parameters."""

    def test_train_method_accepts_no_force_param(self):
        """VocabularyPreferenceLearner.train() should not accept force= kwarg."""
        import inspect

        from src.core.vocabulary.preference_learner import VocabularyPreferenceLearner

        sig = inspect.signature(VocabularyPreferenceLearner.train)
        param_names = list(sig.parameters.keys())
        assert "force" not in param_names, "train() should not have a 'force' parameter"

    def test_train_method_accepts_feedback_manager(self):
        """train() should accept optional feedback_manager parameter."""
        import inspect

        from src.core.vocabulary.preference_learner import VocabularyPreferenceLearner

        sig = inspect.signature(VocabularyPreferenceLearner.train)
        param_names = list(sig.parameters.keys())
        assert "feedback_manager" in param_names

    def test_settings_registry_no_force_in_train_call(self):
        """settings_registry.py should not pass force=True to train()."""
        from pathlib import Path

        settings_path = (
            Path(__file__).parent.parent / "src" / "ui" / "settings" / "settings_registry.py"
        )
        content = settings_path.read_text(encoding="utf-8")
        assert "train(force=" not in content, (
            "settings_registry.py should not pass force= to train()"
        )

    def test_train_callable_without_args(self):
        """train() should be callable with no arguments (uses global singleton)."""
        from src.core.vocabulary.preference_learner import VocabularyPreferenceLearner

        learner = VocabularyPreferenceLearner.__new__(VocabularyPreferenceLearner)
        # Mock the feedback manager to avoid needing real data
        with patch("src.core.vocabulary.preference_learner.get_feedback_manager") as mock_fm:
            mock_fm.return_value.export_training_data.return_value = []
            # Should not raise TypeError
            result = learner.train()
            assert result is False  # No data -> training fails gracefully


# ============================================================================
# Fix 3: NER progress callback -- time-based floor
# ============================================================================


class TestNerProgressTimeBased:
    """Test that NER progress callback fires on time-based interval, not just percentage."""

    def _make_algo_with_mock_nlp(self, num_chunks):
        """Create NERAlgorithm with mocked nlp that returns empty docs."""
        from src.core.vocabulary.algorithms.ner_algorithm import NERAlgorithm

        algo = NERAlgorithm.__new__(NERAlgorithm)
        algo.exclude_list = set()
        algo.user_exclude_list = set()
        algo.medical_terms = set()
        algo.common_words_blacklist = set()
        algo.frequency_dataset = {}
        algo.frequency_rank_map = {}
        algo.rarity_threshold = 50000

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.ents = []
        mock_doc.__iter__ = MagicMock(return_value=iter([]))

        mock_nlp = MagicMock()
        mock_nlp.pipe.return_value = [mock_doc] * num_chunks
        algo._nlp = mock_nlp

        return algo

    def test_progress_fires_every_10_percent(self):
        """Progress callback should fire at 10% intervals (existing behavior)."""
        callback = MagicMock()
        algo = self._make_algo_with_mock_nlp(20)
        chunks = ["The quick brown fox." for _ in range(20)]

        algo.extract("dummy", chunks=chunks, progress_callback=callback)

        # Should have fired at 10%, 20%, ..., 100% = ~10 times
        assert callback.call_count >= 9, (
            f"Expected ~10 progress callbacks for 20 chunks, got {callback.call_count}"
        )

    def test_progress_fires_on_time_interval(self):
        """Progress callback should fire at least every 30s even if <10% done.

        With 1000 chunks, percentage-based triggers fire every 100 chunks (10%).
        We simulate 31 seconds passing every 50 chunks so that time-based triggers
        fire at chunks 50, 150, 250, etc. -- between the percentage triggers.
        """
        callback = MagicMock()
        algo = self._make_algo_with_mock_nlp(1000)
        chunks = ["word." for _ in range(1000)]

        # Track calls to fake_time to simulate gradual time passage
        call_count = [0]
        base_time = 1000000.0

        def fake_time():
            """Each call simulates ~0.7 seconds passing (31s per ~44 calls)."""
            call_count[0] += 1
            # First 2 calls are for start_time and last_report_time initialization
            if call_count[0] <= 2:
                return base_time
            # Each subsequent call = ~0.7s, so every ~44 calls = ~31 seconds
            return base_time + (call_count[0] - 2) * 0.7

        with patch("src.core.vocabulary.algorithms.ner_algorithm.time") as mock_time:
            mock_time.time = fake_time
            mock_time.sleep = MagicMock()
            algo.extract("dummy", chunks=chunks, progress_callback=callback)

        # With time-based triggers every 30s PLUS percentage-based every 10%,
        # we should get more callbacks than the 10 from percentage alone
        assert callback.call_count > 10, (
            f"Expected >10 callbacks (time-based triggers), got {callback.call_count}"
        )

    def test_progress_callback_error_logged_not_raised(self):
        """If callback raises, it should be caught and logged, not crash extraction."""

        def bad_callback(*args):
            raise ValueError("Callback error!")

        algo = self._make_algo_with_mock_nlp(20)
        chunks = ["word." for _ in range(20)]

        # Should not raise
        result = algo.extract("dummy", chunks=chunks, progress_callback=bad_callback)
        assert result is not None

    def test_no_callback_no_error(self):
        """Extraction without progress_callback should work fine."""
        algo = self._make_algo_with_mock_nlp(5)
        chunks = ["word." for _ in range(5)]

        result = algo.extract("dummy", chunks=chunks)
        assert result is not None
        assert result.metadata["chunks_processed"] == 5
