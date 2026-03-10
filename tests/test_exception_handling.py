"""
Tests for exception handling fixes across the codebase.

Covers:
- Category 1: Silent exception swallowing (now logs instead of pass)
- Category 2: Unprotected I/O operations (now wrapped in try/except)
- Category 3: IndexError risks (bounds checking added)
- Category 4: KeyError risks (column validation added)
- Category 5: None-safety / AttributeError risks (safe access)
- Category 6: Division by zero (early returns on empty data)
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

# =========================================================================
# Category 1: Silent exception swallowing - verify logging occurs
# =========================================================================


class TestDocumentServiceExceptionLogging:
    """Test that document_service logs exceptions instead of silently swallowing."""

    def test_preprocessing_settings_logs_on_failure(self, caplog):
        """_get_preprocessing_settings should log and return {} on failure."""
        from src.services.document_service import DocumentService

        with caplog.at_level(logging.DEBUG):
            # Patch the import inside _get_preprocessing_settings to raise
            with patch.dict("sys.modules", {"src.user_preferences": None}):
                result = DocumentService._get_preprocessing_settings()

        assert result == {}
        assert any("Failed to load preprocessing settings" in r.message for r in caplog.records)

    def test_preprocessing_settings_returns_dict_on_success(self):
        """_get_preprocessing_settings should return settings dict on success."""
        from src.services.document_service import DocumentService

        mock_prefs = MagicMock()
        mock_prefs.get.return_value = True

        mock_module = MagicMock()
        mock_module.get_user_preferences.return_value = mock_prefs

        with patch.dict("sys.modules", {"src.user_preferences": mock_module}):
            result = DocumentService._get_preprocessing_settings()

        assert isinstance(result, dict)
        assert len(result) > 0


class TestTooltipManagerExceptionLogging:
    """Test that tooltip_manager logs exceptions instead of silent pass."""

    def test_close_active_logs_on_destroy_error(self, caplog):
        """close_active should log when tooltip.destroy() fails."""
        from src.ui.tooltip_manager import TooltipManager

        manager = TooltipManager.__new__(TooltipManager)
        manager._active_tooltip = MagicMock()
        manager._active_owner = None
        manager._active_tooltip.winfo_exists.return_value = True
        manager._active_tooltip.destroy.side_effect = RuntimeError("Widget destroyed")

        with caplog.at_level(logging.DEBUG):
            manager.close_active()

        assert manager._active_tooltip is None
        assert any("Tooltip cleanup failed" in r.message for r in caplog.records)

    def test_close_active_succeeds_normally(self):
        """close_active should work when tooltip exists and destroys cleanly."""
        from src.ui.tooltip_manager import TooltipManager

        manager = TooltipManager.__new__(TooltipManager)
        manager._active_tooltip = MagicMock()
        manager._active_owner = "test"
        manager._active_tooltip.winfo_exists.return_value = True

        manager.close_active()
        assert manager._active_tooltip is None
        assert manager._active_owner is None

    def test_close_active_noop_when_no_tooltip(self):
        """close_active should be safe to call with no active tooltip."""
        from src.ui.tooltip_manager import TooltipManager

        manager = TooltipManager.__new__(TooltipManager)
        manager._active_tooltip = None
        manager._active_owner = None

        manager.close_active()  # Should not raise
        assert manager._active_tooltip is None


# =========================================================================
# Category 2: Unprotected I/O - verify error handling
# =========================================================================


class TestCorpusManagerIOProtection:
    """Test that corpus_manager wraps write_text in try/except."""

    def test_preprocess_file_raises_on_write_failure(self, tmp_path):
        """_preprocess_file should raise OSError with helpful message on write failure."""
        from src.core.vocabulary.corpus_manager import CorpusManager

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        mgr = CorpusManager(corpus_dir=corpus_dir, cache_dir=cache_dir)

        # Create a test file
        test_file = corpus_dir / "test.txt"
        test_file.write_text("Some text content for testing.")

        # Mock the entire preprocessing pipeline and make write_text fail
        with (
            patch("src.core.extraction.RawTextExtractor") as mock_ext_cls,
            patch("src.core.sanitization.CharacterSanitizer") as mock_san_cls,
            patch("src.core.preprocessing.create_default_pipeline") as mock_pipe,
        ):
            mock_ext_cls.return_value.extract.return_value = {
                "success": True,
                "text": "hello world",
            }
            mock_san_cls.return_value.sanitize.return_value = "hello world"
            mock_pipe.return_value.process.return_value = "hello world"

            # Patch Path.write_text to fail with PermissionError
            original_write = type(test_file).write_text

            def failing_write(self_path, *args, **kwargs):
                if "_preprocessed" in str(self_path):
                    raise PermissionError("Access denied")
                return original_write(self_path, *args, **kwargs)

            with patch.object(type(test_file), "write_text", failing_write):
                with pytest.raises(OSError, match="Failed to save preprocessed text"):
                    mgr.preprocess_file(test_file)


class TestDiagnoseMLIOProtection:
    """Test that diagnose_ml wraps I/O operations."""

    def test_corrupt_csv_handled(self, tmp_path, capsys):
        """analyze_feedback should handle corrupt CSV files gracefully."""
        import scripts.diagnose_ml as diag

        # Create a corrupt CSV (binary garbage)
        csv_path = tmp_path / "corrupt.csv"
        csv_path.write_bytes(b"\x00\x01\x02\xff\xfe")

        with patch.object(diag, "DEFAULT_FEEDBACK", csv_path):
            with patch.object(diag, "USER_FEEDBACK", tmp_path / "nonexistent.csv"):
                result = diag.analyze_feedback()

        # Should not crash - returns None because error is caught
        assert result is None
        captured = capsys.readouterr()
        assert "ERROR" in captured.out

    def test_model_load_handles_missing_file(self, tmp_path, capsys):
        """analyze_model should handle missing model file gracefully."""
        import scripts.diagnose_ml as diag

        with patch.object(diag, "MODEL_PATH", tmp_path / "nonexistent.pkl"):
            diag.analyze_model()

        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()


# =========================================================================
# Category 3: IndexError risks - verify bounds checking
# =========================================================================


class TestGpuDetectorBoundsChecking:
    """Test GPU detector handles edge-case nvidia-smi output."""

    def test_empty_stdout_returns_none(self):
        """nvidia-smi with empty stdout should return None from _detect_gpu_cli."""
        from src.core.utils.gpu_detector import _detect_gpu_cli

        with (
            patch(
                "shutil.which",
                side_effect=lambda cmd: "nvidia-smi" if cmd == "nvidia-smi" else None,
            ),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="")
            result = _detect_gpu_cli()
            # Empty stdout means no GPU detected
            assert result is None

    def test_valid_nvidia_smi_output(self):
        """nvidia-smi with valid output should parse correctly."""
        from src.core.utils.gpu_detector import _detect_gpu_cli

        with (
            patch(
                "shutil.which",
                side_effect=lambda cmd: "nvidia-smi" if cmd == "nvidia-smi" else None,
            ),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="NVIDIA GeForce RTX 3080, 10240\n",
            )
            result = _detect_gpu_cli()
            assert result is not None
            assert result["gpu_name"] == "NVIDIA GeForce RTX 3080"
            assert result["vram_bytes"] == 10240 * 1024 * 1024

    def test_single_column_nvidia_output(self):
        """nvidia-smi with only GPU name (no VRAM) should not crash."""
        from src.core.utils.gpu_detector import _detect_gpu_cli

        with (
            patch(
                "shutil.which",
                side_effect=lambda cmd: "nvidia-smi" if cmd == "nvidia-smi" else None,
            ),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="NVIDIA GeForce RTX 3080\n",
            )
            result = _detect_gpu_cli()
            assert result is not None
            assert result["gpu_name"] == "NVIDIA GeForce RTX 3080"
            assert result["vram_bytes"] == 0  # No VRAM column, defaults to 0


class TestPreferenceFeaturesBoundsChecking:
    """Test preference_learner_features handles edge-case strings."""

    MOCK_FREQ = {"the": 0.0}

    @pytest.fixture
    def mock_deps(self):
        """Mock all external dependencies for feature extraction."""
        with (
            patch(
                "src.core.vocabulary.preference_learner_features._load_scaled_frequencies",
                return_value=self.MOCK_FREQ,
            ),
            patch(
                "src.core.vocabulary.preference_learner_features._load_names_datasets",
                return_value=({"john"}, {"smith"}),
            ),
            patch(
                "src.core.vocabulary.preference_learner_features._get_name_country_data",
                return_value=({"john": 5}, 20),
            ),
            patch(
                "src.core.vocabulary.preference_learner_features.get_user_preferences",
                return_value=MagicMock(get=lambda k, d=None: d),
            ),
            patch(
                "src.core.vocabulary.preference_learner_features._log_rarity_score",
                side_effect=lambda x: x,
            ),
            patch(
                "src.core.vocabulary.preference_learner_features.compute_adjusted_mean",
                return_value=0.5,
            ),
            patch(
                "src.core.vocabulary.rarity_filter.is_common_word",
                side_effect=lambda word, top_n=200000: word in ("the",),
            ),
        ):
            from src.core.vocabulary.preference_learner_features import extract_features

            yield extract_features

    def test_empty_string_term_raises_valueerror(self, mock_deps):
        """Empty string term should raise ValueError (validated at function start)."""
        extract_features = mock_deps
        term_data = {"Term": "", "occurrences": 1, "is_person": 0, "algorithms": "NER"}
        with pytest.raises(ValueError, match="non-empty"):
            extract_features(term_data)

    def test_whitespace_only_term(self, mock_deps):
        """Whitespace-only term should not raise IndexError."""
        extract_features = mock_deps
        term_data = {"Term": "   ", "occurrences": 1, "is_person": 0, "algorithms": "NER"}
        result = extract_features(term_data)
        assert result is not None

    def test_single_char_term(self, mock_deps):
        """Single character term should work without IndexError."""
        extract_features = mock_deps
        term_data = {"Term": "X", "occurrences": 1, "is_person": 0, "algorithms": "NER"}
        result = extract_features(term_data)
        assert result is not None

    def test_term_with_trailing_punctuation(self, mock_deps):
        """Term ending in punctuation should detect it safely."""
        extract_features = mock_deps
        term_data = {"Term": "Smith:", "occurrences": 3, "is_person": 1, "algorithms": "NER"}
        result = extract_features(term_data)
        assert result is not None

    def test_term_with_leading_digit(self, mock_deps):
        """Term starting with a digit should detect it safely."""
        extract_features = mock_deps
        term_data = {"Term": "3M", "occurrences": 2, "is_person": 0, "algorithms": "RAKE"}
        result = extract_features(term_data)
        assert result is not None


# =========================================================================
# Category 4: KeyError risks - verify column validation
# =========================================================================


class TestDiagnoseMLColumnValidation:
    """Test that diagnose_ml validates required columns."""

    def test_missing_columns_returns_none(self, tmp_path, capsys):
        """analyze_feedback should return None when required columns are missing."""
        import scripts.diagnose_ml as diag

        csv_path = tmp_path / "bad_feedback.csv"
        csv_path.write_text("wrong_col1,wrong_col2\n1,2\n")

        with patch.object(diag, "DEFAULT_FEEDBACK", csv_path):
            with patch.object(diag, "USER_FEEDBACK", tmp_path / "nonexistent.csv"):
                result = diag.analyze_feedback()

        assert result is None
        captured = capsys.readouterr()
        assert "Missing required columns" in captured.out

    def test_empty_dataframe_returns_early(self, tmp_path, capsys):
        """analyze_feedback should handle empty DataFrames gracefully."""
        import scripts.diagnose_ml as diag

        csv_path = tmp_path / "empty_feedback.csv"
        csv_path.write_text("feedback,term,occurrences\n")

        with patch.object(diag, "DEFAULT_FEEDBACK", csv_path):
            with patch.object(diag, "USER_FEEDBACK", tmp_path / "nonexistent.csv"):
                result = diag.analyze_feedback()

        assert result is not None
        assert len(result) == 0
        captured = capsys.readouterr()
        assert "No data to analyze" in captured.out

    def test_valid_columns_work(self, tmp_path, capsys):
        """analyze_feedback should work when required columns are present."""
        import scripts.diagnose_ml as diag

        csv_path = tmp_path / "good_feedback.csv"
        csv_path.write_text("feedback,term,occurrences,is_person,algorithms\n1,Smith,5,1,NER\n")

        with patch.object(diag, "DEFAULT_FEEDBACK", csv_path):
            with patch.object(diag, "USER_FEEDBACK", tmp_path / "nonexistent.csv"):
                result = diag.analyze_feedback()

        assert result is not None
        assert len(result) == 1


# =========================================================================
# Category 5: None-safety - verify safe access
# =========================================================================

# TestFocusExtractorNoneSafety removed — src/core/prompting/ deprecated


class TestLayoutAnalyzerNoneReturn:
    """Test that layout_analyzer callers handle None returns."""

    def test_detect_zones_returns_none_for_short_doc(self):
        """detect_zones should return None for documents with too few pages."""
        from src.core.extraction.layout_analyzer import LayoutAnalyzer

        analyzer = LayoutAnalyzer()
        doc = MagicMock()
        doc.__len__ = MagicMock(return_value=2)

        result = analyzer.detect_zones(doc)
        assert result is None

    def test_detect_zones_returns_none_for_insufficient_samples(self):
        """detect_zones should return None when not enough sample pages."""
        from src.core.extraction.layout_analyzer import LayoutAnalyzer

        analyzer = LayoutAnalyzer()
        doc = MagicMock()
        doc.__len__ = MagicMock(return_value=1)

        result = analyzer.detect_zones(doc)
        assert result is None


# =========================================================================
# Category 6: Division by zero - verify early returns
# =========================================================================


class TestDiagnoseMLDivisionSafety:
    """Test that diagnose_ml handles zero-length data without division errors."""

    def test_no_feedback_files_returns_none(self, tmp_path):
        """analyze_feedback should return None when no files exist."""
        import scripts.diagnose_ml as diag

        with patch.object(diag, "DEFAULT_FEEDBACK", tmp_path / "nonexistent1.csv"):
            with patch.object(diag, "USER_FEEDBACK", tmp_path / "nonexistent2.csv"):
                result = diag.analyze_feedback()

        assert result is None

    def test_valid_data_no_division_crash(self, tmp_path):
        """analyze_feedback should compute percentages without ZeroDivisionError."""
        import scripts.diagnose_ml as diag

        csv_path = tmp_path / "feedback.csv"
        csv_path.write_text(
            "feedback,term,occurrences,is_person,algorithms\n1,Smith,5,1,NER\n-1,the,100,0,RAKE\n"
        )

        with patch.object(diag, "DEFAULT_FEEDBACK", csv_path):
            with patch.object(diag, "USER_FEEDBACK", tmp_path / "nonexistent.csv"):
                result = diag.analyze_feedback()

        assert result is not None
        assert len(result) == 2

    def test_single_row_data(self, tmp_path):
        """analyze_feedback should work with just one row."""
        import scripts.diagnose_ml as diag

        csv_path = tmp_path / "feedback.csv"
        csv_path.write_text("feedback,term,occurrences,is_person,algorithms\n1,Smith,5,1,NER\n")

        with patch.object(diag, "DEFAULT_FEEDBACK", csv_path):
            with patch.object(diag, "USER_FEEDBACK", tmp_path / "nonexistent.csv"):
                result = diag.analyze_feedback()

        assert result is not None
        assert len(result) == 1


# =========================================================================
# Integration: Verify no import violations from our changes
# =========================================================================


class TestNoImportViolations:
    """Verify our logging additions don't introduce import violations."""

    def test_tooltip_manager_import(self):
        """tooltip_manager should import logging without violations."""
        from src.ui.tooltip_manager import logger

        assert logger is not None
        assert logger.name == "src.ui.tooltip_manager"

    def test_document_service_import(self):
        """document_service should import cleanly."""
        from src.services.document_service import DocumentService

        assert DocumentService is not None

    def test_system_monitor_has_logger(self):
        """system_monitor should have logger available."""
        from src.ui.system_monitor import logger

        assert logger is not None

    def test_corpus_widget_has_logger(self):
        """corpus_widget should have logger available."""
        from src.ui.settings.corpus_widget import logger

        assert logger is not None
