"""
Tests for session version header logging and old log file purging.

Covers:
- setup_logging() writes a version session header
- purge_old_logs() deletes files older than retention setting
- purge_old_logs() respects "keep forever" (0 days)
- purge_old_logs() handles missing dirs, unreadable files
- log_retention_days setting is registered in SettingsRegistry
"""

import logging
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Session version header
# ---------------------------------------------------------------------------


class TestSessionVersionHeader:
    """setup_logging() should emit a version line after handlers are attached."""

    def test_version_header_in_log(self, tmp_path, caplog):
        """Session start line includes the app version string."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        with (
            patch("src.config.LOGS_DIR", logs_dir),
            caplog.at_level(logging.DEBUG, logger="src"),
        ):
            # Reset handler state so setup_logging() runs fresh
            root = logging.getLogger("src")
            original_handlers = root.handlers[:]
            root.handlers.clear()

            import src.logging_config as lc

            old_fh = lc._file_handler
            old_cf = lc._category_filter
            lc._file_handler = None
            lc._category_filter = None

            try:
                lc.setup_logging()

                version_lines = [r.message for r in caplog.records if "session start" in r.message]
                assert len(version_lines) == 1
                assert "CasePrepd v" in version_lines[0]
            finally:
                # Restore original state so other tests aren't affected
                for h in root.handlers[:]:
                    if h not in original_handlers:
                        root.removeHandler(h)
                        h.close()
                root.handlers = original_handlers
                lc._file_handler = old_fh
                lc._category_filter = old_cf

    def test_version_matches_package_version(self, tmp_path, caplog):
        """The logged version matches src.__version__."""
        from src import __version__

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        with (
            patch("src.config.LOGS_DIR", logs_dir),
            caplog.at_level(logging.DEBUG, logger="src"),
        ):
            root = logging.getLogger("src")
            original_handlers = root.handlers[:]
            root.handlers.clear()

            import src.logging_config as lc

            old_fh = lc._file_handler
            old_cf = lc._category_filter
            lc._file_handler = None
            lc._category_filter = None

            try:
                lc.setup_logging()

                version_lines = [r.message for r in caplog.records if "session start" in r.message]
                assert __version__ in version_lines[0]
            finally:
                for h in root.handlers[:]:
                    if h not in original_handlers:
                        root.removeHandler(h)
                        h.close()
                root.handlers = original_handlers
                lc._file_handler = old_fh
                lc._category_filter = old_cf


# ---------------------------------------------------------------------------
# purge_old_logs()
# ---------------------------------------------------------------------------


class TestPurgeOldLogs:
    """purge_old_logs() deletes main_log_*.txt files past retention."""

    @pytest.fixture
    def logs_dir(self, tmp_path):
        """Create a temp logs directory with old and new log files."""
        d = tmp_path / "logs"
        d.mkdir()
        return d

    def _create_log_file(self, logs_dir: Path, name: str, age_days: int) -> Path:
        """Create a main_log file and backdate its mtime."""
        p = logs_dir / name
        p.write_text("log content", encoding="utf-8")
        old_time = time.time() - (age_days * 86400) - 60  # extra minute buffer
        os.utime(p, (old_time, old_time))
        return p

    def test_deletes_old_files(self, logs_dir):
        """Files older than retention days are deleted."""
        old_file = self._create_log_file(logs_dir, "main_log_20250101_120000.txt", 100)
        new_file = logs_dir / "main_log_20260210_120000.txt"
        new_file.write_text("recent", encoding="utf-8")

        class FakePrefs:
            def get(self, key, default=None):
                if key == "log_retention_days":
                    return "30"
                return default

        with (
            patch("src.user_preferences.get_user_preferences", return_value=FakePrefs()),
            patch("src.config.LOGS_DIR", logs_dir),
        ):
            from src.logging_config import purge_old_logs

            deleted = purge_old_logs()

        assert deleted == 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_keeps_all_when_zero_retention(self, logs_dir):
        """Retention of 0 means 'keep forever' -- nothing deleted."""
        old_file = self._create_log_file(logs_dir, "main_log_20240101_120000.txt", 500)

        class FakePrefs:
            def get(self, key, default=None):
                if key == "log_retention_days":
                    return "0"
                return default

        with (
            patch("src.user_preferences.get_user_preferences", return_value=FakePrefs()),
            patch("src.config.LOGS_DIR", logs_dir),
        ):
            from src.logging_config import purge_old_logs

            deleted = purge_old_logs()

        assert deleted == 0
        assert old_file.exists()

    def test_ignores_non_matching_files(self, logs_dir):
        """Only main_log_*.txt files are considered for deletion."""
        old_log = self._create_log_file(logs_dir, "main_log_20240101_120000.txt", 200)
        # These do NOT match main_log_*.txt
        other = logs_dir / "caseprepd.log"
        other.write_text("structured log", encoding="utf-8")
        os.utime(other, (time.time() - 999 * 86400, time.time() - 999 * 86400))

        readme = logs_dir / "README.txt"
        readme.write_text("info", encoding="utf-8")
        os.utime(readme, (time.time() - 999 * 86400, time.time() - 999 * 86400))

        class FakePrefs:
            def get(self, key, default=None):
                if key == "log_retention_days":
                    return "7"
                return default

        with (
            patch("src.user_preferences.get_user_preferences", return_value=FakePrefs()),
            patch("src.config.LOGS_DIR", logs_dir),
        ):
            from src.logging_config import purge_old_logs

            deleted = purge_old_logs()

        assert deleted == 1
        assert not old_log.exists()
        assert other.exists()
        assert readme.exists()

    def test_nothing_to_delete(self, logs_dir):
        """Returns 0 when all files are within retention window."""
        recent = logs_dir / "main_log_20260210_120000.txt"
        recent.write_text("recent", encoding="utf-8")

        class FakePrefs:
            def get(self, key, default=None):
                if key == "log_retention_days":
                    return "30"
                return default

        with (
            patch("src.user_preferences.get_user_preferences", return_value=FakePrefs()),
            patch("src.config.LOGS_DIR", logs_dir),
        ):
            from src.logging_config import purge_old_logs

            deleted = purge_old_logs()

        assert deleted == 0
        assert recent.exists()

    def test_empty_directory(self, logs_dir):
        """Returns 0 when logs directory is empty."""

        class FakePrefs:
            def get(self, key, default=None):
                if key == "log_retention_days":
                    return "7"
                return default

        with (
            patch("src.user_preferences.get_user_preferences", return_value=FakePrefs()),
            patch("src.config.LOGS_DIR", logs_dir),
        ):
            from src.logging_config import purge_old_logs

            deleted = purge_old_logs()

        assert deleted == 0

    def test_defaults_to_90_on_preference_error(self, logs_dir):
        """Falls back to 90-day retention when preferences are unavailable."""
        old_89 = self._create_log_file(logs_dir, "main_log_20251101_120000.txt", 89)
        old_91 = self._create_log_file(logs_dir, "main_log_20251001_120000.txt", 91)

        with (
            patch(
                "src.user_preferences.get_user_preferences",
                side_effect=Exception("prefs unavailable"),
            ),
            patch("src.config.LOGS_DIR", logs_dir),
        ):
            from src.logging_config import purge_old_logs

            deleted = purge_old_logs()

        assert deleted == 1
        assert old_89.exists()  # 89 < 90, kept
        assert not old_91.exists()  # 91 > 90, deleted

    def test_deletes_multiple_old_files(self, logs_dir):
        """Multiple old files are all deleted in one pass."""
        old1 = self._create_log_file(logs_dir, "main_log_20240101_120000.txt", 400)
        old2 = self._create_log_file(logs_dir, "main_log_20240601_120000.txt", 250)
        old3 = self._create_log_file(logs_dir, "main_log_20250101_120000.txt", 100)
        new1 = logs_dir / "main_log_20260210_120000.txt"
        new1.write_text("recent", encoding="utf-8")

        class FakePrefs:
            def get(self, key, default=None):
                if key == "log_retention_days":
                    return "30"
                return default

        with (
            patch("src.user_preferences.get_user_preferences", return_value=FakePrefs()),
            patch("src.config.LOGS_DIR", logs_dir),
        ):
            from src.logging_config import purge_old_logs

            deleted = purge_old_logs()

        assert deleted == 3
        assert not old1.exists()
        assert not old2.exists()
        assert not old3.exists()
        assert new1.exists()

    def test_logs_purge_count(self, logs_dir, caplog):
        """A log message is emitted when files are purged."""
        self._create_log_file(logs_dir, "main_log_20240101_120000.txt", 200)

        class FakePrefs:
            def get(self, key, default=None):
                if key == "log_retention_days":
                    return "7"
                return default

        with (
            patch("src.user_preferences.get_user_preferences", return_value=FakePrefs()),
            patch("src.config.LOGS_DIR", logs_dir),
            caplog.at_level(logging.DEBUG, logger="src.logging_config"),
        ):
            from src.logging_config import purge_old_logs

            purge_old_logs()

        purge_msgs = [r.message for r in caplog.records if "Purged" in r.message]
        assert len(purge_msgs) == 1
        assert "1 old log file" in purge_msgs[0]

    def test_no_log_when_nothing_purged(self, logs_dir, caplog):
        """No purge log message when zero files deleted."""
        recent = logs_dir / "main_log_20260210_120000.txt"
        recent.write_text("recent", encoding="utf-8")

        class FakePrefs:
            def get(self, key, default=None):
                if key == "log_retention_days":
                    return "7"
                return default

        with (
            patch("src.user_preferences.get_user_preferences", return_value=FakePrefs()),
            patch("src.config.LOGS_DIR", logs_dir),
            caplog.at_level(logging.DEBUG, logger="src.logging_config"),
        ):
            from src.logging_config import purge_old_logs

            purge_old_logs()

        purge_msgs = [r.message for r in caplog.records if "Purged" in r.message]
        assert len(purge_msgs) == 0

    def test_handles_unreadable_file_gracefully(self, logs_dir):
        """An OSError on stat() doesn't crash the purge."""
        self._create_log_file(logs_dir, "main_log_20240101_100000.txt", 200)
        self._create_log_file(logs_dir, "main_log_20240101_120000.txt", 200)

        class FakePrefs:
            def get(self, key, default=None):
                if key == "log_retention_days":
                    return "7"
                return default

        with (
            patch("src.user_preferences.get_user_preferences", return_value=FakePrefs()),
            patch("src.config.LOGS_DIR", logs_dir),
        ):
            from src.logging_config import purge_old_logs

            # Patch Path.stat to always raise -> 0 deleted, no crash
            with patch.object(Path, "stat", side_effect=OSError("permission denied")):
                deleted = purge_old_logs()
                assert deleted == 0

    def test_seven_day_retention(self, logs_dir):
        """7-day retention keeps files from 5 days ago, deletes 10 days ago."""
        recent = self._create_log_file(logs_dir, "main_log_20260207_120000.txt", 5)
        old = self._create_log_file(logs_dir, "main_log_20260202_120000.txt", 10)

        class FakePrefs:
            def get(self, key, default=None):
                if key == "log_retention_days":
                    return "7"
                return default

        with (
            patch("src.user_preferences.get_user_preferences", return_value=FakePrefs()),
            patch("src.config.LOGS_DIR", logs_dir),
        ):
            from src.logging_config import purge_old_logs

            deleted = purge_old_logs()

        assert deleted == 1
        assert recent.exists()
        assert not old.exists()


# ---------------------------------------------------------------------------
# Settings registry: log_retention_days
# ---------------------------------------------------------------------------


class TestLogRetentionSetting:
    """log_retention_days is registered in SettingsRegistry."""

    def test_setting_registered(self):
        """log_retention_days appears in the registry."""
        from src.ui.settings.settings_registry import SettingsRegistry

        setting = SettingsRegistry.get_setting("log_retention_days")
        assert setting is not None

    def test_setting_in_logging_category(self):
        """log_retention_days is in the Logging category."""
        from src.ui.settings.settings_registry import SettingsRegistry

        setting = SettingsRegistry.get_setting("log_retention_days")
        assert setting.category == "Logging"

    def test_setting_default_is_90(self):
        """Default retention is 90 days."""
        from src.ui.settings.settings_registry import SettingsRegistry

        setting = SettingsRegistry.get_setting("log_retention_days")
        assert setting.default == "90"

    def test_setting_has_keep_forever_option(self):
        """'Keep forever' (value 0) is one of the dropdown options."""
        from src.ui.settings.settings_registry import SettingsRegistry

        setting = SettingsRegistry.get_setting("log_retention_days")
        option_values = [v for _, v in setting.options]
        assert "0" in option_values

    def test_setting_has_four_options(self):
        """Dropdown has exactly 4 retention choices."""
        from src.ui.settings.settings_registry import SettingsRegistry

        setting = SettingsRegistry.get_setting("log_retention_days")
        assert len(setting.options) == 4
