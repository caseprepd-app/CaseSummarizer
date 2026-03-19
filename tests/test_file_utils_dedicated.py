"""
Dedicated tests for src/file_utils.py.

Covers atomic file replacement with retry logic for Windows/Dropbox locking:
safe_replace() retry on PermissionError, exhaustion, concurrent use, etc.
"""

import os
import threading
from unittest.mock import patch

import pytest

from src.file_utils import safe_replace


def test_atomic_write_creates_file(tmp_path):
    """Write via safe_replace to a new destination. Verify content arrives."""
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("hello world", encoding="utf-8")

    safe_replace(str(src), str(dst))

    assert dst.read_text(encoding="utf-8") == "hello world"
    assert not src.exists()  # os.replace removes source


def test_atomic_write_overwrites(tmp_path):
    """Overwrite an existing destination. Verify new content replaces old."""
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    dst.write_text("old content", encoding="utf-8")
    src.write_text("new content", encoding="utf-8")

    safe_replace(str(src), str(dst))

    assert dst.read_text(encoding="utf-8") == "new content"


def test_atomic_write_preserves_on_failure(tmp_path):
    """If os.replace always fails, the original destination stays intact."""
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    dst.write_text("precious data", encoding="utf-8")
    src.write_text("replacement", encoding="utf-8")

    with patch("src.file_utils.os.replace", side_effect=PermissionError("locked")):
        with pytest.raises(PermissionError):
            safe_replace(str(src), str(dst))

    assert dst.read_text(encoding="utf-8") == "precious data"


def test_retry_on_permission_error(tmp_path):
    """PermissionError on first 2 attempts, success on 3rd. Verify file arrives."""
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("retry content", encoding="utf-8")

    call_count = {"n": 0}
    real_replace = os.replace

    def flaky_replace(s, d):
        """Fail twice, then succeed."""
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise PermissionError("Dropbox lock")
        real_replace(s, d)

    with patch("src.file_utils.os.replace", side_effect=flaky_replace):
        with patch("src.file_utils.time.sleep"):  # skip delay
            safe_replace(str(src), str(dst))

    assert dst.read_text(encoding="utf-8") == "retry content"
    assert call_count["n"] == 3


def test_retry_exhaustion(tmp_path):
    """All retries fail. Verify PermissionError is raised, no infinite loop."""
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("data", encoding="utf-8")

    with patch("src.file_utils.os.replace", side_effect=PermissionError("locked")):
        with patch("src.file_utils.time.sleep"):
            with pytest.raises(PermissionError, match="locked"):
                safe_replace(str(src), str(dst))


def test_retry_exhaustion_attempt_count(tmp_path):
    """Verify exactly _REPLACE_MAX_ATTEMPTS calls are made before giving up."""
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("data", encoding="utf-8")

    call_count = {"n": 0}

    def counting_replace(s, d):
        """Count and always fail."""
        call_count["n"] += 1
        raise PermissionError("locked")

    with patch("src.file_utils.os.replace", side_effect=counting_replace):
        with patch("src.file_utils.time.sleep"):
            with pytest.raises(PermissionError):
                safe_replace(str(src), str(dst))

    from src.file_utils import _REPLACE_MAX_ATTEMPTS

    assert call_count["n"] == _REPLACE_MAX_ATTEMPTS


def test_concurrent_writes(tmp_path):
    """Two threads writing to different files simultaneously both succeed."""
    results = {}
    errors = []

    def writer(name):
        """Write a file in a thread and record success or failure."""
        try:
            src = tmp_path / f"{name}_src.txt"
            dst = tmp_path / f"{name}_dst.txt"
            src.write_text(f"content-{name}", encoding="utf-8")
            safe_replace(str(src), str(dst))
            results[name] = dst.read_text(encoding="utf-8")
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=writer, args=("a",))
    t2 = threading.Thread(target=writer, args=("b",))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not errors, f"Thread errors: {errors}"
    assert results["a"] == "content-a"
    assert results["b"] == "content-b"


def test_directory_creation_not_implicit(tmp_path):
    """safe_replace does not create missing parent dirs; OS error is raised."""
    src = tmp_path / "src.txt"
    src.write_text("data", encoding="utf-8")
    dst = tmp_path / "nonexistent" / "subdir" / "dst.txt"

    with pytest.raises(OSError):
        safe_replace(str(src), str(dst))
