"""
Integration tests for import rule verification.

Verifies that the import rules from PIPELINE.md are respected.
"""

import subprocess
import sys
from pathlib import Path


class TestImportRules:
    """Tests for import rule compliance."""

    def test_no_import_violations(self):
        """Verify import rules from PIPELINE.md are respected."""
        # Get project root
        project_root = Path(__file__).parent.parent.parent

        # Run the find_violations.py script
        result = subprocess.run(
            [sys.executable, str(project_root / "find_violations.py"), "--quick"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        # Check for violations
        # The script should exit with 0 if no violations found
        assert result.returncode == 0 or "No violations found" in result.stdout, (
            f"Import violations found:\n{result.stdout}\n{result.stderr}"
        )
