"""
Tests for the GUI polish pass — verifying button styling, labels, disabled
states, dead code removal, tab renames, and updated warning text.

These tests read source files as text (no running Tk instance needed).
"""

from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src"


def _read(relative_path: str) -> str:
    """Read a source file and return its content."""
    return (SRC.parent / relative_path).read_text(encoding="utf-8")


# =========================================================================
# 1. Clear All button danger styling
# =========================================================================


class TestClearAllDangerStyling:
    """Clear All button uses danger (red) styling, not gray."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.text = _read("src/ui/window_layout.py")

    def test_clear_all_uses_danger_style(self):
        """Clear All button uses BUTTON_STYLES['danger']."""
        idx = self.text.index("self.clear_files_btn = ctk.CTkButton")
        block = self.text[idx : idx + 400]
        assert '**BUTTON_STYLES["danger"]' in block

    def test_clear_all_no_gray_fg_color(self):
        """Clear All button no longer uses gray fg_color."""
        idx = self.text.index("self.clear_files_btn = ctk.CTkButton")
        block = self.text[idx : idx + 400]
        assert 'fg_color=("gray70", "gray30")' not in block


# =========================================================================
# 2. Default questions label
# =========================================================================


class TestDefaultQuestionsLabel:
    """Default questions checkbox starts with generic text."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.text = _read("src/ui/window_layout.py")

    def test_no_zero_count_in_initial_label(self):
        """Label does NOT say 'Ask 0 default questions'."""
        assert "Ask 0 default questions" not in self.text

    def test_generic_initial_label(self):
        """Label says 'Ask default questions' (count set later at runtime)."""
        assert 'text="Ask default questions"' in self.text


# =========================================================================
# 3. Follow-up entry starts disabled
# =========================================================================


class TestFollowupEntryDisabled:
    """Follow-up entry starts disabled until Q&A completes."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.text = _read("src/ui/window_layout.py")

    def test_followup_entry_starts_disabled(self):
        """followup_entry is created with state='disabled'."""
        idx = self.text.index("self.followup_entry = ctk.CTkEntry")
        block = self.text[idx : idx + 400]
        assert 'state="disabled"' in block

    def test_no_diagnostic_comment(self):
        """No leftover diagnostic comment about typing issue."""
        assert "TEST: Starting enabled to diagnose typing issue" not in self.text


# =========================================================================
# 4. Dead code removal from qa_panel.py
# =========================================================================


class TestQAPanelDeadCodeRemoved:
    """Verify deleted methods are gone from qa_panel.py."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.text = _read("src/ui/qa_panel.py")

    def test_no_create_followup_pane(self):
        """_create_followup_pane method is removed."""
        assert "def _create_followup_pane" not in self.text

    def test_no_toggle_followup_pane(self):
        """_toggle_followup_pane method is removed."""
        assert "def _toggle_followup_pane" not in self.text

    def test_no_submit_followup(self):
        """_submit_followup method is removed."""
        assert "def _submit_followup" not in self.text

    def test_no_poll_followup_result(self):
        """_poll_followup_result method is removed."""
        assert "def _poll_followup_result" not in self.text

    def test_no_queue_import(self):
        """Unused queue import is removed."""
        assert "import queue" not in self.text

    def test_no_threading_import(self):
        """Unused threading import is removed."""
        assert "import threading" not in self.text

    def test_followup_comment_present(self):
        """Comment confirms follow-up is in main_window, not here."""
        assert "Follow-up input is in the main window" in self.text


# =========================================================================
# 5. Tab name renames
# =========================================================================


class TestTabNameRenames:
    """Tab names are 'Vocabulary', 'Questions', 'Summary'."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.dyn = _read("src/ui/dynamic_output.py")

    def test_vocabulary_tab_exists(self):
        """'Vocabulary' tab is added."""
        assert 'tabview.add("Vocabulary")' in self.dyn

    def test_questions_tab_exists(self):
        """'Questions' tab is added."""
        assert 'tabview.add("Questions")' in self.dyn

    def test_summary_tab_exists(self):
        """'Summary' tab is added."""
        assert 'tabview.add("Summary")' in self.dyn

    def test_no_old_names_and_vocab_tab(self):
        """Old 'Names & Vocab' tab name is gone."""
        assert '"Names & Vocab"' not in self.dyn

    def test_no_old_ask_questions_tab(self):
        """Old 'Ask Questions' tab name is gone."""
        assert '"Ask Questions"' not in self.dyn

    def test_main_window_uses_new_tab_name(self):
        """main_window.py references 'Questions' tab, not 'Ask Questions'."""
        mw = _read("src/ui/main_window.py")
        assert 'tabview.set("Questions")' in mw
        assert 'tabview.set("Ask Questions")' not in mw

    def test_help_dialog_uses_new_tab_names(self):
        """Help dialog references new tab names."""
        help_text = _read("src/ui/help_about_dialogs.py")
        assert "- Vocabulary:" in help_text
        assert "- Questions:" in help_text
        assert "- Names & Vocab:" not in help_text


# =========================================================================
# 6. Stale checkbox text fixed
# =========================================================================


class TestStaleCheckboxText:
    """Export warning messages use updated text, not old checkbox reference."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.text = _read("src/ui/qa_panel.py")

    def test_no_old_checkbox_message(self):
        """Old 'Click the checkboxes in the Include column' is gone."""
        assert "Click the checkboxes in the Include column" not in self.text

    def test_new_select_all_message(self):
        """New message mentions 'Select All'."""
        assert "Use 'Select All' or click individual results" in self.text

    def test_five_export_methods_use_new_text(self):
        """All 5 export methods use the updated warning text."""
        methods = [
            "_export_to_csv",
            "_export_to_txt",
            "_export_to_word",
            "_export_to_pdf",
            "_export_to_html",
        ]
        for method_name in methods:
            assert f"def {method_name}" in self.text, f"{method_name} not found"

        # Count occurrences of the new message (should be 5)
        count = self.text.count("Use 'Select All' or click individual results to include them.")
        assert count == 5, f"Expected 5 occurrences, found {count}"


# =========================================================================
# Stale comment cleanup
# =========================================================================


class TestStaleCommentCleanup:
    """Comments referencing deleted QAPanel methods are updated."""

    def test_main_window_no_qapanel_submit_followup_ref(self):
        """main_window.py doesn't reference QAPanel._submit_followup."""
        text = _read("src/ui/main_window.py")
        assert "QAPanel._submit_followup" not in text

    def test_task_mixin_no_qapanel_submit_followup_ref(self):
        """task_mixin.py doesn't reference QAPanel._submit_followup."""
        text = _read("src/ui/main_window_helpers/task_mixin.py")
        assert "QAPanel._submit_followup" not in text
