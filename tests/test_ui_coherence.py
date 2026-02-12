"""
Tests for UI text coherence — verifying that user-facing strings
reference actual current buttons, checkboxes, settings, and features.

These tests read source files as text and check that references match
the actual UI element definitions, without needing a running Tk instance.
"""

from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src"
UI = SRC / "ui"


def _read(relative_path: str) -> str:
    """Read a source file and return its content."""
    return (SRC.parent / relative_path).read_text(encoding="utf-8")


# =========================================================================
# Source-of-truth: actual UI element labels from window_layout.py
# =========================================================================


class TestActualUIElements:
    """Verify the source-of-truth labels we test against."""

    def test_add_files_button(self):
        """Main file button says '+ Add Files'."""
        text = _read("src/ui/window_layout.py")
        assert 'text="+ Add Files"' in text

    def test_perform_tasks_button(self):
        """Main action button says 'Perform N Tasks'."""
        text = _read("src/ui/window_layout.py")
        assert 'text="Perform 2 Tasks"' in text

    def test_vocab_checkbox(self):
        """Vocabulary checkbox says 'Extract Vocabulary'."""
        text = _read("src/ui/window_layout.py")
        assert 'text="Extract Vocabulary"' in text

    def test_qa_checkbox(self):
        """Q&A checkbox says 'Ask Questions'."""
        text = _read("src/ui/window_layout.py")
        assert 'text="Ask Questions"' in text

    def test_summary_checkbox(self):
        """Summary checkbox says 'Generate Summary'."""
        text = _read("src/ui/window_layout.py")
        assert 'text="Generate Summary"' in text

    def test_results_tab_names(self):
        """Results tabs are 'Names & Vocab', 'Ask Questions', 'Summary'."""
        text = _read("src/ui/dynamic_output.py")
        assert '"Names & Vocab"' in text
        assert '"Ask Questions"' in text
        assert '"Summary"' in text


# =========================================================================
# Help dialog coherence (help_about_dialogs.py)
# =========================================================================


class TestHelpDialogCoherence:
    """Help dialog text matches actual UI elements."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.text = _read("src/ui/help_about_dialogs.py")

    def test_step1_mentions_drag_and_drop(self):
        """Step 1 mentions drag-and-drop."""
        assert "Drag files" in self.text

    def test_step1_mentions_add_files_button(self):
        """Step 1 mentions the actual '+ Add Files' button."""
        assert "+ Add Files" in self.text

    def test_step3_checkbox_extract_vocabulary(self):
        """Step 3 checkbox matches actual label 'Extract Vocabulary'."""
        assert "- Extract Vocabulary" in self.text

    def test_step3_checkbox_ask_questions(self):
        """Step 3 checkbox matches actual label 'Ask Questions'."""
        assert "- Ask Questions" in self.text

    def test_step3_checkbox_generate_summary(self):
        """Step 3 matches actual label 'Generate Summary'."""
        assert "- Generate Summary" in self.text

    def test_step3_no_stale_checkbox_names(self):
        """No old checkbox names from dead code."""
        assert "Extract Names & Vocabulary" not in self.text
        assert "Enable Q&A" not in self.text

    def test_step4_perform_tasks_button(self):
        """Step 4 references 'Perform Tasks' not 'Generate Outputs'."""
        assert "Perform Tasks" in self.text
        assert "Generate Outputs" not in self.text

    def test_results_tab_name_matches(self):
        """Results tab name in help matches actual tab 'Names & Vocab'."""
        assert "Names & Vocab:" in self.text

    def test_ctrl_f_shortcut_listed(self):
        """Ctrl+F shortcut is listed."""
        assert "Ctrl+F" in self.text

    def test_no_phantom_llm_enhancement_setting(self):
        """No reference to phantom 'LLM enhancement in settings'."""
        assert "LLM enhancement in settings" not in self.text

    def test_gliner_tip_present(self):
        """Tips mention GLiNER under Settings > Vocabulary."""
        assert "GLiNER" in self.text
        assert "Settings > Vocabulary" in self.text


# =========================================================================
# Error message coherence
# =========================================================================


class TestErrorMessageCoherence:
    """Error/warning messages reference actual UI elements."""

    def test_questions_not_ready_references_perform_tasks(self):
        """'Questions Not Ready' dialog references 'Perform Tasks' button."""
        text = _read("src/ui/main_window_helpers/task_mixin.py")
        assert "Click 'Perform Tasks'" in text

    def test_questions_not_ready_no_phantom_status(self):
        """No reference to phantom 'Questions and answers ready' status."""
        for path in [
            "src/ui/main_window.py",
            "src/ui/main_window_helpers/task_mixin.py",
        ]:
            text = _read(path)
            assert "Questions and answers ready" not in text

    def test_questions_not_ready_uses_search_index(self):
        """Uses 'search index' phrasing instead of phantom status message."""
        text = _read("src/ui/main_window_helpers/task_mixin.py")
        assert "search index" in text

    def test_corpus_disabled_path_correct(self):
        """Corpus disabled message doesn't reference 'Settings > Manage Corpus'."""
        text = _read("src/ui/main_window.py")
        assert "Settings > Manage Corpus" not in text

    def test_corpus_disabled_mentions_manage_button(self):
        """Corpus disabled message mentions the Manage button."""
        text = _read("src/ui/main_window.py")
        assert "Manage button" in text


# =========================================================================
# Q&A panel coherence
# =========================================================================


class TestQAPanelCoherence:
    """Q&A panel references match reality."""

    def test_edit_questions_fallback_no_stale_path(self):
        """No reference to removed 'Edit Default Questions' button."""
        text = _read("src/ui/qa_panel.py")
        assert "Edit Default Questions" not in text

    def test_edit_questions_points_to_settings_qa(self):
        """Fallback message directs to Settings > Q&A."""
        text = _read("src/ui/qa_panel.py")
        assert "Settings > Q&A" in text


# =========================================================================
# Settings tooltip coherence
# =========================================================================


class TestSettingsTooltipCoherence:
    """Settings tooltips reference actual UI elements."""

    def test_summary_length_no_slider_reference(self):
        """Summary length tooltip doesn't reference removed slider."""
        text = _read("src/ui/settings/settings_registry.py")
        assert "summary length slider in the main window" not in text

    def test_summary_length_tooltip_exists(self):
        """Summary length tooltip describes word count."""
        text = _read("src/ui/settings/settings_registry.py")
        assert "Default target word count for generated summaries" in text
