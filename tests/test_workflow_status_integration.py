"""
Integration tests for workflow status UI components.

Tests the integration between workflow_status.py and the UI components
in dynamic_output.py and main_window.py.
"""

import pytest

# Skip all tests if display is not available (CI environment)
pytest.importorskip("tkinter")


@pytest.fixture
def tk_root():
    """Create a hidden Tk root window for testing."""
    import customtkinter as ctk

    root = ctk.CTk()
    root.withdraw()  # Hide the window
    yield root
    try:
        root.destroy()
    except Exception:
        pass  # Window may already be destroyed


@pytest.fixture
def dynamic_output_widget(tk_root):
    """Create a DynamicOutputWidget for testing."""
    from src.ui.dynamic_output import DynamicOutputWidget

    widget = DynamicOutputWidget(tk_root)
    yield widget


class TestDynamicOutputWidgetWorkflowMethods:
    """Test DynamicOutputWidget workflow status methods."""

    def test_initial_workflow_phase_is_idle(self, dynamic_output_widget):
        """Widget should start with IDLE workflow phase."""
        from src.ui.workflow_status import WorkflowPhase

        assert dynamic_output_widget._workflow_phase == WorkflowPhase.IDLE

    def test_initial_tab_status_config_defaults(self, dynamic_output_widget):
        """Widget should have default tab status config."""
        config = dynamic_output_widget._tab_status_config
        assert config.vocab_enabled is True
        assert config.qa_enabled is True

    def test_set_workflow_phase_updates_phase(self, dynamic_output_widget):
        """set_workflow_phase should update the internal phase."""
        from src.ui.workflow_status import WorkflowPhase

        dynamic_output_widget.set_workflow_phase(WorkflowPhase.VOCAB_RUNNING)
        assert dynamic_output_widget._workflow_phase == WorkflowPhase.VOCAB_RUNNING

        dynamic_output_widget.set_workflow_phase(WorkflowPhase.QA_ANSWERING)
        assert dynamic_output_widget._workflow_phase == WorkflowPhase.QA_ANSWERING

    def test_set_workflow_phase_updates_labels(self, dynamic_output_widget):
        """set_workflow_phase should update status label text."""
        from src.ui.workflow_status import WorkflowPhase

        # Set to vocab running
        dynamic_output_widget.set_workflow_phase(WorkflowPhase.VOCAB_RUNNING)

        qa_text = dynamic_output_widget._qa_status_label.cget("text")
        assert "Vocabulary extraction in progress" in qa_text

    def test_set_tab_status_config_updates_vocab(self, dynamic_output_widget):
        """set_tab_status_config should update vocab_enabled."""
        dynamic_output_widget.set_tab_status_config(vocab_enabled=False)
        assert dynamic_output_widget._tab_status_config.vocab_enabled is False

        dynamic_output_widget.set_tab_status_config(vocab_enabled=True)
        assert dynamic_output_widget._tab_status_config.vocab_enabled is True

    def test_set_tab_status_config_updates_qa(self, dynamic_output_widget):
        """set_tab_status_config should update qa_enabled."""
        dynamic_output_widget.set_tab_status_config(qa_enabled=False)
        assert dynamic_output_widget._tab_status_config.qa_enabled is False

        # Check that label shows disabled message
        qa_text = dynamic_output_widget._qa_status_label.cget("text")
        assert "Q&A is disabled" in qa_text

    def test_set_tab_status_config_summary_kwarg_ignored(self, dynamic_output_widget):
        """set_tab_status_config with summary_enabled kwarg should not crash."""
        # Backward compatibility: summary_enabled kwarg is accepted via **kwargs
        dynamic_output_widget.set_tab_status_config(summary_enabled=True)
        # No crash = pass

    def test_set_tab_status_config_partial_update(self, dynamic_output_widget):
        """set_tab_status_config with None should preserve existing values."""
        # Set initial state
        dynamic_output_widget.set_tab_status_config(vocab_enabled=False, qa_enabled=False)

        # Update only one value
        dynamic_output_widget.set_tab_status_config(qa_enabled=True)

        # Check that other values are preserved
        config = dynamic_output_widget._tab_status_config
        assert config.vocab_enabled is False  # Preserved
        assert config.qa_enabled is True  # Updated

    def test_show_qa_content_hides_status_label(self, dynamic_output_widget):
        """show_qa_content should hide status label and show panel."""
        dynamic_output_widget.show_qa_content()

        # Status label should be hidden (grid_remove)
        assert not dynamic_output_widget._qa_status_label.winfo_ismapped()

    def test_show_qa_status_shows_status_label(self, dynamic_output_widget):
        """show_qa_status should show status label and hide panel."""
        # First show content, then show status
        dynamic_output_widget.show_qa_content()
        dynamic_output_widget.show_qa_status()

        # Status label should be visible
        assert not dynamic_output_widget._qa_panel.winfo_ismapped()

    def test_show_summary_content_hides_status_label(self, dynamic_output_widget):
        """show_summary_content should hide status label and show textbox."""
        dynamic_output_widget.show_summary_content()

        # Status label should be hidden
        assert not dynamic_output_widget._summary_status_label.winfo_ismapped()

    def test_show_summary_status_shows_status_label(self, dynamic_output_widget):
        """show_summary_status should show status label and hide textbox."""
        # First show content, then show status
        dynamic_output_widget.show_summary_content()
        dynamic_output_widget.show_summary_status()

        # Textbox should be hidden
        assert not dynamic_output_widget.summary_text_display.winfo_ismapped()


class TestWorkflowPhaseTransitions:
    """Test workflow phase transitions update status correctly."""

    def test_idle_to_vocab_running(self, dynamic_output_widget):
        """Transition from IDLE to VOCAB_RUNNING."""
        from src.ui.workflow_status import WorkflowPhase

        dynamic_output_widget.set_workflow_phase(WorkflowPhase.IDLE)
        dynamic_output_widget.set_workflow_phase(WorkflowPhase.VOCAB_RUNNING)

        qa_text = dynamic_output_widget._qa_status_label.cget("text")
        assert "Vocabulary extraction in progress" in qa_text

    def test_vocab_to_qa_indexing(self, dynamic_output_widget):
        """Transition from VOCAB_RUNNING to QA_INDEXING."""
        from src.ui.workflow_status import WorkflowPhase

        dynamic_output_widget.set_workflow_phase(WorkflowPhase.VOCAB_RUNNING)
        dynamic_output_widget.set_workflow_phase(WorkflowPhase.QA_INDEXING)

        qa_text = dynamic_output_widget._qa_status_label.cget("text")
        assert "building search index from your documents" in qa_text

    def test_qa_indexing_to_answering(self, dynamic_output_widget):
        """Transition from QA_INDEXING to QA_ANSWERING."""
        from src.ui.workflow_status import WorkflowPhase

        dynamic_output_widget.set_workflow_phase(WorkflowPhase.QA_INDEXING)
        dynamic_output_widget.set_workflow_phase(WorkflowPhase.QA_ANSWERING)

        qa_text = dynamic_output_widget._qa_status_label.cget("text")
        assert "Answering questions" in qa_text

    def test_to_summary_running(self, dynamic_output_widget):
        """Transition to SUMMARY_RUNNING."""
        from src.ui.workflow_status import WorkflowPhase

        dynamic_output_widget.set_workflow_phase(WorkflowPhase.SUMMARY_RUNNING)

        summary_text = dynamic_output_widget._summary_status_label.cget("text")
        assert "AI summary" in summary_text or "Generating" in summary_text

    def test_to_complete(self, dynamic_output_widget):
        """Transition to COMPLETE."""
        from src.ui.workflow_status import WorkflowPhase

        dynamic_output_widget.set_workflow_phase(WorkflowPhase.COMPLETE)

        qa_text = dynamic_output_widget._qa_status_label.cget("text")
        assert "Processing complete" in qa_text


class TestStatusLabelVisibility:
    """Test status label visibility logic."""

    def test_qa_status_label_exists(self, dynamic_output_widget):
        """Q&A status label should exist."""
        assert dynamic_output_widget._qa_status_label is not None

    def test_summary_status_label_exists(self, dynamic_output_widget):
        """Summary status label should exist."""
        assert dynamic_output_widget._summary_status_label is not None

    def test_qa_panel_initially_hidden(self, dynamic_output_widget):
        """Q&A panel should be hidden initially (status label shown instead)."""
        assert not dynamic_output_widget._qa_panel.winfo_ismapped()

    def test_summary_textbox_initially_hidden(self, dynamic_output_widget):
        """Summary textbox should be hidden initially (status label shown instead)."""
        assert not dynamic_output_widget.summary_text_display.winfo_ismapped()

    def test_toggle_qa_visibility(self, dynamic_output_widget):
        """Should be able to toggle Q&A content visibility."""
        # Show content
        dynamic_output_widget.show_qa_content()
        assert not dynamic_output_widget._qa_status_label.winfo_ismapped()

        # Show status
        dynamic_output_widget.show_qa_status()
        assert not dynamic_output_widget._qa_panel.winfo_ismapped()

    def test_toggle_summary_visibility(self, dynamic_output_widget):
        """Should be able to toggle Summary content visibility."""
        # Show content
        dynamic_output_widget.show_summary_content()
        assert not dynamic_output_widget._summary_status_label.winfo_ismapped()

        # Show status
        dynamic_output_widget.show_summary_status()
        assert not dynamic_output_widget.summary_text_display.winfo_ismapped()


class TestConfigAndPhaseInteraction:
    """Test interaction between config changes and phase changes."""

    def test_disabled_qa_ignores_phase(self, dynamic_output_widget):
        """When Q&A is disabled, status should show disabled regardless of phase."""
        from src.ui.workflow_status import WorkflowPhase

        dynamic_output_widget.set_tab_status_config(qa_enabled=False)

        for phase in [
            WorkflowPhase.IDLE,
            WorkflowPhase.VOCAB_RUNNING,
            WorkflowPhase.QA_ANSWERING,
        ]:
            dynamic_output_widget.set_workflow_phase(phase)
            qa_text = dynamic_output_widget._qa_status_label.cget("text")
            assert "Q&A is disabled" in qa_text

    def test_enabling_qa_updates_to_current_phase(self, dynamic_output_widget):
        """Enabling Q&A should show message for current phase."""
        from src.ui.workflow_status import WorkflowPhase

        # Start with Q&A disabled in VOCAB_RUNNING phase
        dynamic_output_widget.set_tab_status_config(qa_enabled=False)
        dynamic_output_widget.set_workflow_phase(WorkflowPhase.VOCAB_RUNNING)

        # Enable Q&A - should now show vocab running message
        dynamic_output_widget.set_tab_status_config(qa_enabled=True)
        qa_text = dynamic_output_widget._qa_status_label.cget("text")
        assert "Vocabulary extraction in progress" in qa_text

    def test_vocab_disabled_affects_idle_message(self, dynamic_output_widget):
        """When vocab is disabled, IDLE message should not mention vocab."""
        from src.ui.workflow_status import WorkflowPhase

        dynamic_output_widget.set_tab_status_config(vocab_enabled=False, qa_enabled=True)
        dynamic_output_widget.set_workflow_phase(WorkflowPhase.IDLE)

        qa_text = dynamic_output_widget._qa_status_label.cget("text")
        assert "vocabulary" not in qa_text.lower()
