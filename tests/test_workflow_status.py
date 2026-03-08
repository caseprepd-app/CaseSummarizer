"""
Tests for workflow status messages (Session 148).

Tests the WorkflowPhase enum, TabStatusConfig dataclass, and status message
functions that provide contextual messages for Q&A and Summary tabs.
"""

from src.ui.workflow_status import (
    TabStatusConfig,
    WorkflowPhase,
    get_qa_tab_status,
    get_summary_tab_status,
)


class TestWorkflowPhase:
    """Test the WorkflowPhase enum."""

    def test_all_phases_exist(self):
        """Verify all expected workflow phases are defined."""
        expected_phases = [
            "IDLE",
            "EXTRACTING_DOCS",
            "VOCAB_RUNNING",
            "QA_INDEXING",
            "QA_ANSWERING",
            "SUMMARY_RUNNING",
            "COMPLETE",
        ]
        actual_phases = [p.name for p in WorkflowPhase]
        assert actual_phases == expected_phases

    def test_phases_are_unique(self):
        """Verify all phases have unique values."""
        values = [p.value for p in WorkflowPhase]
        assert len(values) == len(set(values))


class TestTabStatusConfig:
    """Test the TabStatusConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TabStatusConfig()
        assert config.vocab_enabled is True
        assert config.qa_enabled is True
        assert config.summary_enabled is False

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = TabStatusConfig(
            vocab_enabled=False,
            qa_enabled=False,
            summary_enabled=True,
        )
        assert config.vocab_enabled is False
        assert config.qa_enabled is False
        assert config.summary_enabled is True


class TestQATabStatus:
    """Test Q&A tab status messages."""

    def test_qa_disabled_message(self):
        """When Q&A is disabled, show appropriate message."""
        config = TabStatusConfig(qa_enabled=False)
        msg = get_qa_tab_status(WorkflowPhase.IDLE, config)
        assert "Q&A is disabled" in msg
        assert "Enable" in msg

    def test_idle_with_vocab_enabled(self):
        """When idle with vocab enabled, mention vocab will run first."""
        config = TabStatusConfig(vocab_enabled=True, qa_enabled=True)
        msg = get_qa_tab_status(WorkflowPhase.IDLE, config)
        assert "after vocabulary extraction" in msg
        assert "Process Documents" in msg

    def test_idle_without_vocab(self):
        """When idle without vocab, don't mention vocab."""
        config = TabStatusConfig(vocab_enabled=False, qa_enabled=True)
        msg = get_qa_tab_status(WorkflowPhase.IDLE, config)
        assert "vocabulary" not in msg.lower()
        assert "Process Documents" in msg

    def test_extracting_docs_phase(self):
        """During document extraction, show extraction status."""
        config = TabStatusConfig(qa_enabled=True)
        msg = get_qa_tab_status(WorkflowPhase.EXTRACTING_DOCS, config)
        assert "Extracting" in msg
        assert "Q&A will begin" in msg

    def test_vocab_running_phase(self):
        """During vocab extraction, show vocab status."""
        config = TabStatusConfig(qa_enabled=True)
        msg = get_qa_tab_status(WorkflowPhase.VOCAB_RUNNING, config)
        assert "Vocabulary extraction in progress" in msg
        assert "Q&A indexing will begin" in msg

    def test_qa_indexing_phase(self):
        """During Q&A indexing, show indexing status."""
        config = TabStatusConfig(qa_enabled=True)
        msg = get_qa_tab_status(WorkflowPhase.QA_INDEXING, config)
        assert "building search index from your documents" in msg

    def test_qa_answering_phase(self):
        """During Q&A answering, show answering status."""
        config = TabStatusConfig(qa_enabled=True)
        msg = get_qa_tab_status(WorkflowPhase.QA_ANSWERING, config)
        assert "Answering questions" in msg
        assert "Results will appear" in msg

    def test_summary_running_phase(self):
        """During summary generation, show Q&A results available."""
        config = TabStatusConfig(qa_enabled=True)
        msg = get_qa_tab_status(WorkflowPhase.SUMMARY_RUNNING, config)
        assert "Summary generation in progress" in msg
        assert "Q&A results" in msg

    def test_complete_phase(self):
        """When complete, show follow-up message."""
        config = TabStatusConfig(qa_enabled=True)
        msg = get_qa_tab_status(WorkflowPhase.COMPLETE, config)
        assert "Processing complete" in msg
        assert "follow-up" in msg


class TestSummaryTabStatus:
    """Test Summary tab status messages."""

    def test_summary_disabled_message(self):
        """When summary is disabled, show appropriate message."""
        config = TabStatusConfig(summary_enabled=False)
        msg = get_summary_tab_status(WorkflowPhase.IDLE, config)
        assert "Summary generation is disabled" in msg
        assert "Enable" in msg

    def test_idle_with_vocab_and_qa(self):
        """When idle with both vocab and Q&A, mention both as prerequisites."""
        config = TabStatusConfig(vocab_enabled=True, qa_enabled=True, summary_enabled=True)
        msg = get_summary_tab_status(WorkflowPhase.IDLE, config)
        assert "vocabulary extraction" in msg
        assert "Q&A" in msg
        assert "Process Documents" in msg

    def test_idle_with_vocab_only(self):
        """When idle with vocab only, mention vocab as prerequisite."""
        config = TabStatusConfig(vocab_enabled=True, qa_enabled=False, summary_enabled=True)
        msg = get_summary_tab_status(WorkflowPhase.IDLE, config)
        assert "vocabulary extraction" in msg
        assert "Q&A" not in msg

    def test_idle_with_qa_only(self):
        """When idle with Q&A only, mention Q&A as prerequisite."""
        config = TabStatusConfig(vocab_enabled=False, qa_enabled=True, summary_enabled=True)
        msg = get_summary_tab_status(WorkflowPhase.IDLE, config)
        assert "Q&A" in msg

    def test_idle_no_prerequisites(self):
        """When idle with no prerequisites, simple message."""
        config = TabStatusConfig(vocab_enabled=False, qa_enabled=False, summary_enabled=True)
        msg = get_summary_tab_status(WorkflowPhase.IDLE, config)
        assert "Process Documents" in msg

    def test_extracting_docs_phase(self):
        """During document extraction, show extraction status."""
        config = TabStatusConfig(summary_enabled=True)
        msg = get_summary_tab_status(WorkflowPhase.EXTRACTING_DOCS, config)
        assert "Extracting" in msg
        assert "Summary generation will begin" in msg

    def test_vocab_running_with_qa(self):
        """During vocab with Q&A enabled, mention both."""
        config = TabStatusConfig(qa_enabled=True, summary_enabled=True)
        msg = get_summary_tab_status(WorkflowPhase.VOCAB_RUNNING, config)
        assert "Vocabulary extraction in progress" in msg
        assert "Q&A" in msg

    def test_vocab_running_without_qa(self):
        """During vocab without Q&A, don't mention Q&A."""
        config = TabStatusConfig(qa_enabled=False, summary_enabled=True)
        msg = get_summary_tab_status(WorkflowPhase.VOCAB_RUNNING, config)
        assert "Vocabulary extraction in progress" in msg
        assert "Q&A" not in msg

    def test_qa_indexing_phase(self):
        """During Q&A indexing, show Q&A status."""
        config = TabStatusConfig(summary_enabled=True)
        msg = get_summary_tab_status(WorkflowPhase.QA_INDEXING, config)
        assert "Q&A indexing in progress" in msg
        assert "Summary will run after" in msg

    def test_qa_answering_phase(self):
        """During Q&A answering, show answering status."""
        config = TabStatusConfig(summary_enabled=True)
        msg = get_summary_tab_status(WorkflowPhase.QA_ANSWERING, config)
        assert "Answering questions" in msg
        assert "Summary will run after" in msg

    def test_summary_running_phase(self):
        """During summary generation, show generating status."""
        config = TabStatusConfig(summary_enabled=True)
        msg = get_summary_tab_status(WorkflowPhase.SUMMARY_RUNNING, config)
        assert "Generating summary" in msg
        assert "several minutes" in msg

    def test_complete_phase(self):
        """When complete, show complete message."""
        config = TabStatusConfig(summary_enabled=True)
        msg = get_summary_tab_status(WorkflowPhase.COMPLETE, config)
        assert "Processing complete" in msg


class TestStatusMessageConsistency:
    """Test consistency across status messages."""

    def test_all_phases_have_qa_messages(self):
        """Every phase should return a non-empty Q&A message when enabled."""
        config = TabStatusConfig(qa_enabled=True)
        for phase in WorkflowPhase:
            msg = get_qa_tab_status(phase, config)
            assert msg, f"Empty message for Q&A phase {phase.name}"

    def test_all_phases_have_summary_messages(self):
        """Every phase should return a non-empty summary message when enabled."""
        config = TabStatusConfig(summary_enabled=True)
        for phase in WorkflowPhase:
            msg = get_summary_tab_status(phase, config)
            assert msg, f"Empty message for Summary phase {phase.name}"

    def test_disabled_messages_dont_change_with_phase(self):
        """When disabled, message should be consistent regardless of phase."""
        qa_config = TabStatusConfig(qa_enabled=False)
        summary_config = TabStatusConfig(summary_enabled=False)

        qa_messages = {get_qa_tab_status(p, qa_config) for p in WorkflowPhase}
        summary_messages = {get_summary_tab_status(p, summary_config) for p in WorkflowPhase}

        # All disabled messages should be the same
        assert len(qa_messages) == 1, "Q&A disabled message varies by phase"
        assert len(summary_messages) == 1, "Summary disabled message varies by phase"
