"""
Tests for workflow status messages.

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

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = TabStatusConfig(
            vocab_enabled=False,
            qa_enabled=False,
        )
        assert config.vocab_enabled is False
        assert config.qa_enabled is False


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
    """Test Summary tab status messages (key sentences mode)."""

    def test_idle_shows_key_sentences_message(self):
        """When idle, show key sentences will appear message."""
        config = TabStatusConfig()
        msg = get_summary_tab_status(WorkflowPhase.IDLE, config)
        assert "Key sentences" in msg
        assert "processed" in msg

    def test_extracting_docs_phase(self):
        """During document extraction, show extraction status."""
        config = TabStatusConfig()
        msg = get_summary_tab_status(WorkflowPhase.EXTRACTING_DOCS, config)
        assert "Extracting" in msg

    def test_vocab_running_phase(self):
        """During vocab extraction, mention key sentences will follow."""
        config = TabStatusConfig()
        msg = get_summary_tab_status(WorkflowPhase.VOCAB_RUNNING, config)
        assert "Vocabulary extraction" in msg
        assert "Key sentences" in msg

    def test_qa_indexing_phase(self):
        """During Q&A indexing, show building index message."""
        config = TabStatusConfig()
        msg = get_summary_tab_status(WorkflowPhase.QA_INDEXING, config)
        assert "search index" in msg.lower() or "Building" in msg

    def test_qa_answering_phase(self):
        """During Q&A answering, mention key sentences coming soon."""
        config = TabStatusConfig()
        msg = get_summary_tab_status(WorkflowPhase.QA_ANSWERING, config)
        assert "Key sentences" in msg or "Answering" in msg

    def test_summary_running_phase(self):
        """During AI summary generation, show generating status."""
        config = TabStatusConfig()
        msg = get_summary_tab_status(WorkflowPhase.SUMMARY_RUNNING, config)
        assert "AI summary" in msg or "Generating" in msg

    def test_complete_phase(self):
        """When complete, show complete message."""
        config = TabStatusConfig()
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
        """Every phase should return a non-empty summary message."""
        config = TabStatusConfig()
        for phase in WorkflowPhase:
            msg = get_summary_tab_status(phase, config)
            assert msg, f"Empty message for Summary phase {phase.name}"

    def test_disabled_qa_messages_consistent(self):
        """When Q&A is disabled, message should be consistent regardless of phase."""
        qa_config = TabStatusConfig(qa_enabled=False)
        qa_messages = {get_qa_tab_status(p, qa_config) for p in WorkflowPhase}
        # All disabled messages should be the same
        assert len(qa_messages) == 1, "Q&A disabled message varies by phase"
