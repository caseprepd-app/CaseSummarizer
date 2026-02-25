"""
Workflow Status Messages for Tab Displays

Provides centralized status message logic for Q&A and Summary tabs.
These messages reflect checkbox states and workflow progress, avoiding
duplication between tabs and the status bar (DRY principle).

The status bar shows granular progress ("Building search index 32/118..."),
while tabs show higher-level context ("Q&A will run after vocabulary extraction").
"""

from dataclasses import dataclass
from enum import Enum, auto


class WorkflowPhase(Enum):
    """Current phase of the document processing workflow."""

    IDLE = auto()  # No processing - waiting for user to click Process
    EXTRACTING_DOCS = auto()  # Document extraction in progress
    VOCAB_RUNNING = auto()  # Vocabulary extraction in progress
    QA_INDEXING = auto()  # Q&A vector store being built
    QA_ANSWERING = auto()  # Q&A questions being answered
    SUMMARY_RUNNING = auto()  # Summary generation in progress
    COMPLETE = auto()  # All processing complete


@dataclass
class TabStatusConfig:
    """Configuration for what features are enabled."""

    vocab_enabled: bool = True
    qa_enabled: bool = True
    summary_enabled: bool = False


def get_qa_tab_status(phase: WorkflowPhase, config: TabStatusConfig) -> str:
    """
    Get the status message to display in the Q&A tab.

    Args:
        phase: Current workflow phase
        config: Which features are enabled

    Returns:
        Status message string for the Q&A tab placeholder area
    """
    if not config.qa_enabled:
        return (
            "Q&A is disabled.\n\n"
            "Check 'Enable Q&A' in the options panel to ask questions about your documents."
        )

    if phase == WorkflowPhase.IDLE:
        if config.vocab_enabled:
            return (
                "Q&A will run after vocabulary extraction.\n\nClick 'Process Documents' to begin."
            )
        else:
            return "Q&A will run when processing begins.\n\nClick 'Process Documents' to begin."

    if phase == WorkflowPhase.EXTRACTING_DOCS:
        return "Extracting text from documents...\n\nQ&A will begin after extraction completes."

    if phase == WorkflowPhase.VOCAB_RUNNING:
        return (
            "Vocabulary extraction in progress...\n\n"
            "Q&A indexing will begin after vocabulary extraction completes."
        )

    if phase == WorkflowPhase.QA_INDEXING:
        return "Building Q&A search index...\n\nThis may take a moment for large documents."

    if phase == WorkflowPhase.QA_ANSWERING:
        return "Answering questions...\n\nResults will appear below as they complete."

    if phase == WorkflowPhase.SUMMARY_RUNNING:
        return "Summary generation in progress...\n\nQ&A results are shown above."

    if phase == WorkflowPhase.COMPLETE:
        # This shouldn't be shown if we have results, but just in case
        return "Processing complete.\n\nAsk follow-up questions using the input below."

    return ""


def get_summary_tab_status(phase: WorkflowPhase, config: TabStatusConfig) -> str:
    """
    Get the status message to display in the Summary tab.

    Args:
        phase: Current workflow phase
        config: Which features are enabled

    Returns:
        Status message string for the Summary tab placeholder area
    """
    if not config.summary_enabled:
        return (
            "Summary generation is disabled.\n\n"
            "Enable 'Generate Summary' in the options panel to create document summaries."
        )

    if phase == WorkflowPhase.IDLE:
        # Build message based on what will run before summary
        prereqs = []
        if config.vocab_enabled:
            prereqs.append("vocabulary extraction")
        if config.qa_enabled:
            prereqs.append("Q&A indexing")

        if prereqs:
            prereq_text = " and ".join(prereqs)
            return f"Summary will run after {prereq_text}.\n\nClick 'Process Documents' to begin."
        else:
            return "Summary will run when processing begins.\n\nClick 'Process Documents' to begin."

    if phase == WorkflowPhase.EXTRACTING_DOCS:
        return (
            "Extracting text from documents...\n\nSummary generation will begin after extraction."
        )

    if phase == WorkflowPhase.VOCAB_RUNNING:
        if config.qa_enabled:
            return (
                "Vocabulary extraction in progress...\n\n"
                "Summary will run after vocabulary and Q&A complete."
            )
        else:
            return (
                "Vocabulary extraction in progress...\n\n"
                "Summary will run after vocabulary extraction completes."
            )

    if phase == WorkflowPhase.QA_INDEXING:
        return "Q&A indexing in progress...\n\nSummary will run after Q&A completes."

    if phase == WorkflowPhase.QA_ANSWERING:
        return "Answering questions...\n\nSummary will run after Q&A completes."

    if phase == WorkflowPhase.SUMMARY_RUNNING:
        return "Generating summary...\n\nThis may take several minutes for large documents."

    if phase == WorkflowPhase.COMPLETE:
        # This shouldn't be shown if we have a summary, but just in case
        return "Processing complete."

    return ""
