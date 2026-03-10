"""
Workflow Status Messages for Tab Displays

Provides centralized status message logic for Search and Key Sentences tabs.
These messages reflect workflow progress, avoiding duplication between tabs
and the status bar (DRY principle).

The status bar shows granular progress ("Building search index 32/118..."),
while tabs show higher-level context ("Search will run after vocabulary extraction").
"""

from dataclasses import dataclass
from enum import Enum, auto


class WorkflowPhase(Enum):
    """Current phase of the document processing workflow."""

    IDLE = auto()  # No processing - waiting for user to click Process
    EXTRACTING_DOCS = auto()  # Document extraction in progress
    VOCAB_RUNNING = auto()  # Vocabulary extraction in progress
    QA_INDEXING = auto()  # Search index being built
    QA_ANSWERING = auto()  # Search queries being answered
    COMPLETE = auto()  # All processing complete


@dataclass
class TabStatusConfig:
    """Configuration for what features are enabled."""

    vocab_enabled: bool = True
    qa_enabled: bool = True  # Always True now (semantic search always on)


def get_qa_tab_status(phase: WorkflowPhase, config: TabStatusConfig) -> str:
    """
    Get the status message to display in the Search tab.

    Args:
        phase: Current workflow phase
        config: Which features are enabled

    Returns:
        Status message string for the Search tab placeholder area
    """
    if phase == WorkflowPhase.IDLE:
        if config.vocab_enabled:
            return (
                "Semantic search will run after vocabulary extraction.\n\n"
                "Click 'Process Documents' to begin."
            )
        else:
            return (
                "Semantic search will run when processing begins.\n\n"
                "Click 'Process Documents' to begin."
            )

    if phase == WorkflowPhase.EXTRACTING_DOCS:
        return (
            "Extracting text from documents...\n\n"
            "Semantic search will begin after extraction completes."
        )

    if phase == WorkflowPhase.VOCAB_RUNNING:
        return (
            "Vocabulary extraction in progress...\n\n"
            "Search indexing will begin after vocabulary extraction completes."
        )

    if phase == WorkflowPhase.QA_INDEXING:
        return (
            "Preparing semantic search \u2014 building search index from your documents...\n\n"
            "This may take a moment for large documents."
        )

    if phase == WorkflowPhase.QA_ANSWERING:
        return "Running searches...\n\nResults will appear below as they complete."

    if phase == WorkflowPhase.COMPLETE:
        return "Processing complete.\n\nAsk follow-up questions using the input below."

    return ""


def get_summary_tab_status(phase: WorkflowPhase, config: TabStatusConfig) -> str:
    """
    Get the status message to display in the Key Sentences tab.

    Key sentences auto-generate after search indexing — no user action needed.

    Args:
        phase: Current workflow phase
        config: Which features are enabled

    Returns:
        Status message string for the Key Sentences tab placeholder area
    """
    if phase == WorkflowPhase.IDLE:
        return "Key sentences will appear here after documents are processed."

    if phase == WorkflowPhase.EXTRACTING_DOCS:
        return "Extracting text from documents..."

    if phase == WorkflowPhase.VOCAB_RUNNING:
        return (
            "Vocabulary extraction in progress...\n\n"
            "Key sentences will follow after search indexing."
        )

    if phase == WorkflowPhase.QA_INDEXING:
        return "Building search index... Key sentences will follow."

    if phase == WorkflowPhase.QA_ANSWERING:
        return "Running searches... Key sentences will appear shortly."

    if phase == WorkflowPhase.COMPLETE:
        return "Processing complete."

    return ""
