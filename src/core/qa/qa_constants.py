"""
Shared constants for Q&A module.

Avoids circular imports between qa_orchestrator.py and answer_generator.py.
"""

UNANSWERED_TEXT = "No relevant information found in the documents."
REJECTION_TEXT = "Confidence in answer too low after verification step, declining to show answer..."
