"""
Shared constants for semantic search module.

Prompt templates are used by semantic_retriever for context window budget calculation.
"""

UNANSWERED_TEXT = "No relevant information found in the documents."
REJECTION_TEXT = "Confidence in answer too low after verification step, declining to show answer..."

# Compact prompt for small context windows (saves ~70 tokens vs full)
COMPACT_SEMANTIC_PROMPT = (
    "Answer using the excerpts below. Different wording is OK "
    "(e.g., 'failed to appear' = 'didn't show up'). "
    "Only say 'not found' if the topic is entirely absent.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

FULL_SEMANTIC_PROMPT = (
    "You are a legal document analyst. Answer the question using the "
    "document excerpts below.\n\n"
    "GUIDELINES:\n"
    "1. Base your answer on information in the excerpts\n"
    "2. The answer may use different wording than the question "
    "(e.g., 'failed to appear' = 'didn't show up', 'penalty' = 'fine')\n"
    "3. If the excerpts address the topic, provide an answer\n"
    "4. Only say 'The documents do not contain this information' if the "
    "topic is entirely absent from the excerpts\n"
    "5. Quote relevant phrases when helpful\n"
    "6. Keep your answer concise (1-3 sentences)\n\n"
    "DOCUMENT EXCERPTS:\n{context}\n\n"
    "QUESTION: {question}\n\n"
    "ANSWER:"
)

# Use compact prompt when context window is this small or smaller
COMPACT_PROMPT_THRESHOLD = 4096

# Progressive follow-up display placeholders
PENDING_RETRIEVAL_TEXT = "Searching documents..."
PENDING_GENERATION_TEXT = "Generating answer..."
