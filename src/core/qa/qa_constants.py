"""
Shared constants for Q&A module.

Avoids circular imports between qa_orchestrator.py and answer_generator.py.
"""

UNANSWERED_TEXT = "No relevant information found in the documents."
REJECTION_TEXT = "Confidence in answer too low after verification step, declining to show answer..."

# Compact prompt for small context windows (saves ~70 tokens vs full)
COMPACT_QA_PROMPT = (
    "Answer using ONLY the excerpts below. "
    "If the answer is not in the excerpts, say: "
    '"The documents do not contain this information."\n\n'
    "{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

FULL_QA_PROMPT = (
    "You are a legal document analyst. Answer the question using ONLY "
    "the information explicitly stated in the document excerpts below.\n\n"
    "STRICT RULES:\n"
    "1. Use ONLY information directly stated in the excerpts\n"
    "2. If the excerpts do not contain the answer, respond: "
    '"The documents do not contain this information."\n'
    "3. If uncertain, say so rather than guessing\n"
    "4. Quote relevant phrases when possible\n"
    "5. Keep your answer concise (1-3 sentences)\n\n"
    "DOCUMENT EXCERPTS:\n{context}\n\n"
    "QUESTION: {question}\n\n"
    "ANSWER:"
)

# Use compact prompt when context window is this small or smaller
COMPACT_PROMPT_THRESHOLD = 4096
