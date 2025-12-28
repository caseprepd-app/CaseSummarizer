"""
Question Flow Manager for LocalScribe Q&A System.

Manages the branching question tree for document analysis.
Loads questions from config/qa_questions.yaml and tracks flow state.

Architecture:
- Questions are defined in YAML with branching logic
- Classification questions have fixed options that determine next question
- Extraction questions are open-ended (LLM generates answer)
- Flow state tracks answered questions and determines what to ask next

Usage:
    from src.core.vector_store.question_flow import QuestionFlowManager

    flow = QuestionFlowManager()
    question = flow.get_current_question()
    flow.record_answer(question['id'], "civil", "The answer from LLM...")
    next_question = flow.get_current_question()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import DEBUG_MODE
from src.core.config import load_yaml_with_fallback
from src.logging_config import debug_log


@dataclass
class QuestionAnswer:
    """Represents an answered question."""

    question_id: str
    question_text: str
    category: str
    answer_value: str  # For classification: option value; for extraction: full answer
    answer_text: str   # Full LLM response
    sources: list[dict] = field(default_factory=list)


@dataclass
class FlowState:
    """Tracks the current state of the question flow."""

    answered: list[QuestionAnswer] = field(default_factory=list)
    current_question_id: str | None = None
    is_complete: bool = False

    def get_answer(self, question_id: str) -> str | None:
        """Get the answer value for a specific question."""
        for qa in self.answered:
            if qa.question_id == question_id:
                return qa.answer_value
        return None


class QuestionFlowManager:
    """
    Manages the branching question flow for document Q&A.

    Loads questions from YAML configuration and tracks which questions
    have been answered to determine the next question in the flow.

    Example:
        flow = QuestionFlowManager()

        while not flow.is_complete():
            question = flow.get_current_question()
            print(f"Q: {question['text']}")

            # Get answer from LLM via QARetriever...
            answer = get_llm_answer(question['text'])

            # For classification questions, extract the option value
            if question['type'] == 'classification':
                value = classify_answer(answer, question['options'])
            else:
                value = answer

            flow.record_answer(question['id'], value, answer)

        # Get all Q&A for display
        results = flow.get_all_answers()
    """

    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "qa_questions.yaml"

    def __init__(self, config_path: Path | None = None):
        """
        Initialize the question flow manager.

        Args:
            config_path: Path to qa_questions.yaml (uses default if None)
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = self._load_config()
        self.questions = {q['id']: q for q in self.config.get('questions', [])}
        self.entry_point = self.config.get('entry_point', 'is_court_case')
        self.state = FlowState(current_question_id=self.entry_point)

        if DEBUG_MODE:
            debug_log(f"[QuestionFlow] Loaded {len(self.questions)} questions from {self.config_path}")

    def _load_config(self) -> dict:
        """Load question configuration from YAML file."""
        return load_yaml_with_fallback(
            self.config_path,
            fallback={'questions': [], 'entry_point': 'is_court_case'},
            log_prefix="[QuestionFlow]"
        )

    def get_current_question(self) -> dict | None:
        """
        Get the current question to ask.

        Returns:
            Question dict with 'id', 'text', 'type', 'options' (if classification),
            or None if flow is complete.
        """
        if self.state.is_complete or not self.state.current_question_id:
            return None

        question = self.questions.get(self.state.current_question_id)
        if not question:
            debug_log(f"[QuestionFlow] Question not found: {self.state.current_question_id}")
            self.state.is_complete = True
            return None

        return question

    def record_answer(
        self,
        question_id: str,
        answer_value: str,
        answer_text: str,
        sources: list[dict] | None = None
    ):
        """
        Record an answer and advance to the next question.

        Args:
            question_id: ID of the question being answered
            answer_value: For classification: option value; for extraction: summary
            answer_text: Full LLM response text
            sources: List of source citations from retrieval
        """
        question = self.questions.get(question_id)
        if not question:
            debug_log(f"[QuestionFlow] Cannot record answer - question not found: {question_id}")
            return

        # Record the answer
        qa = QuestionAnswer(
            question_id=question_id,
            question_text=question['text'],
            category=question.get('category', 'General'),
            answer_value=answer_value,
            answer_text=answer_text,
            sources=sources or []
        )
        self.state.answered.append(qa)

        if DEBUG_MODE:
            debug_log(f"[QuestionFlow] Recorded answer for '{question_id}': {answer_value[:50]}...")

        # Determine next question
        next_question_id = self._get_next_question_id(question, answer_value)

        if next_question_id:
            self.state.current_question_id = next_question_id
            if DEBUG_MODE:
                debug_log(f"[QuestionFlow] Next question: {next_question_id}")
        else:
            self.state.is_complete = True
            self.state.current_question_id = None
            if DEBUG_MODE:
                debug_log("[QuestionFlow] Flow complete - no more questions")

    def _get_next_question_id(self, question: dict, answer_value: str) -> str | None:
        """
        Determine the next question based on the answer.

        For classification questions, looks up the answer in options.
        For extraction questions, uses the 'next' field directly.

        Args:
            question: The question that was just answered
            answer_value: The answer value (option value or extracted text)

        Returns:
            Next question ID, or None if flow should end
        """
        # Check if this is a terminal question
        if question.get('terminal', False):
            return None

        # For classification questions, find the matching option
        if question.get('type') == 'classification' and 'options' in question:
            for option in question['options']:
                if option['value'].lower() == answer_value.lower():
                    return option.get('next')

            # If no exact match, try fuzzy matching or use first option as default
            if DEBUG_MODE:
                debug_log(f"[QuestionFlow] No exact match for '{answer_value}', using default")

            # Default to first option's next if available
            if question['options']:
                return question['options'][0].get('next')

        # For extraction questions or fallback, use direct 'next' field
        return question.get('next')

    def is_complete(self) -> bool:
        """Check if the question flow is complete."""
        return self.state.is_complete

    def get_all_answers(self) -> list[QuestionAnswer]:
        """Get all recorded answers in order."""
        return self.state.answered

    def get_answers_by_category(self) -> dict[str, list[QuestionAnswer]]:
        """
        Get answers grouped by category.

        Returns:
            Dict mapping category name to list of QuestionAnswer objects
        """
        by_category: dict[str, list[QuestionAnswer]] = {}
        for qa in self.state.answered:
            if qa.category not in by_category:
                by_category[qa.category] = []
            by_category[qa.category].append(qa)
        return by_category

    def get_progress(self) -> tuple[int, int | None]:
        """
        Get flow progress.

        Returns:
            Tuple of (answered_count, estimated_remaining)
            estimated_remaining may be None if branching makes it unpredictable
        """
        answered = len(self.state.answered)

        # Estimate remaining based on current path
        # This is approximate since branching can change the path
        if self.state.is_complete:
            return answered, 0

        # Count questions in current branch (rough estimate)
        remaining = 0
        current_id = self.state.current_question_id
        visited = set()

        while current_id and current_id not in visited:
            visited.add(current_id)
            remaining += 1
            q = self.questions.get(current_id)
            if not q or q.get('terminal'):
                break
            # Follow first option or direct next
            if q.get('type') == 'classification' and q.get('options'):
                current_id = q['options'][0].get('next')
            else:
                current_id = q.get('next')

        return answered, remaining

    def reset(self):
        """Reset the flow to start over."""
        self.state = FlowState(current_question_id=self.entry_point)
        if DEBUG_MODE:
            debug_log("[QuestionFlow] Flow reset to entry point")

    def get_category_order(self) -> list[str]:
        """Get the display order for categories."""
        return self.config.get('category_order', [])

    @staticmethod
    def classify_answer(answer_text: str, options: list[dict]) -> str:
        """
        Attempt to classify an LLM answer into one of the options.

        Uses simple keyword matching. For production, could use
        embeddings similarity or another LLM call.

        Args:
            answer_text: The LLM's response
            options: List of option dicts with 'value' and 'label'

        Returns:
            The best matching option value, or 'unclear' if no match
        """
        answer_lower = answer_text.lower()

        # Direct value matching
        for option in options:
            value = option['value'].lower()
            label = option.get('label', '').lower()

            # Check if option value or label appears in answer
            if value in answer_lower or label in answer_lower:
                return option['value']

        # Keyword-based matching for common cases
        keyword_map = {
            'yes': ['yes', 'court case', 'legal proceeding', 'lawsuit', 'litigation'],
            'no': ['no', 'not a court', 'not related'],
            'criminal': ['criminal', 'crime', 'felony', 'misdemeanor', 'prosecution', 'indictment'],
            'civil': ['civil', 'plaintiff', 'defendant', 'lawsuit', 'damages', 'complaint'],
            'administrative': ['administrative', 'agency', 'regulatory', 'board', 'commission'],
        }

        for option in options:
            keywords = keyword_map.get(option['value'].lower(), [])
            for keyword in keywords:
                if keyword in answer_lower:
                    return option['value']

        # Default to 'unclear' if available, otherwise first option
        for option in options:
            if option['value'] == 'unclear':
                return 'unclear'

        return options[0]['value'] if options else 'unclear'
