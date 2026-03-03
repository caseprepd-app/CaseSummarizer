"""
Default Questions Manager

Manages the list of default Q&A questions with enable/disable state.
Questions are stored in a JSON file with their enabled status.

Supports checkbox-based question management in Settings UI.
"""

import json
import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Default questions file location
DEFAULT_QUESTIONS_PATH = (
    Path(__file__).parent.parent.parent.parent / "config" / "default_questions.json"
)

# Legacy text file (for migration)
LEGACY_QUESTIONS_PATH = (
    Path(__file__).parent.parent.parent.parent / "config" / "qa_default_questions.txt"
)


@dataclass
class DefaultQuestion:
    """A default question with its enabled state."""

    text: str
    enabled: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DefaultQuestion":
        return cls(text=data.get("text", ""), enabled=data.get("enabled", True))


class DefaultQuestionsManager:
    """
    Manages default Q&A questions with persistence.

    Features:
    - Load/save questions with enabled/disabled state
    - Add, remove, reorder questions
    - Migrate from legacy text file format
    - Get only enabled questions for execution
    """

    def __init__(self, config_path: Path | None = None):
        """
        Initialize the manager.

        Args:
            config_path: Path to JSON config file (default: config/default_questions.json)
        """
        self.config_path = config_path or DEFAULT_QUESTIONS_PATH
        self._questions: list[DefaultQuestion] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        """Load questions from JSON file, migrating from legacy if needed."""
        if self.config_path.exists():
            self._load_from_json()
        elif LEGACY_QUESTIONS_PATH.exists():
            self._migrate_from_legacy()
        else:
            self._create_default_questions()

    def _load_from_json(self):
        """Load questions from JSON file."""
        try:
            with open(self.config_path, encoding="utf-8") as f:
                data = json.load(f)

            self._questions = [DefaultQuestion.from_dict(q) for q in data.get("questions", [])]

            logger.debug("Loaded %d questions from JSON", len(self._questions))

        except Exception as e:
            logger.error("Error loading JSON: %s", e)
            self._create_default_questions()

    def _migrate_from_legacy(self):
        """Migrate from legacy text file format."""
        try:
            questions = []
            with open(LEGACY_QUESTIONS_PATH, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        questions.append(DefaultQuestion(text=line, enabled=True))

            self._questions = questions
            self._save()  # Save in new format

            logger.debug("Migrated %d questions from legacy txt", len(questions))

        except Exception as e:
            logger.error("Error migrating from legacy: %s", e)
            self._create_default_questions()

    def _create_default_questions(self):
        """Create default set of questions."""
        self._questions = [
            DefaultQuestion("What is this case about?", True),
            DefaultQuestion("What are the main allegations?", True),
            DefaultQuestion("Who are the plaintiffs?", True),
            DefaultQuestion("Who are the defendants?", True),
            DefaultQuestion("What is the date of the incident?", True),
        ]
        self._save()

        logger.debug("Created default questions")

    def _save(self):
        """Save questions to JSON file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            data = {"questions": [q.to_dict() for q in self._questions]}

            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug("Saved %d questions", len(self._questions))

        except Exception as e:
            logger.error("Error saving: %s", e)

    # =========================================================================
    # Public API
    # =========================================================================

    def get_all_questions(self) -> list[DefaultQuestion]:
        """Get all questions (enabled and disabled)."""
        return self._questions.copy()

    def get_enabled_questions(self) -> list[str]:
        """Get only enabled question texts for execution."""
        return [q.text for q in self._questions if q.enabled]

    def get_enabled_count(self) -> int:
        """Get count of enabled questions."""
        return sum(1 for q in self._questions if q.enabled)

    def get_total_count(self) -> int:
        """Get total count of questions."""
        return len(self._questions)

    def set_enabled(self, index: int, enabled: bool):
        """
        Enable or disable a question by index.

        Args:
            index: Question index (0-based)
            enabled: True to enable, False to disable
        """
        with self._lock:
            if 0 <= index < len(self._questions):
                self._questions[index].enabled = enabled
                self._save()

    def add_question(self, text: str, enabled: bool = True) -> int:
        """
        Add a new question.

        Args:
            text: Question text
            enabled: Initial enabled state

        Returns:
            Index of the new question
        """
        text = text.strip()
        if not text:
            return -1

        with self._lock:
            self._questions.append(DefaultQuestion(text=text, enabled=enabled))
            self._save()
            return len(self._questions) - 1

    def remove_question(self, index: int) -> bool:
        """
        Remove a question by index.

        Args:
            index: Question index (0-based)

        Returns:
            True if removed, False if index invalid
        """
        with self._lock:
            if 0 <= index < len(self._questions):
                del self._questions[index]
                self._save()
                return True
            return False

    def update_question(self, index: int, text: str) -> bool:
        """
        Update question text.

        Args:
            index: Question index (0-based)
            text: New question text

        Returns:
            True if updated, False if index invalid
        """
        text = text.strip()
        if not text:
            return False

        with self._lock:
            if 0 <= index < len(self._questions):
                self._questions[index].text = text
                self._save()
                return True
            return False

    def move_question(self, from_index: int, to_index: int) -> bool:
        """
        Move a question to a new position.

        Args:
            from_index: Current index
            to_index: Target index

        Returns:
            True if moved, False if indices invalid
        """
        with self._lock:
            if not (0 <= from_index < len(self._questions)):
                return False
            if not (0 <= to_index < len(self._questions)):
                return False
            if from_index == to_index:
                return True

            question = self._questions.pop(from_index)
            self._questions.insert(to_index, question)
            self._save()
            return True

    def replace_all(self, questions_data: list[dict]) -> None:
        """
        Replace all questions in a single disk write.

        Args:
            questions_data: List of dicts with 'text' and optional 'enabled' keys
        """
        with self._lock:
            self._questions = [
                DefaultQuestion(text=q["text"], enabled=q.get("enabled", True))
                for q in questions_data
                if q.get("text", "").strip()
            ]
            self._save()

    def reload(self):
        """Reload questions from file."""
        self._load()


# Singleton instance
_manager: DefaultQuestionsManager | None = None
_manager_lock = threading.Lock()


def get_default_questions_manager() -> DefaultQuestionsManager:
    """Get the singleton DefaultQuestionsManager instance."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = DefaultQuestionsManager()
    return _manager
