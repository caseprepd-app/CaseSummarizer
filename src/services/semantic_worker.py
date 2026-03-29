"""
Semantic Search Worker.

Background worker for running semantic search questions against
documents using FAISS vector search and retrieval-based extraction.
"""

import logging
from pathlib import Path
from queue import Queue

from src.services.base_worker import BaseWorker
from src.services.queue_messages import QueueMessage

logger = logging.getLogger(__name__)


class SemanticWorker(BaseWorker):
    """
    Background worker for semantic search document querying.

    Runs default questions against the document using FAISS vector search
    and retrieval-based extraction.

    Signals sent to ui_queue:
    - ('semantic_progress', (current, total, question)) - Question being processed
    - ('semantic_result', SemanticResult) - Single result ready
    - ('semantic_complete', list[SemanticResult]) - All questions processed
    - ('error', str) - Error occurred

    Example:
        worker = SemanticWorker(
            vector_store_path=Path("./vector_stores/case_123"),
            embeddings=embeddings_model,
            ui_queue=ui_queue,
            answer_mode="extraction"
        )
        worker.start()
    """

    def __init__(
        self,
        vector_store_path: Path,
        embeddings,
        ui_queue: Queue,
        answer_mode: str = "extraction",
        questions: list[str] | None = None,
        use_default_questions: bool = False,
    ):
        """
        Initialize semantic search worker.

        Args:
            vector_store_path: Path to FAISS index directory
            embeddings: HuggingFaceEmbeddings model
            ui_queue: Queue for UI communication
            answer_mode: Deprecated — always overridden to "extraction".
            questions: Custom questions to ask (None = use defaults from YAML)
            use_default_questions: If True, load questions from semantic_default_questions.txt
        """
        super().__init__(ui_queue)
        self.vector_store_path = Path(vector_store_path)
        self.embeddings = embeddings
        self.answer_mode = "extraction"
        self.custom_questions = questions
        self.use_default_questions = use_default_questions
        self.results: list = []

    def execute(self):
        """Execute semantic search in background thread."""
        from src.core.semantic import SemanticOrchestrator

        logger.debug("Starting semantic search")

        # Initialize orchestrator
        orchestrator = SemanticOrchestrator(
            vector_store_path=self.vector_store_path,
            embeddings=self.embeddings,
        )

        # Determine which questions to ask and whether they are default questions
        if self.custom_questions:
            # User provided specific custom questions
            questions = self.custom_questions
            is_default = False
            logger.debug("Using %s custom questions", len(questions))
        else:
            # Use enabled questions from DefaultQuestionsManager (respects user toggles)
            questions = orchestrator.load_default_questions_from_txt()
            is_default = True
            logger.debug("Using %s enabled default questions", len(questions))

        total = len(questions)
        if total == 0:
            logger.debug("No questions to process")
            self.ui_queue.put(QueueMessage.semantic_complete([]))
            return

        logger.debug("Processing %s questions", total)

        # Process questions sequentially (extraction mode is fast enough)
        self.results = self._process_questions_sequentially(
            orchestrator, questions, is_default, total
        )

        # Send completion signal with all results
        self.ui_queue.put(QueueMessage.semantic_complete(self.results))
        logger.info("All %s questions processed successfully", total)

    def _process_questions_sequentially(
        self, orchestrator, questions: list[str], is_default: bool, total: int
    ) -> list:
        """
        Process search questions sequentially.

        Extraction mode is fast enough that parallelization is unnecessary.
        Results are streamed to UI as they complete via semantic_result messages.

        Args:
            orchestrator: SemanticOrchestrator instance
            questions: List of questions to ask
            is_default: Whether these are default questions
            total: Total number of questions for progress tracking

        Returns:
            List of SemanticResult objects in original question order
        """
        logger.debug("Processing %s question(s) sequentially", len(questions))
        results = []
        for i, question in enumerate(questions):
            self.check_cancelled()

            # Report progress
            truncated_q = question[:50] + "..." if len(question) > 50 else question
            self.ui_queue.put(QueueMessage.semantic_progress(i, total, truncated_q))

            try:
                result = orchestrator._ask_single_question(
                    question, is_followup=False, is_default=is_default
                )
                results.append(result)
                self.ui_queue.put(QueueMessage.semantic_result(result))
                logger.debug("Q%s/%s complete: %s chars", i + 1, total, len(result.answer))
            except Exception as e:
                logger.warning("Question %s/%s failed: %s", i + 1, total, e)

        return results
