"""
Semantic Search Orchestrator for CasePrepd.

Coordinates the search process: loading questions, performing vector search,
and extracting citation excerpts. Manages the list of SemanticResult objects
for display and export.

Architecture:
- Loads default questions from semantic_questions.yaml
- Uses SemanticRetriever for FAISS similarity search
- Extracts focused citation excerpts via embedding similarity
- Tracks include_in_export flag for selective export

Note: AnswerGenerator and HallucinationVerifier were removed (Mar 2026).
quick_answer is always empty; citation contains the retrieved excerpt.

Integration:
- Used by SemanticWorker for background processing
- Results displayed in Search tab UI widget
- Exportable to TXT with checkbox-based selection
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.config import load_yaml_with_fallback
from src.core.semantic.semantic_constants import (
    UNANSWERED_TEXT,
)
from src.core.vector_store.semantic_retriever import RetrievalResult, SemanticRetriever

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass

# Default questions YAML path (relative to this file: src/core/semantic/ -> config/)
DEFAULT_QUESTIONS_PATH = (
    Path(__file__).parent.parent.parent.parent / "config" / "semantic_questions.yaml"
)
DEFAULT_QUESTIONS_TXT_PATH = (
    Path(__file__).parent.parent.parent.parent / "config" / "semantic_default_questions.txt"
)


@dataclass
class SemanticResult:
    """
    Single question-answer pair with metadata.

    CSV-style output with three main columns:
    - Question: The question that was asked
    - Quick Answer: Reserved (always empty since LLM removal, Mar 2026)
    - Citation: Raw text excerpts from document retrieval (source material)

    Attributes:
        question: The question that was asked
        quick_answer: Always empty (LLM answer generation removed Mar 2026)
        citation: Raw retrieved text excerpts from BM25+/vector search
        include_in_export: Whether to include this result in export (default: True)
        source_summary: Human-readable source citation (e.g., "complaint.pdf, page 3")
        relevance: Relevance score from vector search (0-1)
        retrieval_time_ms: Time taken for vector search
        is_followup: Whether this is a user-asked follow-up question
        is_default_question: Whether this question came from the default questions list
    """

    question: str
    quick_answer: str = ""  # Deprecated — always empty string
    citation: str = ""  # Raw retrieved text from BM25+/vector search
    include_in_export: bool = True
    source_summary: str = ""
    relevance: float = 0.0
    retrieval_time_ms: float = 0.0
    is_followup: bool = False
    is_default_question: bool = False  # Marks questions from default list

    @property
    def answer(self) -> str:
        """Backward compatibility: returns quick_answer (or citation as fallback)."""
        return self.quick_answer or self.citation

    @property
    def is_answered(self) -> bool:
        """Whether this question received a meaningful answer from the documents."""
        from src.core.semantic.semantic_constants import UNANSWERED_TEXT

        return not (self.relevance == 0.0 and self.quick_answer == UNANSWERED_TEXT)

    @property
    def is_exportable(self) -> bool:
        """Whether this result meets quality thresholds for export."""
        import math

        from src.config import SEMANTIC_EXPORT_RELEVANCE_FLOOR
        from src.core.semantic.semantic_constants import UNANSWERED_TEXT

        # Filter NaN or non-finite relevance
        if not math.isfinite(self.relevance):
            return False
        # Filter unanswered
        if self.relevance == 0.0 and self.quick_answer == UNANSWERED_TEXT:
            return False
        # Retrieval relevance must meet floor
        if self.relevance < SEMANTIC_EXPORT_RELEVANCE_FLOOR:
            return False
        return True


@dataclass
class QuestionDef:
    """Question definition from YAML config."""

    id: str
    text: str
    category: str
    question_type: str  # "classification" or "extraction"


class SemanticOrchestrator:
    """
    Coordinates semantic search process: vector search + answer generation.

    Manages the full semantic search workflow:
    1. Load questions from YAML config
    2. For each question, perform vector similarity search
    3. Generate answer from retrieved context
    4. Track results with export flags

    Example:
        orchestrator = SemanticOrchestrator(vector_store_path, embeddings)
        results = orchestrator.run_default_questions()

        # User can toggle include_in_export
        results[0].include_in_export = False

        # Ask follow-up question
        followup = orchestrator.ask_followup("What injuries were claimed?")
    """

    def __init__(
        self,
        vector_store_path: Path,
        embeddings,
        answer_mode: str = "extraction",
        questions_path: Path | None = None,
    ):
        """
        Initialize semantic search orchestrator.

        Args:
            vector_store_path: Path to FAISS index directory
            embeddings: HuggingFaceEmbeddings model for query encoding
            answer_mode: Ignored (kept for backward compat). Always uses extraction.
            questions_path: Path to questions YAML (default: config/semantic_questions.yaml)
        """
        self.vector_store_path = Path(vector_store_path)
        self.embeddings = embeddings
        self.answer_mode = "extraction"
        self.questions_path = questions_path or DEFAULT_QUESTIONS_PATH

        # Initialize retriever
        self.retriever = SemanticRetriever(self.vector_store_path, self.embeddings)

        # Results storage
        self.results: list[SemanticResult] = []

        # Load questions
        self._questions: list[QuestionDef] = []
        self._load_questions()

        logger.debug("Initialized with %d questions", len(self._questions))

    def _load_questions(self) -> None:
        """Load question definitions from YAML config."""
        config = load_yaml_with_fallback(
            self.questions_path, fallback={}, log_prefix="[SemanticOrchestrator]"
        )

        if not config or "questions" not in config:
            logger.debug("No questions found in YAML config")
            return

        for q in config["questions"]:
            self._questions.append(
                QuestionDef(
                    id=q.get("id", ""),
                    text=q.get("text", ""),
                    category=q.get("category", "General"),
                    question_type=q.get("type", "extraction"),
                )
            )

        logger.debug("Loaded %d questions", len(self._questions))

    def load_default_questions_from_txt(self) -> list[str]:
        """
        Load enabled default questions from the DefaultQuestionsManager.

        Now uses JSON-based manager with enable/disable support.
        Falls back to legacy txt file if manager fails.

        Returns:
            List of enabled question strings
        """
        try:
            from src.core.semantic.default_questions_manager import get_default_questions_manager

            manager = get_default_questions_manager()
            questions = manager.get_enabled_questions()

            total = manager.get_total_count()
            logger.debug("Loaded %d/%d enabled default questions", len(questions), total)

            return questions

        except Exception as e:
            logger.error("Error loading from manager, falling back to txt: %s", e, exc_info=True)

            # Fallback to legacy txt file
            if not DEFAULT_QUESTIONS_TXT_PATH.exists():
                return []

            questions = []
            try:
                with open(DEFAULT_QUESTIONS_TXT_PATH, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            questions.append(line)
                return questions
            except Exception as e:
                logger.error("Failed to load questions from txt file: %s", e, exc_info=True)
                return []

    def get_default_questions(self) -> list[str]:
        """
        Get list of default question texts.

        Returns:
            List of question strings to ask
        """
        return [q.text for q in self._questions]

    def run_default_questions(self, progress_callback=None) -> list[SemanticResult]:
        """
        Run all default questions against the document.

        Args:
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of SemanticResult objects (also stored in self.results)
        """
        self.results = []
        questions = self.get_default_questions()
        total = len(questions)

        for i, question in enumerate(questions):
            if progress_callback:
                progress_callback(i, total)

            result = self._ask_single_question(question, is_followup=False, is_default=True)
            self.results.append(result)

            logger.debug(
                "Q%d/%d: %s... -> %d chars", i + 1, total, question[:40], len(result.answer)
            )

        if progress_callback:
            progress_callback(total, total)

        return self.results

    def ask_followup(self, question: str) -> SemanticResult:
        """
        Ask a single follow-up question.

        Args:
            question: User's follow-up question

        Returns:
            SemanticResult (also appended to self.results)
        """
        result = self._ask_single_question(question, is_followup=True)
        self.results.append(result)
        return result

    def _ask_single_question(
        self, question: str, is_followup: bool = False, is_default: bool = False
    ) -> SemanticResult:
        """
        Ask a single question and generate both quick_answer and citation.

        Produces CSV-style output with:
        - citation: Raw text from BM25+/vector retrieval (always populated)
        - quick_answer: Extraction-based answer from retrieved chunks

        Args:
            question: The question to ask
            is_followup: Whether this is a user-initiated follow-up
            is_default: Whether this question is from the default questions list

        Returns:
            SemanticResult with quick_answer, citation, and metadata
        """
        # Retrieve relevant context (this becomes the citation)
        retrieval_result = self.retriever.retrieve_context(question)

        # Log retrieval summary
        if retrieval_result.context:
            logger.debug(
                "Retrieved %d chunks (%d chars) in %.0fms",
                retrieval_result.chunks_retrieved,
                len(retrieval_result.context),
                retrieval_result.retrieval_time_ms,
            )
        else:
            logger.debug(
                "No context retrieved for: '%s...' (chunks_retrieved=%d, time=%.0fms)",
                question[:50],
                retrieval_result.chunks_retrieved,
                retrieval_result.retrieval_time_ms,
            )

        # Check if retrieval quality is sufficient to attempt answering
        from src.config import RETRIEVAL_RELEVANCE_GATE

        best_score = max((s.relevance_score for s in retrieval_result.sources), default=0.0)
        has_quality_context = retrieval_result.context and best_score >= RETRIEVAL_RELEVANCE_GATE

        if has_quality_context:
            # Citation: focused excerpt via embedding similarity
            from src.config import SEMANTIC_CITATION_MAX_CHARS
            from src.core.semantic.citation_excerpt import extract_citation_excerpt

            citation = extract_citation_excerpt(
                context=retrieval_result.context.strip(),
                question=question,
                embeddings=self.embeddings,
                max_chars=SEMANTIC_CITATION_MAX_CHARS,
            )

            source_summary = self.retriever.get_relevant_sources_summary(retrieval_result)
            relevance = self._calculate_relevance(retrieval_result)
        else:
            # Low retrieval scores = question likely unanswerable from these documents
            if retrieval_result.context and best_score < RETRIEVAL_RELEVANCE_GATE:
                logger.debug(
                    "Retrieval quality gate: best_score=%.3f < gate=%s -- treating as unanswerable",
                    best_score,
                    RETRIEVAL_RELEVANCE_GATE,
                )
            citation = "No relevant excerpts found in documents."
            source_summary = ""
            relevance = 0.0

        return SemanticResult(
            question=question,
            quick_answer="",
            citation=citation,
            include_in_export=has_quality_context,
            source_summary=source_summary,
            relevance=relevance,
            retrieval_time_ms=retrieval_result.retrieval_time_ms,
            is_followup=is_followup,
            is_default_question=is_default,
        )

    def retrieve_for_question(self, question: str, is_followup: bool = True) -> SemanticResult:
        """
        Phase 1 of split follow-up flow: retrieve context only.

        Returns a partial SemanticResult with citation/source/relevance populated
        but quick_answer set to a placeholder. Stashes raw retrieval context
        as _retrieval_context for phase 2.

        Args:
            question: The question to search for
            is_followup: Whether this is a user follow-up (default True)

        Returns:
            Partial SemanticResult with placeholder answer
        """
        retrieval_result = self.retriever.retrieve_context(question)

        if retrieval_result.context:
            logger.debug(
                "Retrieved %d chunks (%d chars) in %.0fms",
                retrieval_result.chunks_retrieved,
                len(retrieval_result.context),
                retrieval_result.retrieval_time_ms,
            )
        else:
            logger.debug(
                "No context retrieved for: '%s...'",
                question[:50],
            )

        from src.config import RETRIEVAL_RELEVANCE_GATE

        best_score = max((s.relevance_score for s in retrieval_result.sources), default=0.0)
        has_quality_context = retrieval_result.context and best_score >= RETRIEVAL_RELEVANCE_GATE

        if has_quality_context:
            from src.config import SEMANTIC_CITATION_MAX_CHARS
            from src.core.semantic.citation_excerpt import extract_citation_excerpt

            citation = extract_citation_excerpt(
                context=retrieval_result.context.strip(),
                question=question,
                embeddings=self.embeddings,
                max_chars=SEMANTIC_CITATION_MAX_CHARS,
            )
            source_summary = self.retriever.get_relevant_sources_summary(retrieval_result)
            relevance = self._calculate_relevance(retrieval_result)

            result = SemanticResult(
                question=question,
                quick_answer="",
                citation=citation,
                include_in_export=True,
                source_summary=source_summary,
                relevance=relevance,
                retrieval_time_ms=retrieval_result.retrieval_time_ms,
                is_followup=is_followup,
            )
        else:
            if retrieval_result.context and best_score < RETRIEVAL_RELEVANCE_GATE:
                logger.debug(
                    "Retrieval quality gate: best_score=%.3f < gate=%s",
                    best_score,
                    RETRIEVAL_RELEVANCE_GATE,
                )
            result = SemanticResult(
                question=question,
                quick_answer=UNANSWERED_TEXT,
                citation="No relevant excerpts found in documents.",
                source_summary="",
                relevance=0.0,
                retrieval_time_ms=retrieval_result.retrieval_time_ms,
                is_followup=is_followup,
                include_in_export=False,
            )

        return result

    def generate_answer_for_result(self, result: SemanticResult) -> SemanticResult:
        """
        Phase 2 of split follow-up flow.

        No-op — answer generation and verification were removed.
        Kept for backward compatibility with the split follow-up flow.

        Args:
            result: SemanticResult from retrieve_for_question()

        Returns:
            The same SemanticResult unchanged
        """
        return result

    def _calculate_relevance(self, retrieval_result: RetrievalResult) -> float:
        """
        Calculate overall relevance score from retrieval results.

        Uses average relevance score of retrieved chunks.

        Args:
            retrieval_result: Result from SemanticRetriever

        Returns:
            Relevance score (0-1)
        """
        if not retrieval_result.sources:
            return 0.0

        avg_score = sum(s.relevance_score for s in retrieval_result.sources) / len(
            retrieval_result.sources
        )
        return round(avg_score, 2)

    def get_exportable_results(self) -> list[SemanticResult]:
        """
        Get results where include_in_export is True.

        Returns:
            Filtered list of SemanticResult objects
        """
        return [r for r in self.results if r.include_in_export]

    def toggle_export(self, index: int) -> bool:
        """
        Toggle include_in_export for a result by index.

        Args:
            index: Index of the result to toggle

        Returns:
            New value of include_in_export
        """
        if 0 <= index < len(self.results):
            self.results[index].include_in_export = not self.results[index].include_in_export
            return self.results[index].include_in_export
        return False

    def clear_results(self) -> None:
        """Clear all stored results."""
        self.results = []

    def export_to_text(self) -> str:
        """
        Format exportable results as plain text (legacy format).

        Returns:
            Formatted text string suitable for TXT export
        """
        exportable = self.get_exportable_results()
        if not exportable:
            return ""

        lines = ["=" * 60, "DOCUMENT SEMANTIC SEARCH RESULTS", "=" * 60, ""]

        for i, result in enumerate(exportable, 1):
            lines.append(f"Q{i}: {result.question}")
            lines.append(f"Quick Answer: {result.quick_answer}")
            lines.append(f"Citation: {result.citation}")
            if result.source_summary:
                lines.append(f"   [Source: {result.source_summary}]")
            lines.append("")

        return "\n".join(lines)

    def export_to_csv(self) -> str:
        """
        Format exportable results as CSV.

        Columns: Question, Quick Answer, Citation, Source

        Returns:
            CSV string with headers
        """
        import csv
        import io

        exportable = self.get_exportable_results()
        if not exportable:
            return ""

        output = io.StringIO()
        writer = csv.writer(output)

        # Header row
        writer.writerow(["Question", "Quick Answer", "Citation", "Source"])

        # Data rows
        for result in exportable:
            writer.writerow(
                [result.question, result.quick_answer, result.citation, result.source_summary]
            )

        return output.getvalue()
