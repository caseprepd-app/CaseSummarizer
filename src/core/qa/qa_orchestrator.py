"""
Q&A Orchestrator for CasePrepd.

Coordinates the Q&A process: loading questions, performing vector search,
and generating answers. Manages the list of QAResult objects for display
and export.

Architecture:
- Loads default questions from qa_questions.yaml
- Uses QARetriever for FAISS similarity search
- Uses AnswerGenerator for answer generation (extraction or Ollama)
- Tracks include_in_export flag for selective export

Integration:
- Used by QAWorker for background processing
- Results displayed in QAPanel UI widget
- Exportable to TXT with checkbox-based selection
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import (
    HALLUCINATION_VERIFICATION_ENABLED,
    RETRIEVAL_CONFIDENCE_GATE,
)
from src.core.config import load_yaml_with_fallback
from src.core.qa.qa_constants import (
    PENDING_GENERATION_TEXT,
    UNANSWERED_TEXT,
)
from src.core.vector_store.qa_retriever import QARetriever, RetrievalResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.core.qa.hallucination_verifier import VerificationResult

# Default questions YAML path (relative to this file: src/core/qa/ -> config/)
DEFAULT_QUESTIONS_PATH = Path(__file__).parent.parent.parent.parent / "config" / "qa_questions.yaml"
DEFAULT_QUESTIONS_TXT_PATH = (
    Path(__file__).parent.parent.parent.parent / "config" / "qa_default_questions.txt"
)


@dataclass
class QAResult:
    """
    Single question-answer pair with metadata.

    CSV-style output with three main columns:
    - Question: The question that was asked
    - Quick Answer: AI-synthesized answer from Ollama (concise, readable)
    - Citation: Raw text excerpts from document retrieval (source material)

    Attributes:
        question: The question that was asked
        quick_answer: Ollama-generated synthesized answer (or fallback message)
        citation: Raw retrieved text excerpts from BM25+/vector search
        include_in_export: Whether to include this Q&A in export (default: True)
        source_summary: Human-readable source citation (e.g., "complaint.pdf, page 3")
        confidence: Relevance score from vector search (0-1)
        retrieval_time_ms: Time taken for vector search
        is_followup: Whether this is a user-asked follow-up question
        is_default_question: Whether this question came from the default questions list
    """

    question: str
    quick_answer: str = ""  # AI-synthesized answer from Ollama
    citation: str = ""  # Raw retrieved text from BM25+/vector search
    include_in_export: bool = True
    source_summary: str = ""
    confidence: float = 0.0
    retrieval_time_ms: float = 0.0
    is_followup: bool = False
    is_default_question: bool = False  # Marks questions from default list
    verification: "VerificationResult | None" = (
        None  # Hallucination verification result (Session 60)
    )

    @property
    def answer(self) -> str:
        """Backward compatibility: returns quick_answer (or citation as fallback)."""
        return self.quick_answer or self.citation

    @property
    def is_answered(self) -> bool:
        """Whether this question received a meaningful answer from the documents."""
        from src.core.qa.qa_constants import UNANSWERED_TEXT

        return not (self.confidence == 0.0 and self.quick_answer == UNANSWERED_TEXT)

    @property
    def is_exportable(self) -> bool:
        """Whether this result meets quality thresholds for export.

        Both retrieval confidence AND verification reliability must meet
        their respective configurable floors. This ensures two independent
        signals confirm the answer quality.
        """
        import math

        from src.config import QA_EXPORT_CONFIDENCE_FLOOR, QA_EXPORT_VERIFICATION_FLOOR
        from src.core.qa.qa_constants import REJECTION_TEXT, UNANSWERED_TEXT

        # Filter NaN or non-finite confidence
        if not math.isfinite(self.confidence):
            return False
        # Filter unanswered
        if self.confidence == 0.0 and self.quick_answer == UNANSWERED_TEXT:
            return False
        # Filter rejected answers
        if self.quick_answer == REJECTION_TEXT:
            return False
        # Retrieval confidence must meet floor
        if self.confidence < QA_EXPORT_CONFIDENCE_FLOOR:
            return False
        # Verification reliability must meet floor (if verification ran)
        if self.verification:
            return self.verification.overall_reliability >= QA_EXPORT_VERIFICATION_FLOOR
        # No verification available — fall back to retrieval confidence only
        return True


@dataclass
class QuestionDef:
    """Question definition from YAML config."""

    id: str
    text: str
    category: str
    question_type: str  # "classification" or "extraction"


class QAOrchestrator:
    """
    Coordinates Q&A process: vector search + answer generation.

    Manages the full Q&A workflow:
    1. Load questions from YAML config
    2. For each question, perform vector similarity search
    3. Generate answer from retrieved context
    4. Track results with export flags

    Example:
        orchestrator = QAOrchestrator(vector_store_path, embeddings)
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
        Initialize Q&A orchestrator.

        Args:
            vector_store_path: Path to FAISS index directory
            embeddings: HuggingFaceEmbeddings model for query encoding
            answer_mode: "extraction" (fast, from context) or "ollama" (LLM-generated)
            questions_path: Path to questions YAML (default: config/qa_questions.yaml)
        """
        self.vector_store_path = Path(vector_store_path)
        self.embeddings = embeddings
        self.answer_mode = answer_mode
        self.questions_path = questions_path or DEFAULT_QUESTIONS_PATH

        # Initialize retriever
        self.retriever = QARetriever(self.vector_store_path, self.embeddings)

        # Initialize answer generator (lazy import to avoid circular deps)
        from src.core.qa.answer_generator import AnswerGenerator

        self.answer_generator = AnswerGenerator(mode=answer_mode)

        # Results storage
        self.results: list[QAResult] = []

        # Hallucination verifier (lazy-loaded on first use)
        self._verifier = None

        # Load questions
        self._questions: list[QuestionDef] = []
        self._load_questions()

        logger.debug("Initialized with %d questions", len(self._questions))
        logger.debug("Answer mode: %s", answer_mode)

    def _load_questions(self) -> None:
        """Load question definitions from YAML config."""
        config = load_yaml_with_fallback(
            self.questions_path, fallback={}, log_prefix="[QAOrchestrator]"
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

        Session 63c: Now uses JSON-based manager with enable/disable support.
        Falls back to legacy txt file if manager fails.

        Returns:
            List of enabled question strings
        """
        try:
            from src.core.qa.default_questions_manager import get_default_questions_manager

            manager = get_default_questions_manager()
            questions = manager.get_enabled_questions()

            total = manager.get_total_count()
            logger.debug("Loaded %d/%d enabled default questions", len(questions), total)

            return questions

        except Exception as e:
            logger.error("Error loading from manager, falling back to txt: %s", e)

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
                logger.error("Failed to load questions from txt file: %s", e)
                return []

    def get_default_questions(self) -> list[str]:
        """
        Get list of default question texts.

        Returns:
            List of question strings to ask
        """
        return [q.text for q in self._questions]

    def run_default_questions(self, progress_callback=None) -> list[QAResult]:
        """
        Run all default questions against the document.

        Args:
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of QAResult objects (also stored in self.results)
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

    def ask_followup(self, question: str) -> QAResult:
        """
        Ask a single follow-up question.

        Args:
            question: User's follow-up question

        Returns:
            QAResult (also appended to self.results)
        """
        result = self._ask_single_question(question, is_followup=True)
        self.results.append(result)
        return result

    def _ask_single_question(
        self, question: str, is_followup: bool = False, is_default: bool = False
    ) -> QAResult:
        """
        Ask a single question and generate both quick_answer and citation.

        Produces CSV-style output with:
        - citation: Raw text from BM25+/vector retrieval (always populated)
        - quick_answer: AI-synthesized answer from Ollama (or fallback)
        - verification: Hallucination verification result (if enabled)

        Args:
            question: The question to ask
            is_followup: Whether this is a user-initiated follow-up
            is_default: Whether this question is from the default questions list

        Returns:
            QAResult with quick_answer, citation, verification, and metadata
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

        verification = None

        # Check if retrieval quality is sufficient to attempt answering
        best_score = max((s.relevance_score for s in retrieval_result.sources), default=0.0)
        has_quality_context = retrieval_result.context and best_score >= RETRIEVAL_CONFIDENCE_GATE

        if has_quality_context:
            # Quick Answer: AI-synthesized from Ollama (generated first, no dependency on citation)
            # Always try Ollama mode for quick_answer, regardless of configured answer_mode
            quick_answer = self._generate_quick_answer(question, retrieval_result.context)

            # Citation: focused ~250-char excerpt via embedding similarity
            from src.config import QA_CITATION_MAX_CHARS
            from src.core.qa.citation_excerpt import extract_citation_excerpt

            citation = extract_citation_excerpt(
                context=retrieval_result.context.strip(),
                question=question,
                embeddings=self.embeddings,
                max_chars=QA_CITATION_MAX_CHARS,
            )

            # Run hallucination verification if enabled (Session 60)
            if HALLUCINATION_VERIFICATION_ENABLED and quick_answer:
                verification = self._verify_answer(quick_answer, retrieval_result.context, question)
                # If answer is rejected, replace with rejection message
                if verification and verification.answer_rejected:
                    from src.core.qa.verification_config import REJECTION_MESSAGE

                    quick_answer = REJECTION_MESSAGE

            source_summary = self.retriever.get_relevant_sources_summary(retrieval_result)
            confidence = self._calculate_confidence(retrieval_result)
        else:
            # Low retrieval scores = question likely unanswerable from these documents
            if retrieval_result.context and best_score < RETRIEVAL_CONFIDENCE_GATE:
                logger.debug(
                    "Retrieval quality gate: best_score=%.3f < gate=%s -- treating as unanswerable",
                    best_score,
                    RETRIEVAL_CONFIDENCE_GATE,
                )
            citation = "No relevant excerpts found in documents."
            quick_answer = UNANSWERED_TEXT
            source_summary = ""
            confidence = 0.0

        return QAResult(
            question=question,
            quick_answer=quick_answer,
            citation=citation,
            include_in_export=has_quality_context,  # Unanswered questions default to unchecked
            source_summary=source_summary,
            confidence=confidence,
            retrieval_time_ms=retrieval_result.retrieval_time_ms,
            is_followup=is_followup,
            is_default_question=is_default,
            verification=verification,
        )

    def retrieve_for_question(self, question: str, is_followup: bool = True) -> QAResult:
        """
        Phase 1 of split follow-up flow: retrieve context only.

        Returns a partial QAResult with citation/source/confidence populated
        but quick_answer set to a placeholder. Stashes raw retrieval context
        as _retrieval_context for phase 2.

        Args:
            question: The question to search for
            is_followup: Whether this is a user follow-up (default True)

        Returns:
            Partial QAResult with placeholder answer
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

        best_score = max((s.relevance_score for s in retrieval_result.sources), default=0.0)
        has_quality_context = retrieval_result.context and best_score >= RETRIEVAL_CONFIDENCE_GATE

        if has_quality_context:
            from src.config import QA_CITATION_MAX_CHARS
            from src.core.qa.citation_excerpt import extract_citation_excerpt

            citation = extract_citation_excerpt(
                context=retrieval_result.context.strip(),
                question=question,
                embeddings=self.embeddings,
                max_chars=QA_CITATION_MAX_CHARS,
            )
            source_summary = self.retriever.get_relevant_sources_summary(retrieval_result)
            confidence = self._calculate_confidence(retrieval_result)

            result = QAResult(
                question=question,
                quick_answer=PENDING_GENERATION_TEXT,
                citation=citation,
                include_in_export=True,
                source_summary=source_summary,
                confidence=confidence,
                retrieval_time_ms=retrieval_result.retrieval_time_ms,
                is_followup=is_followup,
            )
            # Stash raw context for phase 2 generation
            result._retrieval_context = retrieval_result.context
        else:
            if retrieval_result.context and best_score < RETRIEVAL_CONFIDENCE_GATE:
                logger.debug(
                    "Retrieval quality gate: best_score=%.3f < gate=%s",
                    best_score,
                    RETRIEVAL_CONFIDENCE_GATE,
                )
            result = QAResult(
                question=question,
                quick_answer=UNANSWERED_TEXT,
                citation="No relevant excerpts found in documents.",
                source_summary="",
                confidence=0.0,
                retrieval_time_ms=retrieval_result.retrieval_time_ms,
                is_followup=is_followup,
                include_in_export=False,
            )
            result._retrieval_context = None

        return result

    def generate_answer_for_result(self, result: QAResult) -> QAResult:
        """
        Phase 2 of split follow-up flow: generate answer for a partial result.

        Takes a QAResult from retrieve_for_question(), calls the answer
        generator and verification, then fills in quick_answer.

        Args:
            result: Partial QAResult with _retrieval_context set

        Returns:
            Updated QAResult with final quick_answer and verification
        """
        context = getattr(result, "_retrieval_context", None)
        if not context:
            logger.debug("No retrieval context on result, returning as-is")
            return result

        # Generate answer
        quick_answer = self._generate_quick_answer(result.question, context)

        # Run hallucination verification if enabled
        verification = None
        if HALLUCINATION_VERIFICATION_ENABLED and quick_answer:
            verification = self._verify_answer(quick_answer, context, result.question)
            if verification and verification.answer_rejected:
                from src.core.qa.verification_config import REJECTION_MESSAGE

                quick_answer = REJECTION_MESSAGE

        result.quick_answer = quick_answer
        result.verification = verification

        # Clean up stashed context
        if hasattr(result, "_retrieval_context"):
            del result._retrieval_context

        return result

    def _generate_quick_answer(self, question: str, context: str) -> str:
        """
        Generate a quick AI-synthesized answer using Ollama.

        Falls back to extraction if Ollama is unavailable.

        Args:
            question: The question to answer
            context: Retrieved document context

        Returns:
            Synthesized answer string
        """
        # Use the configured answer generator
        answer = self.answer_generator.generate(question, context)
        return answer

    def _verify_answer(self, answer: str, context: str, question: str):
        """
        Run hallucination verification on the generated answer.

        Uses LettuceDetect to identify potentially hallucinated spans.
        Lazy-loads the verifier on first use (~150MB model download).

        Args:
            answer: Generated answer to verify
            context: Retrieved context (source documents)
            question: Original question

        Returns:
            VerificationResult with spans and reliability score
        """
        if self._verifier is None:
            from src.core.qa.hallucination_verifier import HallucinationVerifier

            self._verifier = HallucinationVerifier()

        return self._verifier.verify(answer, context, question)

    def _calculate_confidence(self, retrieval_result: RetrievalResult) -> float:
        """
        Calculate overall confidence score from retrieval results.

        Uses average relevance score of retrieved chunks.

        Args:
            retrieval_result: Result from QARetriever

        Returns:
            Confidence score (0-1)
        """
        if not retrieval_result.sources:
            return 0.0

        avg_score = sum(s.relevance_score for s in retrieval_result.sources) / len(
            retrieval_result.sources
        )
        return round(avg_score, 2)

    def get_exportable_results(self) -> list[QAResult]:
        """
        Get results where include_in_export is True.

        Returns:
            Filtered list of QAResult objects
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

        lines = ["=" * 60, "DOCUMENT Q&A SUMMARY", "=" * 60, ""]

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
