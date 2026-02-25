"""
Answer Generator for CasePrepd Q&A System.

Generates answers from retrieved document context using two modes:
1. Extraction mode - Fast, deterministic sentence extraction
2. Ollama mode - AI-generated synthesis using local LLM

Architecture:
- Extraction: Uses sentence segmentation + keyword matching for speed
- Ollama: Uses OllamaModelManager for quality responses

The extraction mode is ideal for quick lookups and ensures reproducibility.
The Ollama mode produces more natural, comprehensive answers but requires
Ollama to be running.
"""

import logging
import re
from enum import Enum

from src.config import (
    QA_MAX_TOKENS,
    QA_TEMPERATURE,
)
from src.core.utils.sentence_splitter import split_sentences

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns
_RE_SOURCE_CITATIONS = re.compile(r"\[[^\]]+\]:")
_RE_WORD_EXTRACT = re.compile(r"\b[a-zA-Z]+\b")
_RE_WHITESPACE = re.compile(r"\s+")


class AnswerMode(Enum):
    """Answer generation mode."""

    EXTRACTION = "extraction"
    OLLAMA = "ollama"


class AnswerGenerator:
    """
    Generates answers from retrieved context.

    Two modes:
    1. "extraction" - Find best matching sentences from context (fast, deterministic)
    2. "ollama" - Use LLM to synthesize answer (slower, more natural)

    Example:
        generator = AnswerGenerator(mode="extraction")
        answer = generator.generate(
            question="Who is the plaintiff?",
            context="[complaint.pdf]: John Smith filed this lawsuit against..."
        )
    """

    def __init__(self, mode: str = "extraction"):
        """
        Initialize answer generator.

        Args:
            mode: "extraction" or "ollama"
        """
        self.mode = AnswerMode(mode) if isinstance(mode, str) else mode
        self._ollama_manager = None

        logger.debug("Initialized with mode: %s", self.mode.value)

    @property
    def ollama_manager(self):
        """Lazy-load Ollama manager to avoid startup overhead."""
        if self._ollama_manager is None:
            from src.core.ai.ollama_model_manager import OllamaModelManager

            self._ollama_manager = OllamaModelManager()
        return self._ollama_manager

    def generate(self, question: str, context: str) -> str:
        """
        Generate an answer from the provided context.

        Args:
            question: The user's question
            context: Retrieved document context (may include source citations)

        Returns:
            Generated answer string
        """
        if not context or not context.strip():
            from src.core.qa.qa_constants import UNANSWERED_TEXT

            return UNANSWERED_TEXT

        logger.debug("generate() called with mode=%s", self.mode.value)

        if self.mode == AnswerMode.EXTRACTION:
            return self._extract_answer(question, context)
        else:
            return self._ollama_answer(question, context)

    def _extract_answer(self, question: str, context: str) -> str:
        """
        Extract the most relevant sentences from context.

        Uses keyword matching and sentence scoring to find the best answer
        directly from the source material. Fast and deterministic.

        Args:
            question: The user's question
            context: Retrieved document context

        Returns:
            Extracted answer (1-3 sentences)
        """
        # Extract question keywords (remove stopwords and short words)
        keywords = self._extract_keywords(question)

        logger.debug("Extraction keywords: %s", keywords)

        # Split context into sentences
        sentences = self._split_sentences(context)

        if not sentences:
            from src.core.qa.qa_constants import UNANSWERED_TEXT

            return UNANSWERED_TEXT

        # Score each sentence by keyword matches
        scored_sentences = []
        for sentence in sentences:
            score = self._score_sentence(sentence, keywords)
            if score > 0:
                scored_sentences.append((score, sentence))

        # Sort by score descending
        scored_sentences.sort(reverse=True)

        if not scored_sentences:
            # No keyword matches = context doesn't address the question
            return "No specific answer found in the documents for this question."

        # Return top 1-3 sentences
        max_sentences = 3
        selected = [s[1] for s in scored_sentences[:max_sentences]]

        # Join and clean
        answer = " ".join(self._clean_sentence(s) for s in selected)

        # Truncate if too long
        if len(answer) > 500:
            logger.warning(
                "Extraction answer truncated from %d to 500 chars for question: %.80s",
                len(answer),
                question,
            )
            answer = answer[:500].rsplit(" ", 1)[0] + "..."

        return answer

    def _ollama_answer(self, question: str, context: str) -> str:
        """
        Generate answer using Ollama LLM with token budget enforcement.

        Computes exact token budget for the context window, applies
        progressive sub-chunking if context exceeds the budget, then
        generates a response via Ollama.

        Args:
            question: The user's question
            context: Retrieved document context

        Returns:
            AI-generated answer
        """
        if not self.ollama_manager.is_connected:
            from src.core.qa.qa_constants import OLLAMA_UNAVAILABLE_TEXT

            logger.debug("Ollama not connected, returning unavailable message")
            return OLLAMA_UNAVAILABLE_TEXT

        from src.core.qa.token_budget import (
            compute_context_budget,
            count_tokens,
            select_best_subchunk,
        )

        context_window = self._get_context_window()
        prompt_template = self._select_prompt_template(context_window)

        # Measure fixed parts of prompt (template without placeholders)
        template_tokens = count_tokens(
            prompt_template.replace("{context}", "").replace("{question}", "")
        )
        question_tokens = count_tokens(question)
        budget = compute_context_budget(
            context_window, template_tokens, question_tokens, QA_MAX_TOKENS
        )

        # If context exceeds budget, use progressive sub-chunking
        context_tokens = count_tokens(context)
        if context_tokens > budget:
            logger.debug(
                "Context (%d tokens) exceeds budget (%d tokens), running sub-chunking",
                context_tokens,
                budget,
            )
            embeddings = self._get_embeddings()
            if embeddings:
                context = select_best_subchunk(context, question, budget, embeddings)
            else:
                from src.core.qa.token_budget import _ensure_fits

                context = _ensure_fits(context, budget)

        prompt = prompt_template.format(context=context, question=question)

        try:
            response = self.ollama_manager.generate_text(
                prompt=prompt, max_tokens=QA_MAX_TOKENS, temperature=QA_TEMPERATURE
            )

            if response and response.strip():
                logger.debug("Ollama returned answer successfully")
                return response.strip()
            else:
                logger.debug("Empty response from Ollama, falling back to extraction")
                return self._extract_answer(question, context)

        except Exception as e:
            logger.debug("Ollama error: %s, falling back to extraction", e)
            return self._extract_answer(question, context)

    def _get_context_window(self) -> int:
        """
        Get effective context window from user preferences.

        Returns:
            Context window size in tokens
        """
        try:
            from src.user_preferences import get_user_preferences

            return get_user_preferences().get_effective_context_size()
        except Exception:
            from src.config import QA_CONTEXT_WINDOW

            return QA_CONTEXT_WINDOW

    def _select_prompt_template(self, context_window: int) -> str:
        """
        Pick compact vs full prompt based on context window size.

        Args:
            context_window: Available context window in tokens

        Returns:
            Prompt template string with {context} and {question} placeholders
        """
        from src.core.qa.qa_constants import (
            COMPACT_PROMPT_THRESHOLD,
            COMPACT_QA_PROMPT,
            FULL_QA_PROMPT,
        )

        if context_window <= COMPACT_PROMPT_THRESHOLD:
            return COMPACT_QA_PROMPT
        return FULL_QA_PROMPT

    def _get_embeddings(self):
        """
        Get embeddings model for sub-chunk similarity scoring.

        Returns:
            HuggingFaceEmbeddings instance or None if unavailable
        """
        try:
            from src.core.retrieval.algorithms.faiss_semantic import get_embeddings_model

            return get_embeddings_model()
        except Exception:
            logger.debug("Could not load embeddings for sub-chunking")
            return None

    def _extract_keywords(self, text: str) -> set[str]:
        """
        Extract significant keywords from text.

        Removes stopwords and short words.

        Args:
            text: Input text

        Returns:
            Set of lowercase keywords
        """
        # Common stopwords to filter
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "also",
            "now",
            "what",
            "who",
            "which",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "them",
            "his",
            "her",
            "its",
            "our",
            "their",
            "my",
            "your",
            "and",
            "but",
            "or",
            "if",
            "because",
            "while",
            "although",
            "though",
            "unless",
        }

        # Extract words using pre-compiled regex
        words = _RE_WORD_EXTRACT.findall(text.lower())

        # Filter
        keywords = {w for w in words if len(w) >= 3 and w not in stopwords}

        return keywords

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Handles common abbreviations and edge cases.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Remove source citations for cleaner sentences (using pre-compiled regex)
        clean_text = _RE_SOURCE_CITATIONS.sub("", text)

        return split_sentences(clean_text)

    def _score_sentence(self, sentence: str, keywords: set[str]) -> int:
        """
        Score a sentence based on keyword matches.

        Args:
            sentence: Sentence to score
            keywords: Keywords to match

        Returns:
            Score (number of keyword matches)
        """
        sentence_lower = sentence.lower()
        sentence_words = set(_RE_WORD_EXTRACT.findall(sentence_lower))

        # Count keyword matches
        matches = keywords & sentence_words

        return len(matches)

    def _clean_sentence(self, sentence: str) -> str:
        """
        Clean a sentence for display.

        Args:
            sentence: Sentence to clean

        Returns:
            Cleaned sentence
        """
        # Remove excessive whitespace (using pre-compiled regex)
        cleaned = _RE_WHITESPACE.sub(" ", sentence).strip()

        # Ensure ends with period
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."

        return cleaned

    def set_mode(self, mode: str) -> None:
        """
        Change the answer generation mode.

        Args:
            mode: "extraction" or "ollama"
        """
        self.mode = AnswerMode(mode)
        logger.debug("Mode changed to: %s", self.mode.value)
