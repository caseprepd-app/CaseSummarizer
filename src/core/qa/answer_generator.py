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

import re
from enum import Enum

from src.config import (
    DEBUG_MODE,
    QA_MAX_TOKENS,
    QA_TEMPERATURE,
)
from src.logging_config import debug_log

# Pre-compiled regex patterns (Session 70 optimization)
# Previously compiled on every call to _split_sentences and _extract_keywords
_RE_SOURCE_CITATIONS = re.compile(r"\[[^\]]+\]:")
_RE_ABBREVIATIONS_TITLES = re.compile(r"(Mr|Mrs|Ms|Dr|Prof|Jr|Sr)\.")
_RE_ABBREVIATIONS_CORP = re.compile(r"(Inc|Corp|Ltd|Co)\.")
_RE_NUMBERS_PERIOD = re.compile(r"(\d+)\.")
_RE_SENTENCE_END = re.compile(r"[.!?]+\s+")
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

        if DEBUG_MODE:
            debug_log(f"[AnswerGenerator] Initialized with mode: {self.mode.value}")

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
            return "No relevant information found in the documents."

        debug_log(f"[AnswerGenerator] generate() called with mode={self.mode.value}")

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

        if DEBUG_MODE:
            debug_log(f"[AnswerGenerator] Extraction keywords: {keywords}")

        # Split context into sentences
        sentences = self._split_sentences(context)

        if not sentences:
            return "No relevant information found in the documents."

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
            answer = answer[:500].rsplit(" ", 1)[0] + "..."

        return answer

    def _ollama_answer(self, question: str, context: str) -> str:
        """
        Generate answer using Ollama LLM.

        Creates a prompt with the question and context, then uses
        Ollama to synthesize a natural language answer.

        Args:
            question: The user's question
            context: Retrieved document context

        Returns:
            AI-generated answer
        """
        # Check if Ollama is connected
        if not self.ollama_manager.is_connected:
            debug_log("[AnswerGenerator] Ollama not connected, falling back to extraction")
            return self._extract_answer(question, context)

        # Build prompt
        prompt = self._build_qa_prompt(question, context)

        try:
            # Generate response
            response = self.ollama_manager.generate_text(
                prompt=prompt, max_tokens=QA_MAX_TOKENS, temperature=QA_TEMPERATURE
            )

            if response and response.strip():
                debug_log("[AnswerGenerator] Ollama returned answer successfully")
                return response.strip()
            else:
                debug_log(
                    "[AnswerGenerator] Empty response from Ollama, falling back to extraction"
                )
                return self._extract_answer(question, context)

        except Exception as e:
            debug_log(f"[AnswerGenerator] Ollama error: {e}, falling back to extraction")
            return self._extract_answer(question, context)

    def _build_qa_prompt(self, question: str, context: str) -> str:
        """
        Build a prompt for Ollama Q&A.

        Args:
            question: The user's question
            context: Retrieved document context

        Returns:
            Formatted prompt string
        """
        return f"""You are a legal document analyst. Answer the question using ONLY the information explicitly stated in the document excerpts below.

STRICT RULES:
1. Use ONLY information directly stated in the excerpts - do not infer, extrapolate, or add outside knowledge
2. If the excerpts do not contain the answer, respond: "The documents do not contain this information."
3. If you are uncertain, say so rather than guessing
4. Quote relevant phrases from the excerpts when possible
5. Keep your answer concise (1-3 sentences)

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""

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

        # Simple sentence splitting
        # Handle common abbreviations (using pre-compiled regex)
        clean_text = _RE_ABBREVIATIONS_TITLES.sub(r"\1<PERIOD>", clean_text)
        clean_text = _RE_ABBREVIATIONS_CORP.sub(r"\1<PERIOD>", clean_text)
        clean_text = _RE_NUMBERS_PERIOD.sub(r"\1<PERIOD>", clean_text)

        # Split on sentence-ending punctuation (using pre-compiled regex)
        sentences = _RE_SENTENCE_END.split(clean_text)

        # Restore periods
        sentences = [s.replace("<PERIOD>", ".").strip() for s in sentences if s.strip()]

        return sentences

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

        # LOG-023: Ensure ends with period
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
        if DEBUG_MODE:
            debug_log(f"[AnswerGenerator] Mode changed to: {self.mode.value}")
