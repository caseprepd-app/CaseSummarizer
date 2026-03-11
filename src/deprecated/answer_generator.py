"""
Answer Generator for CasePrepd Semantic Search.

Generates answers from retrieved document context using extraction mode:
keyword-based sentence selection from retrieved chunks.

Previous Ollama LLM mode moved to src/deprecated/answer_generator.py.
"""

import logging
import re

from src.core.utils.sentence_splitter import split_sentences

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns
_RE_SOURCE_CITATIONS = re.compile(r"\[[^\]]+\]:")
_RE_WORD_EXTRACT = re.compile(r"\b[a-zA-Z]+\b")
_RE_WHITESPACE = re.compile(r"\s+")


class AnswerGenerator:
    """
    Generates answers by extracting relevant sentences from context.

    Uses keyword matching and sentence scoring to find the best answer
    directly from the source material. Fast and deterministic.

    Example:
        generator = AnswerGenerator()
        answer = generator.generate(
            question="Who is the plaintiff?",
            context="[complaint.pdf]: John Smith filed this lawsuit against..."
        )
    """

    def __init__(self, mode: str = "extraction", **kwargs):
        """
        Initialize answer generator.

        Args:
            mode: Ignored (kept for backward compatibility). Always uses extraction.
            **kwargs: Ignored (absorbs legacy ollama_manager param).
        """
        logger.debug("Initialized (extraction mode)")

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

        return self._extract_answer(question, context)

    def _extract_answer(self, question: str, context: str) -> str:
        """
        Extract the most relevant sentences from context.

        Args:
            question: The user's question
            context: Retrieved document context

        Returns:
            Extracted answer (1-3 sentences)
        """
        keywords = self._extract_keywords(question)

        logger.debug("Extraction keywords: %s", keywords)

        sentences = self._split_sentences(context)

        if not sentences:
            from src.core.qa.qa_constants import UNANSWERED_TEXT

            return UNANSWERED_TEXT

        scored_sentences = []
        for sentence in sentences:
            score = self._score_sentence(sentence, keywords)
            if score > 0:
                scored_sentences.append((score, sentence))

        scored_sentences.sort(reverse=True)

        if not scored_sentences:
            return "No specific answer found in the documents for this question."

        max_sentences = 3
        selected = [s[1] for s in scored_sentences[:max_sentences]]

        answer = " ".join(self._clean_sentence(s) for s in selected)

        if len(answer) > 500:
            logger.warning(
                "Extraction answer truncated from %d to 500 chars for question: %.80s",
                len(answer),
                question,
            )
            answer = answer[:500].rsplit(" ", 1)[0] + "..."

        return answer

    def _extract_keywords(self, text: str) -> set[str]:
        """
        Extract significant keywords from text.

        Args:
            text: Input text

        Returns:
            Set of lowercase keywords
        """
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

        words = _RE_WORD_EXTRACT.findall(text.lower())
        keywords = {w for w in words if len(w) >= 3 and w not in stopwords}

        return keywords

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
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
        cleaned = _RE_WHITESPACE.sub(" ", sentence).strip()
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned

    def set_mode(self, mode: str) -> None:
        """
        No-op for backward compatibility.

        Args:
            mode: Ignored
        """
        pass
