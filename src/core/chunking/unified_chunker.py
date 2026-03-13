"""
Unified Recursive Sentence Chunker with Token Enforcement

Replaced semantic chunking (Mar 2026) with recursive sentence splitting.

Research findings supporting this change:
- Vecta Feb 2026 benchmark: Recursive 512-token = 69% accuracy, Semantic = 54%
  https://www.runvecta.com/blog/we-benchmarked-7-chunking-strategies-most-advice-was-wrong
- NAACL 2025: Fixed 200-word chunks match or beat semantic chunking
  https://aclanthology.org/2025.icnlsp-1.15.pdf
- 2026 RAG Performance Paradox: Simpler strategies outperform complex AI-driven methods
  https://ragaboutit.com/the-2026-rag-performance-paradox-why-simpler-chunking-strategies-are-outperforming-complex-ai-driven-methods/
- Firecrawl 2026: Practical defaults 256-512 tokens, 10-20% overlap
  https://www.firecrawl.dev/blog/best-chunking-strategies-rag
- Cohere: For transcripts, split on speaker turns, keep one speaker's content together
  https://docs.cohere.com/page/chunking-strategies

This module provides a single chunking service that:
1. Resolves coreferences (pronouns → names) for self-contained chunks
2. Injects paragraph breaks at speaker-turn boundaries (transcript-aware)
3. Splits text at sentence boundaries using NUPunkt legal-aware splitter
4. Carries forward overlap tokens between chunks for boundary context
5. Enforces min/max token limits (merge undersized, split oversized)
6. Caches chunks in memory for reuse by all downstream consumers
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

from src.config import (
    UNIFIED_CHUNK_ENCODING,
    UNIFIED_CHUNK_MAX_TOKENS,
    UNIFIED_CHUNK_MIN_TOKENS,
    UNIFIED_CHUNK_OVERLAP_TOKENS,
    UNIFIED_CHUNK_TARGET_TOKENS,
)

logger = logging.getLogger(__name__)

# Chunk token limits from config
DEFAULT_MIN_TOKENS = UNIFIED_CHUNK_MIN_TOKENS
DEFAULT_TARGET_TOKENS = UNIFIED_CHUNK_TARGET_TOKENS
DEFAULT_MAX_TOKENS = UNIFIED_CHUNK_MAX_TOKENS
DEFAULT_OVERLAP_TOKENS = UNIFIED_CHUNK_OVERLAP_TOKENS

# tiktoken encoding from config
TIKTOKEN_ENCODING = UNIFIED_CHUNK_ENCODING


@dataclass
class UnifiedChunk:
    """
    Represents a single text chunk with token-aware metadata.

    Used by both search indexing and key excerpts systems.
    """

    chunk_num: int
    text: str
    token_count: int
    word_count: int
    char_count: int
    source_file: str | None = None
    section_name: str | None = None

    # Metadata for downstream processing
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate and recalculate counts if needed."""
        if self.word_count == 0:
            self.word_count = len(self.text.split())
        if self.char_count == 0:
            self.char_count = len(self.text)


class UnifiedChunker:
    """
    Unified recursive sentence chunker with token enforcement.

    Features:
    - Recursive sentence splitting using NUPunkt legal-aware boundaries
    - Speaker-turn boundary detection for court transcripts
    - Token-based overlap between chunks for boundary context
    - Token counting via tiktoken for accurate sizing
    - Post-processing to enforce min/max token constraints
    - Caching of chunks for reuse by multiple consumers

    Usage:
        chunker = UnifiedChunker()
        chunks = chunker.chunk_text(document_text, source_file="complaint.pdf")
    """

    def __init__(
        self,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        tiktoken_encoding: str = TIKTOKEN_ENCODING,
        apply_coreference: bool = True,
    ):
        """
        Initialize the unified chunker.

        Args:
            min_tokens: Minimum tokens per chunk (smaller chunks get merged)
            target_tokens: Target token count for ideal chunks
            max_tokens: Maximum tokens per chunk (larger chunks get split)
            overlap_tokens: Tokens to carry forward between chunks
            tiktoken_encoding: Encoding name for tiktoken (default: cl100k_base)
            apply_coreference: Whether to resolve pronouns before chunking
        """
        self.min_tokens = min_tokens
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.apply_coreference = apply_coreference

        # Initialize tiktoken encoder
        try:
            self.encoder = tiktoken.get_encoding(tiktoken_encoding)
            logger.debug("Initialized tiktoken with encoding: %s", tiktoken_encoding)
        except Exception as e:
            logger.error("Failed to initialize tiktoken: %s", e, exc_info=True)
            raise

        # Coreference resolver (lazy-loaded on first use)
        self._coref_resolver = None

        # Cache for processed chunks (keyed by source identifier)
        self._chunk_cache: dict[str, list[UnifiedChunk]] = {}

        # Cache for token counts to avoid repeated encoding
        self._token_count_cache: dict[str, int] = {}

    def _get_coref_resolver(self):
        """
        Lazy-load coreference resolver on first use.

        Returns:
            CoreferenceResolver instance, or None if unavailable
        """
        if self._coref_resolver is None:
            from src.core.preprocessing.coreference_resolver import CoreferenceResolver

            self._coref_resolver = CoreferenceResolver()
            logger.debug("Initialized coreference resolver for chunking")
        return self._coref_resolver

    def _resolve_coreferences(self, text: str) -> str:
        """
        Resolve pronouns to named antecedents in text.

        Runs coreference resolution on full document text before chunking.
        This improves retrieval by making chunks self-contained
        (e.g., "He testified..." becomes "Dr. Smith testified...").

        Args:
            text: Full document text

        Returns:
            Text with pronouns replaced by their antecedents
        """
        if not self.apply_coreference:
            return text

        resolver = self._get_coref_resolver()
        if resolver is None:
            return text

        try:
            result = resolver.process(text)
            if result.changes_made > 0:
                logger.info(
                    "Coreference resolution: %d pronouns resolved before chunking",
                    result.changes_made,
                )
            return result.text
        except Exception as e:
            logger.warning("Coreference resolution failed, using original text: %s", e)
            return text

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken with caching.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        cache_key = hash(text)

        if cache_key in self._token_count_cache:
            return self._token_count_cache[cache_key]

        count = len(self.encoder.encode(text))
        self._token_count_cache[cache_key] = count
        return count

    def chunk_text(
        self,
        text: str,
        source_file: str | None = None,
        use_cache: bool = True,
    ) -> list[UnifiedChunk]:
        """
        Chunk text using sentence boundaries with token enforcement and overlap.

        Args:
            text: Full document text to chunk
            source_file: Optional source filename for metadata
            use_cache: Whether to use/store in cache

        Returns:
            List of UnifiedChunk objects
        """
        start_time = time.time()
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        cache_key = f"{source_file or 'unknown'}_{text_hash}"

        # Check cache first
        if use_cache and cache_key in self._chunk_cache:
            logger.debug("Returning cached chunks for %s", source_file)
            return self._chunk_cache[cache_key]

        # Validate input
        if not text or not text.strip():
            logger.error("Empty text provided to unified chunker")
            return []

        # Step 0: Coreference resolution (before chunking)
        text = self._resolve_coreferences(text)

        # Step 1: Inject speaker-turn boundaries for transcripts
        from src.core.chunking.transcript_boundaries import inject_speaker_boundaries

        text = inject_speaker_boundaries(text)

        total_tokens = self.count_tokens(text)
        logger.info(
            "Starting unified chunking: %s tokens, %s words", total_tokens, len(text.split())
        )

        # Step 2: Sentence-based splitting
        raw_chunks = self._split_at_sentences(text, self.target_tokens)
        logger.debug("Initial sentence splitting produced %s chunks", len(raw_chunks))

        # Step 3: Token enforcement (split oversized, merge undersized)
        enforced_chunks = self._enforce_token_limits(raw_chunks)
        logger.debug("After token enforcement: %s chunks", len(enforced_chunks))

        # Step 4: Convert to UnifiedChunk objects
        final_chunks = []
        for i, chunk_text in enumerate(enforced_chunks):
            token_count = self.count_tokens(chunk_text)
            chunk = UnifiedChunk(
                chunk_num=i + 1,
                text=chunk_text,
                token_count=token_count,
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text),
                source_file=source_file,
                section_name=self._detect_section(chunk_text),
                metadata={
                    "source": source_file,
                    "chunk_index": i,
                    "total_chunks": len(enforced_chunks),
                },
            )
            final_chunks.append(chunk)

        # Cache results
        if use_cache:
            self._chunk_cache[cache_key] = final_chunks

        # Log statistics
        total_time = time.time() - start_time
        token_counts = [c.token_count for c in final_chunks]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

        logger.info(
            "Unified chunking complete: %s chunks, avg %.0f tokens/chunk, %.2fs",
            len(final_chunks),
            avg_tokens,
            total_time,
        )

        return final_chunks

    def _paragraph_chunk(self, text: str) -> list[str]:
        """
        Fallback paragraph-based chunking.

        Splits on double newlines (paragraph boundaries).
        """
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _enforce_token_limits(self, chunks: list[str]) -> list[str]:
        """
        Enforce min/max token limits on chunks.

        - Split chunks exceeding max_tokens at sentence boundaries
        - Merge chunks below min_tokens with neighbors

        Returns:
            List of text strings with enforced token limits
        """
        # Step 1: Split oversized chunks
        split_chunks = []
        for chunk in chunks:
            token_count = self.count_tokens(chunk)

            if token_count > self.max_tokens:
                sub_chunks = self._split_at_sentences(chunk, self.target_tokens)
                split_chunks.extend(sub_chunks)
            else:
                split_chunks.append(chunk)

        # Step 2: Merge undersized chunks
        merged_chunks = self._merge_small_chunks(split_chunks)

        return merged_chunks

    def _split_at_sentences(self, text: str, target_tokens: int) -> list[str]:
        """
        Split text at sentence boundaries, preserving original whitespace.

        Uses NUPunkt legal-aware sentence boundary detection with character
        spans so chunks are extracted as substrings of the original text,
        keeping newlines and formatting intact.
        """
        from src.core.utils.sentence_splitter import split_sentence_spans

        spans = split_sentence_spans(text)

        chunks = []
        current_spans = []  # list of (sentence, (start, end))
        current_tokens = 0

        for span_item in spans:
            sentence = span_item[0].strip()
            if not sentence:
                continue

            sentence_tokens = self.count_tokens(sentence)

            if sentence_tokens > self.max_tokens:
                if current_spans:
                    chunks.append(self._extract_chunk(text, current_spans))
                    current_spans = []
                    current_tokens = 0
                chunks.append(text[span_item[1][0] : span_item[1][1]].strip())
                continue

            if current_tokens + sentence_tokens > target_tokens and current_spans:
                chunks.append(self._extract_chunk(text, current_spans))

                if self.overlap_tokens > 0:
                    overlap, overlap_tokens = self._get_overlap_spans(current_spans)
                    current_spans = overlap + [span_item]
                    current_tokens = overlap_tokens + sentence_tokens
                else:
                    current_spans = [span_item]
                    current_tokens = sentence_tokens
            else:
                current_spans.append(span_item)
                current_tokens += sentence_tokens

        if current_spans:
            chunks.append(self._extract_chunk(text, current_spans))

        return chunks

    def _extract_chunk(self, text: str, spans: list) -> str:
        """
        Extract a chunk as a substring of the original text.

        Preserves single newlines (Q/A line breaks) but collapses
        multi-newline runs (from speaker boundary injection or page joins)
        down to a single newline for clean display.
        """
        first_start = spans[0][1][0]
        last_end = spans[-1][1][1]
        chunk = text[first_start:last_end].strip()
        return re.sub(r"\n{2,}", "\n", chunk)

    def _get_overlap_spans(self, spans: list) -> tuple[list, int]:
        """
        Get trailing spans from a chunk for overlap into the next chunk.

        Walks backward through spans until overlap_tokens is reached.

        Returns:
            Tuple of (overlap_spans, overlap_token_count)
        """
        overlap = []
        token_count = 0

        for span_item in reversed(spans):
            sent_tokens = self.count_tokens(span_item[0].strip())
            if token_count + sent_tokens > self.overlap_tokens and overlap:
                break
            overlap.insert(0, span_item)
            token_count += sent_tokens

        return overlap, token_count

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """
        Merge chunks that are below minimum token threshold.

        Merges with the following chunk when possible.
        """
        if not chunks:
            return []

        merged = []
        current_chunk = chunks[0]
        current_tokens = self.count_tokens(current_chunk)

        for next_chunk in chunks[1:]:
            next_tokens = self.count_tokens(next_chunk)

            # If current is small and combined wouldn't exceed max, merge
            if current_tokens < self.min_tokens:
                combined_tokens = current_tokens + next_tokens
                if combined_tokens <= self.max_tokens:
                    current_chunk = current_chunk + "\n\n" + next_chunk
                    current_tokens = combined_tokens
                    continue

            # Save current and move to next
            merged.append(current_chunk)
            current_chunk = next_chunk
            current_tokens = next_tokens

        # Don't forget the last chunk
        merged.append(current_chunk)

        return merged

    def _detect_section(self, text: str) -> str | None:
        """
        Detect section name from chunk text using common legal patterns.

        Returns section name if detected, None otherwise.
        """
        patterns = [
            r"^(?:FIRST|SECOND|THIRD|FOURTH|FIFTH)\s+(?:CAUSE\s+OF\s+ACTION|CLAIM)",
            r"^(?:COUNT|CLAIM)\s+(?:ONE|TWO|THREE|FOUR|FIVE|[IVX]+|\d+)",
            r"^(?:WHEREFORE|PRAYER\s+FOR\s+RELIEF)",
            r"^(?:INTRODUCTION|BACKGROUND|STATEMENT\s+OF\s+FACTS)",
            r"^(?:ALLEGATIONS|AFFIRMATIVE\s+DEFENSES)",
            r"^(?:DIRECT|CROSS|REDIRECT)\s+EXAMINATION",
            r"^Q\.\s+",
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:200], re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(0)[:50]

        return None

    def chunk_pdf(self, file_path: Path, source_file: str | None = None) -> list[UnifiedChunk]:
        """
        Load and chunk a PDF file.

        Args:
            file_path: Path to the PDF file
            source_file: Optional source filename for metadata

        Returns:
            List of UnifiedChunk objects
        """
        from langchain_community.document_loaders import PyPDFLoader

        source = source_file or file_path.name

        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()

            full_text = "\n\n".join(doc.page_content for doc in documents)
            logger.debug("Loaded %s pages from PDF: %s", len(documents), source)

            return self.chunk_text(full_text, source_file=source)

        except Exception as e:
            logger.error("Failed to chunk PDF %s: %s", file_path, e, exc_info=True)
            return []

    def clear_cache(self):
        """Clear the chunk cache and token count cache."""
        self._chunk_cache.clear()
        self._token_count_cache.clear()
        logger.debug("Cleared unified chunker cache")

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "cached_documents": len(self._chunk_cache),
            "total_cached_chunks": sum(len(chunks) for chunks in self._chunk_cache.values()),
            "cached_token_counts": len(self._token_count_cache),
        }


def create_unified_chunker(
    min_tokens: int | None = None,
    target_tokens: int | None = None,
    max_tokens: int | None = None,
    apply_coreference: bool | None = None,
) -> UnifiedChunker:
    """
    Factory function to create a UnifiedChunker instance.

    If token sizes are not provided, uses optimal fixed sizes based on RAG research.

    Args:
        min_tokens: Minimum tokens per chunk (auto-scaled if None)
        target_tokens: Target token count (auto-scaled if None)
        max_tokens: Maximum tokens per chunk (auto-scaled if None)
        apply_coreference: Whether to resolve pronouns before chunking

    Returns:
        Configured UnifiedChunker instance
    """
    # Auto-scale if no explicit values provided
    if min_tokens is None and target_tokens is None and max_tokens is None:
        try:
            from src.user_preferences import get_user_preferences

            prefs = get_user_preferences()
            sizes = prefs.get_effective_chunk_sizes()
            min_tokens = sizes["min_tokens"]
            target_tokens = sizes["target_tokens"]
            max_tokens = sizes["max_tokens"]

            logger.debug(
                "Auto-scaled: min=%s, target=%s, max=%s (context=%s)",
                min_tokens,
                target_tokens,
                max_tokens,
                sizes["context_window"],
            )
        except Exception as e:
            # Fallback to defaults if preferences unavailable
            logger.debug("Auto-scale failed (%s), using defaults", e)
            min_tokens = DEFAULT_MIN_TOKENS
            target_tokens = DEFAULT_TARGET_TOKENS
            max_tokens = DEFAULT_MAX_TOKENS
    else:
        # Use provided values with defaults for any not specified
        min_tokens = min_tokens or DEFAULT_MIN_TOKENS
        target_tokens = target_tokens or DEFAULT_TARGET_TOKENS
        max_tokens = max_tokens or DEFAULT_MAX_TOKENS

    # Read coreference preference if not explicitly provided
    if apply_coreference is None:
        try:
            from src.user_preferences import get_user_preferences

            prefs = get_user_preferences()
            apply_coreference = prefs.get("coreference_enabled", False)
        except Exception:
            apply_coreference = False

    return UnifiedChunker(
        min_tokens=min_tokens,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        apply_coreference=apply_coreference,
    )
