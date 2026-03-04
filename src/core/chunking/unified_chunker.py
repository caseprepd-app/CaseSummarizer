"""
Unified Semantic Chunker with Token Enforcement

This module provides a single chunking service that:
1. Uses semantic chunking (LangChain SemanticChunker with gradient breakpoints)
2. Enforces token limits using tiktoken for accurate counting
3. Caches chunks in memory for reuse by all downstream consumers

Based on RAG research (2024-2025), chunk sizes are FIXED at 400-1000
tokens regardless of context window. Larger context = more chunks retrieved,
not bigger chunks. Research sources:
- Chroma: 200-400 tokens for best precision
- arXiv: 512-1024 tokens for analytical queries
- Firecrawl: 400-512 tokens as starting point
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    SEMANTIC_CHUNKER_EMBEDDING_MODEL,
    SEMANTIC_CHUNKER_MODEL_LOCAL_PATH,
    UNIFIED_CHUNK_ENCODING,
    UNIFIED_CHUNK_MAX_TOKENS,
    UNIFIED_CHUNK_MIN_TOKENS,
    UNIFIED_CHUNK_TARGET_TOKENS,
)

logger = logging.getLogger(__name__)

# Chunk token limits (should match config.py defaults).
# The embedding model (nomic-embed-text-v1.5) has an 8,192-token context window,
# so even the max chunk size (1000 tokens) fits comfortably within that limit.
# Chunks are sized for retrieval quality, not to fill the embedding window.
DEFAULT_MIN_TOKENS = UNIFIED_CHUNK_MIN_TOKENS  # Fallback: 400
DEFAULT_TARGET_TOKENS = UNIFIED_CHUNK_TARGET_TOKENS  # Fallback: 700
DEFAULT_MAX_TOKENS = UNIFIED_CHUNK_MAX_TOKENS  # Fallback: 1000

# tiktoken encoding from config
TIKTOKEN_ENCODING = UNIFIED_CHUNK_ENCODING  # Fallback: "cl100k_base"

# Resolve semantic chunker model: prefer bundled, fall back to HF download
if SEMANTIC_CHUNKER_MODEL_LOCAL_PATH.exists():
    _SEMANTIC_CHUNKER_MODEL = str(SEMANTIC_CHUNKER_MODEL_LOCAL_PATH)
else:
    _SEMANTIC_CHUNKER_MODEL = SEMANTIC_CHUNKER_EMBEDDING_MODEL


@dataclass
class UnifiedChunk:
    """
    Represents a single text chunk with token-aware metadata.

    Used by both LLM extraction and Q&A indexing systems.
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
    Unified semantic chunker with token enforcement.

    Features:
    - Semantic chunking using embedding-based breakpoint detection
    - Token counting via tiktoken for accurate LLM context fit
    - Post-processing to enforce min/max token constraints
    - Caching of chunks for reuse by multiple consumers

    Usage:
        chunker = UnifiedChunker()
        chunks = chunker.chunk_text(document_text, source_file="complaint.pdf")
        # Chunks are now available for both LLM extraction and Q&A indexing
    """

    def __init__(
        self,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tiktoken_encoding: str = TIKTOKEN_ENCODING,
        apply_coreference: bool = True,
    ):
        """
        Initialize the unified chunker.

        Args:
            min_tokens: Minimum tokens per chunk (smaller chunks get merged)
            target_tokens: Target token count for ideal chunks
            max_tokens: Maximum tokens per chunk (larger chunks get split)
            tiktoken_encoding: Encoding name for tiktoken (default: cl100k_base)
            apply_coreference: Whether to resolve pronouns before chunking (default: True)
        """
        self.min_tokens = min_tokens
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.apply_coreference = apply_coreference

        # Initialize tiktoken encoder
        try:
            self.encoder = tiktoken.get_encoding(tiktoken_encoding)
            logger.debug("Initialized tiktoken with encoding: %s", tiktoken_encoding)
        except Exception as e:
            logger.error("Failed to initialize tiktoken: %s", e, exc_info=True)
            raise

        # Initialize semantic chunker components
        self._init_semantic_chunker()

        # Coreference resolver (lazy-loaded on first use)
        self._coref_resolver = None

        # Cache for processed chunks (keyed by source identifier)
        self._chunk_cache: dict[str, list[UnifiedChunk]] = {}

        # Cache for token counts to avoid repeated encoding
        self._token_count_cache: dict[str, int] = {}

    def _init_semantic_chunker(self):
        """Initialize LangChain semantic chunking components."""
        logger.debug("Initializing semantic chunker components...")
        init_start = time.time()

        try:
            # Use bundled model if available, otherwise fall back to HF download
            self.embeddings = HuggingFaceEmbeddings(
                model_name=_SEMANTIC_CHUNKER_MODEL, model_kwargs={"device": "cpu"}
            )
            self.semantic_chunker = SemanticChunker(
                self.embeddings, breakpoint_threshold_type="gradient"
            )
            logger.debug(
                "%s took %.2fs", "Semantic chunker initialization", time.time() - init_start
            )
        except Exception as e:
            logger.error("Failed to initialize semantic chunker: %s", e, exc_info=True)
            self.embeddings = None
            self.semantic_chunker = None

    def _get_coref_resolver(self):
        """
        Lazy-load coreference resolver on first use.

        Returns:
            CoreferenceResolver instance, or None if unavailable
        """
        if self._coref_resolver is None:
            # Lazy import to avoid circular dependencies
            from src.core.preprocessing.coreference_resolver import CoreferenceResolver

            self._coref_resolver = CoreferenceResolver()
            logger.debug("Initialized coreference resolver for chunking")
        return self._coref_resolver

    def _resolve_coreferences(self, text: str) -> str:
        """
        Resolve pronouns to named antecedents in text.

        Runs coreference resolution on full document text before chunking.
        This improves Q&A retrieval by making chunks self-contained
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

        Uses hash(text) as cache key which accounts for the full content.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        cache_key = hash(text)

        # Check cache
        if cache_key in self._token_count_cache:
            return self._token_count_cache[cache_key]

        # Calculate and cache
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
        Chunk text using semantic boundaries with token enforcement.

        Args:
            text: Full document text to chunk
            source_file: Optional source filename for metadata
            use_cache: Whether to use/store in cache

        Returns:
            List of UnifiedChunk objects
        """
        start_time = time.time()
        # Use deterministic hash instead of non-deterministic hash()
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
        # Resolves pronouns to named entities so chunks are self-contained
        # e.g., "He testified..." becomes "Dr. Smith testified..."
        text = self._resolve_coreferences(text)

        total_tokens = self.count_tokens(text)
        logger.info(
            "Starting unified chunking: %s tokens, %s words", total_tokens, len(text.split())
        )

        # Step 1: Semantic chunking
        if self.semantic_chunker:
            raw_chunks = self._semantic_chunk(text)
        else:
            # Fallback to paragraph-based chunking
            raw_chunks = self._paragraph_chunk(text)

        logger.debug("Initial semantic chunking produced %s chunks", len(raw_chunks))

        # Step 2: Token enforcement (split oversized, merge undersized)
        enforced_chunks = self._enforce_token_limits(raw_chunks)
        logger.debug("After token enforcement: %s chunks", len(enforced_chunks))

        # Step 3: Convert to UnifiedChunk objects
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

    def _semantic_chunk(self, text: str) -> list[str]:
        """
        Apply semantic chunking to split text at meaningful boundaries.

        Returns:
            List of text strings (raw chunks before token enforcement)
        """
        try:
            # Create a Document for LangChain
            doc = Document(page_content=text)
            semantic_docs = self.semantic_chunker.split_documents([doc])
            return [d.page_content for d in semantic_docs]
        except Exception as e:
            logger.error(
                "Semantic chunking failed: %s, falling back to paragraph chunking", e, exc_info=True
            )
            return self._paragraph_chunk(text)

    def _paragraph_chunk(self, text: str) -> list[str]:
        """
        Fallback paragraph-based chunking when semantic chunking fails.

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
                # Split at sentence boundaries
                sub_chunks = self._split_at_sentences(chunk, self.target_tokens)
                split_chunks.extend(sub_chunks)
            else:
                split_chunks.append(chunk)

        # Step 2: Merge undersized chunks
        merged_chunks = self._merge_small_chunks(split_chunks)

        return merged_chunks

    def _split_at_sentences(self, text: str, target_tokens: int) -> list[str]:
        """
        Split text at sentence boundaries to reach target token count.

        Uses NUPunkt legal-aware sentence boundary detection.
        """
        from src.core.utils.sentence_splitter import split_sentences

        sentences = split_sentences(text)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = self.count_tokens(sentence)

            # If single sentence exceeds max, include it anyway (rare edge case)
            if sentence_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                chunks.append(sentence)
                continue

            # Check if adding this sentence exceeds target
            if current_tokens + sentence_tokens > target_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

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
        # Common legal document section patterns
        patterns = [
            r"^(?:FIRST|SECOND|THIRD|FOURTH|FIFTH)\s+(?:CAUSE\s+OF\s+ACTION|CLAIM)",
            r"^(?:COUNT|CLAIM)\s+(?:ONE|TWO|THREE|FOUR|FIVE|[IVX]+|\d+)",
            r"^(?:WHEREFORE|PRAYER\s+FOR\s+RELIEF)",
            r"^(?:INTRODUCTION|BACKGROUND|STATEMENT\s+OF\s+FACTS)",
            r"^(?:ALLEGATIONS|AFFIRMATIVE\s+DEFENSES)",
            r"^(?:DIRECT|CROSS|REDIRECT)\s+EXAMINATION",
            r"^Q\.\s+",  # Q&A format in depositions
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:200], re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(0)[:50]  # Limit section name length

        return None

    def chunk_pdf(self, file_path: Path, source_file: str | None = None) -> list[UnifiedChunk]:
        """
        Load and chunk a PDF file.

        Args:
            file_path: Path to the PDF file
            source_file: Optional source filename for metadata (uses file_path.name if not provided)

        Returns:
            List of UnifiedChunk objects
        """
        if not self.semantic_chunker:
            logger.error("Semantic chunker not initialized. Cannot chunk PDF.")
            return []

        source = source_file or file_path.name

        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()

            # Combine all pages into single text
            full_text = "\n\n".join(doc.page_content for doc in documents)
            logger.debug("Loaded %s pages from PDF: %s", len(documents), source)

            # Use standard text chunking
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
    apply_coreference: bool = True,
) -> UnifiedChunker:
    """
    Factory function to create a UnifiedChunker instance.

    If token sizes are not provided, uses optimal fixed sizes based on RAG research.
    Chunk sizes are FIXED at 400-1000 tokens (research-based optimal range).

    Args:
        min_tokens: Minimum tokens per chunk (auto-scaled if None)
        target_tokens: Target token count (auto-scaled if None)
        max_tokens: Maximum tokens per chunk (auto-scaled if None)
        apply_coreference: Whether to resolve pronouns before chunking (default: True)

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

    return UnifiedChunker(
        min_tokens=min_tokens,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        apply_coreference=apply_coreference,
    )
