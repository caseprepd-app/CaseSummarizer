"""
Unified Semantic Chunker with Token Enforcement (Session 45, Session 67)

This module provides a single chunking service that:
1. Uses semantic chunking (LangChain SemanticChunker with gradient breakpoints)
2. Enforces token limits using tiktoken for accurate counting
3. Caches chunks in memory for reuse by all downstream consumers

Session 67: Based on RAG research (2024-2025), chunk sizes are FIXED at 400-1000
tokens regardless of context window. Larger context = more chunks retrieved,
not bigger chunks. Research sources:
- Chroma: 200-400 tokens for best precision
- arXiv: 512-1024 tokens for analytical queries
- Firecrawl: 400-512 tokens as starting point
"""

import hashlib
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
    UNIFIED_CHUNK_ENCODING,
    UNIFIED_CHUNK_MAX_TOKENS,
    UNIFIED_CHUNK_MIN_TOKENS,
    UNIFIED_CHUNK_TARGET_TOKENS,
)
from src.logging_config import debug_log, debug_timing, error, info

# Fallback values if config import fails (should match config.py)
DEFAULT_MIN_TOKENS = UNIFIED_CHUNK_MIN_TOKENS  # Fallback: 400
DEFAULT_TARGET_TOKENS = UNIFIED_CHUNK_TARGET_TOKENS  # Fallback: 700
DEFAULT_MAX_TOKENS = UNIFIED_CHUNK_MAX_TOKENS  # Fallback: 1000

# tiktoken encoding from config
TIKTOKEN_ENCODING = UNIFIED_CHUNK_ENCODING  # Fallback: "cl100k_base"


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
    ):
        """
        Initialize the unified chunker.

        Args:
            min_tokens: Minimum tokens per chunk (smaller chunks get merged)
            target_tokens: Target token count for ideal chunks
            max_tokens: Maximum tokens per chunk (larger chunks get split)
            tiktoken_encoding: Encoding name for tiktoken (default: cl100k_base)
        """
        self.min_tokens = min_tokens
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens

        # Initialize tiktoken encoder
        try:
            self.encoder = tiktoken.get_encoding(tiktoken_encoding)
            debug_log(f"Initialized tiktoken with encoding: {tiktoken_encoding}")
        except Exception as e:
            error(f"Failed to initialize tiktoken: {e}")
            raise

        # Initialize semantic chunker components
        self._init_semantic_chunker()

        # Cache for processed chunks (keyed by source identifier)
        self._chunk_cache: dict[str, list[UnifiedChunk]] = {}

        # Cache for token counts to avoid repeated encoding
        self._token_count_cache: dict[str, int] = {}

    def _init_semantic_chunker(self):
        """Initialize LangChain semantic chunking components."""
        debug_log("Initializing semantic chunker components...")
        init_start = time.time()

        try:
            # Use same embedding model as existing chunking_engine.py
            self.embeddings = HuggingFaceEmbeddings(
                model_name=SEMANTIC_CHUNKER_EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
            )
            self.semantic_chunker = SemanticChunker(
                self.embeddings, breakpoint_threshold_type="gradient"
            )
            debug_timing("Semantic chunker initialization", time.time() - init_start)
        except Exception as e:
            error(f"Failed to initialize semantic chunker: {e}")
            self.embeddings = None
            self.semantic_chunker = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken with caching.

        Uses a hash-based cache key combining first 100 chars, last 100 chars,
        and total length to avoid full string hashing while still being unique.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        # Build cache key from text fingerprint (avoids hashing full text)
        text_len = len(text)
        if text_len > 200:
            # Use first 100 + last 100 chars + length as fingerprint
            cache_key = f"{hash(text[:100])}-{hash(text[-100:])}-{text_len}"
        else:
            # Short text: hash the whole thing
            cache_key = f"{hash(text)}-{text_len}"

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
        # PERF-010: Use deterministic hash instead of non-deterministic hash()
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        cache_key = f"{source_file or 'unknown'}_{text_hash}"

        # Check cache first
        if use_cache and cache_key in self._chunk_cache:
            debug_log(f"Returning cached chunks for {source_file}")
            return self._chunk_cache[cache_key]

        # Validate input
        if not text or not text.strip():
            error("Empty text provided to unified chunker")
            return []

        total_tokens = self.count_tokens(text)
        info(f"Starting unified chunking: {total_tokens} tokens, {len(text.split())} words")

        # Step 1: Semantic chunking
        if self.semantic_chunker:
            raw_chunks = self._semantic_chunk(text)
        else:
            # Fallback to paragraph-based chunking
            raw_chunks = self._paragraph_chunk(text)

        debug_log(f"Initial semantic chunking produced {len(raw_chunks)} chunks")

        # Step 2: Token enforcement (split oversized, merge undersized)
        enforced_chunks = self._enforce_token_limits(raw_chunks)
        debug_log(f"After token enforcement: {len(enforced_chunks)} chunks")

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

        info(
            f"Unified chunking complete: {len(final_chunks)} chunks, "
            f"avg {avg_tokens:.0f} tokens/chunk, {total_time:.2f}s"
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
            error(f"Semantic chunking failed: {e}, falling back to paragraph chunking")
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

        Uses regex to find sentence endings (., !, ?) followed by space or newline.
        """
        # Sentence boundary pattern
        sentence_pattern = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_pattern.split(text)

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
            error("Semantic chunker not initialized. Cannot chunk PDF.")
            return []

        source = source_file or file_path.name

        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()

            # Combine all pages into single text
            full_text = "\n\n".join(doc.page_content for doc in documents)
            debug_log(f"Loaded {len(documents)} pages from PDF: {source}")

            # Use standard text chunking
            return self.chunk_text(full_text, source_file=source)

        except Exception as e:
            error(f"Failed to chunk PDF {file_path}: {e}")
            return []

    def clear_cache(self):
        """Clear the chunk cache and token count cache."""
        self._chunk_cache.clear()
        self._token_count_cache.clear()
        debug_log("Cleared unified chunker cache")

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
) -> UnifiedChunker:
    """
    Factory function to create a UnifiedChunker instance.

    If token sizes are not provided, uses optimal fixed sizes based on RAG research.
    Session 67: Chunk sizes are FIXED at 400-1000 tokens (research-based optimal range).

    Args:
        min_tokens: Minimum tokens per chunk (auto-scaled if None)
        target_tokens: Target token count (auto-scaled if None)
        max_tokens: Maximum tokens per chunk (auto-scaled if None)

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

            debug_log(
                f"[UnifiedChunker] Auto-scaled: min={min_tokens}, "
                f"target={target_tokens}, max={max_tokens} "
                f"(context={sizes['context_window']:,})"
            )
        except Exception as e:
            # Fallback to defaults if preferences unavailable
            debug_log(f"[UnifiedChunker] Auto-scale failed ({e}), using defaults")
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
    )
