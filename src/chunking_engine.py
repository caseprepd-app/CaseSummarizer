"""
Intelligent Document Chunking Engine

This module implements intelligent text chunking for long documents using:
1. Paragraph-aware splitting (respects paragraph boundaries)
2. Section detection (uses regex patterns to identify document structure)
3. Adaptive batch sizing (respects min/max word counts)
4. Semantic chunking for PDF documents using LangChain.
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

import warnings

from src.config import CHUNK_OVERLAP_FRACTION
from src.core.config import load_yaml
from src.logging_config import debug_log, debug_timing, error, info


@dataclass
class Chunk:
    """Represents a single text chunk with metadata."""
    chunk_num: int
    text: str
    word_count: int
    section_name: str | None = None

    def __post_init__(self):
        """Validate chunk data."""
        if self.word_count != len(self.text.split()):
            # Recalculate to ensure consistency
            self.word_count = len(self.text.split())


class ChunkingEngine:
    """
    Intelligent document chunking engine.

    Features:
    - Paragraph-aware splitting
    - Section detection via regex patterns
    - Adaptive batch sizing (min/max constraints)
    - Semantic chunking for PDFs
    - Preserves document structure
    """

    def __init__(self, config_path: Path = None):
        """
        Initialize chunking engine.

        .. deprecated::
            ChunkingEngine is deprecated. Use UnifiedChunker from
            src.core.chunking.unified_chunker for token-aware semantic chunking.

        Args:
            config_path: Path to chunking_config.yaml. If None, uses default.
        """
        warnings.warn(
            "ChunkingEngine is deprecated. Use UnifiedChunker from "
            "src.core.chunking.unified_chunker for token-aware semantic chunking.",
            DeprecationWarning,
            stacklevel=2
        )

        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "chunking_config.yaml"

        self.config = self._load_config(config_path)
        self.patterns = self._load_patterns()
        self.compiled_patterns = self._compile_patterns()

        # Initialize LangChain components
        debug_log("Initializing LangChain components for semantic chunking...")
        init_start = time.time()
        try:
            # IMPORTANT: Explicitly set device='cpu' to avoid "meta tensor" errors
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            self.semantic_chunker = SemanticChunker(
                self.embeddings, breakpoint_threshold_type="gradient"
            )
            debug_timing("LangChain component initialization", time.time() - init_start)
        except Exception as e:
            error(f"Failed to initialize LangChain components: {e}")
            # This might happen if models need to be downloaded. The app can
            # continue with text-based chunking but PDF chunking will fail.
            self.embeddings = None
            self.semantic_chunker = None


    def _load_config(self, config_path: Path) -> dict:
        """Load configuration from YAML file."""
        return load_yaml(config_path, log_prefix="[ChunkingEngine]")

    def _load_patterns(self) -> list[str]:
        """Load regex patterns from patterns file."""
        patterns_file = Path(__file__).parent.parent / self.config['chunking']['patterns_file']

        patterns = []
        try:
            with open(patterns_file) as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        patterns.append(line)

            debug_log(f"Loaded {len(patterns)} chunking patterns from {patterns_file}")
            return patterns
        except Exception as e:
            error(f"Failed to load patterns from {patterns_file}: {e}")
            return []

    def _compile_patterns(self) -> list[re.Pattern]:
        """Compile all regex patterns."""
        compiled = []
        for pattern in self.patterns:
            try:
                compiled.append(re.compile(pattern, re.MULTILINE | re.IGNORECASE))
            except re.error as e:
                error(f"Invalid regex pattern '{pattern}': {e}")
        return compiled

    def chunk_pdf(self, file_path: Path, max_tokens: int) -> list[Chunk]:
        """
        Load and chunk a PDF using semantic chunking with a "safety split".

        Args:
            file_path: Path to the PDF file.
            max_tokens: The maximum number of tokens a chunk can have.

        Returns:
            List of Chunk objects.
        """
        start_time = time.time()
        info(f"Starting semantic PDF chunking for {file_path}...")

        if not self.semantic_chunker or not self.embeddings:
            error("Semantic chunking components are not initialized. Cannot chunk PDF.")
            return []

        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            debug_log(f"Loaded {len(documents)} pages from PDF.")

            semantic_docs = self.semantic_chunker.split_documents(documents)
            debug_log(f"Split PDF into {len(semantic_docs)} initial semantic chunks.")

            # --- Safety Split Logic ---
            final_docs = []
            # A simple rule of thumb: 1 token ~ 4 characters
            max_chars = max_tokens * 4

            secondary_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chars,
                chunk_overlap=int(max_chars * CHUNK_OVERLAP_FRACTION),
                length_function=len,
            )

            for doc in semantic_docs:
                if len(doc.page_content) > max_chars:
                    debug_log(f"Semantic chunk is too large ({len(doc.page_content)} chars > {max_chars} chars). Applying safety split.")
                    sub_texts = secondary_splitter.split_text(doc.page_content)

                    for sub_text in sub_texts:
                        final_docs.append(Document(page_content=sub_text, metadata=doc.metadata))
                    debug_log(f"  - Split oversized chunk into {len(sub_texts)} sub-chunks.")
                else:
                    final_docs.append(doc)

            if len(final_docs) > len(semantic_docs):
                debug_log(f"Safety split increased chunk count from {len(semantic_docs)} to {len(final_docs)}.")

            # Convert final Document objects to the project's Chunk dataclass
            chunks = []
            for i, doc in enumerate(final_docs):
                word_count = len(doc.page_content.split())
                chunk = Chunk(
                    chunk_num=i + 1,
                    text=doc.page_content,
                    word_count=word_count,
                    section_name=f"Semantic Chunk {i+1}"
                )
                chunks.append(chunk)

            total_time = time.time() - start_time
            debug_timing("Semantic PDF chunking", total_time)
            info(f"Created {len(chunks)} total chunks from PDF document.")
            return chunks

        except Exception as e:
            error(f"Failed to chunk PDF {file_path}: {e}")
            return []

    def _split_into_paragraphs(self, text: str) -> list[tuple[str, int]]:
        """
        Split text into paragraphs while preserving word counts.

        Returns:
            List of (paragraph_text, word_count) tuples
        """
        # Split on double newlines (standard paragraph break)
        # Also handle single newlines followed by uppercase (common in PDFs)
        paragraphs_raw = re.split(r'\n\s*\n', text)

        paragraphs = []
        for para in paragraphs_raw:
            para = para.strip()
            if para:  # Skip empty paragraphs
                word_count = len(para.split())
                paragraphs.append((para, word_count))

        debug_log(f"Split text into {len(paragraphs)} paragraphs")
        return paragraphs

    def _detect_section(self, text: str) -> str | None:
        """
        Detect section name from paragraph text using regex patterns.

        Returns:
            Section name if matched, None otherwise
        """
        for pattern in self.compiled_patterns:
            match = pattern.search(text)
            if match:
                # Extract the matched text as section name (first 50 chars)
                matched_text = match.group(0)
                section_name = matched_text[:50] if len(matched_text) > 50 else matched_text
                return section_name
        return None

    def _build_chunks(self, paragraphs: list[tuple[str, int]]) -> list[Chunk]:
        """
        Build chunks from paragraphs respecting size and boundary constraints.

        Args:
            paragraphs: List of (text, word_count) tuples

        Returns:
            List of Chunk objects
        """
        config_chunking = self.config['chunking']
        max_words = config_chunking['max_chunk_words']
        # min_words and hard_limit loaded for potential future use
        _min_words = config_chunking['min_chunk_words']
        _hard_limit = config_chunking['max_chunk_words_hard_limit']

        chunks = []
        current_chunk_text = []
        current_chunk_words = 0
        current_section = None
        chunk_num = 1

        for para_text, para_words in paragraphs:
            # Detect section if it's a new one
            detected_section = self._detect_section(para_text)

            # Check if we should start a new chunk
            should_start_new = False

            # Reason 1: Section boundary and we have content
            if detected_section and current_chunk_text:
                should_start_new = True

            # Reason 2: Size limit reached
            elif current_chunk_words + para_words > max_words and current_chunk_text:
                should_start_new = True

            if should_start_new:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk_text)
                chunk = Chunk(
                    chunk_num=chunk_num,
                    text=chunk_text,
                    word_count=current_chunk_words,
                    section_name=current_section
                )
                chunks.append(chunk)
                debug_log(f"Created chunk {chunk_num}: {current_chunk_words} words, section='{current_section}'")

                # Start new chunk
                current_chunk_text = [para_text]
                current_chunk_words = para_words
                current_section = detected_section
                chunk_num += 1
            else:
                # Add to current chunk
                current_chunk_text.append(para_text)
                current_chunk_words += para_words

                # Update section if detected in this paragraph
                if detected_section:
                    current_section = detected_section

        # Don't forget the last chunk
        if current_chunk_text:
            chunk_text = "\n\n".join(current_chunk_text)
            chunk = Chunk(
                chunk_num=chunk_num,
                text=chunk_text,
                word_count=current_chunk_words,
                section_name=current_section
            )
            chunks.append(chunk)
            debug_log(f"Created chunk {chunk_num}: {current_chunk_words} words, section='{current_section}'")

        info(f"Created {len(chunks)} total chunks from document")
        return chunks

    def chunk_text(self, text: str) -> list[Chunk]:
        """
        Split document text into intelligent chunks.

        Args:
            text: Full document text

        Returns:
            List of Chunk objects with metadata
        """
        start_time = time.time()
        debug_log("Starting ChunkingEngine.chunk_text()...")

        # Validate input
        if not text or not text.strip():
            error("Empty text provided to chunking engine")
            return []

        original_words = len(text.split())
        debug_log(f"Input document: {original_words} words")

        # Step 1: Split into paragraphs
        step_start = time.time()
        debug_log("Step 1: Splitting text into paragraphs...")
        paragraphs = self._split_into_paragraphs(text)
        step_time = time.time() - step_start
        debug_timing("Paragraph splitting", step_time)

        if not paragraphs:
            error("No paragraphs extracted from text")
            return []

        debug_log(f"Extracted {len(paragraphs)} paragraphs")

        # Step 2: Build chunks
        step_start = time.time()
        debug_log("Step 2: Building chunks from paragraphs...")
        chunks = self._build_chunks(paragraphs)
        step_time = time.time() - step_start
        debug_timing("Chunk building", step_time)

        debug_log(f"Created {len(chunks)} chunks")

        # Step 3: Validate chunks
        total_words = sum(chunk.word_count for chunk in chunks)

        if total_words != original_words:
            # This can happen due to whitespace differences; log but don't fail
            debug_log(f"Word count mismatch: original={original_words}, chunked={total_words}")

        total_time = time.time() - start_time
        debug_timing("ChunkingEngine.chunk_text() complete", total_time)

        return chunks


def create_chunking_engine(config_path: Path = None) -> ChunkingEngine:
    """Factory function to create a ChunkingEngine instance."""
    return ChunkingEngine(config_path)

