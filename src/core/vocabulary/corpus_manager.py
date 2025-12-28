"""
Corpus Manager for BM25 Algorithm

Manages a corpus of previous transcripts for BM25-based vocabulary extraction.
The corpus provides a baseline of "normal" vocabulary, allowing BM25 to identify
terms that are unusually frequent in the current document.

Key responsibilities:
1. Manage corpus folder (scan, count documents)
2. Extract text from corpus documents (reuses RawTextExtractor)
3. Build and cache IDF (Inverse Document Frequency) index
4. Provide IDF lookups for BM25 scoring

The IDF index is cached to JSON and only rebuilt when the corpus folder changes.

Privacy: All processing is local - no documents or data are sent externally.
"""

import hashlib
import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import CACHE_DIR
from src.logging_config import debug_log
from src.core.utils.tokenizer import tokenize, TokenizerConfig


@dataclass
class CorpusFile:
    """Information about a file in the corpus."""
    path: Path
    name: str
    is_preprocessed: bool
    preprocessed_path: Path | None
    size_bytes: int
    modified_at: datetime | None

# Supported file extensions for corpus documents
SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.rtf'}


class CorpusManager:
    """
    Manages corpus of previous transcripts for BM25 algorithm.

    Stores documents in: %APPDATA%/LocalScribe/corpus/
    Caches IDF index in: %APPDATA%/LocalScribe/cache/bm25_idf_index.json

    Example:
        manager = CorpusManager()
        if manager.is_corpus_ready():
            idf = manager.get_idf("spondylosis")  # Returns IDF score
    """

    def __init__(self, corpus_dir: Path | None = None, cache_dir: Path | None = None):
        """
        Initialize corpus manager.

        Args:
            corpus_dir: Directory containing corpus documents.
                       Defaults to %APPDATA%/LocalScribe/corpus/
            cache_dir: Directory for caching IDF index.
                      Defaults to %APPDATA%/LocalScribe/cache/
        """
        # Import here to avoid circular imports
        from src.config import CORPUS_DIR

        self.corpus_dir = Path(corpus_dir) if corpus_dir else CORPUS_DIR
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR

        # IDF index: {term: idf_score}
        self._idf_index: dict[str, float] = {}
        self._doc_count: int = 0
        self._vocab_size: int = 0
        self._last_build_time: str | None = None

        # Cache metadata
        self._cache_file = self.cache_dir / "bm25_idf_index.json"
        self._corpus_hash: str | None = None

        # Ensure directories exist
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Try to load cached index
        self._load_cache()

    def get_document_count(self) -> int:
        """
        Count supported documents in corpus folder.

        Returns:
            Number of PDF, TXT, and RTF files in corpus folder
        """
        if not self.corpus_dir.exists():
            return 0

        # Use set to avoid double-counting on case-insensitive filesystems (Windows)
        files = set()
        for ext in SUPPORTED_EXTENSIONS:
            files.update(self.corpus_dir.glob(f"*{ext}"))
            files.update(self.corpus_dir.glob(f"*{ext.upper()}"))

        return len(files)

    def is_corpus_ready(self, min_docs: int = 5) -> bool:
        """
        Check if corpus has enough documents for BM25.

        Args:
            min_docs: Minimum number of documents required

        Returns:
            True if corpus has at least min_docs documents
        """
        return self.get_document_count() >= min_docs

    def get_idf(self, term: str) -> float:
        """
        Get IDF score for a term.

        Args:
            term: The term to look up (case-insensitive)

        Returns:
            IDF score. Returns high value (10.0) for OOV terms,
            indicating they are very rare/unusual.
        """
        # Ensure index is built
        if not self._idf_index:
            self.build_idf_index()

        lower_term = term.lower().strip()

        # OOV terms get high IDF (they're rare in corpus)
        if lower_term not in self._idf_index:
            return 10.0  # Max IDF for completely unknown terms

        return self._idf_index.get(lower_term, 10.0)

    def build_idf_index(self, force_rebuild: bool = False) -> bool:
        """
        Build IDF index from all corpus documents.

        Steps:
        1. Scan corpus folder for supported files
        2. Extract text from each document
        3. Tokenize and count document frequencies
        4. Calculate IDF scores
        5. Cache to JSON

        Args:
            force_rebuild: If True, rebuild even if cache is valid

        Returns:
            True if index was built successfully
        """
        # Check if rebuild is needed
        current_hash = self._compute_corpus_hash()

        if not force_rebuild and self._corpus_hash == current_hash and self._idf_index:
            debug_log("[BM25] Using cached IDF index (corpus unchanged)")
            return True

        debug_log("[BM25] Building IDF index from corpus...")
        start_time = time.time()

        # Get all document files
        doc_files = self._get_corpus_files()
        if not doc_files:
            debug_log("[BM25] No documents found in corpus folder")
            return False

        # Document frequency counter: {term: num_docs_containing_term}
        doc_freq: Counter = Counter()
        total_docs = 0

        # Process each document
        for doc_path in doc_files:
            try:
                text = self._extract_text(doc_path)
                if not text:
                    continue

                # Get unique terms in this document
                terms = set(self._tokenize(text))

                # Increment document frequency for each term
                for term in terms:
                    doc_freq[term] += 1

                total_docs += 1

            except Exception as e:
                debug_log(f"[BM25] Error processing {doc_path.name}: {e}")
                continue

        if total_docs == 0:
            debug_log("[BM25] No documents could be processed")
            return False

        # Calculate IDF for each term
        # BM25 IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
        self._idf_index = {}
        for term, df in doc_freq.items():
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
            self._idf_index[term] = round(idf, 4)

        self._doc_count = total_docs
        self._vocab_size = len(self._idf_index)
        self._corpus_hash = current_hash
        self._last_build_time = datetime.now().isoformat()

        elapsed = time.time() - start_time
        debug_log(
            f"[BM25] Built IDF index: {self._vocab_size} terms from "
            f"{total_docs} documents in {elapsed:.2f}s"
        )

        # Save to cache
        self._save_cache()

        return True

    def get_corpus_stats(self) -> dict[str, Any]:
        """
        Get statistics about the corpus for UI display.

        Returns:
            Dictionary with doc_count, vocab_size, last_updated, corpus_path
        """
        return {
            "doc_count": self.get_document_count(),
            "vocab_size": self._vocab_size,
            "last_updated": self._last_build_time,
            "corpus_path": str(self.corpus_dir),
            "is_ready": self.is_corpus_ready(),
        }

    def _get_corpus_files(self) -> list[Path]:
        """Get list of supported document files in corpus folder."""
        # Use set to avoid duplicates on case-insensitive filesystems (Windows)
        files = set()
        for ext in SUPPORTED_EXTENSIONS:
            files.update(self.corpus_dir.glob(f"*{ext}"))
            files.update(self.corpus_dir.glob(f"*{ext.upper()}"))
        return sorted(files)

    def _extract_text(self, file_path: Path) -> str:
        """
        Extract text from a document file.

        Uses RawTextExtractor for consistent extraction across formats.

        Args:
            file_path: Path to the document

        Returns:
            Extracted text content
        """
        try:
            from src.core.extraction import RawTextExtractor

            extractor = RawTextExtractor()
            result = extractor.extract(str(file_path))

            if result.get("success"):
                return result.get("text", "")
            else:
                debug_log(f"[BM25] Extraction failed for {file_path.name}: {result.get('error')}")
                return ""

        except Exception as e:
            debug_log(f"[BM25] Error extracting {file_path.name}: {e}")
            return ""

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into lowercase words.

        Uses shared tokenizer for consistent processing with BM25Algorithm.

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase word tokens
        """
        return tokenize(text)

    def _compute_corpus_hash(self) -> str:
        """
        Compute hash of corpus folder contents for cache invalidation.

        Uses file names and modification times to detect changes.

        Returns:
            MD5 hash string representing corpus state
        """
        files = self._get_corpus_files()
        if not files:
            return "empty"

        # Build string of filenames and modification times
        content_parts = []
        for f in sorted(files):
            mtime = f.stat().st_mtime
            content_parts.append(f"{f.name}:{mtime}")

        content_str = "|".join(content_parts)
        return hashlib.md5(content_str.encode()).hexdigest()

    def _save_cache(self) -> bool:
        """
        Save IDF index to cache file.

        Returns:
            True if save succeeded
        """
        try:
            cache_data = {
                "version": 1,
                "corpus_hash": self._corpus_hash,
                "doc_count": self._doc_count,
                "vocab_size": self._vocab_size,
                "last_build_time": self._last_build_time,
                "idf_index": self._idf_index,
            }

            with open(self._cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f)

            debug_log(f"[BM25] Saved IDF cache to {self._cache_file}")
            return True

        except Exception as e:
            debug_log(f"[BM25] Error saving cache: {e}")
            return False

    def _load_cache(self) -> bool:
        """
        Load IDF index from cache file.

        Returns:
            True if cache was loaded and is valid
        """
        if not self._cache_file.exists():
            return False

        try:
            with open(self._cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Validate cache version
            if cache_data.get("version") != 1:
                debug_log("[BM25] Cache version mismatch, will rebuild")
                return False

            # Check if corpus has changed
            cached_hash = cache_data.get("corpus_hash")
            current_hash = self._compute_corpus_hash()

            if cached_hash != current_hash:
                debug_log("[BM25] Corpus changed since last build, will rebuild")
                return False

            # Load cached data
            self._corpus_hash = cached_hash
            self._doc_count = cache_data.get("doc_count", 0)
            self._vocab_size = cache_data.get("vocab_size", 0)
            self._last_build_time = cache_data.get("last_build_time")
            self._idf_index = cache_data.get("idf_index", {})

            debug_log(
                f"[BM25] Loaded cached IDF index: {self._vocab_size} terms "
                f"from {self._doc_count} documents"
            )
            return True

        except Exception as e:
            debug_log(f"[BM25] Error loading cache: {e}")
            return False

    def invalidate_cache(self):
        """Force cache invalidation on next access."""
        self._corpus_hash = None
        self._idf_index = {}

    def get_average_doc_length(self) -> int:
        """
        Get approximate average document length in tokens.

        Used by BM25 for length normalization.

        Returns:
            Average document length (defaults to 5000 if unknown)
        """
        # For now, use a reasonable default for legal transcripts
        # Could be computed during index building if needed
        return 5000

    # =========================================================================
    # Preprocessing Methods (Session 29 - Multi-Corpus Support)
    # =========================================================================

    def get_corpus_files_with_status(self) -> list[CorpusFile]:
        """
        Get list of corpus files with preprocessing status.

        Returns:
            List of CorpusFile objects with preprocessing status
        """
        result = []

        for file_path in self._get_corpus_files():
            # Skip already-preprocessed text files
            if "_preprocessed" in file_path.stem:
                continue

            preprocessed_path = self._get_preprocessed_path(file_path)
            is_preprocessed = preprocessed_path.exists()

            try:
                stat = file_path.stat()
                modified_at = datetime.fromtimestamp(stat.st_mtime)
                size_bytes = stat.st_size
            except Exception:
                modified_at = None
                size_bytes = 0

            result.append(CorpusFile(
                path=file_path,
                name=file_path.name,
                is_preprocessed=is_preprocessed,
                preprocessed_path=preprocessed_path if is_preprocessed else None,
                size_bytes=size_bytes,
                modified_at=modified_at,
            ))

        return result

    def needs_preprocessing(self, file_path: Path) -> bool:
        """
        Check if a file needs preprocessing.

        A file needs preprocessing if its _preprocessed.txt version doesn't exist.

        Args:
            file_path: Path to the source document

        Returns:
            True if preprocessing is needed
        """
        preprocessed_path = self._get_preprocessed_path(file_path)
        return not preprocessed_path.exists()

    def preprocess_file(self, file_path: Path) -> Path:
        """
        Preprocess a corpus document and save as _preprocessed.txt.

        Steps:
        1. Extract text (RawTextExtractor)
        2. Sanitize (CharacterSanitizer)
        3. Preprocess (PreprocessingPipeline - headers, footers, line numbers, title pages)
        4. Save as {stem}_preprocessed.txt

        Args:
            file_path: Path to the source document

        Returns:
            Path to the preprocessed text file

        Raises:
            Exception: If extraction or preprocessing fails
        """
        from src.core.extraction import RawTextExtractor
        from src.core.preprocessing import PreprocessingPipeline
        from src.core.sanitization import CharacterSanitizer

        debug_log(f"[Corpus] Preprocessing: {file_path.name}")

        # Step 1: Extract text
        extractor = RawTextExtractor()
        result = extractor.extract(str(file_path))

        if not result.get("success"):
            raise Exception(f"Extraction failed: {result.get('error', 'Unknown error')}")

        raw_text = result.get("text", "")
        if not raw_text.strip():
            raise Exception("Extracted text is empty")

        # Step 2: Sanitize (fix encoding, mojibake, etc.)
        sanitizer = CharacterSanitizer()
        clean_text = sanitizer.sanitize(raw_text)

        # Step 3: Preprocess (remove headers, footers, line numbers, title pages)
        pipeline = PreprocessingPipeline()
        final_text = pipeline.process(clean_text)

        # Step 4: Save as _preprocessed.txt
        output_path = self._get_preprocessed_path(file_path)
        output_path.write_text(final_text, encoding="utf-8")

        debug_log(f"[Corpus] Saved preprocessed text: {output_path.name} ({len(final_text)} chars)")

        return output_path

    def preprocess_pending(self) -> int:
        """
        Preprocess all pending files in the corpus.

        Returns:
            Number of files successfully preprocessed
        """
        files = self.get_corpus_files_with_status()
        pending = [f for f in files if not f.is_preprocessed]

        if not pending:
            debug_log("[Corpus] No pending files to preprocess")
            return 0

        debug_log(f"[Corpus] Preprocessing {len(pending)} pending files...")

        success_count = 0
        for corpus_file in pending:
            try:
                self.preprocess_file(corpus_file.path)
                success_count += 1
            except Exception as e:
                debug_log(f"[Corpus] Error preprocessing {corpus_file.name}: {e}")

        debug_log(f"[Corpus] Preprocessed {success_count}/{len(pending)} files")

        # Invalidate cache since corpus content changed
        if success_count > 0:
            self.invalidate_cache()

        return success_count

    def get_preprocessed_text(self, file_path: Path) -> str:
        """
        Get preprocessed text for a file, preprocessing if needed.

        Args:
            file_path: Path to the source document

        Returns:
            Preprocessed text content
        """
        preprocessed_path = self._get_preprocessed_path(file_path)

        if not preprocessed_path.exists():
            self.preprocess_file(file_path)

        return preprocessed_path.read_text(encoding="utf-8")

    def _get_preprocessed_path(self, file_path: Path) -> Path:
        """
        Get the path where preprocessed text should be stored.

        Args:
            file_path: Path to the source document

        Returns:
            Path for the _preprocessed.txt file
        """
        return file_path.parent / f"{file_path.stem}_preprocessed.txt"


# Global singleton instance
_corpus_manager: CorpusManager | None = None


def get_corpus_manager() -> CorpusManager:
    """
    Get the global CorpusManager singleton.

    Returns:
        CorpusManager instance
    """
    global _corpus_manager
    if _corpus_manager is None:
        _corpus_manager = CorpusManager()
    return _corpus_manager
