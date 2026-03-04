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
import logging
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import CACHE_DIR, CORPUS_DIR
from src.core.utils.tokenizer import tokenize

logger = logging.getLogger(__name__)

# Corpus limits and thresholds
MAX_CORPUS_DOCS = 25  # Maximum documents allowed in a corpus
MIN_CORPUS_DOCS = 5  # Minimum documents for corpus to be "ready"
CORPUS_COMMON_THRESHOLD = 0.64  # 64% - term must appear in this % of docs to be "common"
CORPUS_COMMON_MIN_OCCURRENCES = 5  # Term must appear at least this many times total


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
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".rtf"}


class CorpusManager:
    """
    Manages corpus of previous transcripts for BM25 algorithm.

    Stores documents in: %APPDATA%/CasePrepd/corpus/
    Caches IDF index in: %APPDATA%/CasePrepd/cache/bm25_idf_index.json

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
                       Defaults to %APPDATA%/CasePrepd/corpus/
            cache_dir: Directory for caching IDF index.
                      Defaults to %APPDATA%/CasePrepd/cache/
        """
        self.corpus_dir = Path(corpus_dir) if corpus_dir else CORPUS_DIR
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR

        # IDF index: {term: idf_score}
        self._idf_index: dict[str, float] = {}
        # Document frequency: {term: num_docs_containing_term}
        self._doc_freq: dict[str, int] = {}
        self._doc_count: int = 0
        self._vocab_size: int = 0
        self._last_build_time: str | None = None

        # Cache metadata
        self._cache_file = self.cache_dir / "bm25_idf_index.json"
        self._corpus_hash: str | None = None

        # Corpus disabled state (when >25 docs added manually)
        self._corpus_disabled: bool = False
        self._disabled_reason: str | None = None

        # Ensure directories exist
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Check if corpus exceeds document limit
        self._check_corpus_limit()

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

    def is_corpus_ready(self, min_docs: int = MIN_CORPUS_DOCS) -> bool:
        """
        Check if corpus has enough documents for BM25.

        Args:
            min_docs: Minimum number of documents required

        Returns:
            True if corpus has at least min_docs documents and is not disabled
        """
        if self._corpus_disabled:
            return False
        return self.get_document_count() >= min_docs

    def is_corpus_disabled(self) -> bool:
        """
        Check if corpus is disabled due to exceeding document limit.

        Returns:
            True if corpus has >25 documents and is disabled
        """
        return self._corpus_disabled

    def get_disabled_reason(self) -> str | None:
        """
        Get the reason why corpus is disabled.

        Returns:
            Error message if disabled, None if not disabled
        """
        return self._disabled_reason

    def _check_corpus_limit(self) -> None:
        """
        Check if corpus exceeds the maximum document limit.

        If >25 documents are in the corpus folder, disable corpus processing.
        This prevents users from manually adding too many files.
        """
        doc_count = self.get_document_count()
        if doc_count > MAX_CORPUS_DOCS:
            self._corpus_disabled = True
            self._disabled_reason = (
                f"Corpus disabled: {doc_count} documents exceeds the "
                f"maximum allowed ({MAX_CORPUS_DOCS}). Remove files to re-enable."
            )
            logger.warning(self._disabled_reason)
        else:
            self._corpus_disabled = False
            self._disabled_reason = None

    def can_add_documents(self, count: int = 1) -> tuple[bool, str | None]:
        """
        Check if adding documents would exceed the corpus limit.

        Args:
            count: Number of documents to add

        Returns:
            Tuple of (can_add, error_message).
            can_add is True if addition is allowed, False otherwise.
            error_message is None if allowed, or the reason why not.
        """
        current_count = self.get_document_count()
        new_total = current_count + count

        if new_total > MAX_CORPUS_DOCS:
            return (
                False,
                f"Cannot add {count} document(s). Current count: {current_count}, "
                f"maximum allowed: {MAX_CORPUS_DOCS}. "
                f"Would result in {new_total} documents.",
            )
        return (True, None)

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

    def get_doc_freq(self, term: str) -> int:
        """
        Get document frequency for a term.

        Returns the number of corpus documents that contain this term.
        Used for corpus familiarity filtering.

        Args:
            term: The term to look up (case-insensitive)

        Returns:
            Number of documents containing this term. Returns 0 for OOV terms.
        """
        # Ensure index is built
        if not self._doc_freq:
            self.build_idf_index()

        lower_term = term.lower().strip()
        return self._doc_freq.get(lower_term, 0)

    def get_total_docs_indexed(self) -> int:
        """
        Get total number of documents used to build the index.

        This may differ from get_document_count() if corpus has changed
        since the index was last built.

        Returns:
            Number of documents that were indexed
        """
        # Ensure index is built
        if not self._idf_index:
            self.build_idf_index()

        return self._doc_count

    def is_corpus_common_term(self, term: str) -> bool:
        """
        Check if a term is common in the corpus (appears frequently).

        A term is considered "common" if:
        1. It appears in >= 64% of corpus documents, AND
        2. It has at least 5 total document occurrences

        This binary feature helps the ML model learn to deprioritize
        domain-common terms that the user likely already knows.

        Args:
            term: The term to evaluate (case-insensitive)

        Returns:
            True if the term is common in the corpus, False otherwise.
            Returns False if corpus is disabled or not ready.
        """
        if self._corpus_disabled:
            return False

        total_docs = self.get_total_docs_indexed()
        if total_docs < MIN_CORPUS_DOCS:
            return False

        doc_freq = self.get_doc_freq(term)

        # Must appear at least CORPUS_COMMON_MIN_OCCURRENCES times
        if doc_freq < CORPUS_COMMON_MIN_OCCURRENCES:
            return False

        # Must appear in >= CORPUS_COMMON_THRESHOLD of documents
        frequency_ratio = doc_freq / total_docs
        return frequency_ratio >= CORPUS_COMMON_THRESHOLD

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
            logger.debug("Using cached IDF index (corpus unchanged)")
            return True

        logger.debug("Building IDF index from corpus...")
        start_time = time.time()

        # Get all document files
        doc_files = self._get_corpus_files()
        if not doc_files:
            logger.debug("No documents found in corpus folder")
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
                logger.warning("Error processing %s: %s", doc_path.name, e)
                continue

        if total_docs == 0:
            logger.debug("No documents could be processed")
            # Set empty index to prevent infinite rebuild loop
            # (would otherwise trigger rebuild on every get_idf call)
            self._idf_index = {"__empty__": 0.0}
            self._doc_freq = {}
            self._doc_count = 0
            self._corpus_hash = current_hash  # Mark as built with this hash
            return False

        # Calculate IDF for each term
        # BM25 IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
        self._idf_index = {}
        for term, df in doc_freq.items():
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
            self._idf_index[term] = round(idf, 4)

        # Store doc_freq for corpus familiarity filtering
        self._doc_freq = dict(doc_freq)
        self._doc_count = total_docs
        self._vocab_size = len(self._idf_index)
        self._corpus_hash = current_hash
        self._last_build_time = datetime.now().isoformat()

        elapsed = time.time() - start_time
        logger.debug(
            "Built IDF index: %d terms from %d documents in %.2fs",
            self._vocab_size,
            total_docs,
            elapsed,
        )

        # Save to cache
        self._save_cache()

        # Check if corpus just became ready and auto-reset ML model
        self._check_corpus_ready_transition()

        return True

    def _check_corpus_ready_transition(self) -> None:
        """
        Check if corpus just became ready.

        If this is the first time the corpus has 5+ documents and an ML model
        exists, reset the model so it can retrain with corpus familiarity features.
        Feedback history is preserved.
        """
        from src.user_preferences import get_user_preferences

        # Check if we're now ready
        if not self.is_corpus_ready():
            return

        # Check if we've already handled this transition
        prefs = get_user_preferences()
        if prefs.get("corpus_was_ever_ready", False):
            return  # Already handled in a previous session

        # First time corpus is ready - mark it and reset model
        prefs.set("corpus_was_ever_ready", True)

        logger.debug("Corpus became ready for the first time (5+ documents)")

        # Auto-reset the ML model if it's trained
        try:
            from src.core.vocabulary.preference_learner import get_meta_learner

            learner = get_meta_learner()
            if learner.is_trained:
                logger.debug("Auto-resetting ML model to incorporate corpus features")
                learner.reset_to_default()
                logger.debug("ML model reset complete - will retrain with corpus features")
        except Exception as e:
            logger.debug("Error resetting ML model: %s", e)

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
            # Use process_document (not extract) - returns dict with 'status' and 'extracted_text'
            result = extractor.process_document(str(file_path))

            if result.get("status") == "success":
                return result.get("extracted_text", "")
            else:
                logger.debug(
                    "Extraction failed for %s: %s", file_path.name, result.get("error_message")
                )
                return ""

        except Exception as e:
            logger.debug("Error extracting %s: %s", file_path.name, e)
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
                "version": 2,  # v2 added doc_freq
                "corpus_hash": self._corpus_hash,
                "doc_count": self._doc_count,
                "vocab_size": self._vocab_size,
                "last_build_time": self._last_build_time,
                "idf_index": self._idf_index,
                "doc_freq": self._doc_freq,  # For corpus familiarity filtering
            }

            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)

            logger.debug("Saved IDF cache to %s", self._cache_file)
            return True

        except Exception as e:
            logger.warning("Error saving cache: %s", e)
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
            with open(self._cache_file, encoding="utf-8") as f:
                cache_data = json.load(f)

            # Validate cache version (accept v1 or v2, rebuild for older)
            cache_version = cache_data.get("version", 0)
            if cache_version < 1:
                logger.debug("Cache version too old, will rebuild")
                return False

            # Check if corpus has changed
            cached_hash = cache_data.get("corpus_hash")
            current_hash = self._compute_corpus_hash()

            if cached_hash != current_hash:
                logger.debug("Corpus changed since last build, will rebuild")
                return False

            # Load cached data
            self._corpus_hash = cached_hash
            self._doc_count = cache_data.get("doc_count", 0)
            self._vocab_size = cache_data.get("vocab_size", 0)
            self._last_build_time = cache_data.get("last_build_time")
            self._idf_index = cache_data.get("idf_index", {})
            # Load doc_freq (may be empty for v1 caches)
            self._doc_freq = cache_data.get("doc_freq", {})

            logger.debug(
                "Loaded cached IDF index: %d terms from %d documents",
                self._vocab_size,
                self._doc_count,
            )
            return True

        except Exception as e:
            logger.debug("Error loading cache: %s", e)
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
    # Preprocessing Methods (Multi-Corpus Support)
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

            result.append(
                CorpusFile(
                    path=file_path,
                    name=file_path.name,
                    is_preprocessed=is_preprocessed,
                    preprocessed_path=preprocessed_path if is_preprocessed else None,
                    size_bytes=size_bytes,
                    modified_at=modified_at,
                )
            )

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
        from src.core.preprocessing import create_default_pipeline
        from src.core.sanitization import CharacterSanitizer

        logger.debug("Preprocessing: %s", file_path.name)

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
        pipeline = create_default_pipeline()
        final_text = pipeline.process(clean_text)

        # Step 4: Save as _preprocessed.txt
        output_path = self._get_preprocessed_path(file_path)
        try:
            output_path.write_text(final_text, encoding="utf-8")
        except OSError as e:
            raise OSError(f"Failed to save preprocessed text to {output_path}: {e}") from e

        logger.debug("Saved preprocessed text: %s (%d chars)", output_path.name, len(final_text))

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
            logger.debug("No pending files to preprocess")
            return 0

        logger.debug("Preprocessing %d pending files...", len(pending))

        success_count = 0
        for corpus_file in pending:
            try:
                self.preprocess_file(corpus_file.path)
                success_count += 1
            except Exception as e:
                logger.debug("Error preprocessing %s: %s", corpus_file.name, e)

        logger.debug("Preprocessed %d/%d files", success_count, len(pending))

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


# Global singleton instance with thread-safe initialization
_corpus_manager: CorpusManager | None = None
_corpus_lock = threading.Lock()


def reset_corpus_manager() -> None:
    """
    Reset the CorpusManager singleton.

    Call this when the active corpus changes so the manager
    gets recreated with the new corpus path. Thread-safe.
    """
    global _corpus_manager
    with _corpus_lock:
        _corpus_manager = None
    logger.debug("Singleton reset - will reload on next access")


def get_corpus_manager() -> CorpusManager:
    """
    Get the global CorpusManager singleton.

    Uses the active corpus path from CorpusRegistry (multi-corpus support).
    Thread-safe with double-check locking pattern.

    Returns:
        CorpusManager instance
    """
    global _corpus_manager

    # Fast path: already initialized
    if _corpus_manager is not None:
        return _corpus_manager

    # Slow path: need to initialize (with lock)
    with _corpus_lock:
        # Double-check after acquiring lock
        if _corpus_manager is None:
            # Get active corpus path from registry (avoids path mismatch with CORPUS_DIR)
            try:
                from src.core.vocabulary.corpus_registry import get_corpus_registry

                registry = get_corpus_registry()
                active_path = registry.get_active_corpus_path()
                _corpus_manager = CorpusManager(corpus_dir=active_path)
                logger.debug("Using active corpus path: %s", active_path)
            except Exception as e:
                # Fallback to default CORPUS_DIR if registry fails
                logger.debug("Registry failed, using default path: %s", e)
                _corpus_manager = CorpusManager()
    return _corpus_manager
