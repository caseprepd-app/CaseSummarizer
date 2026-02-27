"""
Corpus Registry for Multi-Corpus Management

Manages multiple named corpora, each with its own directory and IDF index.
Allows users to create, delete, combine, and switch between corpora.

Key responsibilities:
1. Track multiple corpus directories via JSON registry
2. Manage active corpus selection (persisted across sessions)
3. Create/delete corpus directories
4. Combine corpora by copying files

Privacy: All data is stored locally in %APPDATA%/CasePrepd/corpora/
"""

import contextlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import APPDATA_DIR, CACHE_DIR
from src.user_preferences import get_user_preferences

logger = logging.getLogger(__name__)

# Base directory for all corpora
CORPORA_DIR = APPDATA_DIR / "corpora"

# Supported file extensions for corpus documents
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".rtf"}


@dataclass
class CorpusInfo:
    """Information about a single corpus."""

    name: str
    path: Path
    doc_count: int
    is_active: bool
    created_at: str | None = None


class CorpusRegistry:
    """
    Manages multiple named corpora with flexible paths.

    Registry stored in: %APPDATA%/CasePrepd/corpora/corpora_registry.json

    Example:
        registry = CorpusRegistry()
        registry.create_corpus("Criminal")
        registry.set_active_corpus("Criminal")
        path = registry.get_corpus_path("Criminal")
    """

    def __init__(self):
        """Initialize the corpus registry."""
        self.registry_file = CORPORA_DIR / "corpora_registry.json"
        self._registry: dict[str, Any] = {}

        # Ensure base directory exists
        CORPORA_DIR.mkdir(parents=True, exist_ok=True)

        # Load existing registry
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from JSON file."""
        if not self.registry_file.exists():
            self._registry = {
                "version": 1,
                "corpora": {},
                "created_at": datetime.now().isoformat(),
            }
            return

        try:
            with open(self.registry_file, encoding="utf-8") as f:
                self._registry = json.load(f)
            logger.debug("Loaded registry with %d corpora", len(self._registry.get("corpora", {})))
        except (json.JSONDecodeError, Exception) as e:
            logger.debug("Error loading registry: %s", e)
            self._registry = {
                "version": 1,
                "corpora": {},
                "created_at": datetime.now().isoformat(),
            }

    def _save_registry(self) -> None:
        """Save registry to JSON file."""
        try:
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(self._registry, f, indent=2, default=str)
            logger.debug("Saved registry")
        except Exception as e:
            logger.debug("Error saving registry: %s", e)

    def create_corpus(self, name: str) -> Path:
        """
        Create a new corpus with the given name.

        Args:
            name: Name for the new corpus (used as folder name)

        Returns:
            Path to the new corpus directory

        Raises:
            ValueError: If corpus with this name already exists
        """
        # Sanitize name for filesystem
        safe_name = self._sanitize_name(name)

        if safe_name in self._registry.get("corpora", {}):
            raise ValueError(f"Corpus '{name}' already exists")

        # Check if this is the first corpus (registry was empty)
        was_empty = not self._registry.get("corpora")

        # Create directory
        corpus_path = CORPORA_DIR / safe_name
        corpus_path.mkdir(parents=True, exist_ok=True)

        # Add to registry
        if "corpora" not in self._registry:
            self._registry["corpora"] = {}

        self._registry["corpora"][safe_name] = {
            "display_name": name,
            "path": str(corpus_path),
            "created_at": datetime.now().isoformat(),
        }

        self._save_registry()
        logger.debug("Created corpus '%s' at %s", name, corpus_path)

        # Auto-activate first corpus created when registry was empty
        if was_empty:
            self.set_active_corpus(name)

        return corpus_path

    def delete_corpus(self, name: str, delete_files: bool = False) -> bool:
        """
        Delete a corpus from the registry.

        Args:
            name: Name of the corpus to delete
            delete_files: If True, also delete the corpus directory and files

        Returns:
            True if deletion succeeded

        Raises:
            ValueError: If corpus doesn't exist or is the only remaining corpus
        """
        safe_name = self._sanitize_name(name)

        if safe_name not in self._registry.get("corpora", {}):
            raise ValueError(f"Corpus '{name}' does not exist")

        # Don't allow deleting the last corpus
        if len(self._registry.get("corpora", {})) <= 1:
            raise ValueError("Cannot delete the last corpus. Create another corpus first.")

        # Get corpus path before removing from registry
        corpus_info = self._registry["corpora"][safe_name]
        corpus_path = Path(corpus_info["path"])

        # Remove from registry
        del self._registry["corpora"][safe_name]

        # If this was the active corpus, switch to another
        active = self.get_active_corpus()
        if active == name:
            remaining = list(self._registry["corpora"].keys())
            if remaining:
                self.set_active_corpus(remaining[0])

        # Delete files if requested
        if delete_files and corpus_path.exists():
            try:
                shutil.rmtree(corpus_path)
                logger.debug("Deleted corpus directory: %s", corpus_path)
            except Exception as e:
                logger.debug("Error deleting directory: %s", e)

        # Delete associated cache file
        cache_file = CACHE_DIR / f"{safe_name}_idf_index.json"
        if cache_file.exists():
            with contextlib.suppress(Exception):
                cache_file.unlink()

        self._save_registry()
        logger.debug("Deleted corpus '%s'", name)

        return True

    def combine_corpora(self, source_names: list[str], new_name: str) -> Path:
        """
        Combine multiple corpora into a new one by copying files.

        Args:
            source_names: List of corpus names to combine
            new_name: Name for the new combined corpus

        Returns:
            Path to the new corpus directory

        Raises:
            ValueError: If new_name already exists or source corpora don't exist
        """
        # Validate sources exist
        for name in source_names:
            safe_name = self._sanitize_name(name)
            if safe_name not in self._registry.get("corpora", {}):
                raise ValueError(f"Source corpus '{name}' does not exist")

        # Create the new corpus
        new_path = self.create_corpus(new_name)

        # Copy files from each source
        copied_count = 0
        for name in source_names:
            source_path = self.get_corpus_path(name)
            if not source_path.exists():
                continue

            for file_path in source_path.iterdir():
                if file_path.is_file():
                    # Handle potential name conflicts
                    dest_path = new_path / file_path.name
                    if dest_path.exists():
                        # Add source corpus name as prefix
                        safe_source = self._sanitize_name(name)
                        dest_path = new_path / f"{safe_source}_{file_path.name}"

                    try:
                        shutil.copy2(file_path, dest_path)
                        copied_count += 1
                    except Exception as e:
                        logger.debug("Error copying %s: %s", file_path.name, e)

        logger.debug(
            "Combined %d corpora into '%s' (%d files)", len(source_names), new_name, copied_count
        )

        return new_path

    def list_corpora(self) -> list[CorpusInfo]:
        """
        List all registered corpora with metadata.

        Returns:
            List of CorpusInfo objects
        """
        result = []
        active = self.get_active_corpus()

        for safe_name, info in self._registry.get("corpora", {}).items():
            corpus_path = Path(info["path"])
            doc_count = self._count_documents(corpus_path)

            result.append(
                CorpusInfo(
                    name=info.get("display_name", safe_name),
                    path=corpus_path,
                    doc_count=doc_count,
                    is_active=(info.get("display_name", safe_name) == active),
                    created_at=info.get("created_at"),
                )
            )

        return result

    def get_corpus_path(self, name: str) -> Path:
        """
        Get the directory path for a named corpus.

        Args:
            name: Name of the corpus

        Returns:
            Path to the corpus directory

        Raises:
            ValueError: If corpus doesn't exist
        """
        safe_name = self._sanitize_name(name)

        if safe_name not in self._registry.get("corpora", {}):
            raise ValueError(f"Corpus '{name}' does not exist")

        return Path(self._registry["corpora"][safe_name]["path"])

    def get_active_corpus(self) -> str | None:
        """
        Get the name of the currently active corpus.

        Returns:
            Name of the active corpus, first corpus if none set, or None if no corpora exist
        """
        corpora = self._registry.get("corpora", {})
        if not corpora:
            return None

        prefs = get_user_preferences()
        active = prefs.get("active_corpus")

        # Validate the active corpus still exists
        if active:
            safe_name = self._sanitize_name(active)
            if safe_name in corpora:
                return active

        # Fall back to first corpus
        first_safe_name = next(iter(corpora.keys()))
        return corpora[first_safe_name].get("display_name", first_safe_name)

    def set_active_corpus(self, name: str) -> None:
        """
        Set the active corpus (persists across sessions).

        Args:
            name: Name of the corpus to make active

        Raises:
            ValueError: If corpus doesn't exist
        """
        safe_name = self._sanitize_name(name)

        if safe_name not in self._registry.get("corpora", {}):
            raise ValueError(f"Corpus '{name}' does not exist")

        prefs = get_user_preferences()
        prefs.set("active_corpus", name)

        # Reset CorpusManager singleton so it picks up the new path
        # Ensures BM25 uses the newly selected corpus
        from src.core.vocabulary.corpus_manager import reset_corpus_manager

        reset_corpus_manager()

        logger.debug("Set active corpus to '%s'", name)

    def get_active_corpus_path(self) -> Path | None:
        """
        Get the path to the currently active corpus.

        Returns:
            Path to the active corpus directory, or None if no corpora exist
        """
        active = self.get_active_corpus()
        if active is None:
            return None
        return self.get_corpus_path(active)

    def corpus_exists(self, name: str) -> bool:
        """Check if a corpus with the given name exists."""
        safe_name = self._sanitize_name(name)
        return safe_name in self._registry.get("corpora", {})

    def _sanitize_name(self, name: str) -> str:
        """
        Sanitize a corpus name for use as a directory name.

        Args:
            name: The display name

        Returns:
            Filesystem-safe name
        """
        # Remove/replace problematic characters
        safe = name.strip()
        for char in ["<", ">", ":", '"', "/", "\\", "|", "?", "*"]:
            safe = safe.replace(char, "_")
        return safe

    def _count_documents(self, corpus_path: Path) -> int:
        """Count supported documents in a corpus directory."""
        if not corpus_path.exists():
            return 0

        count = 0
        seen = set()
        for ext in SUPPORTED_EXTENSIONS:
            for file_path in corpus_path.glob(f"*{ext}"):
                # Skip preprocessed files
                if "_preprocessed" in file_path.stem:
                    continue
                if file_path.name.lower() not in seen:
                    seen.add(file_path.name.lower())
                    count += 1
            for file_path in corpus_path.glob(f"*{ext.upper()}"):
                if "_preprocessed" in file_path.stem:
                    continue
                if file_path.name.lower() not in seen:
                    seen.add(file_path.name.lower())
                    count += 1

        return count


# Global singleton instance
_corpus_registry: CorpusRegistry | None = None


def get_corpus_registry() -> CorpusRegistry:
    """
    Get the global CorpusRegistry singleton.

    Returns:
        CorpusRegistry instance
    """
    global _corpus_registry
    if _corpus_registry is None:
        _corpus_registry = CorpusRegistry()
    return _corpus_registry
