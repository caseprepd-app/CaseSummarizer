# Changelog

All notable changes to CasePrepd will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.20] - 2026-03-18

### Fixed
- Key excerpts extraction errors now report to UI instead of silently returning empty results
- Missing `hasattr` guards on `set_status()` calls in copy-to-clipboard and save-to-file

### Changed
- Relocated shared worker modules (base_worker, queue_messages, silly_messages, status_reporter) from `src/ui/` to `src/services/` to fix backward import direction
- Added `services_imports_ui` check to import violation finder

### Infrastructure
- 30 new tests covering module relocation, error reporting, and hasattr guards

## [1.0.19] - 2026-03-18

### Added
- Combined export with filtered table sorting and score floor controls
- Merged Search Export tab into unified Export tab

### Fixed
- Sticky tooltip bug; refactored tooltips to follow best practices
- Export dropdown disabled during export to prevent rapid clicks

### Changed
- Renamed semantic search "confidence" to "relevance" and raised thresholds
- Swept stale tooltips and removed dead settings

## [1.0.18] - 2026-03-17

### Fixed
- Crash in corpus dialog when clicking an empty treeview row
- Crash in model import when backup file already deleted (`missing_ok=True`)
- Import rule violation: `semantic_question_editor` now imports from `src.config` instead of `src.core.config`
- Semantic index wait status showing "0m elapsed" instead of "30s elapsed"
- Cancellation during semantic indexing now gives the thread a 5-second grace period

### Added
- 99 new tests covering critical gaps identified in test audit:
  - Worker execute() methods (ProcessingWorker, SemanticWorker, ProgressiveExtractionWorker)
  - VocabularyExtractor orchestration (extract, extract_progressive, parallel algorithms)
  - Key excerpts daemon thread (_spawn_key_sentences)
  - WorkerProcessManager crash recovery and lifecycle
  - MainWindow message dispatch and dead subprocess detection
  - Transcript speaker boundary injection
  - Adjusted mean rarity calculator
  - Frequency data loader with thread safety
- Test isolation fix: frequency data cache reset prevents cross-test pollution

### Infrastructure
- Removed stale extraction_prompts/prompts paths from PyInstaller spec
- README.md included in GitHub Releases

## [1.0.17] - 2026-03-16

### Fixed
- Runtime KeyError in theme.py from stale hallucination-verification color references
- Circular import chain (user_preferences → services → model_io → preference_learner → user_preferences)
- Unreachable error-handler recovery branch for preprocessing failures
- Console window flash on worker subprocess and splash screen spawns
- `sanitize()` tuple unpacking in corpus_manager
- 15 test failures from stale mocks, deleted code references, and subprocess timing

### Removed
- ~2,000 lines of dead code from Ollama/LLM/coreference removal
- GPU detection (no longer needed after LLM removal)

## [1.0.16] - 2026-03-16

### Fixed
- YAKE stopwords missing from PyInstaller bundle (crashed on "No such file: stopwords_noLang.txt")
- `RawTextExtractor` API rename: `.extract()` → `.process_document()`, updated key names in document_service and corpus_manager
- Installer now produces single .exe (DiskSpanning=no instead of .exe + .bin split)

### Added
- Console window suppression tests for subprocess spawns and splash screen

## [1.0.15] - 2026-03-16

### Changed
- Pinned all dependency versions for reproducible builds
- Deprecated session persistence (each run is now self-contained)

### Fixed
- Three race conditions in worker subprocess lock sections
- Five bugs found during codebase sweep
- `MIN_LINE_LENGTH` for short transcript lines
- `page_count=None` crash for non-PDF files in key excerpts spawner

### Removed
- Ollama/LLM integration (app is now pure retrieval: FAISS + cross-encoder)
- fastcoref and lettucedetect dependencies (freed ~350MB)
- Coreference settings from UI
- 18 obsolete documentation files

### Infrastructure
- Moved project to `caseprepd-app` GitHub organization
- Added GitHub Pages website replacing Google Sites
- Added version bump script (`scripts/bump_version.py`)
- Added this changelog
