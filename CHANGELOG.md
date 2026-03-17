# Changelog

All notable changes to CasePrepd will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
