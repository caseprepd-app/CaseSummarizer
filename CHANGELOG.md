# Changelog

All notable changes to CasePrepd will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
