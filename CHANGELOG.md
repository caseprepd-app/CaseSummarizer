# Changelog

All notable changes to CasePrepd will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.31] - 2026-03-29

### Fixed
- Settings changes (relevance thresholds, citation limits) now take effect immediately without restarting the app
- Removed duplicate search input from result panels — one search bar, no confusion
- Transcript detection now samples first and middle of document, catching transcripts with long preambles (previously limited to first 20K characters)
- Combined export dropdown now includes Word (.docx) option that was missing despite backend support
- Config file paths centralized via src/core/paths.py — eliminates fragile __file__-based path chains in 9 files

### Changed
- Renamed internal `_d()` config function to `_frozen_setting()` / `get_live_setting()` for clarity
- find_violations.py now enforces centralized path resolution (flags __file__-based config paths)

### Infrastructure
- New src/core/paths.py: single source of truth for config/data/assets directory resolution
- 7 new transcript boundary tests covering all document size categories

## [1.0.30] - 2026-03-29

### Fixed
- Stop button now resets all pending flags (_key_sentences_pending, _followup_pending), preventing infinite poll loops and blocked follow-up searches after cancellation
- Clear Files shows status message when blocked during processing instead of failing silently
- Follow-up status bar displays citation length instead of always-zero quick_answer length
- Search cancellation correctly logged as user action, not misreported as indexing failure
- Default questions loaded from single source (DefaultQuestionsManager) instead of dual YAML/JSON paths
- ask_question() return type annotation corrected to SemanticResult
- Silent exception handlers in export dropdowns now log errors instead of bare pass
- "Key Sentences" → "Key Excerpts" in README and Help dialog
- 26 pre-existing test failures repaired (stale imports of removed constants and parameters)

### Changed
- Workers split into individual modules: processing_worker.py, semantic_worker.py, progressive_extraction_worker.py (workers.py is now a re-export shim)
- Stale LLM/answer-generation references updated across 12 docstrings and comments
- CSV export no longer includes empty "Quick Answer" column
- HTML export omits empty Answer div when quick_answer is blank

### Removed
- Dead code: REJECTION_TEXT, hallucination verification branch (70 lines), chunk_pdf(), _paragraph_chunk(), 12 dead config constants (summary, license, BM25_WEIGHT, RERANKER_TOP_K)
- pypdf dependency (only consumer was deleted chunk_pdf method)
- Dead data/keywords directory reference from spec and asset audit
- Deleted test for removed token_budget module and chunking_config.yaml

### Infrastructure
- 18 new tests covering CSV export columns, HTML quick_answer guard, cancellation vs failure, default questions routing, type annotations
- 5 existing tests hardened with real behavioral assertions and boundary edge cases
- Suite: 3760 passed, 10 skipped, 1 xfailed

## [1.0.29] - 2026-03-28

### Changed
- Single-chunk retrieval: semantic search now returns only the top-1 reranked chunk instead of assembling multiple chunks — eliminates wasted computation and relevance score dilution
- Adaptive retrieval k: candidate pool scales with corpus size (small documents retrieve fewer candidates); config value is ceiling, not fixed target
- Dynamic mean-cutoff reranking: cross-encoder reranker keeps only chunks above the mean score (capped at top_k) instead of always returning a fixed count

### Fixed
- Score explainer crash when feature count changed (mock feature vector size mismatch)
- Export All stub key mismatch ("Summary" → "Key Excerpts") causing empty summary in Word/PDF exports
- Build readiness test missing gibberish_detector hidden imports
- Feature vector size assertions updated from 53 to 55

### Infrastructure
- 10 new test files covering export modules, header/footer remover, export service, semantic orchestrator, hybrid retriever (~170 new tests)
- Deprecated multi-chunk context assembly code moved to src/deprecated/
- Chunk-level progress during semantic indexing

## [1.0.28] - 2026-03-27

### Added
- Key excerpts ranked by interestingness score (vocab hits, person entities, word rarity, rejected term penalty, gibberish penalty) — most useful excerpts now sort to the top
- Markov chain gibberish detection with spell-checker fallback for vocabulary filtering
- ftfy text cleaning for OCR artifacts and encoding issues
- Normalized gibberish scores as ML features (worst + mean per term)

### Fixed
- Export (HTML/PDF/Word) was missing key excerpts due to wrong output key lookup
- Raised Markov gibberish hard-filter threshold from 4.0 to 5.5 to reduce false positives

### Infrastructure
- GitHub Pages version auto-updates during rebuild

## [1.0.27] - 2026-03-25

### Fixed
- TopicRank fails on end-user machines without git.exe

### Changed
- Added download and website links to README

## [1.0.26] - 2026-03-24

### Fixed
- Thread-safe singleton initialization for UserPreferences and AIService
- Hash collision in vocabulary treeview item IDs
- Splash screen orphan process on early exit
- Scanned page detection, corpus double-count, excerpt boundary errors
- Bundled scipy for TopicRank + pytextrank dependency diagnostics

### Removed
- Unused SystemMonitor widget and all references

### Infrastructure
- Test isolation improvements — singleton reset between tests

## [1.0.25] - 2026-03-24

### Fixed
- Increased default chunk sizes for fuller search and excerpt passages
- Bundled pytextrank dependency tree + 5 bug fixes from program sweep

### Added
- Bundled VC++ Redistributable in installer for zero-friction setup

### Infrastructure
- Full test audit — removed outdated tests, strengthened shallow tests, added 502 new tests

## [1.0.24] - 2026-03-23

### Fixed
- Semantic search failing on end-user machines — nomic embedding model's custom architecture code (`trust_remote_code`) was not bundled, causing offline loading to fail with "couldn't connect to huggingface.co"

### Infrastructure
- Bundle nomic-bert custom Python code (`configuration_hf_nomic_bert.py`, `modeling_hf_nomic_bert.py`) with patched `config.json` for fully offline model loading
- Download scripts auto-fetch and patch custom code after model download
- `validate_models.py` now requires custom code files, failing early if missing
- 21 new tests for offline model readiness and custom code bundling

## [1.0.23] - 2026-03-22

### Fixed
- Embedding model no longer attempts HuggingFace downloads — `local_files_only=True` enforced unconditionally
- Dev mode now mirrors production exactly: all assets load from project directory, not system fallbacks
- NLTK restricted to single bundled path (no system corpus fallback)
- Tiktoken and HuggingFace cache env vars force-set instead of setdefault

### Added
- Asset audit log at startup — every model, dataset, and binary logged with BUNDLED/SYSTEM/MISSING provenance
- Pre-build validation script (`scripts/validate_models.py`) catches missing or truncated assets before PyInstaller

### Infrastructure
- Frozen-mode guards on all spaCy and ML model loading — RuntimeError with reinstall guidance instead of silent fallback

## [1.0.22] - 2026-03-19

### Added
- Vocabulary quality indicator turns yellowish-green when some algorithms fail, showing users results may be incomplete

### Fixed
- Graceful degradation when individual vocabulary algorithms or semantic questions fail — partial results shown instead of crash
- Bundled app no longer falls back to system Python dependencies at runtime
- Missing psutil dependency added to build
- Remaining user-facing error messages cleaned up (no raw tracebacks)

### Infrastructure
- Test gap audit: strengthened 40+ weak/tautological assertions across 22 existing test files
- 9 new edge case test files covering vocabulary extractor, chunker, worker process, worker manager, hybrid retriever, PDF extractor, file_utils, status_reporter, and citation_excerpt
- Net +316 tests (3,078 → 3,394)

## [1.0.21] - 2026-03-18

### Fixed
- Fixed "Resource punkt not found" error on end-user machines — RAKE now uses bundled nupunkt sentence splitter instead of NLTK punkt data
- Bundled tiktoken BPE encoding data so token counting works offline without internet
- Improved error messages when bundled spaCy models are missing — users now see reinstall guidance instead of cryptic OSError tracebacks

### Removed
- Removed punkt_tab from NLTK data bundle (no longer needed, saves disk space)

### Infrastructure
- Full audit of system dependencies to ensure standalone installer has zero runtime downloads
- Added tests for tiktoken offline loading and bundled cache validation
- model_loader now logs a warning (not silent debug) when falling back from bundled to HuggingFace

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
