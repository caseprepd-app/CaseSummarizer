# Changelog

All notable changes to CasePrepd are documented here.
Format: user-facing changes first, internal/dev changes in a separate section.

## [1.0.10] - 2026-02-25

### Improved
- GPU-aware per-chunk read timeout for Ollama heartbeat detection
- All LLM inference timeouts increased to 15 hours (prevents timeout on long documents with slow hardware)

### Internal
- Ollama best-practices audit — 7 issues fixed (client API, error handling, keep-alive)
- Removed dead timeout constants and deprecated code (~190 lines)

## [1.0.9] - 2026-02-24

### Fixed
- Default questions now answered even when checkbox is unchecked

### Internal
- Removed 9 dead code items (~660 lines) and 7 unused settings
- Eliminated Performance tab from Settings
- Renamed internal classes for clarity (DictionaryTextValidator, VocabularyReconciler)
- Code cleanup across 79 files (stale comments, session markers)

## [1.0.8] - 2026-02-23

### Fixed
- Race condition: follow-up polling could loop forever if subprocess crashed (now 30s timeout)
- Race condition: unprotected worker state access across threads (added locking)
- Q&A index could get stuck at "Building Q&A Index..." forever

## [1.0.7] - 2026-02-23

### Fixed
- Splash screen in frozen builds now uses environment variable instead of argv (more reliable)

## [1.0.6] - 2026-02-22

### Improved
- Added comprehensive IPC and worker logging for easier debugging

### Fixed
- Eliminated duplicate splash screen in frozen builds
- Fixed worker readiness handshake timing

## [1.0.5] - 2026-02-22

### Fixed
- Production stability: robust splash kill, preserved vocabulary casing, added missing traceback import
- Subprocess IPC race conditions and early-exit in Q&A processing

## [1.0.4] - 2026-02-22

### Fixed
- Prevented duplicate splash screen on startup
- Added worker readiness handshake (app waits for subprocess to be ready before sending work)

## [1.0.3] - 2026-02-21

### Fixed
- Q&A polling could die silently after subprocess refactor
- Cross-encoder model loading error with sentence-transformers 3.0+

### Internal
- Removed ~1900 lines of dead code (queue_message_handler, workflow_orchestrator, VocabularyWorker)
- Added 105 new tests for IPC message flows and handler coverage

## [1.0.2] - 2026-02-21

(Version bump only — no user-facing changes)

## [1.0.1] - 2026-02-21

### Added
- Worker subprocess architecture (heavy processing runs in separate process for stability)
- Splash screen with randomized images during startup
- Universal display scaling for 4K and high-DPI monitors
- Progressive follow-up display (retrieval shown before generation completes)
- Bundled Tesseract + Poppler for zero-setup OCR
- LLM vocabulary enhancement moved to Settings (default off)
- Ollama readiness gates on LLM checkboxes with small-model warning
- Dynamic version in About dialog
- Graceful shutdown guards and resource cleanup

### Fixed
- GUI freeze/thrashing during window resize while processing
- "Not Responding" during window resize and result display
- Vocab column widths snapping back after user drag
- Text clipping on high-DPI displays (font-metrics-based row height)
- Q&A follow-up textbox disappearing on tab switch
- Name deduplication safety gates (common-word, ambiguity, shared last names)
- Summary feature wired up to MultiDocSummaryWorker (was stubbed out)
- VRAM detection for >4GB GPUs, train() TypeError, NER progress floor
- Silent model downloads blocked in production
- NLTK bundle extraction for PyInstaller reliability
- Inno Setup 6.4.3 compatibility fix

### Internal
- Thread-safe concurrency patterns across production files
- Removed dead UI code (quadrant_builder, OutputOptionsWidget, ModelSelectionWidget)
- Removed WordNet definitions from vocabulary output (unhelpful for legal terms)
- Comprehensive logging enabled by default (auto-delete after 90 days)
