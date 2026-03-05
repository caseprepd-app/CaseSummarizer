# Changelog

All notable changes to CasePrepd are documented here.
Format: user-facing changes first, internal/dev changes in a separate section.

## [1.0.15] - 2026-03-04

### Added
- Document preview tab for viewing extracted/preprocessed text
- Term-in-Context viewer for vocabulary table
- Document tab pagination for large documents
- Default regex overrides for vocabulary indicator patterns
- Word count in session stats
- Files shown in table immediately as purple "Extracting..." rows
- User indicator patterns contribute to rule-based quality score (+/-5)

### Improved
- NER extraction ~40% faster by disabling unused spaCy components (including TopicRank)
- Performance audit: optimized data structures, I/O, caching, and logging (9 items)
- DRY refactoring: vocab field constants, export helpers, and model loading centralized
- Pipeline no longer capitalizes common words like "drywall"
- Q&A follow-up entry uses three-state placeholder with color coding

### Fixed
- 17-bug sweep across UI, core, and services layers
- 16-bug audit: silent config, stuck UI, cache/file races, precision errors
- 11-bug audit: re-entrancy guards, crash recovery, export details, lock gaps
- 36-bug audit: 6 HIGH, 15 MEDIUM, 15 LOW issues across full pipeline
- Deep bug audit round 2: state management, data flow, name dedup, config
- Exception handling audit: crash guards, logging, tracebacks
- Text flattening in reconciliation fixed
- Skipped vocab terms excluded from all export paths
- "Ask default questions" checkbox state now passed to worker subprocess
- ExportService return type annotations corrected

### Internal
- Re-entrancy guards on all GUI button handlers
- DRY: VF constants class, shared export helpers, centralized model loader
- Repo root decluttered: docs moved, junk deleted, README updated

## [1.0.14] - 2026-02-28

### Improved
- Rules engine rebalanced: rules floor raised to 45% (ML capped at 55%) so rules act as stable guardrails
- Tiered person name boost: multi-word rare names get +15, multi-word common +12, single-word +5
- Gentler occurrence curve (log10 * 18, cap 35) reduces score inflation from high-frequency terms
- Tougher artifact penalties: all-caps -12, leading digit -8, single letter -15, trailing punctuation -5
- Algorithm confidence boost: high-confidence YAKE/KeyBERT/RAKE/BM25 scores add up to +6 quality points
- All 8 algorithms (NER, RAKE, BM25, TopicRank, MedicalNER, GLiNER, YAKE, KeyBERT) now visible in vocabulary table columns and settings

### Fixed
- Algorithm scores (YAKE, KeyBERT, RAKE, BM25, TopicRank) were always 0 due to result merger burying scores in nested metadata — now promoted to top level
- Feedback CSV only tracked 3 of 8 algorithms — now records all 8 detection flags and 5 numeric scores
- algo_count was undercounting (summed only NER/RAKE/BM25) — now counts all 8 algorithms
- YAKE score aggregation in multi-doc merge treated "no YAKE data" as perfect score
- Vocabulary exports (Word, PDF) only included 3 algorithm columns — now includes all 8

### Internal
- Meta-learner ML weight thresholds reduced: 0/25/35/45/55% (was 0/40/50/60/80%)
- Count bins consolidated to 5: 1 | 2-3 | 4-6 | 7-20 | 21+ (simpler, avoids sparse data)
- Non-linear algorithm agreement tiers: +4 (2 algos), +8 (3), +12 (4+)
- New SCORE_ALGO_CONFIDENCE_BOOST config constant (default 6, tunable 0-15)
- RAKE and BM25 scores pulled through full pipeline (single-doc + multi-doc)
- Developer feedback CSV cleared and header updated with new columns (feature dimensions changed)
- Stale trained model deleted (incompatible with new feature vector)
- 38 new tests: 31 for quality score rules engine, 4 for score promotion, 3 for feedback recording
- Result merger `if d.get(key)` falsiness bug fixed (filtered legitimate 0.0 scores)

## [1.0.13] - 2026-02-26

### Improved
- Corpus dropdown shows "None" placeholder instead of auto-creating empty "General" corpus

### Internal
- Removed ~3,600 lines of dead code: 5 stale mixin files, unused dialogs, OllamaAIWorkerManager, 3 dead vocab table mixins, 6 uncalled functions
- DRY refactored Q&A and vocabulary export methods (5 methods → 1 helper each)
- Fixed dedup bug in vocabulary term exclusion (could add duplicates to exclusion list)
- New shared Google word frequency loader (frequency_data.py) with thread-safe caching
- Moved PENDING_ANSWER_TEXT constant from deleted mixin to main_window.py
- Updated 11 test files: deleted stale tests, redirected imports, fixed file path references
- 2784 tests passing (1 xfailed, 2 xpassed)

## [1.0.12] - 2026-02-26

### Improved
- Tab labels renamed for clarity: "Vocabulary", "Questions", "Summary"
- "Clear All" button uses purple (caution) styling instead of plain gray
- Follow-up question input starts disabled until Q&A is ready
- Default questions checkbox shows generic label until count is loaded
- Export warning messages now reference "Select All" instead of stale "checkboxes"

### Internal
- Deleted 4 dead follow-up methods from qa_panel.py (~120 lines)
- Fixed 2 stale comments referencing deleted QAPanel._submit_followup
- Added "caution" button style (purple) to theme.py
- Added test_gui_polish_pass.py (25 tests) covering all polish changes
- Updated test_ui_coherence.py for new tab names

## [1.0.11] - 2026-02-26

### Added
- Cumulative file input: add files from multiple folders without replacing previous selections
- Per-file remove button (✕) in the file table

### Improved
- Status bar now shows errors in orange text with 5-second hold
- All error/failure paths consistently use orange status bar styling
- Timer and controls now properly reset when extraction worker fails to start

### Internal
- Status bar audit: routed all updates through set_status/set_status_error
- Fixed timer running forever if worker subprocess failed during extraction
- Added AST-based enforcement test to prevent direct status_label.configure() calls
- Added Status Bar Rule to CLAUDE.md
- Synced file_mixin.py _poll_queue with _qa_answering_active flag

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
