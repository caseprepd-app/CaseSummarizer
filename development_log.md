# Development Log

## Session 46 - Q&A Async Implementation Verification (2025-12-11)

**Objective:** Verify Q&A async implementation was completed after previous session interruption.

### Investigation Results

**Status:** ✅ COMPLETE - No work required

The Q&A async follow-up implementation was fully completed in Session 45 Part 2. Verified the following components:

| Component | File | Status |
|-----------|------|--------|
| Async state variables | `qa_panel.py:96-99` | ✅ `_followup_queue`, `_followup_thread`, `_polling_active` |
| Background thread execution | `qa_panel.py:487-497` | ✅ `threading.Thread` with queue-based results |
| Queue polling mechanism | `qa_panel.py:503-532` | ✅ `_poll_followup_result()` with `after(100, ...)` |
| UI state management | `qa_panel.py:481-482, 526-528` | ✅ Disable/enable during processing |
| Callback wiring | `dynamic_output.py:628-634` | ✅ Connected to MainWindow |
| Backend handler | `main_window.py:895-939` | ✅ `_ask_followup_for_qa_panel()` |

No TODOs, FIXMEs, or incomplete code markers found in Q&A-related files.

---

## Session 45 - Streamlined Vocabulary-First Architecture (2025-12-10)

**Objective:** Refocus application on court reporter's core needs (names + vocabulary), implement unified semantic chunking with token enforcement, and simplify UI.

### Business Objective Refocus

Court reporters need to prep for cases. They really need:
1. **Names of people involved** (parties, witnesses, doctors, attorneys)
2. **Technical vocabulary** (medical terms, legal terminology)

Everything else (summaries, Q&A) is nice-to-have but not essential.

### Key Changes

#### New: Unified Semantic Chunker (`src/chunking/unified_chunker.py`)
- **Semantic chunking** with LangChain SemanticChunker (gradient breakpoints)
- **Token enforcement** via tiktoken (cl100k_base encoding)
- **Target**: 400-1200 tokens per chunk (900 target)
- **Single chunking pass** serves both LLM extraction and Q&A indexing

#### Updated: Combined LLM Extraction (`src/extraction/llm_extractor.py`)
- New `LLMPerson` dataclass for people with roles (plaintiff, defendant, attorney, etc.)
- Combined extraction prompt extracts **both people and vocabulary** in one LLM call
- New method `extract_from_unified_chunks()` for UnifiedChunk objects
- ONE prompt per chunk (not 4 like Case Briefing) - 4x faster

#### Updated: Reconciler (`src/vocabulary/reconciler.py`)
- New `ReconciledPerson` dataclass for reconciled people
- New `reconcile_people()` method for NER + LLM people reconciliation
- `combined_to_csv_data()` method for unified table output

#### Updated: UI Components
- **widgets.py**: Simplified checkboxes:
  - "Extract Names & Vocabulary" (ON by default)
  - "Enable Q&A" (ON by default)
  - "Generate Summary" (OFF by default)
- **dynamic_output.py**:
  - Distinct output pane background color
  - Dropdown options: "Names & Vocabulary" | "Q&A" | "Summary"
  - Progress badge: "Initial results (NER only)" → "Enhanced results (NER + LLM)"

#### Updated: Vector Store Builder (`src/vector_store/vector_store_builder.py`)
- New `create_from_unified_chunks()` method for UnifiedChunk objects

### Files Created
| File | Purpose |
|------|---------|
| `src/chunking/__init__.py` | Chunking module init |
| `src/chunking/unified_chunker.py` | Semantic chunking + token enforcement |
| `config/extraction_prompts/combined_extraction.txt` | Combined people + vocabulary prompt |

### Files Modified
| File | Change |
|------|--------|
| `requirements.txt` | Added tiktoken>=0.5.0 |
| `src/config.py` | Added UNIFIED_CHUNK_* settings |
| `src/extraction/__init__.py` | Export LLMPerson |
| `src/extraction/llm_extractor.py` | Combined extraction, UnifiedChunk support |
| `src/vocabulary/reconciler.py` | People reconciliation, combined output |
| `src/vector_store/vector_store_builder.py` | UnifiedChunk support |
| `src/ui/widgets.py` | Simplified checkboxes |
| `src/ui/dynamic_output.py` | New dropdown, progress badge, distinct color |
| `PROJECT_OVERVIEW.md` | Session 45 architecture description |

### Session 45 Part 2 - Progressive UI & Async Q&A (Continued)

Implemented the three-phase progressive output architecture and fixed several UI responsiveness issues.

#### Progressive Extraction Worker (`src/ui/workers.py`)
- New `ProgressiveExtractionWorker` class runs three phases in sequence
- Phase 1: NER extraction (~5 seconds) - results appear immediately
- Phase 2: Q&A vector store indexing (background thread, parallel with Phase 3)
- Phase 3: LLM enhancement (slow, minutes) - updates table when complete
- Queue-based communication with main thread for UI updates

#### UI Responsiveness Fixes
| Issue | Cause | Fix |
|-------|-------|-----|
| GUI lockup when switching to Q&A mode | `style.theme_use("default")` called repeatedly | Added class-level `_style_configured` flag to only configure style once |
| Common words in NER results (all, age, are) | `extract_with_llm()` bypassed `_post_process()` filtering | Added `_filter_reconciled_terms()` with frequency rank + blocklist filtering |
| GUI freeze during Q&A follow-up | `on_ask_followup` callback ran synchronously | Made `_submit_followup()` async with background thread + queue polling |

#### Files Modified
| File | Change |
|------|--------|
| `src/ui/workers.py` | Added ProgressiveExtractionWorker |
| `src/ui/main_window.py` | Progressive extraction integration, handlers for ner_complete/qa_ready/llm_complete |
| `src/ui/qa_panel.py` | Async follow-up with threading + queue polling, style caching |
| `src/ui/dynamic_output.py` | Style caching, clearer Q&A ready message |
| `src/vocabulary/vocabulary_extractor.py` | Added `_filter_reconciled_terms()` for common word filtering |

### Remaining Work (Next Session)
- Case Briefing deprecation (hide from UI, keep code for reference)
- End-to-end testing with sample documents

### Architecture Pattern: Three-Phase Progressive Output
```
User clicks "Process Documents"
    ↓
[Phase 1: NER] ~5 seconds
    ↓
TABLE APPEARS: Initial names + vocabulary (NER-only results)
    ↓
[Phase 2: Q&A Indexing] ~10-30 seconds (background)
    ↓
Q&A PANEL ACTIVATES: User can ask questions
    ↓
[Phase 3: LLM Enhancement] ~2-10 minutes (background)
    ↓
TABLE UPDATES: Reconciled results, "Both" items sorted to top
```

---

## Session 44 - NER+LLM Vocabulary + LlamaIndex Q&A + GUI Refocus (2025-12-09)

**Objective:** Fix NER+LLM vocabulary extraction, add LlamaIndex query transformation for Q&A, and refocus GUI on vocabulary as primary feature.

### Part 1: NER+LLM Vocabulary Extraction Debug & Fix

**Problem Found:** LLM vocabulary extraction was returning 0 terms despite the model producing valid output.

**Root Cause:** The Ollama model (gemma3:1b) sometimes outputs duplicate JSON keys:
```json
{"terms": [...], "terms": [...]}
```
Python's `json.loads()` keeps only the LAST duplicate key, losing earlier terms.

**Solution:** Added Strategy 0 in `_parse_json_response()` to detect and merge duplicate "terms" arrays before standard JSON parsing.

| Component | Change |
|-----------|--------|
| `src/ai/ollama_model_manager.py` | Added duplicate terms merge strategy |
| `src/vocabulary/vocabulary_extractor.py:321` | Fixed method typo: `_lookup_definition` → `_get_definition` |

**Result:** NER+LLM reconciliation now works correctly:
- Both: 5 terms (found by both NER and LLM - highest confidence)
- NER only: 2 terms (John Smith, ORIF)
- LLM only: 7 terms (medical terminology like "deep vein thrombosis")

### Part 2: LlamaIndex Query Transformation

**Purpose:** Expand vague reporter questions into specific search terms for better retrieval.

**Example:**
- Input: "What happened to the person?"
- Output: ["What happened to the person?", "Identify all parties involved", "Explore allegations of liability", "Search for damages and injuries"]

| Component | Change |
|-----------|--------|
| `src/retrieval/query_transformer.py` | **NEW** - LlamaIndex + Ollama query expansion |
| `src/vector_store/qa_retriever.py` | Integrated QueryTransformer in retrieve_context() |
| `src/config.py` | Added QUERY_TRANSFORM_ENABLED/VARIANTS/TIMEOUT |
| `requirements.txt` | Added llama-index-core, llama-index-llms-ollama |

**Architecture:** QueryTransformer layers ON TOP of existing BM25+/FAISS retrieval (doesn't replace).

### Part 3: GUI Refocus - Vocabulary First

**Change:** Reordered OutputOptionsWidget to prioritize vocabulary extraction over summaries.

**Before:**
```
Summary Length slider (always visible)
☐ Individual Summaries
☐ Meta-Summary
☑ Rare Word List (CSV)
☐ Q&A (hidden unless experimental)
```

**After:**
```
Primary Outputs:
☑ Vocabulary List (NER + LLM)  ← First, ON by default
☐ Document Q&A                  ← Always visible

Summary Options:
☐ Individual Summaries
☐ Meta-Summary
[Summary Length slider]         ← Only visible when summary checked
```

### Files Modified

| File | Changes |
|------|---------|
| `src/ai/ollama_model_manager.py` | Duplicate terms merge strategy |
| `src/vocabulary/vocabulary_extractor.py` | Method typo fix |
| `src/retrieval/query_transformer.py` | **NEW** - LlamaIndex query expansion |
| `src/retrieval/__init__.py` | Export QueryTransformer |
| `src/vector_store/qa_retriever.py` | Query transformation integration |
| `src/config.py` | QUERY_TRANSFORM_* settings |
| `src/ui/widgets.py` | GUI refocus (vocabulary first) |
| `requirements.txt` | llama-index packages |

### Patterns Established

- **Query Transformation Pattern:** Transform → retrieve for each variant → deduplicate & merge by score
- **Found By Column:** "Both" > "NER" > "LLM" confidence ordering (Both sorted first)
- **Conditional UI:** Summary length slider only visible when summary checkbox is selected

---

## Session 42 - Architecture Decision + Performance Fix (2025-12-03)

**Objective:** Finalize chunking architecture decision and fix Case Briefing performance.

### Part 1: Architecture Decision ✅

**Question:** Should Case Briefing use `ChunkingEngine` (semantic gradient) or keep the separate `DocumentChunker`?

**Decision:** Keep `DocumentChunker` for extraction.

**Key Findings:**
- Neither chunker uses true semantic/embedding-based splitting — both are regex-based
- `DocumentChunker` has 45 legal-specific patterns vs. 8 in `ChunkingEngine`
- Legal section structure matters for extraction (PARTIES vs. ALLEGATIONS have different legal meaning)

### Part 2: Performance Fix - Dynamic Worker Scaling

**Problem Found:** User tested Case Briefing with real documents:
- 155 chunks generated from legal documents
- Processing 7/155 chunks took 7 minutes (~1 min/chunk)
- Root cause: Hardcoded `max_workers=2` regardless of system resources

**Solution:** Dynamic worker calculation based on system resources.

| Component | Change |
|-----------|--------|
| `src/system_resources.py` | **NEW** - Calculates optimal workers based on CPU/RAM |
| Settings slider | Changed from dropdown (25/50/75%) to slider (25-100%) |
| `src/briefing/extractor.py` | Uses `get_optimal_workers()` instead of hardcoded 2 |

**Performance Impact (12-core, 16GB RAM machine):**
- Before: 2 workers (hardcoded)
- After: 6 workers at 75% usage setting
- Result: **~3x faster Case Briefing extraction**

**User-Configurable:** Settings → Performance → "System resource usage" slider

### Files Modified

| File | Changes |
|------|---------|
| `src/system_resources.py` | NEW: `get_optimal_workers()`, `get_resource_summary()` |
| `src/briefing/extractor.py` | Dynamic worker calculation |
| `src/ui/settings/settings_registry.py` | Slider for resource usage (25-100%) |
| `src/user_preferences.py` | Validation for `resource_usage_pct` |

### Research Findings

- Ollama context window defaults to 2048 tokens (we're passing `num_ctx` correctly)
- GPU acceleration NOT worth pursuing for business laptops (iGPU support is fragile)
- Focus on CPU optimization via dynamic parallelization

### Part 3: Prompt Engineering Improvements

**Problem Found:** Fact-checking revealed hallucinations in extraction output:
- Example names from JSON schema ("John Smith", "Dr. Wilson") appeared in results
- Some defendants misclassified as plaintiffs
- Non-parties incorrectly labeled as parties

**Research-Based Solution:** Implemented best practices from Google's Gemma 3 documentation:

1. **Few-shot prompting** (3 concrete examples vs. rules)
   - Research shows 10-12% accuracy improvement over zero-shot
   - Google: "Using examples to show the model a pattern to follow is more effective than using examples to show the model an anti-pattern to avoid"

2. **External prompt file** for easy iteration
   - Prompt moved to `config/briefing_extraction_prompt.txt`
   - Can modify prompts without changing code
   - Follows "configuration over hardcoding" principle

3. **Added vocabulary extraction field**
   - Extracts medical/legal terminology for layperson reference
   - Aggregated and deduplicated across chunks
   - Displayed in formatted output

**Files Modified:**

| File | Changes |
|------|---------|
| `config/briefing_extraction_prompt.txt` | NEW: External prompt with 3 few-shot examples |
| `src/briefing/extractor.py` | Loads prompt from file, removed hardcoded prompt |
| `src/briefing/aggregator.py` | Added `_aggregate_vocabulary()` method |
| `src/briefing/formatter.py` | Added `_format_vocabulary()` method |

**Prompt Design (based on Gemma 3 research):**
- 3 diverse examples: complaint, defense document, medical records
- Consistent JSON format across all examples
- Positive patterns (show what to do) instead of negative rules
- Direct and concise (Gemma favors directness over verbosity)

---

## Session 40 - Case Briefing Testing, Bug Discovery & Fix (2025-12-03)

**Objective:** Test Case Briefing Generator with real court documents.

### Part 1: Bug Discovery

**Input:** 5 legal documents (complaint, answers, bill of particulars, transcript)
**Output:**
- Case Type: "Not determined"
- Parties: "Not identified"
- Allegations: "None identified"
- Processing Time: 316 seconds

| Check | Result |
|-------|--------|
| LLM with clean input | ✅ Correctly extracts parties/allegations |
| Actual document extraction | ❌ Returns empty arrays |
| Chunk count | ⚠️ 5 docs → 5 chunks (1 per doc) |

### Part 2: Root Cause Identified

**The Bug:** `DocumentChunker._split_into_paragraphs()` only split on double newlines (`\n\s*\n`).

**Why It Failed:**
1. OCR/PDF extracted text uses single newlines throughout
2. When no double newlines exist, entire 43,262-char document → 1 "paragraph"
3. Chunking logic couldn't split an oversized first paragraph (empty list check failed)
4. LLM couldn't process 43K chars → returned empty JSON arrays

### Part 3: Fix Applied

**File Modified:** `src/briefing/chunker.py`

| Method | Change |
|--------|--------|
| `_split_into_paragraphs()` | Added fallback to line-based splitting when paragraphs exceed max_chars |
| `_split_on_lines()` | NEW - Groups single-newline lines into target-sized segments |
| `_force_split_oversized()` | NEW - Last-resort split at sentence/word boundaries |

**Three-Tier Splitting Strategy:**
1. Double newlines (standard paragraphs) → if any > max_chars...
2. Single newlines (OCR fallback) → if still > max_chars...
3. Character-based at sentence/word boundaries

### Test Results

```
Before fix: 43,262 chars → 1 chunk
After fix:  43,262 chars → 24 chunks (avg 1,753 chars, max 1,802 chars)
```

All 224 tests pass.

### Next Steps

- Re-test Case Briefing through UI with real documents
- Verify extraction produces party/allegation data

---

## Session 39 - Case Briefing UI Integration + Phase 4 Optimizations (2025-12-02)

**Objective:** Integrate Case Briefing Generator into the UI and add Phase 4 optimizations.

### Part 1: UI Integration

**Files Modified:**

| File | Changes |
|------|---------|
| `src/ui/workers.py` | Added `BriefingWorker` class (~70 lines) |
| `src/ui/main_window.py` | Added briefing methods and flow integration |
| `src/ui/dynamic_output.py` | Added Case Briefing display support |

**Key Implementation:**

1. **BriefingWorker** - Background thread for briefing generation:
   - Inherits from `threading.Thread` with `daemon=True`
   - Queue-based communication for progress/completion
   - Uses `BriefingOrchestrator` + `BriefingFormatter` pipeline

2. **MainWindow Integration:**
   - `_start_briefing_task()`, `_poll_briefing_queue()`, `_on_briefing_complete()`
   - Modified task flow to use briefing instead of legacy Q&A

3. **DynamicOutputWidget Updates:**
   - Added `briefing_text` and `briefing_sections` parameters
   - Added `_display_briefing()` method for textbox display
   - Case Briefing shown first in dropdown when available

---

### Part 2: Phase 4 Optimizations

**Files Modified:**

| File | Changes |
|------|---------|
| `src/briefing/extractor.py` | Added parallelization + improved prompts |

**1. Parallelization (ThreadPoolExecutor):**

```python
def extract_batch(
    chunks, progress_callback=None,
    parallel=True,      # NEW: Enable parallel processing
    max_workers=2       # NEW: Conservative default for Ollama
) -> list[ChunkExtraction]
```

- Uses `concurrent.futures.ThreadPoolExecutor` for I/O-bound extraction
- Thread-safe progress tracking with `threading.Lock`
- Results ordered by `chunk_id` regardless of completion order
- Falls back to sequential for single chunk or if disabled

**Expected Performance Improvement:**
- 3 chunks: ~142s → ~80s (2 workers processing in parallel)
- Scales with chunk count; limited by Ollama throughput

**2. Prompt Tuning for Party Identification:**

Added explicit guidance to extraction prompt:
- Clear definitions: PLAINTIFFS = filed lawsuit, DEFENDANTS = being sued
- Case caption pattern: "NAME, Plaintiff, v. NAME, Defendant"
- Domain hints: medical malpractice → patient=plaintiff, doctor=defendant
- Improved schema examples with realistic medical malpractice names

**Before:** LLM confused plaintiff/defendant in 50% of tests
**After:** Explicit rules should significantly improve accuracy

---

### Summary

The Case Briefing Generator UI integration is complete, pending real-world testing:
1. ✅ UI displays briefing in output widget
2. ✅ Background processing doesn't block UI
3. ✅ Parallel extraction (2 workers default)
4. ✅ Improved prompts for party identification

**Status:** Work in Progress - needs end-to-end testing with real documents

---

## Session 38 - Case Briefing Generator Phase 3 (2025-12-02)

**Objective:** Implement the orchestration and formatting phases - complete the core pipeline.

### Phase 3 Implementation

**New Files Added to `src/briefing/`:**

| File | Lines | Purpose |
|------|-------|---------|
| `orchestrator.py` | ~290 | Coordinates full pipeline with progress callbacks |
| `formatter.py` | ~310 | Formats output for display and export |

### Key Components

1. **BriefingOrchestrator** - Main entry point:
   - `generate_briefing(documents)` runs the full pipeline
   - Progress callback system for UI updates
   - Timing breakdown per phase
   - `is_ready()` checks Ollama availability

2. **BriefingFormatter** - Output formatting:
   - Plain text format for display
   - Markdown format for rich export
   - Section-based dict for UI panels
   - Configurable metadata inclusion

3. **Data Classes:**
   - `BriefingResult` - Complete pipeline result with timing
   - `FormattedBriefing` - Formatted output with sections

### End-to-End Test Results

Tested with sample complaint + answer documents:
- **Total time:** 142s (sequential, will improve with parallelization)
- **Chunks processed:** 3
- **Successful extractions:** 2/3
- **Case type detection:** ✅ Medical Malpractice
- **Party identification:** Needs prompt tuning (confused plaintiff/defendant)
- **Allegations/Defenses:** ✅ Correctly extracted

### Pipeline Complete

The core Map-Reduce pipeline is now fully functional:
```
Documents → Chunk → Extract → Aggregate → Synthesize → Format
              ↓         ↓          ↓           ↓          ↓
           Phase 1   Phase 1    Phase 2    Phase 2    Phase 3
```

### Remaining Work

- **Phase 4:** Date weighting + parallelization + prompt tuning
- **UI Integration:** BriefingWorker + panel display (can be separate session)

---

## Session 37 - Case Briefing Generator Phase 2 (2025-12-02)

**Objective:** Implement the REDUCE and SYNTHESIS phases of the Map-Reduce pattern.

### Phase 2 Implementation

**New Files Added to `src/briefing/`:**

| File | Lines | Purpose |
|------|-------|---------|
| `aggregator.py` | ~420 | Merge and deduplicate extracted data |
| `synthesizer.py` | ~270 | Generate "WHAT HAPPENED" narrative |

### Key Components

1. **DataAggregator** - Merges ChunkExtraction objects with:
   - Fuzzy name matching (0.85 similarity threshold) to deduplicate people
   - Text similarity-based deduplication for allegations/defenses
   - Case type voting from multiple hints
   - Name normalization (removes titles, standardizes spacing)
   - Category priority (PARTY > MEDICAL > WITNESS > OTHER)

2. **NarrativeSynthesizer** - Generates narrative summaries:
   - LLM-based synthesis with Ollama (temperature=0.3 for natural prose)
   - Template-based fallback if LLM unavailable
   - `format_people_section()` for "Names to Know" output
   - Configurable target word count (~200 words default)

3. **Data Classes:**
   - `AggregatedBriefingData` - Unified output with all case data
   - `PersonEntry` - Individual person with canonical name, aliases, role
   - `SynthesisResult` - Narrative with success/method metadata

### Pattern: Fuzzy Name Matching

```python
# Names like these are recognized as the same person:
"Dr. John Smith, MD" → "John Smith" → "J. Smith"
```

Uses `difflib.SequenceMatcher` for similarity, plus component overlap detection.

### Remaining Phases

- **Phase 3:** BriefingOrchestrator + BriefingFormatter + UI integration
- **Phase 4:** Date-based transcript weighting + parallelization

---

## Session 36 - Case Briefing Generator Phase 1 (2025-12-02)

**Objective:** Replace Q&A system with LLM-First structured extraction for generating Case Briefing Sheets.

### Architecture Decision

After extensive planning and research, decided on **LLM-First Map-Reduce pattern**:
- **Current Q&A system problem:** BM25+ is a search algorithm, but the use case requires structured extraction
- **New approach:** Direct LLM extraction with JSON schema for structured output
- **Industry validation:** Research confirms Map-Reduce is the standard for long-document processing

### Business Context

Court reporters need a "Case Briefing Sheet" for 30-minute prep before proceedings:
- Case type and parties
- What happened (narrative)
- Allegations and defenses
- Names to expect (grouped by role)
- Medical vocabulary

### Phase 1 Implementation (This Session)

**New Package Created:** `src/briefing/`

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | ~30 | Package with exports |
| `chunker.py` | ~280 | Section-aware document splitting |
| `extractor.py` | ~260 | Per-chunk LLM extraction |

**Key Components:**

1. **DocumentChunker** - Splits legal documents with awareness of:
   - Legal section headers (CAUSES OF ACTION, WHEREFORE, etc.)
   - Document type detection (complaint, answer, transcript)
   - Paragraph boundaries
   - Target ~1800 chars per chunk for context window

2. **ChunkExtractor** - Extracts structured JSON from each chunk:
   - Parties (plaintiffs/defendants)
   - Allegations and defenses
   - Names mentioned (with role and category)
   - Key facts and dates
   - Case type hints

3. **OllamaModelManager.generate_structured()** - New method for JSON output:
   - Uses Ollama v0.5+ `format: "json"` mode
   - Temperature=0 for deterministic extraction
   - Robust JSON parsing with fallback strategies

### Remaining Phases

- **Phase 2:** DataAggregator (merge/deduplicate) + NarrativeSynthesizer
- **Phase 3:** BriefingOrchestrator + BriefingFormatter + UI integration
- **Phase 4:** Date-based transcript weighting + parallelization

### Pattern: Map-Reduce for Long Documents

```
Documents → Chunking → [MAP: Extract per-chunk] → [REDUCE: Merge] → [SYNTHESIZE: Narrative] → Format
```

This pattern is industry-standard for handling documents longer than LLM context windows.

---

## Session 34 - Project Root Cleanup (2025-12-01)

**Objective:** Organize project root directory - move files to proper directories, clean up artifacts, and improve project structure.

### Part 1: Test File Cleanup

**Problem:** 8 test files in project root (unclear status, cluttering root)

**Solution:**
1. Deleted 2 orphaned ONNX test files (deprecated backend):
   - `test_onnx_simple.py`
   - `test_phi3_summary.py`

2. Created `tests/manual/` for 6 integration tests requiring Ollama:
   - `test_debug_mode.py`
   - `test_model_generation.py`
   - `test_model_quick.py`
   - `test_ollama_workflow.py`
   - `test_prompts.py`
   - `test_slider_config.py`

3. Created `tests/manual/README.md` documenting manual test procedures

### Part 2: Root Directory Organization

**Files Moved:**

| From | To | Code Updates |
|------|-----|--------------|
| `check_spacy.py` | `scripts/check_spacy.py` | None |
| `download_onnx_models.py` | `scripts/download_onnx_models.py` | None |
| `test_simple_case.txt` | `tests/sample_docs/test_simple_case.txt` | 1 file |
| `Word_rarity-count_1w.txt` | `data/frequency/Word_rarity-count_1w.txt` | 1 file |

**Code Path Updates:**
- `src/config.py:173` - Word frequency file path updated
- `tests/manual/test_ollama_workflow.py:87` - Test data path updated

**Files Created:**
- `scripts/README.md` - Documents utility scripts

**Configuration Updates:**
- `.gitignore` - Added `debug_flow.txt`, `generated_summary.txt`, `nul` (Windows artifact)
- `.gitignore` - Added exception for `!data/frequency/Word_rarity-count_1w.txt`

### Pattern: Project Root Organization

```
CaseSummarizer/
├── .gitignore, pytest.ini, ruff.toml, requirements.txt  # Config
├── README.md, ARCHITECTURE.md, TODO.md, etc.            # Docs
├── src/                    # Source code
├── tests/                  # Tests (unit + manual + sample_docs)
├── config/                 # Prompts and settings
├── data/                   # Data files (word frequencies)
└── scripts/                # Development utilities
```

Only essential config and docs in root - all code and data in directories.

### User Testing Note

Tested application workflow. Confirmed that the two-phase workflow is working correctly:
1. **Phase 1:** Add Files → extracts text (timer runs during extraction)
2. **Phase 2:** Perform Tasks → runs Q&A/Vocabulary/Summary (timer runs during tasks)

Timer correctly stops after each phase completes.

---

## Session 33 - Codebase Organization & Cleanup (2025-12-01)

**Objective:** Review and improve codebase organization, fix naming inconsistencies, and ensure documentation is up to date.

### Phase 1: Critical Cleanup

Deleted orphaned/obsolete files:
- `src/1_ingestion/` - Empty directory from failed numeric prefix attempt
- `src/vocabulary/vocabulary_extractor_backup.py` - Stale backup file
- Duplicate `SettingsDialog` class from `src/ui/dialogs.py` (newer version in `src/ui/settings/`)
- 15 `.tmp.*` temporary files

### Phase 2: Logging Standardization

Updated `src/extraction/raw_text_extractor.py` to use canonical logging import:
- Old: `from src.utils import Timer, debug, error, info, warning`
- New: `from src.logging_config import Timer, debug, error, info, warning`

The backward-compat wrapper in `src/utils/logger.py` remains for any external code.

### Phase 3: Created `src/prompting/` Package

Consolidated 4 orphan prompt-related files from `src/` root into a proper package:

| Old Location | New Location |
|--------------|--------------|
| `src/prompt_adapters.py` | `src/prompting/adapters.py` |
| `src/prompt_focus_extractor.py` | `src/prompting/focus_extractor.py` |
| `src/prompt_template_manager.py` | `src/prompting/template_manager.py` |
| `src/prompt_config.py` | `src/prompting/config.py` |

Created unified facade API in `src/prompting/__init__.py`:
```python
from src.prompting import (
    PromptTemplateManager, AIFocusExtractor, MultiDocPromptAdapter,
    PromptConfig, get_prompt_config,
)
```

Updated all consumers to use new import paths:
- `src/ai/ollama_model_manager.py`
- `src/ai/summary_post_processor.py`
- `src/summarization/document_summarizer.py`
- `src/summarization/multi_document_orchestrator.py`
- `src/ui/main_window.py`
- `src/ui/workers.py`
- Test files

### Phase 4: UI Code Splitting

**4.1 - main_window.py refactored:**
- Created `src/ui/window_layout.py` with `WindowLayoutMixin`
- Extracted 6 UI creation methods (~260 lines) into the mixin
- `MainWindow` now inherits from `WindowLayoutMixin, ctk.CTk`
- Clean separation: layout code in mixin, business logic in main_window.py

**4.2 - workers.py assessed:**
- Reviewed `workers.py` (651 lines) - under 1000-line threshold
- 5 well-organized, cohesive worker classes (~75-170 lines each)
- Decision: No split needed - file is appropriately sized

### Pattern: Mixin for Layout Separation

```python
# window_layout.py
class WindowLayoutMixin:
    def _create_header(self): ...
    def _create_main_panels(self): ...
    def _create_status_bar(self): ...

# main_window.py
class MainWindow(WindowLayoutMixin, ctk.CTk):
    # Business logic only
```

This pattern separates visual layout from event handling/business logic.

### Files Changed

**Created:**
- `src/prompting/__init__.py` - Unified prompting API
- `src/prompting/adapters.py` - Stage-specific prompt generation
- `src/prompting/focus_extractor.py` - AI focus extraction
- `src/prompting/template_manager.py` - Template loading/management
- `src/prompting/config.py` - Prompt parameters
- `src/ui/window_layout.py` - UI layout mixin

**Deleted:**
- `src/1_ingestion/` (empty directory)
- `src/vocabulary/vocabulary_extractor_backup.py`
- `src/prompt_adapters.py`
- `src/prompt_focus_extractor.py`
- `src/prompt_template_manager.py`
- `src/prompt_config.py`
- Old `SettingsDialog` in `src/ui/dialogs.py`

---

## Session 32 - Unified Package APIs & Architecture Docs (2025-12-01)

**Objective:** Reorganize codebase to better match program flow (Input → Vocabulary → Q&A → Summaries) through unified package APIs rather than file moves.

### Problem Solved

User found code hard to follow because Q&A functionality was split across 3 packages (`qa/`, `vector_store/`, `retrieval/`). Rather than physically moving files (which would require 80+ import changes), we created **unified facade APIs**.

### Unified Q&A API (`src/qa/__init__.py`)

Everything Q&A-related is now importable from `src.qa`:

```python
from src.qa import (
    # Orchestration
    QAOrchestrator, QAResult, AnswerGenerator, AnswerMode,
    # Storage
    VectorStoreBuilder, QARetriever, QuestionFlowManager,
    # Retrieval
    HybridRetriever, ChunkMerger, BaseRetrievalAlgorithm,
)
```

The package re-exports from `src.vector_store` and `src.retrieval`, providing a single entry point.

### Unified Summarization API (`src/summarization/__init__.py`)

All summarization functionality accessible from `src.summarization`:

```python
from src.summarization import (
    ProgressiveSummarizer, ChunkingEngine, Chunk,  # Core
    MultiDocumentOrchestrator, MultiDocumentSummaryResult,  # Multi-doc
)
```

Orphan files (`progressive_summarizer.py`, `chunking_engine.py`) are re-exported without physical move.

### Architecture Documentation Updates

Updated `ARCHITECTURE.md` with:
- New Mermaid diagram showing hybrid retrieval architecture
- Hybrid retrieval section explaining BM25+/FAISS combination
- File directory updated with retrieval package entries
- Unified API documentation

### Note on Numeric Directory Prefixes

Initial plan was to use numbered prefixes (`1_ingestion/`, `2_vocabulary/`) for visual program flow, but **Python package names cannot start with digits**. Abandoned this approach in favor of unified APIs.

### Test Results

- **224 tests passing**
- Pre-existing UI test issue with `test_ui_startup.py` excluded (unrelated)

---

## Session 31 - Hybrid BM25+ Retrieval System (2025-12-01)

**Objective:** Replace FAISS-only Q&A retrieval with a hybrid BM25+ / FAISS system to fix "no information found" results. The semantic-only approach was failing because the embedding model wasn't trained on legal terminology.

### Architecture

Created a new `src/retrieval/` package mirroring the vocabulary extraction pattern:

| Component | Location | Purpose |
|-----------|----------|---------|
| `BaseRetrievalAlgorithm` | `src/retrieval/base.py` | ABC for all retrieval algorithms |
| `BM25PlusRetriever` | `src/retrieval/algorithms/bm25_plus.py` | Lexical search with BM25+ scoring |
| `FAISSRetriever` | `src/retrieval/algorithms/faiss_semantic.py` | Semantic search with embeddings |
| `ChunkMerger` | `src/retrieval/chunk_merger.py` | Weighted result combination |
| `HybridRetriever` | `src/retrieval/hybrid_retriever.py` | Coordinates multiple algorithms |

### Why BM25+ Over Pure Semantic Search

- **Legal language is precise** - exact terms matter ("plaintiff" means plaintiff)
- **Embedding model not trained on legal text** - `all-MiniLM-L6-v2` doesn't understand legal terminology
- **BM25+ scores are more reliable** - deterministic, no neural model needed
- **Future ML integration** - same pattern as vocabulary extraction allows user preference learning

### Algorithm Weights (from config.py)

```python
RETRIEVAL_ALGORITHM_WEIGHTS = {
    "BM25+": 1.0,   # Primary - exact term matching
    "FAISS": 0.5,   # Secondary - semantic can help but less reliable
}
RETRIEVAL_MIN_SCORE = 0.1  # Lower than before (was 0.5)
```

### Key Changes

- **New package:** `src/retrieval/` with 6 new files
- **Refactored:** `src/vector_store/qa_retriever.py` now uses HybridRetriever internally
- **Config:** Added `RETRIEVAL_*` settings to `src/config.py`
- **Dependency:** Added `rank-bm25>=0.2.2` to requirements.txt
- **Tests:** Added 17 new tests in `tests/test_hybrid_retrieval.py`

### Test Results

- **224 tests passing** (207 existing + 17 new hybrid retrieval tests)
- Pre-existing UI test issue with `test_ui_startup.py` (unrelated to this session)

---

## Session 30 - Q&A/Vocabulary Integration Fixes (2025-12-01)

**Objective:** Wire up the Q&A and vocabulary systems to the new CustomTkinter UI and fix integration bugs.

### Fixed Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| UI freeze after processing | HuggingFaceEmbeddings loading blocked main thread | Moved to background `threading.Thread` |
| Placeholder Q&A code | `_start_qa_task()` was using fake QAResult dicts | Wired up real QAWorker with proper vector store |
| Vocabulary not in dropdown | `if self._outputs.get("key"):` returned False for empty `[]` | Changed to `is not None` check |
| Dropdown labels unclear | No indication of result count | Added counts: "Rare Word List (N terms)", "Q&A Results (N)" |

### Code Changes

**`src/ui/main_window.py`:**
- Added imports: `QAWorker`, `VectorStoreBuilder`
- Added Q&A infrastructure: `_embeddings`, `_vector_store_path`, `_qa_results`
- Added debug logging to `_start_vocabulary_extraction()`
- Rewrote `_start_qa_task()` with background thread for embeddings loading
- Added `_qa_init_complete()`, `_poll_qa_queue()`, `_on_qa_complete()` methods
- Updated `_ask_followup()` to use real QAOrchestrator

**`src/ui/dynamic_output.py`:**
- Changed `_refresh_dropdown()` truthiness checks to `is not None`
- Added item counts to dropdown labels
- Updated `_on_dropdown_change()` to use `startswith()` matching

### Known Issues (documented in TODO.md)

- 🔴 Vocabulary returns 0 terms (`combine_document_texts` may return empty)
- 🔴 Q&A answers all say "no information found" (relevance scores near 0 or negative)
- 🟡 Need to switch from FAISS cosine to BM25+ for better retrieval
- 🟡 Corpus name validation needed (prevent invalid filesystem characters)

---

## Session 29 - Q&A-First UI Pivot + Multi-Corpus Architecture (2025-11-30)

**Objective:** Strategic pivot from summarization-first to Q&A-first tool. Complete UI restructure with multi-corpus management system.

### Strategic Changes

The application has pivoted from a summarization-first tool to a Q&A-first document analysis tool. Key reasons:
- Court reporters need quick case familiarization (30+ minute summaries are impractical)
- Q&A results serve as handoff documents for colleagues
- Corpus-based vocabulary requires user's own transcripts to identify what's "unusual"

### New Components Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/vocabulary/corpus_registry.py` | 415 | Multi-corpus management with JSON registry |
| `src/ui/corpus_dialog.py` | 785 | Full-featured corpus management dialog |
| `src/ui/main_window.py` | 765 | Complete CustomTkinter rewrite with two-panel layout |

### Main Window Restructure

Migrated from PySide6 to CustomTkinter with new two-panel layout:
- **Header:** App title + Corpus dropdown + Manage button + Settings button
- **Warning Banner:** Shown when no corpus configured
- **Left Panel:** Session Documents list + Task checkboxes + "Perform N Tasks" button
- **Right Panel:** Results display with output selector + Follow-up input
- **Status Bar:** Status text + Corpus info + Processing timer

### Multi-Corpus Features

1. **CorpusRegistry:** Manages multiple named corpora (Criminal, Civil, etc.)
2. **Preprocessing:** `_preprocessed.txt` suffix for cached extracted text
3. **UI Integration:** Corpus dropdown in header, Manage button opens full dialog
4. **Persistence:** Active corpus stored in user preferences

### Corpus Dialog Features

- Educational header explaining what a corpus is
- BM25 Wikipedia link for curious users
- Create, delete, combine corpora
- Add files with immediate preprocessing
- View preprocessing status (✓ Ready / ⏳ Pending)
- Open folder in system explorer

### Task Workflow Changes

- **Q&A (default ON):** Fast case familiarization
- **Vocabulary (default ON):** Uses corpus for BM25
- **Summary (default OFF):** Warning dialog about 30+ minute processing

---

## Session 28 - Q&A Results Display Bug Fix (2025-11-30)

**Objective:** Fix Q&A results not appearing in the UI despite status showing "14 questions answered."

### Root Cause

The Q&A message handlers in [queue_message_handler.py](src/ui/queue_message_handler.py) were referencing `self.main_window.dynamic_output`, but the actual widget attribute is `self.main_window.summary_results`. This attribute name mismatch prevented Q&A results from reaching the display widget.

### Fix Applied

| File | Line | Change |
|------|------|--------|
| `src/ui/queue_message_handler.py` | 380-381 | `dynamic_output` → `summary_results` |
| `src/ui/queue_message_handler.py` | 400-401 | `dynamic_output` → `summary_results` |

### Verification

- ✅ All 207 tests pass
- ✅ All Q&A module imports work correctly
- ✅ 20 Q&A-specific tests pass

### What Now Works

- Q&A results appear in "Q&A Results" dropdown option
- QAPanel displays questions and answers
- Follow-up questions update the display correctly

---

## Session 27 - Q&A Panel Feature with Vector Search (2025-11-30)

**Objective:** Implement complete Q&A UI leveraging the FAISS vector search infrastructure (Session 24).

### Architecture

| File | Lines | Purpose |
|------|-------|---------|
| `src/qa/__init__.py` | 25 | Package exports |
| `src/qa/qa_orchestrator.py` | 210 | Coordinates Q&A: question loading, vector search, answer generation |
| `src/qa/answer_generator.py` | 250 | Two modes: extraction (fast) vs Ollama (AI-synthesized) |
| `src/ui/qa_panel.py` | 420 | Plain text Q&A display with toggle list, follow-up input |
| `src/ui/qa_question_editor.py` | 380 | GUI for editing default questions (YAML) |
| `src/ui/workers.py` | +115 | QAWorker for background processing |
| `tests/test_qa_orchestrator.py` | 280 | 20 tests for Q&A system |

### Key Features

1. **Dual Answer Modes:** Extraction (keyword matching, fast) vs Ollama (AI synthesis)
2. **Q&A Panel UI:** Scrollable display, checkbox toggles for export, follow-up input
3. **Question Editor:** Add/Edit/Delete/Reorder questions, saves to YAML
4. **Settings Integration:** New "Q&A" tab with answer mode and auto-run options

---

## Session 26 - BM25 Corpus-Based Vocabulary Extraction (2025-11-30)

**Objective:** Add BM25 as third vocabulary algorithm identifying terms unusual compared to user's corpus.

### Implementation

- `src/vocabulary/corpus_manager.py` (430 lines): Corpus folder, IDF index, caching
- `src/vocabulary/algorithms/bm25_algorithm.py` (210 lines): BM25 scoring
- Auto-enables when corpus has ≥5 documents in `%APPDATA%/LocalScribe/corpus/`
- Algorithm weights: NER (1.0), BM25 (0.8), RAKE (0.7)
- 20 new tests pass; full suite: 187 tests

---

## Session 25 - Multi-Algorithm Vocabulary with ML Feedback Learning (2025-11-30)

**Objective:** Extensible multi-algorithm vocabulary extraction with user feedback learning.

### Architecture Refactor

Refactored monolithic `vocabulary_extractor.py` (1336→580 lines, 57% reduction):

| File | Purpose |
|------|---------|
| `src/vocabulary/algorithms/base.py` | ABC, CandidateTerm dataclass |
| `src/vocabulary/algorithms/ner_algorithm.py` | spaCy NER extraction |
| `src/vocabulary/algorithms/rake_algorithm.py` | RAKE keyword extraction |
| `src/vocabulary/result_merger.py` | Weighted confidence combination |
| `src/vocabulary/feedback_manager.py` | CSV-based feedback storage |
| `src/vocabulary/meta_learner.py` | Logistic regression for preference learning |

### ML Feedback System

- 👍/👎 columns in vocabulary table for user feedback
- Features: quality_score, frequencies, algorithm flags, type one-hot
- Training threshold: 30 samples minimum, retrain every 10 new samples
- New dependencies: `rake-nltk`, `scikit-learn`

---

## Session 24 - Q&A Infrastructure with FAISS Vector Search (2025-11-30)

**Objective:** RAG-based Q&A for legal documents with source citations.

### Implementation

- Chose FAISS over ChromaDB (file-based, no database config needed)
- `src/vector_store/vector_store_builder.py`: Creates FAISS indexes
- `src/vector_store/qa_retriever.py`: Retrieves context for questions
- `config/qa_questions.yaml`: 14 branching questions for case analysis
- Vector store created automatically after extraction, saved to `%APPDATA%/LocalScribe/vector_stores/`
- Uses `all-MiniLM-L6-v2` embeddings, 500-char chunks with 50 overlap

---

## Session 23 - Vocabulary CSV Quality Improvements (2025-11-29)

**Objective:** Make vocabulary CSV usable by reducing noise and adding quality scoring.

### Key Changes

| Change | Impact |
|--------|--------|
| Min occurrence filter (2+) | 30-40% noise reduction |
| Raised rarity threshold (150K→180K) | 10-15% reduction |
| OCR error pattern filtering | 5% reduction |
| Quality Score column (0-100) | Excel filtering |
| In-Case Freq & Freq Rank columns | Statistical signals |
| Export format options (all/basic/terms) | User preference |

**Design Decision:** PERSON entities exempt from min occurrence filter (party names critical even if mentioned once).

---

## Session 22 - UI Improvements & Documentation (2025-11-29)

### Features

- Processing timer with elapsed time display (⏱ 0:45 format)
- Button state feedback ("Generating..." during processing)
- Metrics CSV logging for future ML duration prediction
- Human-readable completion time (11m 20s vs 680.6s)
- `format_duration()` utility function

### Documentation

- Merged `scratchpad.md` → `TODO.md`
- Created `ARCHITECTURE.md` with Mermaid diagrams

---

## Sessions 20-21 - Multi-Document Summarization & Thread-Through Architecture (2025-11-29)

### Session 20: Hierarchical Map-Reduce

Implemented proper multi-document summarization (previously naive concatenation truncated ~97% of content):
- **Map Phase:** Each document processed through ProgressiveDocumentSummarizer
- **Reduce Phase:** Individual summaries combined into meta-summary
- New package: `src/summarization/` with result types, orchestrator, document summarizer
- 16 new tests pass

### Session 21: Thread-Through Focus

User's selected template now guides entire pipeline (not just final output):
- `PromptFocusExtractor`: AI extracts focus areas from template
- `MultiDocPromptAdapter`: Threads focus through chunk→doc→meta stages
- Content-hash caching for efficiency
- 22 new tests pass

---

## Sessions 16-19 - Performance & Settings (2025-11-28)

### Session 16: GUI Performance & Preprocessing

- Async batch insertion with pagination (50 rows initial, "Load More")
- Background GC threads
- Smart preprocessing pipeline (HeaderFooterRemover, LineNumberRemover, QAConverter)
- 16 new preprocessing tests

### Session 17: Ollama Context Window Fix

- Aligned chunking to 2048 token context window (max 1000 words/chunk)
- Added `num_ctx` to API payload
- Ruff linting: fixed 17 critical bugs, 342→0 issues

### Session 18: Parallel Document Processing

- `src/parallel/` module with Strategy Pattern
- ThreadPoolStrategy for production, SequentialStrategy for testing
- ProgressAggregator for throttled UI updates
- 2.5-3x speedup for multi-document processing
- 30 new tests

### Session 19: Extensible Settings GUI

- Registry-based settings system (add setting = one register() call)
- Tabbed dialog with SliderSetting, CheckboxSetting, DropdownSetting widgets
- Tooltip system with hover delay
- 6 settings across Performance/Summarization/Vocabulary tabs

---

## Sessions 13-15 - Vocabulary Quality & GUI Responsiveness (2025-11-28)

### Session 13: UI Locking & Cancel Button

- Lock/unlock controls during processing
- Red cancel button stops all workers
- Batch queue processing (10 messages/100ms cycle)
- Configurable vocab display limits (150 rows default)

### Session 14: Vocabulary Performance

- Chunked processing with `nlp.pipe()` (50KB chunks)
- Disabled unused spaCy components (3x speedup)
- NLTK download timeout (15 seconds)
- Fixed medical term filtering

### Session 15: NER Quality Improvements

- Upgraded to `en_core_web_lg` model (4% better accuracy)
- Document frequency filtering (dynamic threshold)
- Pattern-based entity validation
- "Unknown" category for ambiguous classifications

---

## Sessions 9-12 - Vocabulary System Redesign (2025-11-27)

### Session 9: Stenographer-Focused Output

Created modular role detection system (`src/vocabulary/role_profiles.py`):
- Categories: Person, Place, Medical, Technical (not academic terms)
- Context-aware role detection (Plaintiff, Treating physician, etc.)
- O(1) cached frequency rank lookup

### Sessions 10-11: Bug Fixes & Filtering

- Fixed common words leaking (threshold 75K→150K)
- Fixed ALL CAPS name misclassification
- Entity cleaning (removes "and/or lung" fragments)
- Legal citation/boilerplate filtering
- Deduplication algorithm (30-40% reduction)

### Session 12: Condensation Policy

Established automatic log condensation rules: 5 recent detailed, 6-20 condensed, 21+ minimal.

---

## Sessions 5-8 - Core Features (2025-11-25 to 2025-11-26)

### Session 8: System Monitor & UI

- Real-time CPU/RAM display with color thresholds
- Tooltip system for quadrant headers
- Dynamic vocabulary table with filtering
- Prompt selection UI with persistent user prompts
- Recursive length enforcement for summaries

### Sessions 5-7: Refactoring

- Created `WorkflowOrchestrator` (business logic separate from UI)
- Unified logging system (`src/logging_config.py`)
- `src/vocabulary/` package created
- Comprehensive debug logging pattern

### Session 6: Vocabulary Workflow

- `VocabularyWorker` class for async extraction
- spaCy auto-download using `sys.executable`
- Config files made optional

---

## Historical Summary (Pre-Session 5)

### Technology Stack Established

- **UI:** CustomTkinter dark theme (replaced broken PyQt6)
- **AI Backend:** Ollama REST API (replaced fragile ONNX Runtime)
- **Concurrency:** ThreadPoolExecutor for I/O-bound operations
- **Configuration:** config.py with DEBUG mode

### Phase Completion

- ✅ Phase 1: Document pre-processing (PDF/TXT/RTF, OCR detection)
- ✅ Phase 2 (Partial): Desktop UI, AI integration, parallel processing
- ⏳ Phase 2.2, 2.4: Document prioritization, license server (post-v1.0)

---

## Current Status

**Application State:** Q&A-first tool - Integration bugs being fixed
**Tests:** 207 passing
**Sessions:** 30 completed
**Last Updated:** 2025-12-01 (Session 30 - Q&A/Vocabulary Integration Fixes)
