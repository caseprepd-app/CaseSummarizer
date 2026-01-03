# Research Log

> **Purpose:** Technical decisions and why they were made. Check here before researching something that may already be decided.
>
> **Format:** Append new entries at the top. Never delete old entries.

---

## Session 74: Auto-Rotation and Document Cropping for Phone Photos — 2026-01-02

**Problem:** Users uploading phone photos of documents need automatic rotation correction (90°/180°/270°) and background cropping before OCR. The existing `ImagePreprocessor` only handled small-angle deskewing (±10°).

### Research Findings

**Libraries Evaluated for Rotation Detection:**

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Tesseract OSD** | Already have pytesseract; detects 0°/90°/180°/270° + confidence score | ~90% accuracy; needs visible text | ✅ Chosen |
| **OpenCV Hough Transform** | Pure OpenCV (already installed) | Complex tuning required | ❌ Overkill |
| **ocropus3-ocrorot** (DL) | Very accurate (0.2° precision) | Heavy dependency | ❌ Overkill |

**Libraries Evaluated for Document Detection/Cropping:**

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **OpenCV Contour Detection** | Already installed; well-documented; handles perspective | Struggles with busy backgrounds | ✅ Chosen |
| **GrabCut** | Handles white backgrounds well | Fails if document corners cropped | ❌ Less reliable |
| **Perspectra** (PyPI) | All-in-one solution | Version 0.1.0 (immature) | ❌ Too new |
| **Deep Learning** (docTR) | Most robust | Heavy dependencies | ❌ Overkill |

### Implementation

Added two new stages at the **beginning** of `ImagePreprocessor.preprocess()`:

**Stage 0a: Orientation Detection** (`_detect_and_correct_orientation`)
- Uses `pytesseract.image_to_osd()` for 90°/180°/270° detection
- Confidence threshold (default 2.0) prevents false corrections
- Gracefully skips if OSD fails (e.g., no text in image)

**Stage 0b: Document Detection** (`_detect_and_crop_document`)
- Canny edge detection → contour finding → largest 4-point contour
- Perspective transform via `cv2.getPerspectiveTransform()` + `cv2.warpPerspective()`
- Minimum area ratio (default 10%) prevents detecting noise as document
- Skips gracefully if no 4-sided document found

**New Pipeline Flow:**
```
Phone Photo
    ↓
[0a] Orientation correction (90°/180°/270°)
    ↓
[0b] Document detection & perspective crop
    ↓
[1-6] Existing preprocessing (grayscale, denoise, CLAHE, binarize, deskew, border)
    ↓
OCR
```

**New Constructor Parameters:**
- `enable_orientation_correction` (default: True)
- `orientation_confidence_threshold` (default: 2.0)
- `enable_document_detection` (default: True)
- `min_document_area_ratio` (default: 0.1)

### Files Modified

- `src/core/extraction/image_preprocessor.py` — Added 2 new stages, updated stats
- `tests/test_image_preprocessor.py` — 8 new tests (27 total, all passing)

### No New Dependencies

Both features use already-installed libraries:
- `pytesseract` — OSD for orientation
- `opencv-python-headless` — Contour detection, perspective transform

### Sources

- [PyImageSearch: Mobile Document Scanner](https://pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/)
- [PyImageSearch: Correcting Text Orientation](https://pyimagesearch.com/2022/01/31/correcting-text-orientation-with-tesseract-and-python/)
- [LearnOpenCV: Automatic Document Scanner](https://learnopencv.com/automatic-document-scanner-using-opencv/)
- [Tesseract ImproveQuality Docs](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html)

---

## Session 66: Test Fixes (Artifact Filter & BM25 Mock) — 2026-01-01

**Fixes:** 2 failing tests now pass (313/313 tests passing)

### 1. Artifact Filter False Positive Fix

**Problem:** `test_extract_deduplication` was failing because "John Smith" was being filtered out by the artifact filter.

**Root Cause:** The artifact filter checks if canonical terms are substrings of other terms. When BM25 extracted single words "john" and "smith" (freq=3 each), these became canonical terms. The filter then detected "john" as a substring of "john smith" and incorrectly removed "John Smith" as an artifact.

**Fix:** Only use multi-word canonical terms (containing spaces) for substring matching. Single-word canonicals cause false positives with legitimate multi-word phrases like names.

```python
# Only include multi-word terms (2+ words) as canonical
if term and " " in term:
    canonical_terms.append(term.lower())
```

**File:** `src/core/vocabulary/artifact_filter.py:62-66`

### 2. BM25 Test Mock Fix

**Problem:** `test_vocabulary_extractor_skips_bm25_when_corpus_empty` was failing because it assumed an empty corpus, but the user's actual corpus had 6 documents.

**Root Cause:** The test created a `VocabularyExtractor()` without mocking the corpus manager, so it used the real user corpus which had documents, causing BM25 to be correctly enabled.

**Fix:** Mock `get_corpus_manager` to return a manager pointing to an empty temp directory, isolating the test from user data.

```python
with patch('src.core.vocabulary.corpus_manager.get_corpus_manager', return_value=empty_manager):
    extractor = VocabularyExtractor()
```

**File:** `tests/test_bm25_algorithm.py:303-324`

---

## Session 65: Export & Copy Buttons + GPU Detection Tests — 2026-01-01

**Features Added:**

### 1. Vocabulary Export CSV Button
Added one-click CSV export for vocabulary results:
- New "Export CSV" button in the output button bar
- Exports to Documents folder with timestamped filename (`vocabulary_YYYYMMDD_HHMMSS.csv`)
- Success dialog offers to open containing folder in Windows Explorer
- Respects user's export format setting (basic/all/terms_only)

**Files:**
- `src/ui/dynamic_output.py` — Added `export_csv_btn` and `_quick_export_vocab_csv()` method

### 2. Q&A Copy to Clipboard Button
Added clipboard copy for Q&A results:
- New "Copy to Clipboard" button in Q&A panel button bar
- Copies selected Q&A pairs in readable text format
- Format: `Q: question\nA: answer\n[Source: ...]\n`
- Shows confirmation with count of copied items

**Files:**
- `src/ui/qa_panel.py` — Added `copy_btn` and `_copy_to_clipboard()` method

### 3. Comprehensive GPU Detection Tests
Created 30 unit tests for Session 64's GPU detection and dynamic context features:

**Test Categories:**
- `TestIsDedicatedGpu` (11 tests) — NVIDIA/AMD keyword detection, integrated GPU exclusion
- `TestOptimalContextSize` (8 tests) — VRAM-to-context tier mapping (4K-64K)
- `TestVramGb` (2 tests) — Byte-to-GB conversion
- `TestGpuStatusText` (2 tests) — Human-readable status strings
- `TestUserPreferencesContextSize` (5 tests) — Preferences integration, validation
- `TestVramContextTiers` (3 tests) — Configuration validation

**Files:**
- `tests/test_gpu_detector.py` — New file with 30 tests (all passing)

---

## Session 64: Move Corpus Settings to Settings Menu — 2025-12-31

**Problem:** Corpus management was scattered across multiple UI locations:
1. Header dropdown for quick switching (useful - keep this)
2. "Manage" button opening separate CorpusDialog
3. Warning banner when corpus < 5 documents
4. BM25-related settings in Vocabulary tab

**Goal:** Consolidate corpus configuration into Settings dialog while keeping quick-switch dropdown in header.

### Solution: Corpus Tab in Settings

**1. Created CorpusSettingsWidget** (`src/ui/settings/settings_widgets.py`):
```
┌─────────────────────────────────────────────────────────────┐
│  Corpus Management                                       ⓘ  │
├─────────────────────────────────────────────────────────────┤
│  📚 What is a Corpus?                                       │
│  A corpus is a collection of YOUR past transcripts...       │
│  ✓ 100% local and offline                                   │
│  ✓ Powers the BM25 vocabulary algorithm                     │
├─────────────────────────────────────────────────────────────┤
│  Current Status:                                            │
│  • Active corpus: My Transcripts                            │
│  • Documents: 23 files                                      │
│  • BM25 Algorithm: ✓ Active (5+ documents)                  │
├─────────────────────────────────────────────────────────────┤
│  [Manage Corpus...]                                         │
└─────────────────────────────────────────────────────────────┘
```

**2. Registered Corpus Tab** (`src/ui/settings/settings_registry.py`):
- New "Corpus" category with CUSTOM widget
- Widget shows status summary and opens CorpusDialog for full management

**3. Updated Header "Manage" Button** (`src/ui/main_window.py`):
- Now opens Settings → Corpus tab instead of CorpusDialog directly
- User can then click "Manage Corpus..." for full dialog

**4. Removed Warning Banner**:
- Deleted `_create_warning_banner()` from `window_layout.py`
- Deleted `_update_corpus_banner()` from `main_window.py`
- Status now visible in Settings → Corpus tab

### Architecture Decision

**Why not embed full CorpusDialog (786 lines) into Settings tab?**
- Avoids code duplication
- Keeps corpus management logic in one place
- Settings tab provides status/context, CorpusDialog handles CRUD
- Pattern matches DefaultQuestionsWidget (summary + action button)

**Settings Tab Order:** Performance → Vocabulary → Corpus → Q&A → Experimental

### GPU Detection Improvements

**Problem:** The gpu-tracker library was producing warnings:
```
WARNING: Neither the nvidia-smi command nor the amd-smi command is installed.
```

**Root Cause:** gpu-tracker required external CLI tools (nvidia-smi, amd-smi) to function properly.

**Solution:** Switch to PyTorch for GPU detection.

1. **Removed `CUDA_VISIBLE_DEVICES=''`** from `main.py`
   - Was forcing CPU-only mode for torch
   - Now allows GPU detection to work properly

2. **Updated `gpu_detector.py`** to use PyTorch as primary:
   - `torch.cuda.is_available()` works without any CLI tools
   - Falls back to nvidia-smi/amd-smi check for edge cases
   - Returns actual GPU name (e.g., "NVIDIA GeForce RTX 3080 detected")

3. **Removed gpu-tracker dependency** from `requirements.txt`
   - PyTorch is already installed (via sentence-transformers)
   - No additional dependencies needed

4. **Added WMI detection** for Windows (detects both NVIDIA and AMD):
   - Queries `Win32_VideoController` via PowerShell
   - Filters by keywords to distinguish dedicated vs integrated GPUs:
     - **Dedicated NVIDIA:** geforce, rtx, gtx, quadro, tesla
     - **Dedicated AMD:** radeon rx, radeon pro, firepro
     - **Integrated (excluded):** intel, uhd graphics, iris, radeon graphics
   - Returns GPU name, vendor, and VRAM

**Detection Order:**
1. PyTorch CUDA (fast, NVIDIA only)
2. WMI Query (Windows, all vendors including AMD)
3. CLI tools (nvidia-smi / amd-smi fallback)

**Result:** Clean GPU detection with no warnings, supports both NVIDIA and AMD, no external dependencies.

### Dynamic LLM Context Window Based on VRAM

**Problem:** LLM context window was hardcoded at 2048 tokens regardless of hardware.
Users with powerful GPUs couldn't leverage larger context windows, while users without
GPUs might experience issues with the default.

**Research Findings:**
- VRAM usage grows linearly with context (about 1GB per 4K tokens for 8B models)
- Ollama defaults to 4096 tokens but supports much larger contexts
- Exceeding VRAM causes 5-20x performance degradation (spills to RAM)
- Larger context = better comprehension of long documents

**Solution:** Dynamic context size based on detected VRAM.

**Conservative VRAM Tiers (with 1.5% safety buffer):**

| VRAM | num_ctx |
|------|---------|
| No GPU / < 6GB | 4,000 |
| 6-8GB | 8,000 |
| 8-12GB | 16,000 |
| 12-16GB | 32,000 |
| 16-24GB | 48,000 |
| 24GB+ | 64,000 |

**Implementation:**

1. **`gpu_detector.py`**: Added `get_optimal_context_size()` and `get_vram_gb()`

2. **`user_preferences.py`**: Added context size preference:
   - `get_context_size_mode()` → "auto" or int
   - `get_effective_context_size()` → resolves "auto" via GPU detection

3. **Settings UI**: Added dropdown in Q&A tab:
   - "Auto (16K - detected 8.0GB VRAM)"
   - Manual options: 4K, 8K, 16K, 32K, 48K, 64K

4. **`ollama_model_manager.py`**: Updated `generate_text()` and `generate_structured()`
   to use dynamic context from user preferences

**User Experience:**
- Auto-detection works out of the box
- Power users can override in Settings → Q&A → Context window size
- If Ollama becomes slow, user can manually reduce context

### BM25/Corpus Path Mismatch Bug Fix

**Problem:** BM25 algorithm was showing as disabled (0 documents) even though the
Corpus dialog showed 6 documents in the "General" corpus. Investigation revealed
a path mismatch between two systems:

| System | Path Used |
|--------|-----------|
| CorpusRegistry | `%APPDATA%/LocalScribe/corpora/General/` (multi-corpus) |
| CorpusManager | `%APPDATA%/LocalScribe/corpus/` (old single-corpus) |

**Root Cause:** When multi-corpus support was added (Session 29), CorpusRegistry
was created to manage corpora in `corpora/{name}/` subdirectories. However, the
CorpusManager singleton continued using the old `CORPUS_DIR` constant from config.py.

**Solution:**
1. Modified `get_corpus_manager()` to use active corpus path from registry:
   ```python
   registry = get_corpus_registry()
   active_path = registry.get_active_corpus_path()
   _corpus_manager = CorpusManager(corpus_dir=active_path)
   ```

2. Added `reset_corpus_manager()` function to invalidate singleton when active
   corpus changes

3. Updated `CorpusRegistry.set_active_corpus()` to call `reset_corpus_manager()`

**Result:** BM25 now correctly sees documents in the active corpus and is enabled
when 5+ documents are present.

---

## Session 63c: Default Questions Management — 2025-12-30

**Problem:** The default Q&A questions system had multiple issues:
1. Default questions weren't actually being executed after document processing
2. Questions could only be edited via external text file (poor UX)
3. No way to enable/disable individual questions without deleting them

### Root Cause: QA Message Routing

The `qa_progress`, `qa_result`, and `qa_complete` messages from QAWorker were being sent to `_ui_queue` but `_handle_queue_message()` had no handlers for them. Messages were silently dropped.

### Solution: Multi-Part Implementation

**1. Fixed QA Message Handlers** (`src/ui/main_window.py`):
```python
elif msg_type == "qa_progress":
    current, total, question = data
    self.set_status(f"Answering default questions: {current + 1}/{total}...")

elif msg_type == "qa_result":
    self._qa_results.append(data)
    self.output_display.update_outputs(qa_results=self._qa_results)

elif msg_type == "qa_complete":
    # Display all results and enable follow-up button
```

**2. Created DefaultQuestionsManager** (`src/core/qa/default_questions_manager.py`):
- JSON-based storage in `config/default_questions.json`
- Each question has `text` and `enabled` fields
- Methods: `get_all_questions()`, `get_enabled_questions()`, `set_enabled()`, `add_question()`, `remove_question()`, `update_question()`, `move_question()`
- Auto-migrates from legacy `qa_default_questions.txt` format
- Singleton pattern via `get_default_questions_manager()`

**3. Added CUSTOM Setting Type** (`src/ui/settings/`):
- New `SettingType.CUSTOM` enum value
- New `widget_factory` field on `SettingDefinition`
- Dialog creates custom widgets via factory function

**4. Created DefaultQuestionsWidget** (`src/ui/settings/settings_widgets.py`):
```
┌─────────────────────────────────────────────────────────┐
│  Default Questions                                   ⓘ  │
├─────────────────────────────────────────────────────────┤
│  ☑ What is this case about?                         [✕] │
│  ☑ What are the main allegations?                   [✕] │
│  ☐ Who are the plaintiffs?                          [✕] │
├─────────────────────────────────────────────────────────┤
│  [+ Add Question]                                       │
└─────────────────────────────────────────────────────────┘
```
- Checkboxes for enable/disable (saves immediately)
- Click question text to edit
- ✕ button to delete with confirmation
- "+ Add Question" button

### Files Modified/Created

**New:**
- `src/core/qa/default_questions_manager.py` — 270 lines
- `config/default_questions.json` — Data file

**Modified:**
- `src/ui/main_window.py` — QA message handlers
- `src/ui/settings/settings_dialog.py` — CUSTOM type handler
- `src/ui/settings/settings_registry.py` — Widget registration, removed old button
- `src/ui/settings/settings_widgets.py` — DefaultQuestionsWidget class
- `src/core/qa/qa_orchestrator.py` — Use new manager
- `src/core/qa/__init__.py` — New exports

### Testing

All 283 tests pass.

---

## Session 63c: GUI Checkbox Redesign — 2025-12-30

**Problem:** The main window checkboxes didn't reflect the current architecture:
1. Order was confusing (Q&A first, Vocabulary second)
2. LLM enhancement was hidden in settings with no visibility in main UI
3. Users couldn't tell if LLM would run or not

### Solution: Expanded Checkbox Layout with LLM Sub-Option

**New layout:**
```
☑ Extract Vocabulary
    ☑ Use LLM Enhancement  ← greyed out if no GPU or disabled
☑ Ask Questions
    ☑ Ask N default questions
☐ Generate Summary (slow)
```

### Implementation

**Files modified:**
- `src/ui/window_layout.py` — Reordered checkboxes, added `vocab_llm_check`
- `src/ui/main_window.py` — Added 4 new methods for LLM state management

**New methods in MainWindow:**
1. `_update_vocab_llm_checkbox_state()` — Sets checkbox state based on settings/GPU
2. `_set_vocab_llm_tooltip(text)` — Updates tooltip explaining why disabled
3. `_on_vocab_check_changed()` — Handles parent checkbox toggle
4. `_on_vocab_llm_check_changed()` — Handles LLM checkbox toggle

**Greying logic:**
| Condition | Result |
|-----------|--------|
| Vocabulary unchecked | Disabled, "Enable 'Extract Vocabulary' first" |
| Setting = "no" | Disabled, "LLM disabled in Settings" |
| Setting = "auto", no GPU | Disabled, "LLM requires dedicated GPU" |
| Setting = "auto", has GPU | Enabled and checked |
| Setting = "yes" | Enabled and checked |

### State Synchronization

- Checkbox state refreshed at startup (`__init__`)
- Checkbox state refreshed after settings dialog closes (`_open_settings`, `_open_model_settings`)
- Task execution reads checkbox state, not settings directly

### Testing

All 283 tests pass. Manual verification:
- Checkbox creation works correctly
- Enable/disable state changes work
- GPU detection integrates properly

---

## Session 63b: Name Regularization for Vocabulary Extraction — 2025-12-30

**Problem:** The NER vocabulary extraction sometimes produced fragmented or duplicated name entries:
1. "Ms. Di Leo" would produce "Di" and "Leo" as separate entries (spaCy splits multi-word names)
2. OCR typos like "Barbr Jenkins" appeared alongside correct "Barbra Jenkins"

### Solution: Post-Processing Filters

Created `src/core/vocabulary/name_regularizer.py` with two filtering strategies:

**1. Fragment Filter**
- Terms in top quartile (by count) are canonical
- Terms in bottom 75% are checked for word-level subsets
- Example: "Di" removed when "Di Leo" exists in top quartile
- Uses set intersection to detect fragments, not substring matching

**2. Typo Filter**
- Uses Levenshtein edit distance algorithm
- Removes terms with 1-character difference from canonical terms
- Example: "Barbr Jenkins" removed when "Barbra Jenkins" exists
- Minimum term length of 5 characters to avoid false positives

### Multi-Pass Architecture

The filter runs multiple passes (default 3) because removing noise allows legitimate terms to "bubble up" into the canonical set:
1. Pass 1: Remove obvious fragments/typos based on initial top quartile
2. Pass 2: Some terms move into top quartile → their fragments/typos can now be caught
3. Pass 3: Final cleanup

Early exit if no changes in a pass (optimization).

### Canonical Term Selection

```python
# Use whichever is larger: fraction-based or minimum count
fraction_index = int(len(vocab) * 0.25)  # top 25%
min_canonical_count = 10  # minimum 10 terms
max_canonical = int(len(vocab) * 0.75)  # cap at 75%
split_index = min(max(min_canonical_count, fraction_index), max_canonical)
```

This ensures:
- Small vocabularies (< 40 items) still have enough canonical terms to catch typos
- Large vocabularies don't include too many terms in the canonical set
- Always leaves at least 25% for filtering

### Integration Point

Added as Step 5b in vocabulary pipeline (after artifact_filter, before final output):
```
Step 5a: Artifact Filter → Step 5b: Name Regularization → Output
```

### Files Created/Modified

**New:**
- `src/core/vocabulary/name_regularizer.py` — 440 lines
- `tests/test_name_regularizer.py` — 31 unit tests

**Modified:**
- `src/core/vocabulary/vocabulary_extractor.py` — Integrated regularize_names()
- `src/core/vocabulary/__init__.py` — Exported new functions

### Performance

- Edit distance: O(m×n) per comparison, but optimized with length-difference pruning
- Typical vocabulary (50-200 terms): < 50ms for all passes
- Memory: Minimal (in-place filtering)

### Example Results

Input vocabulary (18 terms):
```
Wagner Doman Leto (50), Robert Wighton (45), Ms. Di Leo (35), ...
Wagner (3), Doman (2), Di (2), Leo (2), Hospital (1)
Robrt Wighton (2), Barbr Jenkins (1), Barbra Jenkinss (1)
```

After regularization (10 terms):
- Fragments removed: Wagner, Doman, Di, Leo, Hospital
- Typos removed: Robrt Wighton, Barbr Jenkins, Barbra Jenkinss
- Legitimate terms preserved: John Smith, Plaintiff

---

## Session 63: OCR Image Preprocessing for Improved Accuracy — 2025-12-30

**Problem:** The OCR system used basic Tesseract with no image preprocessing. Scanned legal documents often have issues like skew, noise, poor contrast, and scanner artifacts that significantly degrade OCR accuracy.

### Research Findings

Studies and documentation show that image preprocessing can improve OCR accuracy by 20-50%:
- [Tesseract documentation](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html) recommends rescaling, binarization, noise removal, and deskewing
- [Research papers](https://www.academia.edu/86875370/Improve_OCR_Accuracy_with_Advanced_Image_Preprocessing_using_Machine_Learning_with_Python) report CER reductions of up to 50% with preprocessing
- Adaptive thresholding performs better than global thresholding for documents with uneven lighting

### Libraries Evaluated

| Library | License | Purpose | Decision |
|---------|---------|---------|----------|
| **opencv-python-headless** | Apache 2.0 | Image preprocessing | ✅ Chosen |
| **deskew** | MIT | Skew detection/correction | ✅ Chosen |
| **scikit-image** | BSD | Additional image processing | ✅ Chosen |
| **EasyOCR** | Apache 2.0 | Alternative OCR engine | ❌ Not needed (preprocessing sufficient) |
| **OCRmyPDF** | MPL 2.0 | PDF OCR tool | ❌ CLI-focused, less flexible |
| **unpaper** | GPL | Document cleanup | ❌ Not pip-installable (C tool) |

### Implementation: 6-Stage Preprocessing Pipeline

Created `src/core/extraction/image_preprocessor.py` with:

1. **Grayscale conversion** — Simplifies image for processing
2. **Noise removal** — `cv2.fastNlMeansDenoising()` removes scanner artifacts
3. **Contrast enhancement (CLAHE)** — Improves text/background separation with adaptive histogram equalization
4. **Adaptive thresholding** — Gaussian-weighted binarization handles uneven lighting better than global Otsu
5. **Deskewing** — Detects and corrects rotational skew (critical for Tesseract line segmentation)
6. **Border padding** — Adds 10px white border (Tesseract requirement)

### Configuration

New settings in `src/config.py`:
```python
OCR_PREPROCESSING_ENABLED = True  # Enable by default
OCR_DENOISE_STRENGTH = 10         # 1-30, higher = more smoothing
OCR_ENABLE_CLAHE = True           # Contrast enhancement
```

### Files Created/Modified

**New:**
- `src/core/extraction/image_preprocessor.py` — `ImagePreprocessor` class, `preprocess_for_ocr()` function
- `tests/test_image_preprocessor.py` — 19 unit tests

**Modified:**
- `src/core/extraction/raw_text_extractor.py` — Integrated preprocessing into `_perform_ocr()`
- `src/config.py` — Added OCR preprocessing configuration
- `requirements.txt` — Added `opencv-python-headless`, `deskew`, `scikit-image`

### Why These Choices

1. **opencv-python-headless vs opencv-python:** Headless version doesn't require GUI dependencies, smaller install
2. **Adaptive thresholding:** Better than Otsu for scanned documents with shadows or uneven scanning
3. **CLAHE:** Contrast Limited Adaptive Histogram Equalization prevents over-amplification of noise
4. **Always-on:** Research shows preprocessing rarely hurts and often helps significantly; no user toggle needed

### Trade-offs

- **Added dependencies:** ~50MB additional install size
- **Processing time:** ~100-200ms per page for preprocessing (acceptable given OCR is slower)
- **Memory:** OpenCV arrays are larger than PIL images temporarily

### Sources

- [Tesseract - Improving Quality](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html)
- [OpenCV Image Thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
- [OCRmyPDF PyPI](https://pypi.org/project/ocrmypdf/)
- [deskew PyPI](https://pypi.org/project/deskew/)
- [EasyOCR vs Tesseract Comparison](https://medium.com/swlh/ocr-engine-comparison-tesseract-vs-easyocr-729be893d3ae)

---

## Session 62b: GPU Auto-Detection, Tooltip Fix & Settings Reorganization — 2025-12-29

### 1. GPU-Based LLM Auto-Detection

**Problem:** Users without GPUs experienced slow vocabulary extraction because LLM phase ran regardless. Manual toggle was buried and non-obvious.

**Solution:** Tri-state `vocab_use_llm` setting with automatic GPU detection:
- **Auto** (default): Detect GPU → use LLM if found, skip otherwise
- **Yes**: Always use LLM (user accepts slower performance)
- **No**: Never use LLM (fast NER-only extraction)

**Implementation:**
- New `src/core/utils/gpu_detector.py` with `has_dedicated_gpu()` function
- Detection methods: `nvidia-smi`, `amd-smi`/`rocm-smi`, `gpu-tracker` library fallback
- Results cached via `@lru_cache` to avoid repeated detection
- `ProgressiveExtractionWorker` checks setting before Phase 3 (LLM)

**Files:**
- `src/core/utils/gpu_detector.py` — GPU detection logic
- `src/core/user_preferences.py` — `get_vocab_llm_mode()`, `set_vocab_llm_mode()` with legacy migration
- `src/ui/workers.py:759-821` — Conditional Phase 3 execution
- `src/ui/settings/settings_registry.py` — Dropdown with GPU status in tooltip

### 2. Global Tooltip Manager (Stacking Bug Fix)

**Problem:** Tooltips weren't disappearing when moving mouse quickly between tooltip-enabled elements. Multiple tooltips would stack up and clutter the interface.

**Root Cause:** Multiple independent tooltip implementations across the codebase, each managing their own state:
- `TooltipIcon` in settings_widgets.py
- `create_tooltip()` in tooltip_helper.py
- Custom tooltips in main_window.py, system_monitor.py

**Solution:** Centralized `TooltipManager` singleton ensuring only ONE tooltip visible at a time:
```python
# Before creating any tooltip:
tooltip_manager.close_active()
# After creating:
tooltip_manager.register(tooltip_window)
# When hiding:
tooltip_manager.unregister(tooltip_window)
```

**Files:**
- `src/ui/tooltip_manager.py` — NEW: Singleton manager
- `src/ui/tooltip_helper.py` — Updated both helper functions
- `src/ui/settings/settings_widgets.py` — Updated TooltipIcon
- `src/ui/main_window.py` — Updated Ollama tooltip
- `src/ui/system_monitor.py` — Updated system monitor tooltip

### 3. Settings Menu Reorganization (6 tabs → 4 tabs)

**Problem:** Settings categories no longer reflected their contents. "Summarization" had only 1 setting. "Advanced" was entirely vocabulary-related. GPU/LLM setting was buried in "Experimental".

**Solution:** Consolidated to 4 logical tabs:

| Old Structure | New Structure |
|--------------|---------------|
| Performance (3) | **Performance (5)** — + LLM mode, + summary length |
| Summarization (1) | *Merged into Performance* |
| Vocabulary (7) | **Vocabulary (11)** — + all filtering thresholds |
| Questions (5) | **Q&A (5)** — renamed |
| Advanced (4) | *Merged into Vocabulary* |
| Experimental (2) | **Experimental (1)** — only briefing remains |

**Implementation:** Changed `category` field in each setting's registration. The declarative registry pattern made this a simple string change with no UI code modifications needed.

**Files:**
- `src/ui/settings/settings_registry.py` — Category reassignments
- `src/ui/settings/settings_dialog.py` — Window width adjustment (750→700px)

---

## Session 62: Settings Menu Improvements & LLM Filtering Fix — 2025-12-29

**Problem:** Users had no way to select Ollama models from the GUI. The "Configure" button in the header opened Settings → Questions tab, but that tab had no model selector. Additionally, LLM-extracted vocabulary terms weren't being filtered as aggressively as NER terms, resulting in many common words passing through.

### 1. Ollama Model Selector (Questions Tab)

Added a dropdown to select installed Ollama models:
- **Dynamic population** — Queries `OllamaModelManager.get_available_models()` at dialog open
- **Graceful fallbacks** — Shows helpful messages when Ollama not running or no models installed
- **Size display** — Format: `gemma3:1b (1.2 GB)`
- **Tooltip** — Advises Gemma 3 for best results, GPU recommended for 7B+ models

**Files:**
- `src/ui/settings/settings_registry.py:527-582` — `_get_ollama_model_options()`, `_set_ollama_model()`, setting definition
- `src/ui/main_window.py:237-270` — Model reload after settings close
- `src/user_preferences.py:275-277` — Validation for `ollama_model` key

### 2. Vocabulary Filtering Controls (Advanced Tab)

Added two new user-configurable settings:
- **Minimum term occurrences** (SPINBOX, 1-5, default 2) — Filter terms appearing fewer than N times
- **Phrase mean commonality** (SLIDER, 0.10-0.90, default 0.40) — Filter phrases where average word is too common

**Files:**
- `src/ui/settings/settings_registry.py:522-560` — Setting definitions
- `src/user_preferences.py:278-287` — Validation for new keys
- `src/core/vocabulary/vocabulary_extractor.py:475,631` — Read `vocab_min_occurrences` from preferences

### 3. LLM Term Filtering Bug Fix

**Problem:** LLM extracted "way more" terms than other algorithms because common words not in the Google Word Frequency dataset bypassed filtering.

**Root Cause:** In `_filter_reconciled_terms()`, the check:
```python
rank = self.frequency_rank_map.get(lower_term)
if rank is not None and rank < self.rarity_threshold:
```
Only filtered words that WERE in the dataset. Words NOT in the dataset (rank=None) passed through unfiltered, assuming they were rare/specialized. LLM extracts many domain terms not in the Google corpus.

**Fix:** Added call to `should_filter_phrase()` from `rarity_filter.py` which uses a different frequency database with better coverage:
```python
from src.core.vocabulary.rarity_filter import should_filter_phrase
if should_filter_phrase(term, is_person):
    continue  # Filter common term
```

**Files:**
- `src/core/vocabulary/vocabulary_extractor.py:624-630` — Added rarity filter check

**Why two databases?**
- Google Word Frequency: ~333K words, good for rank-based single-word filtering
- Scaled Frequencies (rarity_filter): Broader coverage, includes word commonality scores
- Using both ensures LLM terms don't slip through gaps

---

## Session 61: GUI Enhancements & Vocabulary Display Fix — 2025-12-29

**Changes Made:**

### 1. Ollama Status Indicator (Status Bar)
Added a small, less prominent Ollama connection indicator in the status bar:
- **Green dot** — Ollama connected and responding
- **Red dot + "(disconnected)"** — Ollama not reachable
- **Tooltip on disconnect** — Hover shows troubleshooting tips ("Is Ollama running?", "Check localhost:11434")

**Files:**
- `src/ui/window_layout.py:312-330` — Widget creation
- `src/ui/main_window.py` — `_update_ollama_status()`, `_setup_ollama_tooltip()`, `_clear_ollama_tooltip()`

### 2. Model Display in Header
Added model name display with parameter count in header bar:
- **Format:** `🤖 gemma3:1b (1B params)`
- **Configure button** — Opens Settings dialog directly to Questions tab
- **Auto-updates** — Refreshes when model changes

**Files:**
- `src/ui/window_layout.py:69-96` — Widget creation
- `src/ui/main_window.py` — `_update_model_display()`, `_format_model_display()`, `_open_model_settings()`
- `src/ui/settings/settings_dialog.py` — Added `initial_tab` parameter

### 3. Vocabulary Table Invisible Text Bug (Critical Fix)

**Problem:** After LLM phase completed, vocabulary table showed "1688 names & terms found" but all rows appeared blank. Only the "Found By" column showed "Both" when a row was selected.

**Root Cause:** The `combined_to_csv_data()` method in `reconciler.py` output dictionary keys that didn't match what the GUI expected:

| Reconciler Output | GUI Expected |
|------------------|--------------|
| `"Name/Term"` | `"Term"` |
| `"Type"` | `"Is Person"` |
| `"Confidence"` | `"Quality Score"` |
| `"Found By"` | `"Found By"` ✓ |

Only "Found By" matched, which is why only "Both" was visible when selecting a row.

**Fix:** Updated `combined_to_csv_data()` to output correct keys matching `GUI_DISPLAY_COLUMNS`.

**Additional fix:** Changed `found_by="Both"` to `found_by="NER, LLM"` to properly list algorithms (legacy "Both" value was from before multi-algorithm support).

**Files:**
- `src/core/vocabulary/reconciler.py:combined_to_csv_data()` — Key mapping fix
- `src/ui/dynamic_output.py` — Handle legacy "Both" as multiple algorithms
- `src/ui/theme.py` — Brightened algorithm colors for dark mode visibility

**Why this happened:** The NER-only phase and LLM phase used different code paths to format output. NER phase used `vocabulary_extractor`'s format (correct keys), while post-LLM used `reconciler.combined_to_csv_data()` (wrong keys).

---

## Hallucination Verification for Q&A Answers — 2025-12-28

**Problem:** The Q&A feature was hallucinating plausible-sounding data. When asked "What were the vital signs?", Gemma 3:1b fabricated specific values (98.6°F, 92 bpm, 130/85 mmHg) that weren't in the source documents. Small LLMs (1B params) are prone to fabrication when they don't know the answer.

**Research:** Investigated hallucination detection approaches:
- **Prompting alone:** Insufficient—model can still fabricate when confident
- **LettuceDetect:** MIT-licensed library using ModernBERT (~570MB model), achieves 79.22% F1 on RAGTruth benchmark
- **Span-level detection:** Returns per-token hallucination probabilities, enabling granular color-coding

**Decision:** Implement LettuceDetect-based verification with color-coded display:

| Hallucination Prob | Color | Tag | Display |
|-------------------|-------|-----|---------|
| < 0.30 | Green | `verify_verified` | Normal text |
| 0.30 – 0.50 | Yellow | `verify_uncertain` | Yellow text |
| 0.50 – 0.70 | Orange | `verify_suspicious` | Orange text |
| 0.70 – 0.85 | Red | `verify_unreliable` | Red text |
| >= 0.85 | Gray | `verify_hallucinated` | ~~Strikethrough~~ |

**Key features:**
1. **Overall reliability score:** Bold header showing percentage and level (HIGH/MEDIUM/LOW)
2. **Answer rejection:** Answers with < 50% reliability are replaced with rejection message
3. **Hidden citation on rejection:** Rejected answers don't show citation/source (confusing UX otherwise)
4. **Legend:** Color key displayed at bottom of Q&A panel

**Bundled model support:**
- Model downloaded to `models/lettucedect-base-modernbert-en-v1/` for installer bundling
- Uses `local_files_only=True` when bundled model exists (no network calls in production)
- Download script: `scripts/download_hallucination_model.py`

**Files created:**
- `src/core/qa/verification_config.py` — Thresholds, category helpers
- `src/core/qa/hallucination_verifier.py` — HallucinationVerifier class
- `scripts/download_hallucination_model.py` — Pre-download for installer

**Files modified:**
- `src/config.py` — `HALLUCINATION_VERIFICATION_ENABLED`, model paths
- `src/ui/theme.py` — Verification colors and text tags
- `src/core/qa/qa_orchestrator.py` — `QAResult.verification` field, `_verify_answer()`
- `src/ui/qa_panel.py` — Color-coded display, reliability header, legend
- `requirements.txt` — Added `lettucedetect>=0.1.6`

**Performance:**
- Model size: ~570MB (larger than initially expected)
- Model load: ~3-5 seconds (once per session, lazy-loaded)
- Per-answer verification: ~100-200ms
- Memory: ~500MB additional when loaded

**Trade-offs:**
- Adds ~570MB to installer size
- First Q&A takes longer (model load)
- May occasionally reject valid answers (false positives)
- Always-on by default (configurable via `HALLUCINATION_VERIFICATION_ENABLED`)

---

## GUI-Configurable Vocabulary Filtering Thresholds — 2025-12-28

**Problem:** The vocabulary filtering thresholds (Session 58) were hardcoded in `config.py`. Users couldn't adjust them without editing code.

**Decision:** Add GUI controls in Settings → **Advanced** tab to let users adjust:

| Setting | Default | Range | Description |
|---------|---------|-------|-------------|
| Single-word filtering threshold | 0.50 | 0.10-0.90 | Filter single words in top X% of English |
| Phrase filtering threshold | 0.50 | 0.10-0.90 | Filter phrases where all words are in top X% |

**Implementation:**

1. **`settings_registry.py`** — Registered 2 new `SettingType.SLIDER` definitions in "Advanced" category
2. **`user_preferences.py`** — Added validation (0.1-0.9 range) for the new keys
3. **`rarity_filter.py`** — `should_filter_phrase()` now reads from `get_user_preferences()` instead of hardcoded config constants

**Bugs fixed during implementation:**

1. **Settings dialog crashed** — `vocab_sort_method` dropdown used plain strings instead of tuples, causing `ValueError` that stopped all tabs after Vocabulary from being created
2. **Float sliders showed "0"** — `SliderSetting` widget used `int(value)` which truncated 0.50 to 0. Fixed with float-aware formatting.
3. **Dialog too narrow** — Widened from 650px to 750px to fit all 6 tabs

**User experience:**
- Settings → **Advanced** tab with dedicated rarity threshold sliders
- Sliders show proper decimal values (0.50, not 0)
- Changes saved when user clicks "Save" button
- Persists to `%APPDATA%/LocalScribe/config/user_preferences.json`

**Why Advanced tab:**
- Separates power-user settings from common vocabulary options
- Easier to find than buried at bottom of Vocabulary tab
- Room for future advanced settings

---

## Vocabulary Filtering: Rank-Based Scoring & Expanded STOPWORDS — 2025-12-28

**Problem:** Common words like "age" were appearing in vocabulary results despite filters. Investigation revealed two issues:

1. **RAKE gap:** RAKE only filtered single words against STOPWORDS (~180 words). Words like "age" not in STOPWORDS passed through unchecked.

2. **Log scaling was unintuitive:** The existing log-based commonality scoring compressed the high end—"age" (0.784) looked similar to "the" (1.0) even though "the" is 175x more common. This made threshold tuning difficult.

**Decision:** Two-part fix:

### Part 1: Centralized Single-Word Filtering
Extended `rarity_filter.py` to filter single words (previously only filtered multi-word phrases). This catches common words that individual algorithms miss.

### Part 2: Rank-Based Scoring (replacing log-based)
Changed from `log(count+1) / log(max+1)` to `rank / total_words`:

| Approach | "the" | "age" | "radiculopathy" |
|----------|-------|-------|-----------------|
| **Log-based** | 1.000 | 0.784 | 0.446 |
| **Rank-based** | 0.0000 | 0.0017 | 0.4990 |

Rank-based scoring directly answers: "What percentage of English words are more common than this?"
- 0.0 = most common word
- 0.5 = median word
- 1.0 = rarest word

**Threshold:** 0.50 (filter top 50% of English vocabulary). Court reporters know common English; they need only specialized terms.

### Part 3: Expanded STOPWORDS
Expanded from ~180 to **581 words**:
- Basic body parts: age, body, head, leg, arm, side, etc.
- Transcript fillers: okay, um, correct, yeah, etc.
- Time words: hour, morning, yesterday, etc.
- Common nouns: state, office, question, answer, etc.

**Code changes:**
- `src/config.py` — Updated thresholds for rank-based system
- `src/core/vocabulary/rarity_filter.py` — Rank-based `_load_scaled_frequencies()`, flipped comparison operators
- `src/core/utils/tokenizer.py` — Expanded STOPWORDS from 180 to 581

**Trade-offs:**
- 50% threshold is aggressive—even "radiculopathy" (rank ~166,000) is at the boundary
- Threshold easily adjustable in `config.py:SINGLE_WORD_COMMONALITY_THRESHOLD`
- Multi-word phrases like "lumbar radiculopathy" may still pass through

---

## Major Codebase Refactoring: GUI/Logic Separation — 2025-12-28

**Problem:** The codebase had accumulated technical debt:
1. Business logic scattered throughout UI components (main_window.py was 1,196 lines)
2. No clear separation between GUI and core logic
3. Duplicate code patterns (9 identical `_load_config()` implementations)
4. Root directory cluttered with temp files and misplaced modules
5. Three different chunking implementations causing confusion

**Decision:** Complete architectural refactoring with layered separation.

**Implementation:**

### New Architecture Pattern
```
src/
├── core/           # ALL business logic (15 packages)
├── services/       # Interface layer (UI → Core)
└── ui/             # User interface only
```

### Phase 0: Root Directory Cleanup
- Deleted temp files: `=0.5.0`, `nul`, `debug_flow.txt`, `*.tmp.*`
- Moved test files to `tests/`: `test_import.py`, `test_imports.py`, `test_qa_metadata.py`
- Moved debug scripts to `scripts/`: `run_debug.py`

### Phase 1: DRY - Shared Config Loader
Created `src/core/config/loader.py` with:
- `load_yaml()` - Consistent YAML loading with error handling
- `load_yaml_with_fallback()` - Returns fallback on error
- `save_yaml()` - Consistent YAML saving

Replaced 9 duplicate implementations across: `question_flow.py`, `qa_orchestrator.py`, `qa_question_editor.py`, `vocabulary_extractor.py`, `prompt_config.py`, `chunking_config.py`, etc.

### Phase 2: Chunking Consolidation
- **UnifiedChunker** designated as CANONICAL (token-aware via tiktoken)
- **ChunkingEngine** removed (Session 71) — `ProgressiveSummarizer` now uses `UnifiedChunker`
- Removed `briefing/chunker.py` (part of deprecated briefing system)

### Phase 3: Business Logic to `src/core/`
Moved all domain packages:
```
src/ai/           → src/core/ai/
src/extraction/   → src/core/extraction/
src/vocabulary/   → src/core/vocabulary/
src/chunking/     → src/core/chunking/
src/qa/           → src/core/qa/
src/retrieval/    → src/core/retrieval/
src/vector_store/ → src/core/vector_store/
src/prompting/    → src/core/prompting/
src/summarization/→ src/core/summarization/
src/preprocessing/→ src/core/preprocessing/
src/sanitization/ → src/core/sanitization/
src/parallel/     → src/core/parallel/
src/briefing/     → src/core/briefing/
src/utils/        → src/core/utils/
```

Updated all imports from `from src.xxx` to `from src.core.xxx`.
Fixed relative path calculations (`Path(__file__).parent.parent.parent` → `.parent.parent.parent.parent`).

### Phase 4: Services Layer
Created `src/services/` as interface between UI and Core:

| Service | Purpose |
|---------|---------|
| `DocumentService` | Document extraction, sanitization, preprocessing |
| `VocabularyService` | Vocabulary extraction with feedback tracking |
| `QAService` | Vector indexing, question answering |
| `SettingsService` | User preferences with convenience properties |

### Phase 5: UI Cleanup
- Removed temp files from `src/ui/`
- Prepared directory structure for future file splitting
- Updated imports to use `src.core.*` paths

### Phase 6: Documentation
- Updated ARCHITECTURE.md with new directory structure
- Added this RESEARCH_LOG entry

**Test Results:** 228 tests passing (97.9%), 5 pre-existing failures unrelated to refactoring.

**Benefits:**
1. Clear separation of concerns — UI knows nothing about business logic internals
2. Testability — Core modules can be tested without UI
3. Maintainability — Changes to logic don't require touching UI code
4. DRY — Shared utilities prevent code duplication
5. Discoverability — Logical package structure makes code easier to find

**Trade-offs:**
- Import paths are longer (`from src.core.extraction` vs `from src.extraction`)
- One-time migration effort for any external scripts

---

## UI Theme Consolidation — 2025-12-27

**Problem:** Font and color definitions were scattered across 12+ UI files as hardcoded `ctk.CTkFont()` calls and raw hex color strings. This made styling inconsistent and difficult to maintain.

**Decision:** Create a centralized theme system in `src/ui/theme.py`.

**Implementation:**
1. Created `theme.py` with:
   - `FONTS` — 16 font definitions as tuples (not CTkFont objects)
   - `COLORS` — 43 semantic color definitions
   - `BUTTON_STYLES` — 5 style presets (primary, secondary, tertiary, danger, disabled)
   - `FRAME_STYLES`, `QA_TEXT_TAGS`, `VOCAB_TABLE_TAGS`, `FILE_STATUS_TAGS`

2. Updated all UI files to import from theme:
   - `window_layout.py`, `widgets.py`, `dialogs.py`
   - `processing_timer.py`, `tooltip_helper.py`, `quadrant_builder.py`
   - `system_monitor.py`, `qa_question_editor.py`, `corpus_dialog.py`
   - `settings/settings_dialog.py`, `settings/settings_widgets.py`

**Why tuples instead of CTkFont objects:**
- CTkFont objects cause issues with `tag_config()` in CTkTextbox
- Tuple fonts `("Segoe UI", 12, "bold")` work universally
- Avoids the font scaling bug documented in the Q&A follow-up entry below

**Result:** All fonts and colors now in one location. Changes to styling require editing only `theme.py`.

---

## RESOLVED: Q&A Follow-up Font Scaling Error — 2025-12-27

**Problem:** When asking a follow-up question in the Q&A tab, users got:
```
Failed to process follow-up: 'font' option forbidden, because would be incompatible with scaling
```

**Root Cause:** CTkTextbox.tag_config() intercepts the `font` keyword argument and raises an error. The Session 55 fix was actually complete, but not verified.

**Solution (Session 55, verified Session 57):**
1. Changed `CTkFont(size=12)` to tuple `("Segoe UI", 12)` in widget constructors
2. Used `cnf={}` parameter for tag_config() to bypass CTkTextbox's font restriction:
   ```python
   # Instead of: tag_config("name", font=(...))  # FAILS
   # Use:        tag_config("name", cnf={"font": (...)})  # WORKS
   ```

**Why the workaround works:**
- CTkTextbox.tag_config() only checks `if "font" in kwargs`
- The `cnf={}` parameter puts font in a nested dict, bypassing the check
- The underlying tkinter Text widget receives and applies the font correctly

**Verification (Session 57):**
- Created test scripts that mimic the exact LocalScribe follow-up flow
- Both simple and realistic tests passed without errors
- The bug was already fixed; RESEARCH_LOG entry was outdated

---

## Full-Corpus Q&A Retrieval — 2025-12-26

**Problem:** Q&A system was hallucinating answers. User asked "What is the accident in this case?" about a slip-and-fall case, but the LLM answered with fabricated details about a "motor vehicle collision."

**Root Cause:** `QA_RETRIEVAL_K = 4` meant only 4 chunks were retrieved per question. The BM25+/FAISS algorithms scored early document chunks (jury instructions, appearances) higher than the actual case facts buried deeper in the transcript. The LLM only saw those 4 irrelevant chunks and hallucinated an answer.

**Decision:** Change to full-corpus retrieval with context window protection.

**Implementation:**
1. `QA_RETRIEVAL_K = None` — Retrieve and rank ALL chunks by relevance
2. Context window protection — Include top-ranked chunks until 80% of `QA_CONTEXT_WINDOW` (4096 tokens) is filled
3. Remaining chunks skipped with debug logging

**Code changes:**
- `src/config.py:392-394` — `QA_RETRIEVAL_K = None` with documentation
- `src/vector_store/qa_retriever.py:261-269` — Handle `K=None` by using total chunk count
- `src/vector_store/qa_retriever.py:296-356` — Token tracking and context window enforcement

**Why this approach:**
- All chunks are scored and ranked, so the most relevant content is always considered
- Context window limit prevents overwhelming the LLM (stays within 4096 tokens)
- Chunks are still ordered by relevance — best matches come first
- Performance impact is minimal (retrieval is fast; LLM time is the bottleneck)

**Trade-offs:**
- Slightly longer retrieval time (~500ms more for 100 chunks)
- Same or slightly longer LLM time (context is similar size due to window limit)
- Much better answer quality — LLM sees the most relevant content from entire corpus

---

## OCR Error and Gibberish Detection — 2025-12-23

**Question:** How to filter OCR errors and gibberish that slip through frequency-based filtering?

**Decision:** Two-layer defense:
1. **Expanded pattern filter** — Regex patterns for obvious OCR artifacts (digits in words, punctuation errors)
2. **Gibberish detector** — Spell-check based detection for nonsense strings (non-PERSON entities only)

**Research Findings:**

Libraries evaluated:
- **gibberish-detector** — Character n-gram Markov model, requires training a model file
- **nostril** — Nonsense String Evaluator, 99%+ accuracy, fast (30-50μs/string)
- **symspellpy** — Fast spell checker (1M+ times faster than standard), frequency dictionary
- **pyspellchecker** — Pure Python, built-in dictionary, simple API ✓ CHOSEN
- **OCRfixr** — BERT-based contextual spellchecker (inactive project, 558 weekly downloads)

**Why pyspellchecker:**
- Built-in English dictionary (no training required)
- Simple API: `spell.unknown()`, `spell.candidates()`
- Distinguishes gibberish from typos: gibberish has NO corrections, typos have corrections
- Lightweight, no ML model dependencies

**Implementation:**

New module `src/utils/gibberish_filter.py`:
- Word is gibberish if: unknown to dictionary AND has no spelling corrections
- Multi-word phrases: gibberish if ANY word is gibberish
- PERSON entities exempt (foreign names like "Nguyen", "Xiaoqing" would incorrectly trigger)

Expanded `OCR_ERROR_FILTER` in `src/utils/pattern_filter.py`:
```python
patterns=(
    r'^[A-Za-z]+-[A-Z][a-z]',     # Line-break artifacts: "Hos-pital"
    r'.*[0-9][A-Za-z]{2,}[0-9]',  # Digit-letter-digit: "3ohn5mith"
    r'[A-Za-z]+[0-9]+[A-Za-z]+',  # NEW: Digit(s) embedded: "Joh3n", "sp1ne"
    r'^[0-9]+[A-Za-z]+',          # NEW: Leading digit(s): "1earn", "3ohn"
    r'[A-Za-z]+[0-9]+$',          # NEW: Trailing digit(s): "learn1"
    r'[A-Za-z]+[;:][A-Za-z]+',    # NEW: Punctuation errors: "John;Smith"
)
```

**Test results:**
| Input | Type | Result |
|-------|------|--------|
| "xkjwqr" | Random letters | FILTERED (gibberish) |
| "asdfgh" | Keyboard mash | FILTERED (gibberish) |
| "Joh3n" | OCR digit error | FILTERED (pattern) |
| "cervical" | Medical term | KEPT |
| "Nguyen" | Foreign name | KEPT (in dictionary) |
| "thier" | Typo | KEPT (has corrections) |

**Code changes:**
- New: `src/utils/gibberish_filter.py` — GibberishFilter class, is_gibberish() function
- Modified: `src/utils/pattern_filter.py:93-104` — Expanded OCR_ERROR_FILTER patterns
- Modified: `src/vocabulary/vocabulary_extractor.py:273-289` — Integration in extract() step 7
- Modified: `src/vocabulary/vocabulary_extractor.py:390-407` — Integration in extract_with_llm() phase 10
- Modified: `requirements.txt` — Added pyspellchecker>=0.8.0

**Sources:**
- [pyspellchecker PyPI](https://pypi.org/project/pyspellchecker/)
- [gibberish-detector PyPI](https://pypi.org/project/gibberish-detector/)
- [Nostril GitHub](https://github.com/casics/nostril)
- [symspellpy Documentation](https://symspellpy.readthedocs.io/en/latest/)

---

## Graduated ML Weight + Source-Based Training — 2025-12-23

**Question:** Should ML have more influence on vocabulary scores once the model is trained with sufficient data?

**Decision:** Two related changes:
1. **Graduated ML Weight** — ML influence on score increases with training corpus size
2. **Source-Based Training** — User feedback weighted higher than shipped default data

**Rationale:**
The ML model already incorporates all rule-based features (quality_score, frequency, rarity, algorithm count) PLUS additional artifact detection features. Once trained, it has strictly more information than the rule-based score alone. The previous ±15 point cap underutilized the model's learning.

**Graduated ML Weight (Scoring):**
| User Samples | ML Weight | Effect |
|--------------|-----------|--------|
| < 30 | 0% | Rules only |
| 30-50 | 45% | Blend |
| 51-99 | 60% | ML-leaning |
| 100-199 | 70% | ML-dominant |
| 200+ | 85% | Near-full ML |

Formula: `score = base_score * (1 - ml_weight) + ml_prob * 100 * ml_weight`

**Source Weighting (Training):**
| User Samples | Default Weight | User Weight | Ratio |
|--------------|----------------|-------------|-------|
| < 30 | 1.0 | 1.0 | Equal |
| 30-99 | 1.0 | 1.3 | User 1.3x |
| 100+ | 1.0 | 2.0 | User 2x |

The conservative user weights (max 2x) reflect healthy skepticism — users may have idiosyncratic preferences, and default data provides stable baseline.

**Two-File Feedback System:**
- `config/default_feedback.csv` — Ships with app (developer's training data)
- `%APPDATA%/LocalScribe/data/feedback/user_feedback.csv` — User's own

**Code changes:**
- `src/config.py:68-88` — ML_WEIGHT_THRESHOLDS, ML_SOURCE_WEIGHTS, file paths
- `src/vocabulary/feedback_manager.py` — Two-file support, export_training_data with source tags
- `src/vocabulary/meta_learner.py` — get_ml_weight(), source weighting in _calculate_sample_weight, sample count storage in pickle
- `src/vocabulary/vocabulary_extractor.py:652-706` — Graduated blend formula in _apply_ml_boost
- `config/default_feedback.csv` — New placeholder file

---

## Phrase Component Rarity Filtering — 2025-12-21

**Question:** Why does RAKE return useless multi-word phrases like "the same", "left side", "read copy"?

**Decision:** Filter multi-word phrases based on the commonality of their component words, using log-scaled Google word frequency data.

**Root Cause:**
RAKE scores phrases by how often words appear together, not by whether the individual words are unusual. Phrases like "the same" score well because they co-occur frequently, but both words are extremely common and provide no vocabulary prep value.

**Implementation:**
New module `src/vocabulary/rarity_filter.py` with centralized filtering:

1. **Log-scaled frequency scoring** — Converts raw Google word counts to 0.0-1.0 range
   - Formula: `log(count + 1) / log(max_count + 1)`
   - Preserves relative frequency differences (rank-based ordering loses this)
   - 0.0 = rare/unknown, 1.0 = extremely common ("the")

2. **Component word analysis** — For multi-word phrases, calculates:
   - `min_commonality`: LOWEST score (the rarest word) — if this is high, ALL words are common
   - `mean_commonality`: Average score (catches phrases where words are generally common)

3. **Configurable thresholds** in `config.py`:
   - `PHRASE_MAX_COMMONALITY_THRESHOLD = 0.75` — Filter if RAREST word exceeds (all words common)
   - `PHRASE_MEAN_COMMONALITY_THRESHOLD = 0.65` — Filter if average exceeds

4. **Exemptions:**
   - Single words (handled by NER rarity check, stopwords)
   - Person names (names like "John Smith" use common words but are valuable)

**Why log scaling instead of rank:**
- Rank only tells position, not magnitude
- "the" vs "a" are ranks 1 and 2, but "the" appears 2x as often — rank hides this
- Log scaling compresses the extreme range while preserving relative differences

**Code changes:**
- New: `src/vocabulary/rarity_filter.py` — `filter_common_phrases()`, `calculate_phrase_component_scores()`
- `src/config.py:232-257` — Threshold constants with documentation
- `src/vocabulary/vocabulary_extractor.py:254-259` — Integration point (step 6)
- `src/vocabulary/vocabulary_extractor.py:371-376` — Integration in `extract_with_llm` (phase 9)
- Removed duplicate hardcoded `common_words` set, now uses shared `STOPWORDS`
- Updated module docstrings in NER, RAKE, BM25 to clarify filtering scope
- Added "Filtering Strategy" section to ARCHITECTURE.md

**Test results:**
| Phrase | Min (rarest) | Mean | Result |
|--------|--------------|------|--------|
| "the same" | 0.81 | 0.90 | FILTER |
| "left side" | 0.79 | 0.79 | FILTER |
| "cervical spine" | 0.62 | 0.64 | KEEP |
| "medical records" | 0.77 | 0.78 | FILTER |
| "lumbar radiculopathy" | 0.45 | 0.52 | KEEP |
| "bilateral effusion" | 0.53 | 0.58 | KEEP |

---

## Stopword Filtering for Vocabulary Extraction — 2025-12-21

**Question:** Why are common words like "same", "left", "bill", "copy", "read" appearing in vocabulary results?

**Decision:** Implement unified stopword filtering across NER and RAKE algorithms using a shared, expanded stopwords list.

**Root Cause (Two issues):**
1. **RAKE algorithm** used `rake_nltk`'s internal stopwords, which differ from our shared list. Single-word results that slipped through weren't being filtered.
2. **NER algorithm** only applied rarity checks to non-PERSON entities. spaCy sometimes tags common words like "bill" as PERSON entities (could be a name), bypassing the rarity check entirely.

**Implementation:**
- Expanded `STOPWORDS` in `src/utils/tokenizer.py` from ~109 to ~250 words
- Added categories: common verbs, nouns, adjectives, legal document terms
- RAKE: Filter single-word results against STOPWORDS in `_clean_phrase()`
- NER: Filter ALL single-word entities (including PERSON) against STOPWORDS
- NER: Stopword check added to `_is_unusual()` for non-entity tokens

**Code changes:**
- `src/utils/tokenizer.py` - Expanded STOPWORDS set
- `src/vocabulary/algorithms/rake_algorithm.py:280-283` - Single-word stopword filter
- `src/vocabulary/algorithms/ner_algorithm.py:201-208` - Entity stopword filter (all types)
- `src/vocabulary/algorithms/ner_algorithm.py:327-328` - Token stopword filter

**Why this approach:**
- Single source of truth for stopwords (shared tokenizer module)
- Filtering at extraction time is more efficient than post-processing
- Applies to both entity extraction and unusual token detection
- Doesn't affect multi-word phrases (e.g., "left shoulder" still extracted)

---

## Vocabulary Table Tag Colors for Algorithm Sources — 2025-12-21

**Question:** Why are RAKE results invisible in the vocabulary table until clicked?

**Decision:** Add color tags for all algorithm sources and update tag selection logic.

**Root Cause:** Session 52 changed "Found By" values from "Both"/"NER"/"LLM" to actual algorithm names like "RAKE", "NER, RAKE", "BM25". The tag logic only handled old values, so RAKE/BM25 results got no foreground color tag (invisible default).

**Implementation:**
- Added tags: `found_rake` (purple #8e44ad), `found_bm25` (orange #e67e22), `found_multi` (green #28a745)
- Updated tag selection to parse comma-separated algorithm names
- Multiple algorithms (2+) get green, single algorithms get their specific color

**Code changes:**
- `src/ui/dynamic_output.py:539-545` - Tag configurations
- `src/ui/dynamic_output.py:691-705` - Tag selection logic

---

## Tab Navigation Replaces Dropdown Menu — 2025-12-17

**Question:** Would tabs reduce GUI glitchiness compared to dropdown menu refreshing?

**Decision:** Replace dropdown navigation with CTkTabview tabs for persistent, visible navigation.

**Root Cause of Glitchiness:** The dropdown approach required destroying and recreating widgets on each selection change. The `_clear_dynamic_content()` method called `grid_remove()` on all children, then `_on_output_selection()` recreated them. This caused:
1. Layout recalculation on every selection
2. Treeview recreation losing scroll position and state
3. Memory allocation/deallocation churn
4. Event loop congestion during transitions

**Implementation:**
- Replaced CTkOptionMenu dropdown with CTkTabview (3 tabs: "Names & Vocab", "Q&A", "Summary")
- Each tab's content is created once and persists (no destroy/recreate cycle)
- Treeview, Q&A panel, and summary textbox are permanently placed in their respective tabs
- Tab switching is handled by CTkTabview's built-in `tkraise()` mechanism (instant, no layout recalc)
- Removed `_on_output_selection()` and `_clear_dynamic_content()` methods (~60 lines)
- Replaced `_refresh_dropdown()` with `_refresh_tabs()` that populates content instead of managing visibility
- Button bar moved below tabs (row 1) for better spatial consistency

**Code changes:**
- `src/ui/dynamic_output.py:127-177` - Tab structure creation
- `src/ui/dynamic_output.py:400-442` - New `_refresh_tabs()` method
- `src/ui/dynamic_output.py:536` - Treeview parent → Names & Vocab tab
- `src/ui/dynamic_output.py:652` - Q&A panel parent → Q&A tab
- `src/ui/dynamic_output.py:169-177` - Summary textbox → Summary tab
- `src/ui/dynamic_output.py:294, 310, 902, 962` - Changed `output_selector.get()` to `tabview.get()`
- `src/ui/queue_message_handler.py:469` - Changed `_refresh_dropdown()` to `_refresh_tabs()`
- Removed methods: `_on_output_selection()`, `_clear_dynamic_content()`
- Removed `_clear_dynamic_content()` calls from: `_display_csv()`, `_display_qa_results()`, `_display_briefing()`

**Why tabs are superior to dropdown:**
1. **Performance:** Frame stacking with `tkraise()` is O(1), no widget recreation needed
2. **State preservation:** Scroll positions, selections, and Q&A panel state persist across tab switches
3. **UX improvement:** All options always visible, no click needed to see what's available
4. **Simpler code:** Removed 2 methods and ~100 lines of visibility management logic
5. **No glitching:** Instant transitions, no layout recalculation or memory churn

**Research findings:**
- CustomTkinter's CTkTabview uses segmented button + frame stacking internally
- Tkinter's `tkraise()` changes Z-order without touching X/Y layout (extremely fast)
- Widget destruction/recreation was the primary cause of the glitchiness, not window resizing
- Tabs are the recommended pattern for multi-view interfaces in tkinter applications

**Sources:**
- [CustomTkinter CTkTabview Documentation](https://customtkinter.tomschimansky.com/documentation/widgets/tabview)
- [Tkinter Frame Stacking with tkraise()](https://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter)
- [CTk Tabview vs Dropdown Performance Comparison](https://www.reddit.com/r/learnpython/comments/11xfqzh/switching_between_frames_in_tkinter/)

---

## Alternating Row Colors for Vocabulary Table — 2025-12-17

**Question:** Can we add grid lines and alternating row colors to make the CSV table easier to read on smaller screens?

**Decision:** Add alternating row background colors (striped rows). Grid lines not needed - column separators provide vertical lines, row colors provide horizontal separation.

**Implementation:**
- Added 'oddrow' and 'evenrow' tag configurations with background colors (#f8f9fa and #ffffff)
- Modified row insertion to combine alternating row tag with existing rating/found tags
- Uses `i % 2` to determine odd vs even rows
- Tags stack: `('oddrow', 'rated_up')` combines both effects

**Code changes:**
- `src/ui/dynamic_output.py:610-612` - Tag configurations
- `src/ui/dynamic_output.py:763-779` - Tag application during insertion

**Why:** Standard Treeview pattern for improving readability. Very simple (10 lines) and doesn't interfere with existing color coding for feedback ratings or extraction sources.

**Sources:**
- [Striped Treeview Rows - Python Tkinter Tutorial](https://tkinter.com/striped-treeview-rows-python-tkinter-gui-tutorial-119/)
- [How to Alternate Row Color in Tkinter Treeview](https://morioh.com/p/5293364ed63f)
- [Different rows colours in treeview tkinter](https://python-forum.io/thread-38757.html)

---

## GUI Glitchiness During Window Resize — 2025-12-17

**Question:** Why does the GUI become glitchy for ~30 seconds when maximizing the window during vocabulary loading?

**Decision:** Implement resize event debouncing to pause batch insertion during window resize.

**Root Cause:** Event loop congestion from three competing operations:
1. Batch insertion using `after()` callbacks (50 rows every 10ms)
2. Window maximize/resize events triggering layout recalculation
3. CustomTkinter widget redrawing overhead

When the user maximized while vocabulary was loading (~156 items = 31 batches), the interleaved callbacks caused visual glitches.

**Implementation:**
- Added `_resize_after_id` and `_batch_insertion_paused` state tracking
- Bound to `<Configure>` event to detect resize/maximize
- Debounce with 100ms delay - cancels pending callbacks on each resize event
- Batch insertion checks pause flag and reschedules if paused
- Resumes automatically 100ms after resize completes

**Code changes:**
- `src/ui/dynamic_output.py:236` - Bind to Configure event
- `src/ui/dynamic_output.py:238-262` - Resize event handlers
- `src/ui/dynamic_output.py:720-722` - Pause check in batch insertion

**Why:** Standard Tkinter best practice for preventing lag during resize. Prevents expensive operations from running repeatedly during resize drag. Based on research from Tcl/Tk community and Python Tkinter forums.

**Sources:**
- [Tk event after resizing a window](https://comp.lang.tcl.narkive.com/LcOYYnom/tk-event-after-resizing-a-window)
- [Recipe 15.16. Responding to Tk Resize Events](https://www.cs.ait.ac.th/~on/O/oreilly/perl/cookbook/ch15_17.htm)
- [Tkinter window lags when resizing](https://gist.github.com/xiaodeaux/e36068d97d2ca37d38c96bcf805c642c)

---

## Q&A Panel Disappearing Bug — 2025-12-17

**Question:** Why does the Q&A panel appear initially but then disappear from the output dropdown menu?

**Decision:** Check `_qa_ready` flag instead of checking for Q&A results, AND trigger dropdown refresh when Q&A becomes ready.

**Root Cause (Two-part bug):**
1. The dropdown refresh logic in `DynamicOutputWidget._refresh_dropdown()` only showed the "Q&A" option when `qa_data` (actual Q&A results) existed. This meant the panel only appeared after the user had already asked questions and received answers, not when the system became ready.
2. When Q&A indexing completed, the `_qa_ready` flag was set but the dropdown was never refreshed to show the new option.

**Implementation:**
- Modified `src/ui/dynamic_output.py:442` to check `main_window._qa_ready` flag
- Modified `src/ui/queue_message_handler.py:469` to call `_refresh_dropdown()` when Q&A becomes ready
- Q&A option now displays as "Q&A (ready)" when vector store is built
- Changes to "Q&A (N results)" once questions are answered
- Maintains separation of concerns: widget queries MainWindow state via `winfo_toplevel()`

**Why:** Q&A system should be accessible as soon as the vector store is indexed, not after the first question is asked. This aligns with user expectations and the "qa_ready" message flow.

**Testing note:** Initial fix was incomplete - dropdown checked the flag but wasn't refreshed when the flag changed. This was caught during live testing.

---

## Trailing Digit Feature for ML Learner — 2025-12-17

**Question:** Should we add "ends with digit" detection to vocabulary ML features?

**Decision:** Yes, add `has_trailing_digit` feature to catch page/line number suffixes.

**Implementation:**
- Added `has_trailing_digit` to `FEATURE_NAMES` in `src/vocabulary/meta_learner.py`
- Feature extraction checks if `term[-1].isdigit()`
- Inserted between `has_leading_digit` and `word_count` in feature vector (now 18 total features)

**Examples caught:**
- "Smith 17" (page number suffix)
- "Di Leo 2" (transcript line artifact)
- "DeMonte 8" (page reference)

**Why:** Complements `has_leading_digit` to filter both prefix and suffix numeric artifacts. These are typically transcript formatting artifacts, not vocabulary terms the user wants to learn.

**Note:** This is a breaking change for existing trained models. Models will need retraining with the new 18-feature schema.

---

## GUI Style Centralization — 2025-12-17

**Question:** First view switch from vocabulary to Q&A causes ~30 second GUI freeze. Subsequent switches are instant.

**Decision:** Move all `ttk.Style().theme_use("default")` calls to app startup in centralized `src/ui/styles.py`.

**Root Cause:** The expensive `theme_use("default")` call was happening lazily when QAPanel or Treeview widgets were first created. This Tkinter internal call triggers layout recalculation across the entire window.

**Implementation:**
- New `src/ui/styles.py` with single `initialize_all_styles()` function
- Called once in `MainWindow.__init__()` before any UI is built
- Configures all 4 Treeview styles: Vocab, QATable, FileReview, QuestionList
- Removed redundant style code from: `dynamic_output.py`, `qa_panel.py`, `widgets.py`, `qa_question_editor.py`

**Why:** App may take 1-2 seconds longer to start (acceptable), but view switching is instant from the beginning.

---

## Ensemble Learning for Vocabulary ML — 2025-12-17

**Question:** Should we add Random Forest alongside Logistic Regression for better vocabulary preference learning?

**Decision:** Yes, with graduated training and confidence-weighted blending.

**Implementation:**
- 30+ samples: Train Logistic Regression only
- 200+ samples: Add Random Forest (23 trees for speed)
- Blending: Each model's prediction weighted by confidence (distance from 0.5)
- Example: LR=0.9 (conf 0.4), RF=0.55 (conf 0.05) → LR gets ~89% weight

**Why confidence-weighted blend over winner-takes-all:** Smoother handling of uncertain predictions. When both models are uncertain (near 0.5), their outputs blend smoothly. When one is highly confident, it dominates appropriately.

**New features added (17 total):**
- `log_count` replaces `in_case_freq` — better discrimination at low counts
- `occurrence_ratio` — document-relative frequency
- Artifact detection: `has_trailing_punctuation`, `has_leading_digit`, `word_count`, `is_all_caps`

---

## Artifact Filter for Vocabulary — 2025-12-17

**Question:** How to handle false positives like "Ms. Di Leo:" and "4 Ms. Di Leo" when "Ms. Di Leo" is correct?

**Decision:** Substring containment filter removes terms that contain high-frequency canonical terms.

**Implementation:** `src/vocabulary/artifact_filter.py`
1. Sort terms by frequency, take top N as canonical (default: top 5% or at least 10)
2. For each non-canonical term, check if any canonical term is a substring
3. If so, remove the containing term (it's likely an artifact with extra punctuation/digits)

**Example:**
- Canonical: "Ms. Di Leo" (high frequency)
- Removed: "Ms. Di Leo:" (contains canonical + trailing punctuation)
- Removed: "4 Ms. Di Leo" (contains canonical + leading digit)

**Why:** NER often picks up artifacts with line numbers or punctuation attached. Filtering by substring containment with canonical terms is simpler and more reliable than complex regex patterns.

---

## DRY Refactoring — 2025-12-16

**Question:** How to eliminate ~495 lines of duplicated code across workers, tokenization, and pattern matching?

**Decision:** Five refactoring efforts:

1. **BaseWorker Class** (`src/ui/base_worker.py`)
   - All 6 workers now extend `BaseWorker` or `CleanupWorker`
   - Provides: daemon setup, stop event, `check_cancelled()`, `send_progress()`, `send_error()`, error handling wrapper
   - `CleanupWorker` adds automatic garbage collection

2. **QueueMessage Factory** (`src/ui/queue_messages.py`)
   - Type-safe message construction: `QueueMessage.progress(50, "msg")` instead of raw tuples
   - `MessageType` constants for all 20 message types
   - Used across workers, orchestrator, message handler

3. **Shared Tokenizer** (`src/utils/tokenizer.py`)
   - Unified tokenization for BM25Algorithm, CorpusManager, BM25PlusRetriever
   - Shared `STOPWORDS` (109 words), `TOKEN_PATTERN`, `TokenizerConfig`
   - Functions: `tokenize(text, config)`, `tokenize_simple(text)`

4. **PatternFilter Utility** (`src/utils/pattern_filter.py`)
   - Centralized regex pattern matching for NER algorithm
   - Pre-built filters: ADDRESS_FILTER, LEGAL_BOILERPLATE_FILTER, VARIATION_FILTER, etc.
   - Helper functions: `matches_entity_filter()`, `matches_token_filter()`, `is_valid_acronym()`

5. **Unified BM25 Parameters** (`src/config.py`)
   - Standardized: `BM25_K1=1.5`, `BM25_B=0.75`, `BM25_DELTA=1.0`
   - Fixed inconsistency: BM25Algorithm had K1=1.2, now uses 1.5 like BM25PlusRetriever

**Why:**
- Eliminated copy-paste code that drifted out of sync (e.g., different stopword sets, K1 values)
- Single source of truth for tokenization behavior
- Consistent error handling across all workers
- Easier to add new workers or message types
- Pre-compiled regex patterns (compiled once at module load, not per-call)

**Source:** All files listed above + `src/ui/workers.py`, `src/vocabulary/algorithms/bm25_algorithm.py`, `src/vocabulary/algorithms/ner_algorithm.py`, `src/vocabulary/corpus_manager.py`, `src/retrieval/algorithms/bm25_plus.py`

---

## Person Name Deduplication — 2025-12-13

**Question:** How to handle duplicate Person names from legal transcripts? Examples: "DI LEO 1 Q", "DI LEO 2", "Diana Di Leo" (same person), and OCR variants like "Arthur Jenkins" vs "Anhur Jenkins".

**Decision:** Two-phase approach in `name_deduplicator.py`:
1. **Strip transcript artifacts first** — Remove Q/A notation, speech attribution, line numbers, honorifics using regex patterns
2. **Fuzzy match remaining variants** — Use `difflib.SequenceMatcher` with 85% threshold to group OCR/typo variants

**Why:** Simple fuzzy matching alone fails for transcript artifacts because "DI LEO 1 Q" and "DI LEO 2" have low character similarity (~60%) despite being the same person. Must strip artifacts first to expose the canonical name, then fuzzy match catches OCR errors.

**Artifact patterns handled:**
- Q/A notation: `DI LEO 1 Q`, `SMITH 2 A`
- Speech attribution: `DI LEO: Objection`
- Trailing numbers: `Di Leo 17`
- Leading numbers: `1 MR SMITH`
- Honorifics: `SMITH MR`

**Source:** `src/vocabulary/name_deduplicator.py`

---

## Chunking Architecture — 2025-12-03

**Question:** Should Case Briefing use semantic gradient chunking (`ChunkingEngine`) or section-aware chunking (`DocumentChunker`)?

**Decision:** Keep separate chunkers. `DocumentChunker` for extraction, `ChunkingEngine` for summarization.

**Why:** Legal section structure matters for extraction (PARTIES vs ALLEGATIONS have different meaning). `DocumentChunker` has 45 legal-specific regex patterns vs 8 in `ChunkingEngine`. Neither uses true embedding-based semantic splitting—both are regex-based.

**UPDATE (Session 71):** `ChunkingEngine` removed. `UnifiedChunker` is now the canonical chunker for all use cases — it uses true embedding-based semantic splitting (LangChain SemanticChunker with gradient breakpoints) plus token enforcement (400-1000 tokens).

---

## Hybrid Retrieval Weights — 2025-12-01

**Question:** How should BM25+ and FAISS scores be weighted for Q&A retrieval?

**Decision:** BM25+ weight 1.0 (primary), FAISS weight 0.5 (secondary), min_score threshold 0.1

**Why:** The embedding model (`all-MiniLM-L6-v2`) isn't trained on legal terminology, so semantic search alone returns "no information found." BM25+ provides reliable exact keyword matching for legal terms like "plaintiff," "defendant," "allegation."

**Source:** `src/config.py` — `RETRIEVAL_ALGORITHM_WEIGHTS`

---

## Gemma Duplicate JSON Keys — 2025-12-09

**Question:** Why does LLM vocabulary extraction return 0 terms despite valid model output?

**Decision:** Added merge strategy (Strategy 0) to detect and combine duplicate "terms" arrays before JSON parsing.

**Why:** Gemma 3 (1b) sometimes outputs `{"terms": [...], "terms": [...]}`. Python's `json.loads()` keeps only the LAST duplicate key, silently losing earlier data.

**Source:** `src/ai/ollama_model_manager.py` — `_parse_json_response()`

---

## Dynamic Worker Scaling — 2025-12-03

**Question:** How many parallel workers should extraction use?

**Decision:** Calculate dynamically based on CPU cores and available RAM, not hardcoded.

**Why:** Hardcoded `max_workers=2` caused 7 minutes for 7/155 chunks (~1 min/chunk). Dynamic scaling on 12-core/16GB machine → 6 workers → ~3x speedup.

**Source:** `src/system_resources.py` — `get_optimal_workers()`

---

## Few-Shot Prompting for Extraction — 2025-12-03

**Question:** How to prevent LLM from hallucinating example names from JSON schema (e.g., "John Smith" appearing in results)?

**Decision:** Use 3 few-shot examples (complaint, answer, medical records) instead of rules/instructions.

**Why:** Google's Gemma documentation says "Show patterns to follow, not anti-patterns to avoid." Research shows 10-12% accuracy improvement over zero-shot. Negative instructions ("don't hallucinate") are ineffective.

**Source:** `config/briefing_extraction_prompt.txt`

---

## Three-Tier Paragraph Splitting — 2025-12-03

**Question:** Why does `DocumentChunker` produce 1 giant chunk for some documents?

**Decision:** Implemented fallback chain: double newlines → single newlines → force split at sentence/word boundaries.

**Why:** OCR output often uses single newlines throughout. Original code only split on `\n\s*\n`, causing entire 43K-char documents to become 1 chunk (too large for LLM context window).

**Source:** `src/briefing/chunker.py` — `_split_into_paragraphs()`, `_split_on_lines()`, `_force_split_oversized()`

---

## Query Transformation — 2025-12-09

**Question:** How to handle vague user questions like "What happened?"

**Decision:** Use LlamaIndex + Ollama to expand queries into 3-4 specific search variants before retrieval.

**Why:** Vague questions don't match specific document text. Expanding "What happened to the person?" → ["injuries sustained", "allegations of liability", "damages claimed"] improves retrieval recall.

**Source:** `src/retrieval/query_transformer.py`

---

## UI Framework Choice — 2025-11-25

**Question:** Which GUI framework for Windows desktop app?

**Decision:** CustomTkinter (replaced broken PyQt6 attempt).

**Why:** CustomTkinter provides modern dark theme out of box, no licensing concerns for commercial use, simpler than Qt for our needs.

---

## AI Backend Choice — 2025-11-25

**Question:** How to run AI models locally?

**Decision:** Ollama REST API (replaced fragile ONNX Runtime attempt).

**Why:** Ollama handles model management, quantization, and GPU/CPU routing automatically. REST API is simple and reliable. Supports any GGUF model.

---

## Vector Store Choice — 2025-11-30

**Question:** ChromaDB vs FAISS for vector storage?

**Decision:** FAISS

**Why:** File-based storage (no database config needed), simpler deployment for Windows installer, well-documented Python API.

---

## NER Model Choice — 2025-11-28

**Question:** Which spaCy model for named entity recognition?

**Decision:** `en_core_web_lg` (large model)

**Why:** 4% better accuracy than medium model on legal entity extraction. Acceptable download size (~560MB). Runs efficiently on CPU.

---

## GUI Testing Strategy — 2026-01-03

**Question:** How to automate testing of the GUI workflow (load PDFs → preprocess → Process Documents)?

**Decision:** Headless worker testing with pytest markers for slow tests.

**Why:**
- Direct worker testing (no GUI rendering) is faster and more reliable than Tkinter event simulation
- Workers communicate via queues which can be monitored with timeouts to detect stuck states
- Pytest `@pytest.mark.slow` allows skipping long tests during quick runs (`pytest -m "not slow"`)
- Discovered BM25 infinite loop bug (wrong method call) via this testing approach

**Files:** `tests/test_gui_workflow.py` — TestHeadlessProgressiveExtraction, TestDiagnostics

---
