# CasePrepd — Project Recommendations

> Usefulness audit conducted March 2026. Work through these one at a time.

---

## Large Changes (New Functionality)

### ~~1. Session Persistence / Case Management~~ Deprecated
CasePrepd is a lock-and-load tool: load files, process, export, done. There's nothing monolithic to persist across sessions. This was originally flagged for iterative workflows, but the app's design is throughput-oriented — each run is self-contained.

### 2. ~~Document Preview Panel~~ ✅ Implemented
Clicking a file row in the left panel shows its extracted text in a "Document" tab (first tab position) with metadata header (pages, word count, confidence, method). Prefers preprocessed text, falls back to raw. Supports Ctrl+F search. Auto-clears when file is removed.

### 3. Term-in-Context Viewer
The vocabulary table shows a term, its score, and occurrence count — but not *where* it appears. Clicking a term should show document excerpts with the term highlighted. The `# Docs` column shows a count but not which documents. This would let users make better keep/skip decisions.

### 4. Progress Estimation for Long Operations
There's a processing timer but no ETA or percentage. Processing typically takes seconds to a few minutes depending on document size. The status bar just says what step is running. Even a rough progress bar would help users gauge remaining time.

### 5. Grow the Developer Dataset
Currently **129 observations** (as of Mar 2026). The ML model activates at 30 and Random Forest joins the ensemble at 40 — both thresholds are well cleared. Continue building toward 150–200 samples across diverse case types for a stronger out-of-box experience for new users.

---

## Medium Changes (Gaps in Existing Features)

### 6. ~~Combined Export to Word/PDF (Not Just HTML)~~ ✅ Implemented
"Export All" file dialog now offers HTML, Word (.docx), and PDF formats. Routes to the existing `export_combined_to_word()` / `export_combined_to_pdf()` methods based on chosen extension. All formats include vocabulary, search results, and key excerpts.

### 7. ~~Undo/Correct Vocabulary Feedback~~ ✅ Already Working
Clicking Keep/Skip toggles the rating on; clicking the same box again sets `feedback=0` which calls `_delete_feedback_from_csv()` and removes the row entirely. The CSV always mirrors the GUI state. Rapid-click safe: toggle decision reads from in-memory cache (instant), CSV writes are under `_file_lock`, and deleting a nonexistent row is a no-op.

### 8. ~~Extractive Summary for CPU-Only Users~~ ✅ Implemented
The Key Excerpts tab extracts the most representative passages via K-means clustering on semantic embeddings — no LLM, no GPU required, runs in seconds. Each cluster centroid represents a distinct topic; the nearest passage is surfaced. This is the extractive approach described here.

### ~~9. Summary Section-Level or Quick Mode (GPU users)~~ Obsolete
This recommendation referenced the LLM summarization feature (Ollama-based, 30+ min). That feature was removed in March 2026. The Key Excerpts tab is now the summary mechanism for all users.

### 10. Default Export Folder Preference
Every export opens a file picker starting from wherever. No setting for a default export directory. Court reporters working on many cases would benefit from a configurable default folder.

### 11. ✅ Dark/Light Theme Toggle — Implemented
All colors converted to (light, dark) tuples. Theme dropdown in Settings > Appearance (Dark/Light/System). Light mode uses off-white palette with eye-catching attention colors. Live switching via reinitialize_styles().

---

## Small Changes (Polish & Bugs)

### 12. Dead Export in `__init__.py` ✅
Removed `"filter_typo_variants"` from `src/core/vocabulary/__init__.py` `__all__` — it referenced a function that didn't exist.

### 13. ExportService Return Type Annotations ✅
Fixed all 10 export methods: `-> bool` changed to `-> tuple[bool, str | None]` with matching docstrings.

### 14. ~~Score Floor Filtering TODO~~ ✅ Resolved
The TODO in `dynamic_output.py` was already gone. `MIN_LINE_LENGTH` lowered from 15 → 2 in `config.py` so short but valid transcript lines ("no", "Q. Yes?", "A. No.") are no longer dropped by the text normalizer.

### 15. Keyboard Shortcuts for Vocabulary Rating
No keyboard shortcut for keep/skip on the selected vocabulary term. Power users rating 100+ terms per case would benefit from a quick keyboard workflow (e.g., `K` for keep, `S` for skip, arrow keys to navigate).

### 16. Word Count / Document Stats ✅
After extraction, show summary statistics (total word count, page count, document lengths) in the left panel document table area. Gives users a sense of scale and helps catch extraction errors.

---

## Not Missing (Confirmed Working)
- Multi-format extraction (PDF, DOCX, TXT, RTF, OCR)
- 6 vocabulary algorithms with ML preference learning
- Hybrid retrieval (FAISS + BM25) with follow-up questions
- 5 export formats per tab (TXT, CSV, Word, PDF, HTML)
- Settings with 30+ tunable parameters
- Subprocess IPC architecture with crash recovery
- Preprocessing pipeline (7 modular steps)
- Corpus management for domain-specific BM25
