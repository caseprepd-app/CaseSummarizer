# CasePrepd — Project Recommendations

> Usefulness audit conducted March 2026. Work through these one at a time.

---

## Large Changes (New Functionality)

### 1. Session Persistence / Case Management
Every session is ephemeral. You load files, process, export, close — everything vanishes. There's no way to:
- Save a case and reopen it later
- See a history of cases you've processed
- Iterate on a case across multiple sessions (e.g., add a new exhibit to an existing case)

This is probably the single biggest gap for a working court reporter.

### 2. Document Preview Panel
After extraction, there's no way to view the cleaned/preprocessed text. Users can see OCR confidence scores in the file table, but can't inspect what the app actually extracted. This matters because:
- Users can't verify preprocessing worked correctly (headers removed, Q/A converted, etc.)
- When vocabulary terms look wrong, there's no way to check the source text
- OCR quality issues are invisible beyond the confidence number

### 3. Term-in-Context Viewer
The vocabulary table shows a term, its score, and occurrence count — but not *where* it appears. Clicking a term should show document excerpts with the term highlighted. The `# Docs` column shows a count but not which documents. This would let users make better keep/skip decisions.

### 4. Progress Estimation for Long Operations
There's a processing timer but no ETA or percentage. Summarization takes 30+ minutes. The status bar just says what step is running. Users have no idea if it'll be 5 more minutes or 50. Even a rough chunk-based progress bar (e.g., "Summarizing chunk 12 of 47") would help.

### 5. Grow the Developer Dataset
Only **34 observations** (14 good, 20 bad). The ML model needs 30 minimum to activate, so the baseline is barely above threshold. The ensemble needs 40 for Random Forest weight. A thin developer dataset means:
- Out-of-box vocabulary filtering is unreliable
- New users get a poor first impression before they've rated enough terms
- Recommend building this to 150–200 samples across diverse case types

---

## Medium Changes (Gaps in Existing Features)

### 6. Combined Export to Word/PDF (Not Just HTML)
"Export All" only produces HTML. The `export_combined_to_word()` and `export_combined_to_pdf()` methods exist in ExportService but aren't wired to any UI button. Court reporters handing off prep documents probably want Word or PDF, not HTML.

### 7. Undo/Correct Vocabulary Feedback
If a user accidentally thumbs-down a good term, there's no undo. They'd have to manually edit the feedback CSV. A simple "undo last rating" or the ability to re-rate a term from the context menu would save frustration.

### 8. Summary Section-Level or Quick Mode
Summary is all-or-nothing: off (0 minutes) or on (30+ minutes). Options to consider:
- Quick summary mode (single pass, shorter, ~5 minutes)
- Per-document summaries vs. multi-document synthesis
- Partial results shown as they're generated (progressive display is in the engine but may not be surfaced in the UI)

### 9. Default Export Folder Preference
Every export opens a file picker starting from wherever. No setting for a default export directory. Court reporters working on many cases would benefit from a configurable default folder.

### 10. Dark/Light Theme Toggle
Currently dark mode only, hardcoded. Some users (especially those working in bright office environments) prefer light mode. CustomTkinter supports both — this is just a settings toggle and theme constant swap.

---

## Small Changes (Polish & Bugs)

### 11. Dead Export in `__init__.py` ✅
Removed `"filter_typo_variants"` from `src/core/vocabulary/__init__.py` `__all__` — it referenced a function that didn't exist.

### 12. ExportService Return Type Annotations ✅
Fixed all 10 export methods: `-> bool` changed to `-> tuple[bool, str | None]` with matching docstrings.

### 13. Score Floor Filtering TODO
`src/ui/dynamic_output.py` has `# TODO: Test score floor filtering for both GUI display and CSV export`. This is the only TODO in the codebase — needs verification and the TODO removed.

### 14. Keyboard Shortcuts for Vocabulary Rating
No keyboard shortcut for keep/skip on the selected vocabulary term. Power users rating 100+ terms per case would benefit from a quick keyboard workflow (e.g., `K` for keep, `S` for skip, arrow keys to navigate).

### 15. Word Count / Document Stats ✅
After extraction, show summary statistics (total word count, page count, document lengths) in the left panel document table area. Gives users a sense of scale and helps catch extraction errors.

---

## Not Missing (Confirmed Working)
- Multi-format extraction (PDF, DOCX, TXT, RTF, OCR)
- 8 vocabulary algorithms with ML preference learning
- Hybrid Q&A retrieval (FAISS + BM25) with follow-up questions
- 5 export formats per tab (TXT, CSV, Word, PDF, HTML)
- Settings with 30+ tunable parameters
- Subprocess IPC architecture with crash recovery
- Preprocessing pipeline (7 modular steps)
- Corpus management for domain-specific BM25
