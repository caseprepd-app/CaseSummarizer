# DRY Violation Audit — 2026-03-13

## Violation 1: ExtractionResult dict (21 locations)

**Problem:** Every extraction function hand-builds the same dict:
`{text, method, confidence, status, error_message, page_count}`.
Success dicts and error dicts are copy-pasted with only values changed.

**New class:** `ExtractionResult` dataclass in `src/core/extraction/extraction_result.py`
with `ExtractionResult.success(...)` and `ExtractionResult.error(...)` factory methods,
plus dict-style `__getitem__`/`get` for backward compatibility.

**Files affected:**
- `src/core/extraction/file_readers.py` — 8 dict constructions (4 success, 4 error)
- `src/core/extraction/ocr_processor.py` — 6 dict constructions (3 success, 3 error)
- `src/core/extraction/raw_text_extractor.py` — 7 dict constructions in `_extract_by_type` and `_process_pdf_inner`

---

## Violation 2: Thread-safe singleton boilerplate (3 locations)

**Problem:** Three modules repeat identical double-checked locking:
`_instance = None`, `_lock = Lock()`, get-with-lock, reset-with-lock.

**New class:** `SingletonHolder` in `src/services/singleton.py`.
Wraps a factory callable, provides `get(*args)` and `reset()`.

**Files affected:**
- `src/user_preferences.py:554-593` — `_user_prefs` / `get_user_preferences` / `reset_singleton`
- `src/services/ai_service.py:19-39,91-101` — `_ai_service_instance` / `__new__` / `reset_singleton`
- `src/services/export_service.py:422-442` — `_export_service` / `get_export_service` / `reset_export_service`

---

## Violation 3: Settings header-with-tooltip widget (7 locations, 5 files)

**Problem:** Every settings widget creates the same 3-widget cluster:
transparent `CTkFrame` → `CTkLabel` (title) → `TooltipIcon`. Each copy is
6-12 lines of identical structure with only the title text, tooltip text,
and font varying.

**New class:** `SectionHeader` widget in `src/ui/settings/section_header.py`.
Takes `title`, `tooltip_text`, and optional `font` keyword.

**Files affected:**
- `src/ui/settings/columns_widget.py:84-101`
- `src/ui/settings/questions_widget.py:63-81`
- `src/ui/settings/patterns_widget.py:57-70`
- `src/ui/settings/indicator_pattern_widget.py:136-147, 174-185`
- `src/ui/settings/corpus_widget.py:65-84`
