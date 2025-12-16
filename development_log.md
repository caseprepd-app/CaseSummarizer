# Development Log

> Timestamped log of significant changes to the codebase.

---

## 2025-12-16 — DRY Refactoring Complete

### New Files Created
- `src/ui/base_worker.py` — BaseWorker and CleanupWorker classes providing common boilerplate for all background workers (daemon setup, stop events, error handling, progress reporting)
- `src/ui/queue_messages.py` — QueueMessage factory and MessageType constants for type-safe inter-thread communication
- `src/utils/tokenizer.py` — Shared tokenization for all BM25/vocabulary operations with unified stopwords (109 words) and configurable settings
- `src/utils/pattern_filter.py` — PatternFilter class and pre-built filters for NER entity/token validation

### Files Modified
- `src/ui/workers.py` — Refactored all 6 workers (ProcessingWorker, VocabularyWorker, QAWorker, MultiDocSummaryWorker, ProgressiveExtractionWorker, BriefingWorker) to extend BaseWorker/CleanupWorker
- `src/vocabulary/algorithms/bm25_algorithm.py` — Now uses shared tokenizer and unified BM25 parameters (K1=1.5)
- `src/vocabulary/algorithms/ner_algorithm.py` — Now uses PatternFilter utilities instead of inline pattern lists
- `src/vocabulary/corpus_manager.py` — Now uses shared tokenizer
- `src/retrieval/algorithms/bm25_plus.py` — Now uses shared tokenizer
- `src/config.py` — Added BM25_K1, BM25_B, BM25_DELTA constants

### Bug Fix
- Fixed BM25 K1 inconsistency: BM25Algorithm had K1=1.2 while BM25PlusRetriever had K1=1.5. Now unified at K1=1.5 via config constants.

### Tests
- 223 tests passed after refactoring
- 1 pre-existing test failure (test_feature_extraction) unrelated to DRY changes

---
