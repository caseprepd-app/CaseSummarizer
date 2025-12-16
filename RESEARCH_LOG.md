# Research Log

> **Purpose:** Technical decisions and why they were made. Check here before researching something that may already be decided.
>
> **Format:** Append new entries at the top. Never delete old entries.

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
