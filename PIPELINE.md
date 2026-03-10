# CasePrepd Pipeline Architecture

> **Purpose:** Defines the ONLY allowed data flow. If code violates this, it's a bug.

## The Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  USER INPUT                                                         │
│  PDF, DOCX, TXT, RTF, PNG, JPG                                      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: EXTRACTION                        src/core/extraction/    │
│  - PDF text (PyMuPDF + pdfplumber)                                  │
│  - OCR (pytesseract + image preprocessing)                          │
│  - File readers (TXT, RTF, DOCX)                                    │
│  OUTPUT: raw text + confidence score per document                   │
└─────────────────────────────────┬───────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: SANITIZATION                      src/core/sanitization/  │
│  - Fix mojibake                                                     │
│  - Unicode normalization                                            │
│  - Handle redactions                                                │
│  OUTPUT: clean text                                                 │
└─────────────────────────────────┬───────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: PREPROCESSING (Frontend Filtering) src/core/preprocessing/│
│  - TitlePageRemover: Remove cover pages                             │
│  - IndexPageRemover: Remove concordance/index pages                 │
│  - HeaderFooterRemover: Remove repetitive headers/footers           │
│  - LineNumberRemover: Remove margin line numbers                    │
│  - TranscriptCleaner: Remove page numbers, inline citations         │
│  - QAConverter: Convert Q./A. to readable format                    │
│  OUTPUT: "CLEAN CONTENT TEXT" - all algorithms receive this         │
└─────────────────────────────────┬───────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4: CHUNKING                          src/core/chunking/      │
│  - Semantic splitting                                               │
│  - Token enforcement (400-1000 tokens)                              │
│  OUTPUT: list of chunks                                             │
└─────────────────────────────────┬───────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 5: ANALYSIS (three parallel paths - NO CROSS-TALK)           │
│                                                                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        │
│  │ VOCABULARY      │ │ SEMANTIC SEARCH │ │ KEY SENTENCES   │        │
│  │ src/core/       │ │ src/core/qa/    │ │ src/core/       │        │
│  │ vocabulary/     │ │ src/core/       │ │ summarization/  │        │
│  │                 │ │ vector_store/   │ │                 │        │
│  │ TEXT FLOW:      │ │ src/core/       │ │ - K-means       │        │
│  │ Full text →     │ │ retrieval/      │ │   clustering    │        │
│  │ NER,RAKE,BM25   │ │                 │ │ - No LLM needed │        │
│  │                 │ │ - FAISS index   │ │                 │        │
│  │ Backend Filter: │ │ - Vector search │ │                 │        │
│  │ - Name dedup    │ │ - Cross-encoder │ │                 │        │
│  │ - Artifact filt │ │   reranking     │ │                 │        │
│  │ - Rarity filter │ │                 │ │                 │        │
│  │ - ML scoring    │ │                 │ │                 │        │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘        │
│           │                   │                   │                 │
└───────────┼───────────────────┼───────────────────┼─────────────────┘
            ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 6: EXPORT                            src/core/export/        │
│  - Format for display                                               │
│  - Word document generation                                         │
│  - PDF generation                                                   │
│  OUTPUT: formatted results                                          │
└─────────────────────────────────┬───────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SERVICES LAYER                             src/services/           │
│  - DocumentService (stages 1-4)                                     │
│  - VocabularyService (stage 5a)                                     │
│  - QAService / Semantic Search (stage 5b)                            │
│  - ExportService (stage 6)                                          │
│  PURPOSE: Only interface between UI and Core                        │
└─────────────────────────────────┬───────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  UI LAYER                                   src/ui/                 │
│  - MainWindow                                                       │
│  - Workers (background threads)                                     │
│  - Widgets                                                          │
│  PURPOSE: Display only. NO business logic.                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Import Rules

### ALLOWED imports (downstream only)

| Module | Can Import From |
|--------|-----------------|
| `src/core/extraction/` | `src/config.py`, `src/core/config/`, stdlib |
| `src/core/sanitization/` | `src/config.py`, `src/core/config/`, stdlib |
| `src/core/preprocessing/` | `src/config.py`, `src/core/config/`, stdlib |
| `src/core/chunking/` | `src/config.py`, `src/core/config/`, stdlib |
| `src/core/vocabulary/` | stages 1-4, `src/core/ai/`, config, stdlib |
| `src/core/qa/` | stages 1-4, config, stdlib |
| `src/core/summarization/` | stages 1-4, config, stdlib |
| `src/core/export/` | stages 1-5, config, stdlib |
| `src/services/` | `src/core/*`, config |
| `src/ui/` | `src/services/`, `src/config.py`, `src/ui/*` |

### FORBIDDEN imports

| Module | Cannot Import From |
|--------|-------------------|
| `src/core/vocabulary/` | `src/core/qa/`, `src/core/summarization/`, `src/ui/` |
| `src/core/qa/` | `src/core/vocabulary/`, `src/core/summarization/`, `src/ui/` |
| `src/core/summarization/` | `src/core/vocabulary/`, `src/core/qa/`, `src/ui/` |
| `src/ui/` | `src/core/*` (must use `src/services/`) |
| Any Core module | `src/ui/` |

### SHARED modules (anyone can import)

```
src/config.py              # Global settings
src/core/config/           # YAML loading utilities  
src/core/ai/               # AI utilities (embeddings, reranking)
src/utils/                 # Pure utility functions
```

## Two-Phase Filtering Model

### Phase 1: Frontend Filtering (Document-Level)

**Purpose:** Remove structural noise BEFORE any algorithm sees the text.

```
Raw Text
    ↓
[Sanitization] - Character-level cleanup
    - Mojibake fixing (ftfy)
    - Unicode normalization
    - Redaction handling
    ↓
[Preprocessing] - Structural noise removal
    - Title page removal
    - Index page removal (IndexPageRemover)
    - Header/footer removal
    - Line number removal
    - Q/A notation conversion
    ↓
"CLEAN CONTENT TEXT" ← All algorithms receive THIS
```

### Phase 2: Backend Filtering (Term-Level)

**Purpose:** Remove vocabulary artifacts AFTER extraction.

```
Extracted Terms
    ↓
[Vocabulary Filter Chain]
    - Name deduplication (fuzzy matching)
    - Artifact removal (substring containment)
    - Rarity filtering (common word removal)
    - Gibberish detection
    ↓
Final Vocabulary
```

### Text Flow Architecture

All vocabulary algorithms (NER, RAKE, BM25) receive **identical** preprocessed text.
This is enforced by design - a text hash is logged to verify consistency.

```
preprocessed_text
    ├── Vocabulary Path: Full text → NER/RAKE/BM25 (identical input)
    ├── Semantic Search: UnifiedChunks → FAISS vector index
    └── Key Sentences: UnifiedChunks → K-means clustering
```

## How to Check Compliance

Run the violation finder script (see find_violations.py) after any changes.

A clean codebase produces: `✓ No violations found`
