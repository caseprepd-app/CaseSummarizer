# LocalScribe Pipeline Architecture

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
│  STAGE 3: PREPROCESSING                     src/core/preprocessing/ │
│  - Remove title pages                                               │
│  - Remove headers/footers                                           │
│  - Remove line numbers                                              │
│  - Clean transcript notation                                        │
│  OUTPUT: content-only text                                          │
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
│  │ VOCABULARY      │ │ Q&A             │ │ SUMMARIZATION   │        │
│  │ src/core/       │ │ src/core/qa/    │ │ src/core/       │        │
│  │ vocabulary/     │ │ src/core/       │ │ summarization/  │        │
│  │                 │ │ vector_store/   │ │                 │        │
│  │ - NER           │ │ src/core/       │ │ - Map-reduce    │        │
│  │ - RAKE          │ │ retrieval/      │ │ - Focus thread  │        │
│  │ - BM25          │ │                 │ │                 │        │
│  │ - LLM enhance   │ │ - Index build   │ │                 │        │
│  │ - Filtering     │ │ - Retrieval     │ │                 │        │
│  │ - ML scoring    │ │ - Answer gen    │ │                 │        │
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
│  - QAService (stage 5b)                                             │
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
| `src/core/qa/` | stages 1-4, `src/core/ai/`, config, stdlib |
| `src/core/summarization/` | stages 1-4, `src/core/ai/`, config, stdlib |
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
src/core/ai/               # Ollama interface
src/core/prompting/        # Prompt templates
src/utils/                 # Pure utility functions
```

## How to Check Compliance

Run the violation finder script (see find_violations.py) after any changes.

A clean codebase produces: `✓ No violations found`
