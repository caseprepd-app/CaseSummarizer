# CasePrepd - Architecture

> For WHAT and WHY, see [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).
> For technical research decisions, see [RESEARCH_LOG.md](RESEARCH_LOG.md).

## 1. Implementation Status

### Fully Implemented ✓
- [x] GUI/Logic separation — UI in `src/ui/`, business logic in `src/core/`
- [x] Document extraction — PDF (digital + OCR), TXT, RTF, DOCX, PNG/JPG
- [x] Character sanitization — 6-stage pipeline
- [x] Smart preprocessing — Headers, footers, line numbers, Q&A notation
- [x] Vocabulary extraction — Dual NER + LLM with reconciliation
- [x] Questions & Answers — Hybrid BM25+/FAISS retrieval
- [x] Hallucination verification — LettuceDetect span-level verification
- [x] Progressive summarization — Chunked map-reduce
- [x] Semantic chunking — Token enforcement via tiktoken (400-1000 tokens/chunk)
- [x] Parallel processing — Dynamic worker scaling
- [x] Settings system — 5-tab dialog with GPU auto-detection
- [x] Export to Word/PDF — Vocabulary and Q&A export

### Not Yet Built ○
- [ ] License server integration
- [ ] Model-aware prompt wrapping
- [ ] Batch processing mode

---

## 2. Layered Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  USER INPUT                                                     │
│  PDF/TXT/RTF/DOCX/PNG/JPG Files, Settings, Questions            │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  src/ui/                 UI LAYER                               │
│  MainWindow, Widgets, QAPanel, DynamicOutput                    │
│  (CustomTkinter - NO business logic)                            │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  src/services/           SERVICES LAYER                         │
│  DocumentService, VocabularyService, QAService, Settings        │
│  (Clean API - thin wrappers coordinating Core modules)          │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  src/core/               CORE LAYER                             │
│  All business logic: extraction, vocabulary, Q&A, etc.          │
└─────────────────────────────────────────────────────────────────┘
```

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| GUI/Logic Separation | UI layer has no business logic; all processing in `src/core/` |
| Services as Interface | `src/services/` provides clean API between UI and Core |
| Non-blocking UI | All heavy processing in background threads |
| Queue-based messaging | Workers communicate via `ui_queue` |

---

## 3. Processing Pipeline

```
PDF/DOCX/TXT → Extraction → Sanitization → Preprocessing → Chunking → Output
```

| Stage | Module | Purpose |
|-------|--------|---------|
| Extraction | `core/extraction/` | PDF (hybrid PyMuPDF + pdfplumber), OCR, file readers |
| Sanitization | `core/sanitization/` | 6-stage Unicode cleanup |
| Preprocessing | `core/preprocessing/` | Headers, footers, page numbers, Q&A notation |
| Chunking | `core/chunking/` | Semantic + token enforcement (400-1000 tokens) |

---

## 4. Vocabulary Extraction

### Three-Phase Flow
1. **Phase 1: Local Algorithms** (~5 sec) — spaCy NER + RAKE + BM25 corpus
2. **Phase 2: Question Indexing** — Build FAISS index (parallel)
3. **Phase 3: LLM Enhancement** (~minutes) — Ollama extraction per chunk

### Key Components

| Component | Purpose |
|-----------|---------|
| `vocabulary/vocabulary_extractor.py` | Orchestrator |
| `vocabulary/algorithms/` | NER, RAKE, BM25 implementations |
| `vocabulary/reconciler.py` | Merge NER + LLM results |
| `vocabulary/filters/` | FilterChain pipeline |
| `vocabulary/meta_learner.py` | ML preference learning from user feedback |

---

## 5. Questions & Answers

### Hybrid Retrieval
- **BM25+** (weight 1.0) — Exact term matching
- **FAISS** (weight 0.5) — Semantic similarity
- Query expansion via LlamaIndex

### Key Components

| Component | Purpose |
|-----------|---------|
| `vector_store/vector_store_builder.py` | Creates FAISS indexes |
| `retrieval/hybrid_retriever.py` | Coordinates BM25+ and FAISS |
| `qa/answer_generator.py` | Generates answers |
| `qa/hallucination_verifier.py` | LettuceDetect verification |

---

## 6. File Directory

```
src/
├── main.py                      # Entry point
├── config.py                    # Global configuration
├── core/                        # ALL BUSINESS LOGIC
│   ├── extraction/              # Document extraction
│   ├── sanitization/            # Unicode cleanup
│   ├── preprocessing/           # Text cleanup
│   ├── chunking/                # Semantic chunking
│   ├── vocabulary/              # Vocabulary extraction
│   ├── retrieval/               # Q&A retrieval algorithms
│   ├── vector_store/            # FAISS indexes
│   ├── qa/                      # Q&A orchestration
│   ├── summarization/           # Multi-doc summarization
│   ├── ai/                      # Ollama integration
│   ├── prompting/               # Prompt management
│   ├── export/                  # Word/PDF export
│   └── parallel/                # Parallel processing
├── services/                    # INTERFACE LAYER
│   ├── document_service.py
│   ├── vocabulary_service.py
│   ├── qa_service.py
│   └── settings_service.py
└── ui/                          # USER INTERFACE ONLY
    ├── main_window.py
    ├── widgets.py
    ├── dynamic_output.py
    ├── qa_panel.py
    ├── workers.py
    └── settings/
```

---

## 7. Development Setup

### Prerequisites
- Python 3.11+
- Ollama at `http://localhost:11434`
- ~4GB disk for spaCy model

### Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### Running
```bash
# Normal
python src/main.py

# Debug mode
set DEBUG=true && python src/main.py
```

### Tests
```bash
# Quick (skip slow tests)
python -m pytest tests/ -v -m "not slow"

# All tests
python -m pytest tests/ -v
```
