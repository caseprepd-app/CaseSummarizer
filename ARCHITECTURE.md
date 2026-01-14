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

## 3. How It Works (User Journey + Technical Flow)

This diagram shows what the user does and what happens behind the scenes.

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║  STEP 1: USER OPENS APP                                                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  USER SEES                          BEHIND THE SCENES                         ║
║  ─────────────────                  ───────────────────                       ║
║  Empty window with:                 • App checks Ollama connection            ║
║  • "+ Add Files" button             • Loads settings (model, corpus)          ║
║  • Task checkboxes (unchecked)      • Initializes AI service                  ║
║  • Empty results panel              • Pre-loads embedding models              ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║  STEP 2: USER SELECTS FILES  (Click "+ Add Files")                            ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  USER SEES                          BEHIND THE SCENES                         ║
║  ─────────────────                  ───────────────────                       ║
║  File picker opens                  Background worker starts:                 ║
║       ↓                                  ↓                                    ║
║  Selects complaint.pdf,             For each file (in parallel):              ║
║  answer.pdf, exhibits.docx          ├─ PDF → PyMuPDF + pdfplumber             ║
║       ↓                             ├─ Images → Tesseract OCR                 ║
║  Progress bar fills                 ├─ DOCX/TXT/RTF → Native readers          ║
║       ↓                             ├─ Calculate OCR confidence               ║
║  File table shows:                  └─ Send progress to UI                    ║
║  ┌──────────────┬───────┐                ↓                                    ║
║  │ File         │ OCR % │           Then for all text:                        ║
║  ├──────────────┼───────┤           ├─ Sanitize (fix unicode, mojibake)       ║
║  │ complaint    │ 99%   │           ├─ Remove headers/footers                 ║
║  │ answer       │ 87%   │           ├─ Remove page numbers                    ║
║  │ exhibits     │ 100%  │           └─ Clean transcript notation              ║
║  └──────────────┴───────┘                                                     ║
║                                                                               ║
║  PIPELINE: Files → Extraction → Sanitization → Preprocessing → Ready         ║
║            (core/extraction/)  (core/sanitization/) (core/preprocessing/)     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║  STEP 3: USER SELECTS TASKS  (Check boxes, click "Perform Tasks")             ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  USER CHOOSES                       DEFAULT SETTINGS                          ║
║  ─────────────                      ────────────────                          ║
║  ☑ Extract Vocabulary               ON  — Names, terms, medical/legal vocab   ║
║  ☑ Ask Questions                    ON  — 6 default questions with answers    ║
║  ☐ Generate Summary                 OFF — Takes 30+ minutes                   ║
║                                                                               ║
║  Button updates: "Perform 2 Tasks"                                            ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║  STEP 4: PROCESSING  (Three phases run, results appear progressively)         ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  PHASE 1 (~5 seconds) ─────────────────────────────────────────────────────   ║
║                                                                               ║
║  USER SEES                          BEHIND THE SCENES                         ║
║  ─────────────────                  ───────────────────                       ║
║  Status: "Running NER..."           Fast local algorithms run:                ║
║       ↓                             ├─ spaCy NER (finds names, orgs, places)  ║
║  Vocabulary table starts            ├─ RAKE (extracts key phrases)            ║
║  filling with terms:                └─ BM25 corpus (finds rare legal terms)   ║
║  • Dr. James Mitchell                    ↓                                    ║
║  • Plaintiff                        Results merged & deduplicated             ║
║  • Femur fracture                        ↓                                    ║
║  (Source: NER+RAKE)                 ~142 terms found quickly                  ║
║                                                                               ║
║  ─────────────────────────────────────────────────────────────────────────    ║
║  PHASE 2 (runs parallel with Phase 1 finishing) ──────────────────────────    ║
║                                                                               ║
║  USER SEES                          BEHIND THE SCENES                         ║
║  ─────────────────                  ───────────────────                       ║
║  Status: "Building index..."        Document chunking:                        ║
║       ↓                             ├─ Split text into 400-1000 token chunks  ║
║  "Ask Follow-up" button             ├─ ~847 chunks created                    ║
║  becomes enabled                    └─ Semantic boundaries preserved          ║
║                                          ↓                                    ║
║                                     Vector embedding:                         ║
║                                     ├─ all-MiniLM-L6-v2 model                 ║
║                                     ├─ Each chunk → 384-dim vector            ║
║                                     └─ Stored in FAISS index                  ║
║                                          ↓                                    ║
║                                     Q&A ready (user can ask questions now)    ║
║                                                                               ║
║  PIPELINE: Preprocessed text → Chunking → Embeddings → FAISS Index            ║
║            (core/chunking/)    (sentence-transformers)  (core/vector_store/)  ║
║                                                                               ║
║  ─────────────────────────────────────────────────────────────────────────    ║
║  PHASE 3 (~5-30 minutes, runs in background) ─────────────────────────────    ║
║                                                                               ║
║  USER SEES                          BEHIND THE SCENES                         ║
║  ─────────────────                  ───────────────────                       ║
║  Status: "LLM: 3/52 chunks..."      For each chunk:                           ║
║       ↓                             ├─ Ollama extracts names & terms          ║
║  Vocabulary table updates           ├─ LLM catches context-dependent terms    ║
║  as LLM finds more terms            └─ Progress sent to UI                    ║
║       ↓                                  ↓                                    ║
║  Final count: 89 terms              Reconciliation:                           ║
║  (duplicates removed,               ├─ Merge NER + LLM results                ║
║   false positives filtered)         ├─ Remove OCR errors & typos              ║
║                                     ├─ Calculate confidence scores            ║
║                                     └─ Tag source documents                   ║
║                                                                               ║
║  PIPELINE: Chunks → Ollama LLM → Reconciler → Filtered Vocabulary             ║
║            (core/ai/)   (core/vocabulary/reconciler.py)   (core/vocabulary/)  ║
║                                                                               ║
║  ─────────────────────────────────────────────────────────────────────────    ║
║  PHASE 4 (Q&A - triggered after indexing ready) ──────────────────────────    ║
║                                                                               ║
║  USER SEES                          BEHIND THE SCENES                         ║
║  ─────────────────                  ───────────────────                       ║
║  Status: "Answering: 2/6..."        For each default question:                ║
║       ↓                             ├─ Hybrid search (BM25+ + FAISS)          ║
║  Q&A panel fills with               ├─ Find relevant chunks                   ║
║  questions and answers:             ├─ Generate answer (extraction or LLM)    ║
║                                     ├─ LettuceDetect hallucination check      ║
║  Q: Who are the parties?            └─ Attach source citations                ║
║  A: Plaintiff John Smith                                                      ║
║     vs. ABC Hospital...                                                       ║
║     [Source: complaint.pdf p.1]                                               ║
║                                                                               ║
║  PIPELINE: Question → Hybrid Retrieval → Answer Gen → Hallucination Check     ║
║            (core/retrieval/)         (core/qa/)       (LettuceDetect)         ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║  STEP 5: RESULTS READY                                                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  USER SEES                          USER CAN                                  ║
║  ─────────────────                  ──────────                                ║
║                                                                               ║
║  VOCABULARY TAB                     • Click terms to see context              ║
║  ┌────────────────┬──────────┬────────────┐                                   ║
║  │ Term           │ Category │ Source     │   • Sort by confidence            ║
║  ├────────────────┼──────────┼────────────┤                                   ║
║  │ Dr. Mitchell   │ Person   │ complaint  │   • Filter by category            ║
║  │ Femur fracture │ Medical  │ answer     │                                   ║
║  │ Negligence     │ Legal    │ exhibits   │   • Give feedback (👍/👎)          ║
║  └────────────────┴──────────┴────────────┘     (trains ML scorer)            ║
║                                                                               ║
║  QUESTIONS TAB                      • Ask follow-up questions                 ║
║  ┌─────────────────────────────────────────┐                                  ║
║  │ Q: What are the key dates?              │   • View source citations        ║
║  │ A: Injury occurred on March 15, 2023... │                                  ║
║  │    [complaint.pdf, p.3]                 │                                  ║
║  └─────────────────────────────────────────┘                                  ║
║                                                                               ║
║  EXPORT OPTIONS                     • Export to Word (.docx)                  ║
║  [Export to Word] [Export to PDF]   • Export to PDF                           ║
║                                     • Combined report                         ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### Summary of Data Flow

```
FILES IN → Extract → Sanitize → Preprocess → Chunk → [3 parallel paths] → OUTPUT
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                               VOCABULARY         Q&A          SUMMARY
                               NER+RAKE+LLM    BM25+FAISS     Map-Reduce
                                    │               │               │
                                    └───────────────┼───────────────┘
                                                    ▼
                                               EXPORT
                                            Word/PDF/Display
```

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
