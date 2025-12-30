# LocalScribe - Architecture

> **Document Type:** Prescriptive (downstream) — Defines HOW the program works.
> For WHAT and WHY, see [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).
> For technical decisions, see [RESEARCH_LOG.md](RESEARCH_LOG.md).


## Quick Navigation

- [Implementation Status](#1-implementation-status)
- [High-Level Overview](#2-high-level-overview)
- [User Interface Layer](#3-user-interface-layer)
- [Services Layer](#4-services-layer) ← NEW
- [Processing Pipeline](#5-processing-pipeline)
- [Vocabulary Extraction](#6-vocabulary-extraction)
- [Questions & Answers System](#7-questions--answers-system)
- [Summarization](#8-summarization)
- [Code Patterns](#9-code-patterns)
- [File Directory](#10-file-directory)
- [Development Setup](#11-development-setup)

---

## 1. Implementation Status

### Fully Implemented ✓

- [x] **GUI/Logic Separation** — All business logic in `src/core/`, services layer in `src/services/` (Session 57)
- [x] **Document extraction** — PDF (digital + OCR), TXT, RTF via pdfplumber, pytesseract
- [x] **Character sanitization** — 6-stage pipeline (mojibake, Unicode, transliteration, redactions, control chars, whitespace)
- [x] **Smart preprocessing** — Title page, headers/footers, line numbers, Q&A notation removal
- [x] **Vocabulary extraction** — Dual NER + LLM extraction with reconciliation, "Found By" column
- [x] **Questions & Answers system** — Hybrid BM25+ / FAISS retrieval, LlamaIndex query expansion
- [x] **Hallucination verification** — LettuceDetect-based span-level verification with color-coded display (Session 60)
- [x] **Progressive summarization** — Chunked map-reduce with focus threading
- [x] **Unified semantic chunking** — Token enforcement via tiktoken (400-1200 tokens/chunk) — CANONICAL
- [x] **Parallel processing** — Dynamic worker scaling based on CPU/RAM
- [x] **Settings system** — Registry-based 4-tab dialog (Performance, Vocabulary, Q&A, Experimental) with GPU auto-detection for LLM (Session 62b)
- [x] **Progressive extraction worker** — Three-phase NER→Q&A→LLM with unified queue routing (Session 48)
- [x] **Shared config loader** — DRY utility for YAML loading (`src/core/config/loader.py`)
- [x] **Default questions management** — JSON-based storage with enable/disable per question, Settings UI widget (Session 63c)
- [x] **Name regularization** — Post-processing filter for vocabulary fragments and OCR typos (Session 63b)
- [x] **Image preprocessing** — Deskew, denoise, contrast enhancement for scanned PDFs (Session 63a)

### Partially Implemented ⚡

- [ ] **Case Briefing Generator** — Functional but being deprecated from UI

### Deprecated ⚠️

- [x] **ChunkingEngine** — Legacy chunker, use `UnifiedChunker` instead (deprecation warning added)

### Not Yet Built ○

- [ ] Document prioritization (truncate by HIGH/MEDIUM/LOW importance)
- [ ] License server integration
- [ ] Model-aware prompt wrapping (Llama vs Gemma vs Mistral formats)
- [ ] Batch processing mode
- [ ] Export to Word/PDF

---

## 2. High-Level Overview

### Layered Architecture (Session 57 Refactoring)

The codebase follows a strict **GUI/Logic Separation** pattern:

```
┌─────────────────────────────────────────────────────────────┐
│  USER INPUT                                                 │
│  PDF/TXT/RTF Files, Settings, Questions                     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  src/ui/                 UI LAYER                           │
│  MainWindow, Widgets, QAPanel, DynamicOutput                │
│  (CustomTkinter - NO business logic)                        │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  src/services/           SERVICES LAYER                     │
│  DocumentService, VocabularyService, QAService, Settings    │
│  (Clean API - thin wrappers coordinating Core modules)      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  src/core/               CORE LAYER                         │
│  All business logic: extraction, vocabulary, Q&A, etc.      │
│  (15 packages - testable without UI)                        │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```mermaid
flowchart TB
    subgraph USER["User Input"]
        Files["PDF/TXT/RTF Files"]
        Settings["Settings & Preferences"]
        Questions["Default Questions"]
    end

    subgraph UI["UI LAYER (src/ui/)"]
        MainWindow["MainWindow"]
        Widgets["FileTable, ModelSelector"]
        Output["DynamicOutput"]
        QAPanel["QAPanel"]
    end

    subgraph SERVICES["SERVICES LAYER (src/services/)"]
        DocService["DocumentService"]
        VocabService["VocabularyService"]
        QAService["QAService"]
        SettingsService["SettingsService"]
    end

    subgraph CORE["CORE LAYER (src/core/)"]
        subgraph EXTRACT["extraction/"]
            RawText["RawTextExtractor"]
        end
        subgraph SANIT["sanitization/"]
            Sanitize["CharacterSanitizer"]
        end
        subgraph PREPROC["preprocessing/"]
            Preprocess["PreprocessingPipeline"]
        end
        subgraph CHUNK["chunking/"]
            Unified["UnifiedChunker"]
        end
        subgraph VOCAB["vocabulary/"]
            VocabExtract["VocabularyExtractor"]
            Reconciler["Reconciler"]
        end
        subgraph QA["qa/ + vector_store/"]
            VectorStore["VectorStoreBuilder"]
            Retriever["HybridRetriever"]
            AnswerGen["AnswerGenerator"]
        end
    end

    Files --> UI
    UI --> SERVICES
    SERVICES --> CORE
    DocService --> EXTRACT
    DocService --> SANIT
    DocService --> PREPROC
    VocabService --> VOCAB
    QAService --> QA
    QAService --> CHUNK
```

### Core Design Principles

| Principle | Implementation |
|-----------|----------------|
| **GUI/Logic Separation** | UI layer has no business logic; all processing in `src/core/` |
| **Services as Interface** | `src/services/` provides clean API between UI and Core |
| **Non-blocking UI** | All heavy processing in background threads (workers) |
| **Queue-based messaging** | Workers communicate via `ui_queue` |
| **Pluggable algorithms** | Registry pattern for vocabulary/retrieval algorithms |
| **DRY** | Shared utilities in `src/core/config/loader.py` |
| **Graceful degradation** | Fallbacks at every stage if components fail |

---

## 3. User Interface Layer

### MainWindow Structure

```mermaid
flowchart TB
    subgraph MainWindow["MainWindow (ctk.CTk)"]
        subgraph Header
            SelectFiles["Select Files Button"]
            FileCount["Files Count Label"]
            CorpusSelect["Corpus Dropdown"]
        end

        subgraph Panels["Two-Panel Layout"]
            subgraph Left["Left Panel"]
                FileTable["FileReviewTable"]
                TaskCheckboxes["Task Checkboxes"]
                ProcessBtn["Process Button"]
            end
            subgraph Right["Right Panel"]
                OutputSelector["Output Dropdown"]
                ResultsView["Results Display"]
                FollowupInput["Follow-up Input"]
            end
        end

        subgraph StatusBar
            StatusLabel["Status"]
            Timer["Processing Timer"]
            SysMon["CPU/RAM Monitor"]
        end

        subgraph Background["Background"]
            Queue["ui_queue"]
            Handler["QueueMessageHandler"]
            Orchestrator["WorkflowOrchestrator"]
        end
    end
```

### UI Components

| Component | File | Purpose |
|-----------|------|---------|
| `MainWindow` | `ui/main_window.py` | Central coordinator, business logic |
| `WindowLayoutMixin` | `ui/window_layout.py` | Layout creation (separated from logic) |
| `FONTS, COLORS, BUTTON_STYLES` | `ui/theme.py` | Centralized theme (fonts, colors, style presets) |
| `initialize_all_styles` | `ui/styles.py` | Centralized ttk style config (prevents UI freeze) |
| `FileReviewTable` | `ui/widgets.py` | File list with status/confidence |
| `ModelSelectionWidget` | `ui/widgets.py` | Model + prompt dropdown |
| `OutputOptionsWidget` | `ui/widgets.py` | Task checkboxes, word count slider |
| `DynamicOutputWidget` | `ui/dynamic_output.py` | Tabbed results (Vocab/Questions/Summary) |
| `QAPanel` | `ui/qa_panel.py` | Questions & answers with follow-up |
| `QueueMessageHandler` | `ui/queue_message_handler.py` | Routes worker messages to UI |
| `WorkflowOrchestrator` | `ui/workflow_orchestrator.py` | Processing state machine |

### Message Types

| Message | Handler | UI Update |
|---------|---------|-----------|
| `progress` | `handle_progress()` | Progress bar + status |
| `file_processed` | `handle_file_processed()` | FileTable row update |
| `ner_complete` | `handle_ner_complete()` | Initial vocab table |
| `qa_ready` | `handle_qa_ready()` | Enable Questions panel |
| `llm_complete` | `handle_llm_complete()` | Enhanced vocab table |
| `summary_result` | `handle_summary_result()` | Summary display |
| `error` | `handle_error()` | Error dialog |

---

## 4. Services Layer

The services layer (`src/services/`) provides a clean API between the UI and Core layers. Each service is a thin wrapper that coordinates multiple core modules.

### Available Services

| Service | Module | Purpose |
|---------|--------|---------|
| `DocumentService` | `document_service.py` | Document extraction, sanitization, preprocessing |
| `VocabularyService` | `vocabulary_service.py` | Vocabulary extraction with feedback tracking |
| `QAService` | `qa_service.py` | Vector index building, question answering |
| `SettingsService` | `settings_service.py` | User preferences with convenience properties |

### Usage Example

```python
from src.services import DocumentService, VocabularyService, QAService

# Process documents
doc_service = DocumentService()
results = doc_service.process_documents(file_paths)
combined_text = doc_service.combine_texts(results)

# Extract vocabulary
vocab_service = VocabularyService()
vocab_data = vocab_service.extract_vocabulary(combined_text)

# Build Q&A index and ask questions
qa_service = QAService()
qa_service.build_index(combined_text)
answer = qa_service.ask_question("Who is the plaintiff?")
```

### Design Rationale

1. **Simplifies UI code** — UI components call one method instead of coordinating multiple modules
2. **Encapsulates complexity** — Services hide the details of module interactions
3. **Enables testing** — Services can be unit tested without UI dependencies
4. **Future-proofs the API** — Internal changes don't affect UI code if the service interface stays stable

---

## 5. Processing Pipeline

### Document Processing Stages

```mermaid
flowchart TB
    Input["User selects files"]

    subgraph Stage1["EXTRACTION"]
        PDF["PDF → pdfplumber"]
        TXT["TXT → direct read"]
        RTF["RTF → striprtf"]
        OCR["Scanned PDF → pytesseract"]
    end

    subgraph Stage2["SANITIZATION"]
        S1["1. Fix mojibake (ftfy)"]
        S2["2. Unicode normalize (NFKC)"]
        S3["3. Transliterate accents"]
        S4["4. Handle redactions"]
        S5["5. Remove control chars"]
        S6["6. Normalize whitespace"]
    end

    subgraph Stage3["PREPROCESSING"]
        P1["TitlePageRemover"]
        P2["HeaderFooterRemover"]
        P3["LineNumberRemover"]
        P4["QAConverter (Q./A. → Question:/Answer:)"]
    end

    subgraph Stage4["CHUNKING"]
        Chunker["UnifiedChunker<br/>Semantic boundaries + token limits"]
    end

    Output["Clean chunks ready for<br/>extraction, Q&A, summarization"]

    Input --> Stage1
    Stage1 --> Stage2
    S1 --> S2 --> S3 --> S4 --> S5 --> S6
    Stage2 --> Stage3
    P1 --> P2 --> P3 --> P4
    Stage3 --> Stage4
    Stage4 --> Output
```

### Extraction Details

| Stage | File | Method |
|-------|------|--------|
| PDF text | `extraction/raw_text_extractor.py` | pdfplumber digital extraction |
| OCR detection | `extraction/raw_text_extractor.py` | Dictionary confidence < 60% triggers OCR |
| OCR processing | `extraction/raw_text_extractor.py` | pdf2image + pytesseract at 300 DPI |
| **OCR preprocessing** | `extraction/image_preprocessor.py` | 6-stage image enhancement (Session 63) |
| Sanitization | `sanitization/character_sanitizer.py` | 6-stage pipeline |
| Preprocessing | `preprocessing/*.py` | Pluggable removers |

### OCR Image Preprocessing (Session 63)

When OCR is triggered for scanned documents, a 6-stage preprocessing pipeline improves accuracy by 20-50%:

```mermaid
flowchart LR
    subgraph Preprocessing["IMAGE PREPROCESSING"]
        I1["1. Grayscale"]
        I2["2. Denoise<br/>(fastNlMeans)"]
        I3["3. CLAHE<br/>(contrast)"]
        I4["4. Adaptive<br/>threshold"]
        I5["5. Deskew<br/>(rotation fix)"]
        I6["6. Border<br/>padding"]
    end

    PDF["Scanned PDF"] --> Images["pdf2image<br/>300 DPI"]
    Images --> I1
    I1 --> I2 --> I3 --> I4 --> I5 --> I6
    I6 --> Tesseract["pytesseract"]
    Tesseract --> Text["Extracted Text"]
```

**Configuration:**
```python
OCR_PREPROCESSING_ENABLED = True  # Enable by default
OCR_DENOISE_STRENGTH = 10         # 1-30, higher = more smoothing
OCR_ENABLE_CLAHE = True           # Contrast enhancement
```

**Libraries:** OpenCV (Apache 2.0), deskew (MIT), scikit-image (BSD)

### Chunking Strategy

The `UnifiedChunker` uses semantic chunking with token enforcement:

1. **Semantic splitting** — LangChain SemanticChunker with gradient breakpoints
2. **Token enforcement** — tiktoken (cl100k_base) ensures 400-1200 tokens per chunk
3. **Single pass** — Same chunks used for LLM extraction AND Q&A indexing

---

## 6. Vocabulary Extraction

### Three-Phase Progressive Architecture

```mermaid
flowchart TB
    Start["User clicks Process"]

    subgraph Phase1["PHASE 1: Local Algorithms (~5 seconds)"]
        NER["spaCy NER<br/>en_core_web_lg"]
        RAKE["RAKE Keywords"]
        BM25["BM25 Corpus<br/>(if 5+ docs)"]
        Merger["ResultMerger"]
        Phase1Result["Initial names + vocab"]
    end

    subgraph Phase2["PHASE 2: Question Indexing (parallel)"]
        Index["Build FAISS index"]
    end

    subgraph Phase3["PHASE 3: LLM Enhancement (~minutes)"]
        LLMExtract["Ollama extraction<br/>per chunk"]
        Reconcile["Reconciler<br/>merge with LLM"]
    end

    Output["Final table with<br/>'Found By' column"]

    Start --> Phase1
    NER --> Merger
    RAKE --> Merger
    BM25 --> Merger
    Merger --> Phase1Result
    Phase1Result -->|"Table appears"| Phase2
    Phase2 -->|"Questions panel activates"| Phase3
    Phase3 --> Output
```

### Algorithm Components

| Component | File | Purpose |
|-----------|------|---------|
| `VocabularyExtractor` | `vocabulary/vocabulary_extractor.py` | Orchestrator |
| `NERAlgorithm` | `vocabulary/algorithms/ner_algorithm.py` | spaCy entity extraction |
| `RAKEAlgorithm` | `vocabulary/algorithms/rake_algorithm.py` | Keyword extraction |
| `BM25Algorithm` | `vocabulary/algorithms/bm25_algorithm.py` | Corpus-based TF-IDF |
| `LLMExtractor` | `extraction/llm_extractor.py` | Ollama-based extraction |
| `Reconciler` | `vocabulary/reconciler.py` | Merge NER + RAKE + BM25 + LLM results |
| `ResultMerger` | `vocabulary/result_merger.py` | Weighted confidence combination |
| `RarityFilter` | `vocabulary/rarity_filter.py` | Filter common phrase components |
| `FeedbackManager` | `vocabulary/feedback_manager.py` | User feedback storage |
| `MetaLearner` | `vocabulary/meta_learner.py` | ML preference learning |

### Filtering Strategy

Filtering is split between algorithm-level and centralized:

**Algorithm-level (single-word filtering):**
- NER: Stopwords, rarity threshold, exclude lists, pattern matching
- RAKE: Score threshold, phrase length, stopword removal for single words
- BM25: Score threshold, stopwords, basic validation

**Centralized (multi-word phrase filtering):**
After all algorithms contribute, `rarity_filter.py` filters phrases where ALL component words are common. This catches phrases like "the same" or "left side" that score well algorithmically but provide no vocabulary value.

**Filter pipeline order:**
1. **Post-process** — Frequency thresholds, ML boost, deduplication
2. **Name deduplication** — Fuzzy matching for OCR variants
3. **Artifact filter** — Substring containment removal
4. **Phrase rarity filter** — Filter if rarest word is still common
5. **Gibberish filter** — Spell-check based nonsense detection (non-persons only)
6. **Sort** — By quality score or rarity

### Algorithm Weights

Centralized in `config.py` for tuning and future ML optimization:

```python
VOCAB_ALGORITHM_WEIGHTS = {
    "NER": 1.0,   # Primary - spaCy, most precise
    "RAKE": 0.7,  # Secondary - good for phrases
    "BM25": 0.8,  # Corpus-based (requires 5+ docs)
}
```

### Per-Algorithm Tracking

Each term tracks which algorithms detected it:
- **NER**, **RAKE**, **BM25** — Boolean columns (Yes/No)
- **Algo Count** — Sum of algorithms that found the term
- UI has "Show Details" toggle to display these columns

### GUI Display Columns

**Standard view:** Term | Score | Is Person | Found By | Keep | Skip

**Extended view (Show Details):** Adds NER, RAKE, BM25, Algo Count columns

The Score column displays the ML-adjusted Quality Score (0-100) that determines term ranking. Users see exactly what the model uses to sort terms.

### Document Confidence

Document confidence measures OCR/extraction quality (0-100%):
- Digital PDFs: Based on % of words found in English dictionary
- OCR PDFs: Based on OCR engine confidence
- TXT/RTF: Based on dictionary word percentage

This confidence propagates to vocabulary extraction as an ML feature, helping the model learn to down-weight terms from poorly-read documents.

### ML Preference Learning

User feedback (thumbs up/down) trains an ensemble model:

```
User Feedback → FeedbackManager (CSV) → VocabularyPreferenceLearner (train)
                                              ↓
New Terms → VocabularyPreferenceLearner (predict) → Quality Score blend
```

**Two-File Feedback System (Session 55):**
- `config/default_feedback.csv` — Ships with app (developer's training data)
- `%APPDATA%/LocalScribe/data/feedback/user_feedback.csv` — User's own feedback

**Graduated ML Weight (Session 55):**
ML influence on final score increases with user's training corpus:

| User Samples | ML Weight | Formula |
|--------------|-----------|---------|
| < 30 | 0% | `score = base_score` |
| 30-50 | 45% | `score = base * 0.55 + ml_prob * 100 * 0.45` |
| 51-99 | 60% | `score = base * 0.40 + ml_prob * 100 * 0.60` |
| 100-199 | 70% | `score = base * 0.30 + ml_prob * 100 * 0.70` |
| 200+ | 85% | `score = base * 0.15 + ml_prob * 100 * 0.85` |

**Source-Based Training Weights (Session 55):**
User feedback weighted higher than default data once user has enough samples:

| User Samples | Default Weight | User Weight | Ratio |
|--------------|----------------|-------------|-------|
| < 30 | 1.0 | 1.0 | Equal |
| 30-99 | 1.0 | 1.3 | User 1.3x |
| 100+ | 1.0 | 2.0 | User 2x |

**Graduated Training:**
- 30+ samples: Logistic Regression only
- 200+ samples: Ensemble mode (LR + Random Forest with 23 trees)

**Ensemble Blending:** When both models are active, predictions use confidence-weighted blending. Each model's vote is weighted by its confidence (distance from 0.5), so more certain predictions have more influence.

**Features used (15 total):**
- `quality_score` — Base quality from algorithm weights
- `log_count` — Log-transformed in-case frequency (better low-count discrimination)
- `occurrence_ratio` — Document-relative frequency
- `freq_rank_normalized` — Rank among all terms
- `num_algorithms` — Count of algorithms that found term
- `has_ner`, `has_rake`, `has_bm25` — Algorithm presence flags
- `is_person` — NER person detection (only reliable type signal)
- `has_trailing_punctuation`, `has_leading_digit`, `has_trailing_digit`, `word_count`, `is_all_caps` — Artifact detection
- `source_doc_confidence` — OCR/extraction quality of source documents (Session 54)

**Time Decay Weighting:**
Older feedback is weighted less to adapt to changing preferences:
- Decay: 1.00 → 0.82 (1yr) → 0.67 (2yr) → 0.55 (3yr)
- Floor: 55% — old feedback always retains significant weight
- Rationale: Most early feedback flags universal false positives (common words) that should persist. Career changes affecting preferences are infrequent.

**Reset options in Settings:**
- "Reset Vocabulary Model" — Restore default model, keep feedback history
- "Reset Model and Clear History" — Complete fresh start

### "Found By" Column

Results show which algorithms detected each term:
- **NER, RAKE, BM25** — Found by all three algorithms (highest confidence)
- **NER, RAKE** — Found by NER and RAKE
- **NER** — Found only by spaCy NER
- **RAKE** — Found only by RAKE keyword extraction

Terms found by more algorithms rank higher in quality score.

---

## 7. Questions & Answers System

### Hybrid Retrieval Architecture

```mermaid
flowchart LR
    Query["User Question"]

    subgraph Transform["QUERY EXPANSION"]
        QT["QueryTransformer<br/>LlamaIndex + Ollama"]
        Variants["3-4 search variants"]
    end

    subgraph Retrieve["PARALLEL RETRIEVAL"]
        BM25["BM25+ (weight 1.0)<br/>Exact term matching"]
        FAISS["FAISS (weight 0.5)<br/>Semantic similarity"]
    end

    subgraph Merge["RESULT MERGING"]
        Merger["ChunkMerger<br/>Weighted scores"]
        Bonus["Multi-algo bonus +0.1"]
    end

    Answer["AnswerGenerator<br/>Extraction or Ollama mode"]

    Query --> Transform
    Transform --> Variants
    Variants --> BM25
    Variants --> FAISS
    BM25 --> Merger
    FAISS --> Merger
    Merger --> Bonus
    Bonus --> Answer
```

### Questions & Answers Components

| Component | File | Purpose |
|-----------|------|---------|
| `VectorStoreBuilder` | `vector_store/vector_store_builder.py` | Creates FAISS indexes |
| `QARetriever` | `vector_store/qa_retriever.py` | Retrieves context |
| `QueryTransformer` | `retrieval/query_transformer.py` | Expands vague questions |
| `HybridRetriever` | `retrieval/hybrid_retriever.py` | Coordinates algorithms |
| `BM25PlusRetriever` | `retrieval/algorithms/bm25_plus.py` | Lexical search |
| `FAISSRetriever` | `retrieval/algorithms/faiss_semantic.py` | Semantic search |
| `ChunkMerger` | `retrieval/chunk_merger.py` | Combines results |
| `QAOrchestrator` | `qa/qa_orchestrator.py` | Coordinates Q&A flow |
| `AnswerGenerator` | `qa/answer_generator.py` | Generates answers |
| `HallucinationVerifier` | `qa/hallucination_verifier.py` | Verifies answers for hallucination |
| `QuestionFlowManager` | `vector_store/question_flow.py` | Branching questions |

### Retrieval Configuration

```python
RETRIEVAL_ALGORITHM_WEIGHTS = {"BM25+": 1.0, "FAISS": 0.5}
RETRIEVAL_MIN_SCORE = 0.1
RETRIEVAL_MULTI_ALGO_BONUS = 0.1
QA_RETRIEVAL_K = None           # None = search ALL chunks, or integer for top-K
QA_CONTEXT_WINDOW = 4096        # Max tokens for LLM context
```

**Full-Corpus Retrieval (Session 56):**
- `QA_RETRIEVAL_K = None` means ALL chunks are scored and ranked by relevance
- Prevents hallucination from incomplete context (e.g., only seeing document headers)
- Context window protection: Top-ranked chunks are included until 80% of context window is filled
- LLM receives the most relevant chunks from the ENTIRE document corpus, not just top-4

**Why BM25+ is primary:** The embedding model (`all-MiniLM-L6-v2`) isn't trained on legal terminology, so semantic search alone often returns "no information found."

### Answer Generation

Two modes available:
- **Extraction mode** — Fast, deterministic sentence extraction from retrieved context
- **Ollama mode** — AI-synthesized answers using local LLM (default)

**Important:** Ollama mode can hallucinate if retrieved context doesn't contain the answer. The LLM may fill gaps with plausible-sounding but incorrect information. Full-corpus retrieval mitigates this by ensuring the most relevant chunks are always included.

### Hallucination Verification (Session 60)

After answer generation, each answer is verified using LettuceDetect:

```mermaid
flowchart LR
    Answer["Generated Answer"]
    Verifier["HallucinationVerifier<br/>LettuceDetect model"]
    Spans["Span-level probabilities"]

    subgraph Display["UI Display"]
        Green["Green < 0.30<br/>Verified"]
        Yellow["Yellow 0.30-0.50<br/>Uncertain"]
        Orange["Orange 0.50-0.70<br/>Suspicious"]
        Red["Red 0.70-0.85<br/>Unreliable"]
        Strike["Strikethrough ≥ 0.85<br/>Hallucinated"]
    end

    Answer --> Verifier
    Verifier --> Spans
    Spans --> Display
```

**Verification Flow:**
1. `AnswerGenerator` produces answer from retrieved context
2. `HallucinationVerifier.verify()` analyzes answer against context
3. Returns `VerificationResult` with span-level hallucination probabilities
4. If overall reliability < 50%, answer is rejected (replaced with rejection message)
5. Rejected answers also hide citation/source (confusing UX otherwise)
6. UI displays color-coded spans + reliability header + legend

**Configuration:**
```python
HALLUCINATION_VERIFICATION_ENABLED = True  # Global toggle
ANSWER_REJECTION_THRESHOLD = 0.50          # Reject if reliability < 50%
```

**Model Bundling:**
- Model stored at `models/lettucedect-base-modernbert-en-v1/` (~570MB)
- Uses `local_files_only=True` when bundled model exists (no network calls)
- Download script: `scripts/download_hallucination_model.py`

---

## 8. Summarization

### Map-Reduce Architecture

```mermaid
flowchart TB
    Docs["Multiple Documents"]

    subgraph Map["MAP PHASE"]
        Chunk["Chunk each document"]
        ChunkSum["Summarize each chunk"]
        DocSum["Combine → Document summary"]
    end

    subgraph Reduce["REDUCE PHASE"]
        Meta["Combine all doc summaries"]
        Final["Final meta-summary"]
    end

    Docs --> Map
    Map --> Reduce
```

### Focus Threading

User's selected prompt template guides every stage:

| Stage | What's Used |
|-------|-------------|
| Focus Extraction | AI extracts emphasis + instructions from template |
| Chunk Prompts | `emphasis` string guides detail capture |
| Document Final | `emphasis` preserves focus in doc summary |
| Meta-Summary | `instructions` structure final output |

### Summarization Components

| Component | File | Purpose |
|-----------|------|---------|
| `ProgressiveSummarizer` | `progressive_summarizer.py` | Chunked summarization |
| `ChunkingEngine` | `chunking_engine.py` | Text chunking (legacy) |
| `UnifiedChunker` | `chunking/unified_chunker.py` | Semantic + token chunking |
| `DocumentSummarizer` | `summarization/document_summarizer.py` | Single doc processing |
| `MultiDocOrchestrator` | `summarization/multi_document_orchestrator.py` | Multi-doc coordination |
| `AIFocusExtractor` | `prompting/focus_extractor.py` | Extract focus from template |
| `MultiDocPromptAdapter` | `prompting/adapters.py` | Stage-specific prompts |

---

## 9. Code Patterns

### Transformation Pipeline Logging

All transformation stages use this pattern for observability:

```python
debug("Starting Stage N: [Stage Name]")
start_time = time.time()
original_len = len(text)

try:
    text = transform(text)
    duration = time.time() - start_time
    debug(f"  ✓ SUCCESS ({duration:.3f}s)")
    debug(f"    Input: {original_len} | Output: {len(text)} | Delta: {len(text) - original_len:+d}")
except Exception as e:
    duration = time.time() - start_time
    debug(f"  ✗ FAILED ({duration:.3f}s) - {type(e).__name__}: {str(e)}")
    raise
```

### Worker Thread Pattern

Background workers extend `BaseWorker` for consistent error handling and cancellation:

```python
from src.ui.base_worker import BaseWorker
from src.ui.queue_messages import QueueMessage

class SomeWorker(BaseWorker):
    def __init__(self, data, ui_queue):
        super().__init__(ui_queue)
        self.data = data

    def execute(self):
        # Check for cancellation at safe points
        self.check_cancelled()
        self.send_progress(50, "Working...")

        result = do_work(self.data)
        self.ui_queue.put(QueueMessage.some_result(result))

    def _cleanup(self):
        # Optional: cleanup resources (called in finally block)
        pass
```

`BaseWorker` provides:
- `stop()` — Signal worker to stop
- `is_stopped` — Check if stop requested
- `check_cancelled()` — Raise InterruptedError if stopped
- `send_progress(pct, msg)` — Send progress update if not stopped
- `send_error(op, exc)` — Log and send error message

For memory-intensive workers, extend `CleanupWorker` which adds automatic garbage collection.

### QueueMessage Factory Pattern

Type-safe queue message construction:

```python
from src.ui.queue_messages import QueueMessage

# Instead of raw tuples:
# ui_queue.put(('progress', (50, 'Working...')))

# Use factory methods:
ui_queue.put(QueueMessage.progress(50, 'Working...'))
ui_queue.put(QueueMessage.error('Something failed'))
ui_queue.put(QueueMessage.file_processed(result))
```

All message types are defined in `src/ui/queue_messages.py` with `MessageType` constants.

### Registry Pattern (Algorithms)

New algorithms register via decorator:

```python
@register_algorithm
class MyNewAlgorithm(BaseAlgorithm):
    name = "my_algorithm"
    weight = 0.6

    def extract(self, text: str) -> AlgorithmResult:
        # Implementation
        pass
```

---

## 10. File Directory

### Source Code (`src/`)

**Architecture Pattern:** GUI/Logic Separation
- `src/core/` - All business logic (AI, extraction, processing)
- `src/services/` - Interface layer between UI and Core
- `src/ui/` - User interface only (no business logic)

```
src/
├── main.py                      # Entry point
├── config.py                    # Global configuration
├── logging_config.py            # Debug logging setup
├── user_preferences.py          # User settings persistence
├── system_resources.py          # CPU/RAM detection for worker scaling
├── categories.py                # Category definitions
├── chunking_engine.py           # Legacy chunker (deprecated)
├── progressive_summarizer.py    # Chunked summarization
│
├── core/                        # ALL BUSINESS LOGIC
│   ├── config/                  # Shared configuration loading
│   │   └── loader.py            # load_yaml(), load_yaml_with_fallback()
│   │
│   ├── ai/                      # Ollama integration
│   │   ├── ollama_model_manager.py  # REST API client
│   │   ├── prompt_formatter.py      # Model-specific formatting
│   │   └── summary_post_processor.py # Length enforcement
│   │
│   ├── extraction/              # Document extraction
│   │   ├── raw_text_extractor.py    # PDF/TXT/RTF → text
│   │   ├── image_preprocessor.py    # OCR image preprocessing (Session 63)
│   │   └── llm_extractor.py         # Ollama-based extraction
│   │
│   ├── sanitization/
│   │   └── character_sanitizer.py   # 6-stage Unicode cleanup
│   │
│   ├── preprocessing/           # Text cleanup
│   │   ├── base.py              # BasePreprocessor ABC
│   │   ├── title_page_remover.py
│   │   ├── header_footer_remover.py
│   │   ├── line_number_remover.py
│   │   └── qa_converter.py      # Q./A. → Question:/Answer:
│   │
│   ├── chunking/                # Text chunking
│   │   └── unified_chunker.py   # Semantic + token enforcement (CANONICAL)
│   │
│   ├── vocabulary/              # Vocabulary extraction
│   │   ├── vocabulary_extractor.py  # Main orchestrator
│   │   ├── reconciler.py            # NER + LLM merge
│   │   ├── result_merger.py         # Algorithm result combination
│   │   ├── name_deduplicator.py     # Person name deduplication
│   │   ├── artifact_filter.py       # Substring containment removal
│   │   ├── name_regularizer.py      # Fragment + typo deduplication (Session 63b)
│   │   ├── rarity_filter.py         # Filter common phrases
│   │   ├── role_profiles.py         # Role detection
│   │   ├── feedback_manager.py      # User feedback CSV
│   │   ├── meta_learner.py          # ML preference learning
│   │   ├── corpus_manager.py        # BM25 corpus
│   │   ├── corpus_registry.py       # Multi-corpus management
│   │   └── algorithms/
│   │       ├── base.py              # ABC + CandidateTerm
│   │       ├── ner_algorithm.py     # spaCy NER
│   │       ├── rake_algorithm.py    # RAKE keywords
│   │       └── bm25_algorithm.py    # Corpus TF-IDF
│   │
│   ├── retrieval/               # Q&A retrieval
│   │   ├── base.py              # ABC + dataclasses
│   │   ├── hybrid_retriever.py  # Coordinates algorithms
│   │   ├── chunk_merger.py      # Weighted combination
│   │   ├── query_transformer.py # LlamaIndex expansion
│   │   └── algorithms/
│   │       ├── bm25_plus.py     # BM25+ lexical
│   │       └── faiss_semantic.py # FAISS semantic
│   │
│   ├── vector_store/            # FAISS indexes
│   │   ├── vector_store_builder.py  # Create indexes
│   │   ├── qa_retriever.py          # Retrieve context
│   │   └── question_flow.py         # Branching questions
│   │
│   ├── qa/                      # Q&A orchestration
│   │   ├── qa_orchestrator.py   # Coordinates flow
│   │   ├── answer_generator.py  # Generate answers
│   │   └── default_questions_manager.py  # JSON-based question storage (Session 63c)
│   │
│   ├── summarization/           # Multi-doc summarization
│   │   ├── result_types.py      # Dataclasses
│   │   ├── document_summarizer.py   # Single doc
│   │   └── multi_document_orchestrator.py
│   │
│   ├── prompting/               # Prompt management
│   │   ├── template_manager.py  # Load templates
│   │   ├── focus_extractor.py   # AI focus extraction
│   │   ├── adapters.py          # Stage-specific prompts
│   │   └── config.py            # Prompt parameters
│   │
│   ├── parallel/                # Parallel processing
│   │   ├── executor_strategy.py # Strategy pattern
│   │   ├── task_runner.py       # Task orchestration
│   │   └── progress_aggregator.py # Throttled updates
│   │
│   └── briefing/                # Case Briefing (deprecated)
│       ├── chunker.py, extractor.py, aggregator.py
│       ├── synthesizer.py, orchestrator.py, formatter.py
│
├── services/                    # INTERFACE LAYER (UI → Core)
│   ├── document_service.py      # Document processing
│   ├── vocabulary_service.py    # Vocabulary extraction
│   ├── qa_service.py            # Q&A operations
│   └── settings_service.py      # User preferences
│
├── ui/                          # USER INTERFACE ONLY
│   ├── main_window.py           # Main window
│   ├── window_layout.py         # Layout mixin
│   ├── theme.py                 # Centralized fonts, colors, style presets
│   ├── styles.py                # Centralized ttk style config
│   ├── widgets.py               # FileTable, ModelSelector, etc.
│   ├── dynamic_output.py        # Results display
│   ├── qa_panel.py              # Q&A panel
│   ├── qa_question_editor.py    # Edit questions dialog
│   ├── corpus_dialog.py         # Corpus management
│   ├── base_worker.py           # BaseWorker/CleanupWorker classes
│   ├── queue_messages.py        # QueueMessage factory + MessageType
│   ├── workers.py               # Background workers (extend BaseWorker)
│   ├── workflow_orchestrator.py # State machine
│   ├── queue_message_handler.py # Message routing
│   ├── processing_timer.py      # Elapsed time
│   ├── system_monitor.py        # CPU/RAM display
│   ├── tooltip_helper.py        # Tooltips
│   └── settings/
│       ├── settings_dialog.py   # Tabbed dialog
│       ├── settings_registry.py # Setting definitions
│       └── settings_widgets.py  # Custom widgets
│
└── utils/
    ├── logger.py                # Backward-compat wrapper
    ├── text_utils.py            # Text utilities
    ├── tokenizer.py             # Shared BM25 tokenization
    ├── pattern_filter.py        # Regex pattern matching for NER/OCR errors
    └── gibberish_filter.py      # Spell-check based gibberish detection
```

### Configuration (`config/`)

```
config/
├── prompts/                     # Summarization templates
│   └── phi-3-mini/
│       ├── factual-summary.txt
│       └── strategic-analysis.txt
├── extraction_prompts/          # LLM extraction prompts
│   ├── combined_extraction.txt  # People + vocab
│   └── vocab_extraction.txt
├── briefing_extraction_prompt.txt # Case Briefing (few-shot)
├── qa_questions.yaml            # Default Q&A questions
├── chunking_config.yaml         # Chunking parameters
├── common_medical_legal.txt     # Vocabulary blacklist
├── medical_terms.txt            # Medical term whitelist
└── legal_exclude.txt            # Legal term blacklist
```

### Tests (`tests/`)

```
tests/
├── test_raw_text_extractor.py
├── test_character_sanitizer.py
├── test_preprocessing.py
├── test_vocabulary_extractor.py
├── test_feedback_ml.py
├── test_bm25_algorithm.py
├── test_hybrid_retrieval.py
├── test_qa_orchestrator.py
├── test_multi_document_summarization.py
├── test_prompt_adapters.py
├── test_progressive_summarizer.py
├── test_parallel.py
└── manual/                      # Require Ollama running
    ├── README.md
    └── test_ollama_workflow.py
```

### User Data (`%APPDATA%/LocalScribe/`)

```
%APPDATA%/LocalScribe/
├── settings.json                # User preferences
├── prompts/                     # Custom prompt templates
├── corpus/                      # BM25 reference corpus
├── vector_stores/               # FAISS indexes (per-session)
├── data/feedback/               # ML feedback CSVs
└── logs/                        # Debug logs
```

---

## 11. Development Setup

### Prerequisites

- Python 3.11+
- Ollama running locally at `http://localhost:11434`
- ~4GB disk space for spaCy model and dependencies

### Installation

```bash
# Clone and enter directory
cd CaseSummarizer

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Unix)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg
```

### Running the Application

```bash
# Normal mode
python src/main.py

# Debug mode (verbose logging)
set DEBUG=true && python src/main.py   # Windows CMD
$env:DEBUG="true"; python src/main.py  # Windows PowerShell
DEBUG=true python src/main.py          # Unix
```

### Running Tests

```bash
# All automated tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_vocabulary_extractor.py -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=html

# Manual tests (require Ollama)
python tests/manual/test_ollama_workflow.py
```

### Linting

```bash
# Run ruff
ruff check src/

# Auto-fix
ruff check src/ --fix
```

---

*Last updated: 2025-12-30 (Session 63)*
