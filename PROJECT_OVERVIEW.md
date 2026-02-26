# CasePrepd - Project Overview

> For technical implementation details, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Documentation

| File | Purpose |
|------|---------|
| **PROJECT_OVERVIEW.md** (this file) | What we're building and why |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | How it's built — diagrams, components, setup |
| **[RESEARCH_LOG.md](RESEARCH_LOG.md)** | Technical research decisions with sources |

---

## 1. Mission

Build a **100% offline, private, commercial Windows desktop application** for court reporters. The app solves the PII/PHI liability problem by ensuring sensitive legal documents *never leave the user's computer*.

**Key Principle:** Process MULTIPLE documents (complaint, answer, exhibits, motions, etc.) to generate ONE comprehensive case-level output—not individual document summaries.

---

## 2. What Court Reporters Actually Need

### Primary Outputs (Default ON)
1. **Names of people involved** — parties, witnesses, doctors, attorneys with their roles
2. **Technical vocabulary** — medical terms, legal terminology, unusual words for prep

### Secondary Output (Default ON)
3. **Ask Questions of Your Documents** — Ask questions, get answers with source citations. Results serve as handoff documents for colleagues.

### Optional Output (Default OFF)
4. **AI-generated summaries** — Comprehensive case synthesis. Off by default due to 30+ minute processing time.

---

## 3. Business Constraints

| Constraint | Rationale |
|------------|-----------|
| **100% offline** | Privacy/liability — no cloud, no data transmission |
| **Windows desktop** | Target users are on Windows business machines |
| **CPU-only processing** | Must work on typical laptops without dedicated GPU |
| **16GB RAM minimum** | AI models and vector stores need memory headroom |
| **Commercial licensing** | App will be sold; all dependencies must allow commercial use |
| **Handle poor-quality scans** | Many legal documents are OCR'd with errors |
| **Standalone installer** | All assets bundled with installer — no post-install downloads (except Ollama) |
| **Ollama user-managed** | User installs Ollama separately and chooses their own LLM models |

---

## 4. Success Criteria

| Criteria | Target |
|----------|--------|
| Case prep time | < 30 minutes from documents to ready |
| Name extraction accuracy | > 95% of parties/witnesses identified |
| Vocabulary usefulness | Technical terms a layperson would need defined |
| Question answer quality | Answers cite specific document sources |
| OCR error filtering | False positives (typos, scan artifacts) filtered out |

---

## 5. Technical Stack (High-Level)

| Component | Technology | License |
|-----------|------------|---------|
| Language | Python 3.11+ | — |
| UI Framework | CustomTkinter | MIT |
| Local AI | Ollama (any GGUF model) | MIT |
| OCR | Tesseract via pytesseract | Apache 2.0 |
| PDF Parsing | pdfplumber | MIT |
| NER | spaCy (en_core_web_lg) | MIT |
| Vector Search | FAISS | MIT |
| Embeddings | sentence-transformers | Apache 2.0 |
| Token Counting | tiktoken | MIT |

**Default AI Model:** Gemma models (Google's terms allow commercial use with attribution).

---

## 6. The 5-Step User Workflow

```
1. FILE INGEST      → User selects documents (PDF, TXT, RTF)
2. PRE-PROCESSING   → App extracts text, provides OCR confidence scores
3. FILE SELECTION   → User reviews quality scores, chooses which to include
4. AI PROCESSING    → Extraction, question indexing, optional summarization
5. OUTPUT DISPLAY   → Vocabulary table, Questions panel, optional Summary
```

The status bar keeps users informed of progress through each pipeline stage.

---

## 7. Future Considerations (Not Yet Prioritized)

- License server for commercial distribution
- Batch processing mode (overnight jobs)

---

*This document defines the north star. Implementation details are in [ARCHITECTURE.md](ARCHITECTURE.md).*
