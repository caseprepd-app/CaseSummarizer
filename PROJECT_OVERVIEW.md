# LocalScribe - Project Overview

> **Document Type:** Prescriptive (upstream) — Defines WHAT we're building and WHY.
> For HOW it's built, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Documentation System

This project uses exactly 3 documentation files:

| File | Purpose | When to Update |
|------|---------|----------------|
| **PROJECT_OVERVIEW.md** (this file) | What, why, and rationale — business goals, constraints, key decisions | When goals/constraints change or major decisions are made |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | How — implementation status, diagrams, code patterns, setup | When code structure changes |
| **[RESEARCH_LOG.md](RESEARCH_LOG.md)** | Research cache — prevents re-searching the same technical questions | After completing any technical research |

**Rules:**
- No other documentation files (no TODO.md, no session logs, no human summaries)
- PROJECT_OVERVIEW.md is the north star — check here for why things are the way they are
- ARCHITECTURE.md tracks what's built (✓) vs what's not (○)
- RESEARCH_LOG.md is a cache — check before searching, update after searching

**When stuck on a coding issue:**
1. Check RESEARCH_LOG.md first — the solution may already be documented
2. If absent or inadequate, do an online search
3. Update RESEARCH_LOG.md with findings so future sessions don't repeat the search

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
5. OUTPUT DISPLAY   → Names & Vocabulary table, Questions panel, optional Summary
```

---

## 7. Future Considerations (Not Yet Prioritized)

- License server for commercial distribution
- Model downloads via Dropbox with quota tracking
- Batch processing mode (overnight jobs)

---

*This document defines the north star. Implementation details are in [ARCHITECTURE.md](ARCHITECTURE.md).*
