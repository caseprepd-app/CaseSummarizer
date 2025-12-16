# LocalScribe - Human Summary

> **For human consumption only.** High-level status report.

---

## Project Status

LocalScribe is a fully functional Windows desktop application for processing legal documents. The core features (document extraction, vocabulary extraction, Q&A system, summarization) are complete and working. A DRY refactoring was just completed that eliminated ~495 lines of duplicated code by creating shared utilities for worker threads, queue messages, tokenization, and pattern filtering. The codebase is now cleaner and more maintainable.

---

## File Directory

### Root
| File | Purpose |
|------|---------|
| `PROJECT_OVERVIEW.md` | Business goals, constraints, success criteria |
| `ARCHITECTURE.md` | Technical architecture, component diagrams, file structure |
| `RESEARCH_LOG.md` | Technical decisions with rationale |
| `development_log.md` | Timestamped changelog |
| `human_summary.md` | This file — status overview |

### Source (`src/`)
| Directory | Purpose |
|-----------|---------|
| `ai/` | Ollama integration, prompt formatting |
| `briefing/` | Case Briefing feature (being deprecated) |
| `chunking/` | Semantic text chunking with token enforcement |
| `extraction/` | PDF/TXT/RTF text extraction |
| `parallel/` | Dynamic worker scaling, progress aggregation |
| `preprocessing/` | Header/footer removal, line number stripping |
| `prompting/` | Template management, focus extraction |
| `qa/` | Q&A orchestration and answer generation |
| `retrieval/` | Hybrid BM25+/FAISS retrieval |
| `sanitization/` | Unicode cleanup, mojibake fixes |
| `summarization/` | Map-reduce document summarization |
| `ui/` | CustomTkinter GUI, workers, message handling |
| `utils/` | Shared tokenizer, pattern filters, logging |
| `vector_store/` | FAISS index building and retrieval |
| `vocabulary/` | NER/RAKE/BM25 extraction, ML feedback |

### Configuration (`config/`)
| Directory/File | Purpose |
|----------------|---------|
| `prompts/` | Summarization prompt templates |
| `extraction_prompts/` | LLM extraction prompts |
| `qa_questions.yaml` | Default Q&A questions |
| `*.txt` | Term whitelists/blacklists |

### Tests (`tests/`)
| Directory/File | Purpose |
|----------------|---------|
| `test_*.py` | Automated unit tests (223 tests) |
| `manual/` | Tests requiring Ollama running |

---

*Last updated: 2025-12-16*
