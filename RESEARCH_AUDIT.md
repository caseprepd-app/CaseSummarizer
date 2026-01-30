# CasePrepd Research Audit — Alternative Approaches & Libraries

> **Date:** 2026-01-29
> **Purpose:** Comprehensive audit of alternative/supplemental approaches across every pipeline stage.
> **Status:** 5 of 18 items implemented (see ✅ markers below). Remaining items are unimplemented.

---

## Quick Reference: Top Recommendations by Impact

| # | Change | Effort | New Deps | Impact |
|---|--------|--------|----------|--------|
| 1 | ✅ **RRF fusion** (replace weighted score merging) | ~10 lines | None | Eliminates hand-tuned BM25/FAISS weights |
| 2 | ✅ **gte-reranker-modernbert-base** (replace bge-reranker-base) | Small | None (same `sentence-transformers`) | 149M params, 8192-token context — sees full chunks |
| 3 | ✅ **TinyLettuce** (three-tier model selection) | Small | Same library | Standard/Fast/Fastest — all three bundled for installer |
| 4 | **GLiNER** (zero-shot NER) | Medium | `gliner` (~200MB) | One model for legal + medical + generic entities |
| 5 | **KeyBERT** (supplement RAKE) | Low | `keybert` (~80MB) | Semantic keyword extraction, different signal |
| 6 | ✅ **pytextrank** (supplement RAKE/BM25) | Very low | `pytextrank` (no model) | Graph-based scoring as spaCy component |
| 7 | **scispaCy** (medical NER) | Low | `en_ner_bc5cdr_md` (~200MB) | Drug name and disease detection |
| 8 | **NUPunkt** (sentence boundaries) | Low | `nupunkt` (no model) | Legal-specific SBD, 91% precision |
| 9 | **Docling** (PDF structure + tables) | Medium | `docling` (~500MB-1GB) | MIT license, best table extraction |
| 10 | **nomic-embed-text-v1.5** (embeddings) | Medium | Model (~500MB) | 8K context, Matryoshka dims, better retrieval |
| 11 | **LanceDB** (replace FAISS) | Medium | `lancedb` | Persistent vector store, metadata filtering |
| 12 | **Late chunking** | Medium | Requires #10 | Preserves cross-chunk context for legal text |

---

## Stage 1: PDF Extraction & OCR

### Current Approach
- PyMuPDF (primary) + pdfplumber (secondary) with word-level voting
- Layout-aware zone clipping for headers/footers/line numbers
- Tesseract OCR fallback with opencv/deskew/scikit-image preprocessing

### Alternatives Evaluated

#### pymupdf4llm
- **What:** PyMuPDF extension that outputs Markdown optimized for LLM/RAG. Handles multi-column layouts, tables, reading order, header/footer detection automatically.
- **License:** AGPL-3.0 (copyleft — requires open-sourcing your app OR purchasing commercial license from Artifex)
- **GPU:** No
- **Size:** ~15MB on top of PyMuPDF
- **Verdict:** Best capabilities for near-zero integration cost, but **AGPL license is a blocker** for commercial distribution without a paid license.

#### Docling (IBM)
- **What:** Full document understanding: layout analysis, reading order, table structure (TableFormer — 97.9% accuracy), code blocks, formulas. Outputs Markdown/HTML/JSON.
- **License:** MIT
- **GPU:** Optional (CPU works)
- **Size:** ~500MB-1GB with models
- **Verdict:** **Strong candidate.** MIT license, strongest table extraction, full structure detection. Could replace or supplement the entire extraction pipeline. Integrates with LangChain/LlamaIndex.

#### PaddleOCR
- **What:** Modular OCR pipeline (detection + orientation + recognition). 80+ languages. Better than Tesseract on complex layouts and degraded scans.
- **License:** Apache 2.0
- **GPU:** Optional
- **Size:** ~150-300MB
- **Verdict:** **Best Tesseract replacement.** The PaddlePaddle framework dependency is friction on Windows, but ONNX inference is available. Significant accuracy improvement on degraded scans.

#### docTR (Mindee)
- **What:** Two-stage OCR (detection + recognition) with PyTorch/TensorFlow backends.
- **License:** Apache 2.0
- **GPU:** Optional
- **Size:** ~200-400MB
- **Verdict:** Good for structured forms/documents. Less broad than PaddleOCR.

#### pdfplumber Table Extraction
- **What:** Already in your stack. `pdfplumber.extract_tables()` supports table extraction with customizable settings.
- **Verdict:** **Check this first** before adding new table extraction dependencies. You may already have this capability unused.

#### camelot-py
- **What:** Dedicated table extraction with Lattice (bordered) and Stream (borderless) modes. Returns pandas DataFrames.
- **License:** MIT
- **Size:** ~20MB (requires Ghostscript)
- **Verdict:** Best dedicated table extractor if pdfplumber tables are insufficient.

#### Skip These
- **Marker-PDF:** GPL + restricted model weights. License incompatible.
- **Surya OCR:** GPL + restricted weights. License incompatible.
- **MinerU:** AGPL. License incompatible.
- **EasyOCR:** No accuracy advantage over Tesseract for English legal docs.
- **TrOCR (Microsoft):** Requires GPU. Only relevant for handwritten text.

---

## Stage 2: Text Sanitization & Cleanup

### Current Approach
- ftfy for mojibake/encoding repair
- unidecode for transliteration
- Custom regex for Unicode normalization, control chars, redactions

### Alternatives Evaluated

#### clean-text
- **What:** Wraps ftfy + adds URL/email/phone stripping, whitespace normalization in a single `clean()` call.
- **License:** Apache 2.0 (note: optional `unidecode` dep is GPL; use `text-unidecode` instead)
- **Verdict:** Could simplify your CharacterSanitizer into fewer lines. Same underlying ftfy.

#### textacy
- **What:** spaCy extension for text preprocessing, normalization, feature extraction.
- **License:** Apache 2.0
- **Verdict:** Heavier than clean-text but more powerful. Only worth it if you already use spaCy extensively (you do).

#### NUPunkt (Sentence Boundary Detection)
- **What:** Legal-specific sentence boundary detection. Pure Python, zero dependencies. 91.1% precision on legal text. Processes 10M chars/sec.
- **License:** MIT
- **Verdict:** **Recommended.** Directly addresses your legal document domain. Better than pySBD for legal text.

#### pySBD
- **What:** Rule-based SBD, handles abbreviations and legal citations.
- **License:** MIT
- **Verdict:** Good but NUPunkt is better for legal text (91% vs 59% precision).

---

## Stage 3: Preprocessing (Header/Footer/Title Removal)

### Current Approach
- Custom regex-based header/footer frequency analysis
- Title page scoring with pattern matching
- Line number regex removal
- Index page detection
- Q/A notation conversion

### Alternatives Evaluated

#### unstructured (Apache 2.0)
- **What:** Auto-partitions documents into typed elements (Title, NarrativeText, ListItem, Table, Header, Footer). 30+ formats.
- **Verdict:** Could replace significant custom regex for header/footer/title detection. However, the open-source version has decreased in quality compared to their paid platform. Evaluate carefully.

#### Transcript-Specific Libraries
- **Finding:** No dedicated open-source library exists for deposition transcript cleanup. Commercial tools (Lexitas, SmartDepo, CaseFleet) are SaaS.
- **Verdict:** Your custom approach is effectively state of the art for open source. Keep it.

---

## Stage 4: Chunking

### Current Approach
- LangChain SemanticChunker with gradient breakpoints
- tiktoken token counting, 400-1000 token chunks

### Alternatives Evaluated

#### Late Chunking (jina-ai)
- **What:** Embed the entire document through a long-context transformer first, then split into chunks via mean pooling over token spans. Preserves cross-chunk context.
- **License:** MIT
- **Why it matters:** Legal documents are full of anaphoric references ("the aforementioned party", "said defendant") that lose meaning at chunk boundaries. Late chunking solves this.
- **Prerequisite:** Requires a long-context embedding model (8K+), such as nomic-embed-text-v1.5.
- **Verdict:** **Most impactful chunking upgrade for legal text.** But requires upgrading the embedding model first.

#### NLTK TextTiling
- **What:** Topic boundary detection using lexical cohesion. Splits at topic shifts rather than fixed sizes.
- **License:** Apache 2.0
- **Verdict:** Simple topic-aware boundaries. No extra dependencies if you already use NLTK (you do).

#### datasketch (Chunk Deduplication)
- **What:** MinHash + LSH for near-duplicate detection. Detects repeated paragraphs across legal documents (boilerplate clauses).
- **License:** MIT
- **Verdict:** **Useful supplement** before summarization to avoid redundant LLM calls on duplicate chunks.

---

## Stage 5a: Vocabulary Extraction (NER + Keywords)

### Current Approach
- spaCy en_core_web_lg for NER
- RAKE (rake-nltk) for keyphrases
- BM25 corpus scoring for unusual terms
- LLM extraction via Ollama
- pyspellchecker for gibberish detection
- Google word frequency for rarity

### Alternatives Evaluated

#### GLiNER (Zero-Shot NER)
- **What:** Specify entity type labels at inference time — "person", "medical_condition", "legal_term", "case_citation" — and it extracts matching spans without training.
- **License:** Apache 2.0
- **Size:** ~200MB (small model)
- **CPU:** Yes, designed for it. ONNX export available.
- **Verdict:** **Strongest single candidate.** One model replaces the need for separate legal + medical + generic NER. Near GPT-4o accuracy on standard benchmarks. `gliner-spacy` integration exists.

#### scispaCy (Medical NER)
- **What:** spaCy pipeline for biomedical text from Allen AI. `en_ner_bc5cdr_md` specializes in drug names and disease mentions (trained on 4409 chemicals, 5818 diseases).
- **License:** Apache 2.0
- **Size:** ~200MB
- **Verdict:** **Best targeted medical NER.** Runs alongside en_core_web_lg. Includes abbreviation detector (Schwartz & Hearst algorithm).

#### KeyBERT
- **What:** Uses BERT embeddings + cosine similarity to find semantically representative keyphrases.
- **License:** MIT
- **Size:** ~80MB (MiniLM backend) or ~8MB (potion-base-8M via Model2Vec)
- **Verdict:** **Strong supplement to RAKE.** Different algorithm yields different keywords. Higher accuracy in benchmarks.

#### pytextrank — ✅ IMPLEMENTED
- **What:** Graph-based keyword scoring using TextRank (PageRank on word co-occurrence). Integrates as a spaCy pipeline component.
- **License:** MIT
- **Size:** No model download (reuses en_core_web_lg already bundled for NER)
- **Verdict:** **Easiest addition** as a 5th extraction algorithm. Zero friction to add.
- **Implementation:** New file `src/core/vocabulary/algorithms/textrank_algorithm.py` with `@register_algorithm("TextRank")`. Weight 0.6 (below NER 1.0, BM25 0.8, RAKE 0.7). Loads a separate en_core_web_lg instance with pytextrank pipe. Gated behind `textrank_enabled` config. Graceful degradation if pytextrank not installed. Added to `requirements.txt`.

#### coreferee (Coreference Resolution)
- **What:** Resolves "he", "she", "they" back to named entities. Hybrid rule + neural approach for spaCy.
- **License:** MIT
- **Verdict:** **Strong supplement** for improving name detection in depositions where witnesses use pronouns.

#### Flair NER (Character-Level)
- **What:** Character-level embeddings with LSTM-CRF. Handles misspelled/unusual words better than spaCy's token-level approach — relevant for OCR output.
- **License:** MIT
- **Size:** ~250MB (ner-fast)
- **Verdict:** Useful supplement for noisy OCR text.

#### BM25+ vs BM25Okapi — ✅ VERIFIED
- **Quick check:** Verify whether your `bm25_algorithm.py` uses BM25Okapi or BM25Plus. Plus fixes edge cases with very long documents (avoids excessive length penalties). One-line change if using Okapi.
- **Result:** The retrieval module (`src/core/retrieval/algorithms/bm25_plus.py`) already imports and uses `BM25Plus` from `rank_bm25`. The vocabulary module (`src/core/vocabulary/algorithms/bm25_algorithm.py`) uses a custom BM25 implementation with configurable k1/b parameters. No changes needed.

#### Skip These
- **YAKE:** GPL-3.0. License incompatible.
- **PKE:** GPL-3.0. License incompatible.
- **Legal-BERT:** Requires fine-tuning, CC-BY-SA license, slow on CPU.
- **Blackstone:** UK-law focused, experimental.
- **Stanza:** Not clearly better than spaCy for this use case.

---

## Stage 5b: Q&A / RAG Retrieval

### Current Approach
- FAISS vector search with all-MiniLM-L6-v2 embeddings
- Custom BM25+ for keyword retrieval
- Hybrid: BM25 weight 1.0, FAISS weight 0.5
- bge-reranker-base cross-encoder reranking
- LlamaIndex + Ollama query expansion
- LettuceDetect (ModernBERT, 150M params) for hallucination detection

### Alternatives Evaluated

#### Reciprocal Rank Fusion (RRF) — ✅ IMPLEMENTED
- **What:** Combines retrieval results by rank position: `score = sum(1/(k+rank))` with k=60. About 10 lines of Python.
- **New deps:** None
- **Verdict:** **Single easiest improvement.** Eliminates score normalization and hand-tuned weights. Immune to score distribution differences between BM25 and FAISS.
- **Implementation:** Replaced `ChunkMerger._calculate_weighted_score()` with weighted RRF in `src/core/retrieval/chunk_merger.py`. Formula: `w/(k+rank)` where k=60 and weights favor semantic search (FAISS=1.0, BM25+=0.9) since reporters ask exploratory questions without knowing exact document terminology. Multi-algorithm bonus removed (RRF handles it naturally). All existing tests updated and passing.

#### FlashRank Reranker
- **What:** ONNX-only reranker. No PyTorch needed for reranking. 4MB (nano) or 86MB (medium).
- **License:** Apache 2.0
- **Verdict:** Superseded — we upgraded to `Alibaba-NLP/gte-reranker-modernbert-base` instead (149M params, 8192-token context, smaller than bge-reranker-base). Uses same `sentence-transformers` CrossEncoder API, no new dependencies. FlashRank remains an option if we want to eliminate PyTorch entirely.

#### TinyLettuce (Hallucination Detection) — ✅ IMPLEMENTED (three-tier user selection)
- **What:** Next-gen LettuceDetect. 17–68M params (vs 150M). Real-time on CPU. Same MIT license, same token-level span detection.
- **Accuracy on real-world data (RAGTruth benchmark):**

| Model | Params | RAGTruth F1 | Synthetic F1 | Setting |
|-------|--------|-------------|--------------|---------|
| LettuceDetect-base (ModernBERT) | 150M | **76.07%** | 87.60% | Standard (default) |
| TinyLettuce-68M (Ettin) | 68M | 74.97% | 92.64% | Fast |
| TinyLettuce-17M (Ettin) | 17M | 68.52% | 90.87% | Fastest |

- **Verdict:** The headline "90% F1" numbers are on synthetic data only. On real-world RAGTruth, the base model leads by 1–7.5 F1 points. Standard remains default; Fast and Fastest offered for resource-constrained users.
- **Implementation:** Three-tier dropdown in Settings > Performance: Standard (recommended), Fast, Fastest (less accurate). `HallucinationVerifier` accepts `model_variant` parameter, reads from user preferences. Bundled model paths for all three variants. Download script (`scripts/download_hallucination_model.py`) updated to fetch all three for installer builds. Weighted RRF also applied (FAISS=1.0, BM25+=0.9) to give semantic search a modest edge.

#### Embedding Model Upgrades

| Model | Params | Dims | Max Tokens | License | Why |
|-------|--------|------|-----------|---------|-----|
| nomic-embed-text-v1.5 | 137M | 64-768 (Matryoshka) | 8192 | Apache 2.0 | 16x longer context. Beats OpenAI ada-002 |
| bge-base-en-v1.5 | 109M | 768 | 512 | MIT | Best general-purpose upgrade |
| BGE-M3 | 568M | 1024 | 8192 | MIT | Produces dense + sparse + multi-vector. Best accuracy |

**Recommended:** nomic-embed-text-v1.5. The 8192 token context is transformative for legal documents. Matryoshka lets you use 256 dims for speed.

#### LanceDB (Replace FAISS)
- **What:** Embedded vector DB (no server). Apache 2.0. Columnar Lance format. Native hybrid search. Pandas-like API. 4MB idle RAM.
- **Verdict:** Adds persistence and metadata filtering that FAISS lacks. "SQLite of vector DBs."

#### ONNX Export + INT8 Quantization
- **What:** Convert embedding model from PyTorch to ONNX runtime with INT8 weights. 2-4x CPU speedup.
- **How:** `pip install optimum[onnxruntime]`, then export.
- **Verdict:** **Do this regardless of which embedding model you use.** Pure performance win.

#### ColBERT (Late Interaction)
- **What:** Per-token embeddings with MaxSim scoring. Cross-encoder quality with bi-encoder speed.
- **Library:** PyLate (MIT, Sentence Transformers ecosystem)
- **Verdict:** Large architectural change. Only pursue if reranking accuracy is a bottleneck.

#### BGE-M3 Unified Hybrid
- **What:** One model produces dense, sparse, and multi-vector representations. Eliminates need for separate BM25 + embedding pipelines.
- **License:** MIT
- **Verdict:** Most elegant long-term solution but large model (568M params).

---

## Stage 5c: Summarization

### Current Approach
- Progressive (rolling) map-reduce via Ollama

### Alternatives Evaluated

#### Tree Summarize (LlamaIndex)
- **What:** Recursive bottom-up: chunks are summarized, summaries are summarized, until one remains. Parallelizable.
- **License:** MIT
- **Verdict:** Better information retention for very long documents. Worth testing vs rolling map-reduce.

#### Refine Chain
- **What:** Each chunk refines the running summary sequentially. Maintains narrative coherence.
- **Verdict:** Better for depositions where chronology matters. Slower (no parallelism). Early-bias risk.

#### DSPy (Prompt Optimization)
- **What:** Programmatic prompt optimization. Define input/output schemas, let DSPy auto-optimize prompts against metrics.
- **License:** MIT
- **Verdict:** Future investment. Significant learning curve but could systematically improve summarization quality.

---

## Stage 6: No Changes Identified
Export stage (Word/PDF generation) was not a focus of this audit.

---

## License Summary

All recommendations above use one of:
- **MIT** — most permissive, no restrictions
- **Apache 2.0** — permissive, requires attribution
- **No license needed** — pure algorithms (RRF, ONNX export)

Explicitly excluded: GPL, AGPL, CC-NC, restricted model weights.

---

## Implementation Roadmap (Suggested Order)

### Quick Wins (1-2 sessions each, no architectural changes)
1. ~~RRF fusion — replace weighted score merging~~ ✅ Done
2. ~~FlashRank reranker — replace bge-reranker-base~~ ✅ Done (upgraded to gte-reranker-modernbert-base)
3. ~~TinyLettuce — upgrade hallucination detector~~ ✅ Done (user-selectable option)
4. ~~pytextrank — add as 5th vocabulary algorithm~~ ✅ Done
5. ~~BM25+ check — verify Okapi vs Plus variant~~ ✅ Done (retrieval uses `BM25Plus` from `rank_bm25`; vocabulary uses custom implementation)
6. ONNX export — quantize current embedding model

### Medium Effort (1-3 sessions each)
7. GLiNER — add zero-shot NER for legal + medical entities
8. KeyBERT — add semantic keyword extraction
9. scispaCy — add medical NER for drug/disease detection
10. NUPunkt — add legal sentence boundary detection
11. nomic-embed-text-v1.5 — upgrade embedding model
12. LanceDB — replace FAISS

### Larger Investments (research + implementation)
13. Late chunking — requires #11 first
14. Docling — evaluate for table extraction + structure detection
15. coreferee — coreference resolution for name detection
16. datasketch — chunk deduplication before summarization
17. Tree Summarize — test against current map-reduce
18. DSPy — systematic prompt optimization
