# Research Log

> **Purpose:** Cache external research so it doesn't have to be repeated. Check here BEFORE searching.
>
> **Format:** Append new entries at the top. Only log actual research with external sources.

---

## TextRank Integration into Vocabulary Pipeline (January 2025)

**Question:** How does TextRank (pytextrank) work, and how should its output integrate into the existing vocabulary quality scoring?

**Background:** TextRank (Mihalcea & Tarau, 2004) applies PageRank to a word co-occurrence graph built from the document. Words that co-occur frequently with other important words get higher centrality scores. The pytextrank library implements this as a spaCy pipeline component.

**Key property — centrality score (0–1):** Each keyphrase gets a `rank` score representing its PageRank centrality in the document's word graph. This is a unique signal not captured by NER (entity recognition), RAKE (co-occurrence statistics within stopword-delimited phrases), or BM25 (corpus-level term importance). A high centrality score means the term is well-connected to other important terms in the document.

**Integration decisions:**
- **Algorithm layer:** TextRank runs as a fourth algorithm alongside NER, RAKE, and BM25. It loads a separate en_core_web_lg instance to avoid mutating the shared NER pipeline.
- **ML features:** Two new features added to the 48-feature ML vector: `has_textrank` (binary) and `textrank_score` (continuous 0–1).
- **Rule-based scoring:** Centrality score contributes up to +8 points to the quality score (`min(textrank_score * 10, 8)`), comparable to person name boost (+10) and multi-algorithm agreement (up to +10). Configurable via `score_textrank_centrality_boost`.

**Sources:**
- Mihalcea, R. & Tarau, P. (2004). "TextRank: Bringing Order into Text." https://aclanthology.org/W04-3252/
- pytextrank documentation: https://derwen.ai/docs/ptr/
- spaCy pipeline components: https://spacy.io/usage/processing-pipelines

---

## RAG Pipeline: Alternative and Supplemental Approaches (January 2025)

**Question:** What alternatives/supplements exist for each component of the current RAG pipeline (embeddings, chunking, retrieval, reranking, hallucination detection, vector store)?

**Current stack:** FAISS + all-MiniLM-L6-v2 + custom BM25+ + BGE-reranker-base + LangChain SemanticChunker + LettuceDetect (ModernBERT)

---

### 1. Embedding Models

| Model | Params | Dimensions | Max Tokens | License | CPU Speed | Notes |
|-------|--------|-----------|------------|---------|-----------|-------|
| all-MiniLM-L6-v2 (current) | 22M | 384 | 512 | Apache 2.0 | ~5-14k sent/sec | Fast but limited on domain-specific/long text |
| **e5-small-v2** | 118M | 384 | 512 | MIT | ~16ms/embed | Outperforms models 70x larger on Top-5 retrieval |
| **bge-base-en-v1.5** | 109M | 768 | 512 | MIT | Good | Best general-purpose upgrade from MiniLM |
| **nomic-embed-text-v1.5** | 137M | 64-768 (Matryoshka) | 8192 | Apache 2.0 | Good | Long context + flexible dimensions. Beats OpenAI ada-002 |
| **nomic-embed-text-v2-moe** | 475M (305M active) | 256-768 | 8192 | Open source | MoE = efficient | Multilingual, MoE architecture |
| **BGE-M3** | 568M | 1024 | 8192 | MIT | Slower (large) | Dense + sparse + multi-vector in one model. Best accuracy but heaviest |

**Recommendation:** nomic-embed-text-v1.5 is the best upgrade path -- 8192 token context (vs 512), Matryoshka dimensions, Apache 2.0, and reasonable CPU performance. BGE-M3 is best if accuracy is paramount and CPU budget allows.

**Sources:**
- https://huggingface.co/spaces/mteb/leaderboard
- https://app.ailog.fr/en/blog/guides/choosing-embedding-models
- https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
- https://www.nomic.ai/blog/posts/nomic-embed-text-v2
- https://huggingface.co/BAAI/bge-m3

---

### 2. Chunking Strategies

| Strategy | Library/Source | How It Works | Pros | Cons |
|----------|--------------|-------------|------|------|
| Semantic (current) | LangChain SemanticChunker | Splits at gradient breakpoints in embedding similarity | Good semantic boundaries | Requires embedding model; can be slow |
| **Late Chunking** | jina-ai/late-chunking (GitHub) | Embed entire document first, THEN split via mean pooling over segments | Preserves cross-chunk context, resolves anaphora | Requires long-context embedding model (8K+ tokens) |
| **Recursive Character** | LangChain RecursiveCharacterTextSplitter | Splits by paragraph > sentence > word hierarchy | LangChain default; good structure preservation | No semantic awareness |
| **Sliding Window + Overlap** | Any splitter with overlap param | Fixed-size chunks with configurable overlap | Simple, predictable chunk sizes | Can split mid-concept |

**Recommendation:** Late chunking is the most promising upgrade -- it solves context loss from anaphoric references ("this case", "the defendant"), which is common in legal text. Requires switching to a long-context embedding model (nomic-embed-text-v1.5 or BGE-M3).

**Sources:**
- https://github.com/jina-ai/late-chunking
- https://arxiv.org/abs/2409.04701
- https://weaviate.io/blog/chunking-strategies-for-rag

---

### 3. Late-Interaction Models (ColBERT)

| Library | What It Does | License | CPU Support | Notes |
|---------|-------------|---------|-------------|-------|
| **PyLate** | ColBERT training + retrieval on Sentence Transformers | MIT | Yes (via ST) | Best CPU option; FastPLAID index |
| **RAGatouille** | Simplified ColBERT wrapper for RAG | MIT | Partial | Easiest API but more GPU-oriented |
| **colbert-ai** | Official Stanford ColBERT library | MIT | Possible but GPU-oriented | Most features, heaviest setup |

**What ColBERT offers:** Token-level late interaction gives cross-encoder-level accuracy with bi-encoder-level indexing speed. ColBERTv2 uses residual compression to reduce storage.

**Recommendation:** PyLate is the best fit (MIT, Sentence Transformers, CPU-native). However, this is a significant architectural change. Consider only if current reranking accuracy is insufficient.

**Sources:**
- https://github.com/lightonai/pylate
- https://github.com/stanford-futuredata/ColBERT
- https://weaviate.io/blog/late-interaction-overview

---

### 4. Rerankers

| Reranker | Size | Dependencies | License | CPU Speed | Notes |
|----------|------|-------------|---------|-----------|-------|
| BGE-reranker-base (current) | ~278M | Transformers + Torch | MIT | Moderate | Solid baseline |
| **FlashRank (nano)** | ~4MB | No Torch/Transformers (ONNX) | Apache 2.0 | Blazing fast | Tiniest reranker. Competitive accuracy |
| **FlashRank (medium)** | ~86MB | No Torch/Transformers (ONNX) | Apache 2.0 | Fast | Better accuracy, still lightweight |
| **rerankers library** | Varies | Unified API | MIT | Varies | Wrapper supporting FlashRank, cross-encoders, ColBERT, T5, etc. |

**Recommendation:** FlashRank eliminates PyTorch overhead for reranking (ONNX only, ~4MB). The `rerankers` library from Answer.AI provides a unified API to swap between backends without code changes.

**Sources:**
- https://github.com/PrithivirajDamodaran/FlashRank
- https://github.com/AnswerDotAI/rerankers

---

### 5. Hallucination Detection

| Tool | Params | Approach | License | CPU | Accuracy |
|------|--------|---------|---------|-----|----------|
| LettuceDetect (current) | ~150M | Token-level ModernBERT | MIT | Yes | F1 79.2% (RAGTruth) |
| **TinyLettuce** | 17-68M | Token-level Ettin encoder | MIT | Yes, real-time | F1 90.87% (synthetic). Outperforms GPT-5-mini |
| **Vectara HHEM-2.1-Open** | ~335M | Sentence-level scoring | Apache 2.0 | GPU preferred | 71.8% (AggreFact) |
| **Bespoke-MiniCheck** | 7B | Claim-level factuality | Non-commercial | GPU required | 77.4% (AggreFact) |

**Recommendation:** TinyLettuce (same team as LettuceDetect) -- 17M params, real-time on CPU, MIT license, better accuracy. Drop-in upgrade.

**Sources:**
- https://huggingface.co/blog/adaamko/tinylettuce
- https://github.com/KRLabsOrg/LettuceDetect
- https://www.vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model

---

### 6. Vector Stores

| Store | Type | License | RAM Usage | Query Speed | Notes |
|-------|------|---------|-----------|-------------|-------|
| FAISS (current) | In-memory library | MIT | High (all in RAM) | Very fast | No metadata filtering, no persistence API |
| **LanceDB** | Embedded DB | Apache 2.0 | 4MB idle, ~150MB search | 40-60ms | "SQLite of vector DBs". Native Python API. Hybrid search |
| **Qdrant (local)** | Embedded or server | Apache 2.0 | ~400MB constant | 20-30ms | Rust-based HNSW. Rich metadata filtering |
| **SQLite-VSS** | SQLite extension | MIT | Low | Moderate | Less community momentum |

**Recommendation:** LanceDB -- embedded, Apache 2.0, very low RAM, native hybrid search, Pandas-like Python API. Replaces FAISS while adding persistence and metadata filtering.

**Sources:**
- https://lancedb.com/
- https://thedataquarry.com/blog/vector-db-1/

---

### 7. Hybrid Retrieval Approaches

| Approach | What It Does | Implementation | Notes |
|----------|-------------|---------------|-------|
| Weighted scores (current) | BM25 * 1.0 + FAISS * 0.5 | Custom | Requires score normalization; hand-tuned weights |
| **Reciprocal Rank Fusion (RRF)** | Combines by rank: `1/(k+rank)` | ~10 lines Python; LlamaIndex QueryFusionRetriever | No normalization needed. Robust to score distribution differences |
| **SPLADE** | Neural sparse expansion via BERT MLM head | SPLADERunner (pip, ONNX, no Torch) | Replaces BM25. Better generalization. **Official weights: CC-NC license** |
| **BGE-M3 hybrid** | Single model: dense + sparse + multi-vector | FlagEmbedding library | One model replaces embedding + BM25. MIT license |

**Recommendation:** RRF is the easiest win -- replace weighted combination with rank-based fusion. ~10 lines, no new deps. BGE-M3's built-in hybrid is the best long-term solution.

**Sources:**
- https://safjan.com/implementing-rank-fusion-in-python/
- https://developers.llamaindex.ai/python/examples/retrievers/reciprocal_rerank_fusion/
- https://pypi.org/project/SPLADERunner/

---

### 8. CPU-Optimized Embedding Inference

| Model | Params | ONNX | Quantization | Notes |
|-------|--------|------|-------------|-------|
| all-MiniLM-L6-v2 (current) | 22M | Yes | Yes | Smallest, fastest, weakest |
| **e5-small-v2** | 118M | Yes | Yes | Best speed/accuracy ratio |
| **bge-small-en-v1.5** | 33M | Yes | Yes | MIT. Small step up from MiniLM |
| **nomic-embed-text-v1.5 (dim=256)** | 137M | Yes | Yes | Matryoshka dim=256 for speed, 8K context |

**Recommendation:** Use ONNX export + INT8 quantization via Hugging Face `optimum` library for 2-4x CPU speedup on any model. nomic-embed-text-v1.5 at dim=256 gives best balance of quality, context length, and speed.

**Sources:**
- https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
- https://research.aimultiple.com/open-source-embedding-models/

---

### Priority Upgrade Roadmap (Easiest to Hardest)

1. **RRF fusion** -- Replace weighted scores with Reciprocal Rank Fusion. No new deps. ~10 lines.
2. **FlashRank** -- Add ONNX reranker. `pip install FlashRank`. No Torch overhead.
3. **TinyLettuce** -- Upgrade hallucination detection. Same library, smaller model, better accuracy.
4. **LanceDB** -- Replace FAISS. Adds persistence, metadata filtering, lower RAM.
5. **nomic-embed-text-v1.5** -- Switch embedding model. 8K context, Matryoshka dims, better accuracy.
6. **Late chunking** -- Implement after long-context embeddings. Major accuracy gain for legal text.
7. **BGE-M3 hybrid** -- Single model for dense+sparse. Most complex change.

---

## NER, Keyword Extraction, and Vocabulary Pipeline Alternatives

**Date:** 2026-01-29

**Question:** What alternatives or supplements exist for the current vocabulary extraction pipeline (spaCy NER, RAKE, BM25, LLM extraction, WordNet, pyspellchecker, Google word frequency)?

**Requirements:** Offline, CPU-friendly, commercial license, Python.

### 1. Legal-Domain NER Models

| Library | What It Does | License | Model Size | CPU? | Verdict |
|---------|-------------|---------|------------|------|---------|
| **Blackstone** | spaCy pipeline for UK legal text. Detects CASENAME, CITATION, INSTRUMENT, PROVISION entities. Trained on case law from 1860s onward. | MIT | ~500MB (spaCy model) | Yes | **Supplement** current NER. Adds legal-specific entity types spaCy misses. However, trained on UK law -- may not align with US legal depositions. Experimental/not production-grade. |
| **Legal-BERT** (`nlpaueb/legal-bert-base-uncased`) | BERT pretrained on legal corpora. Can be used as spaCy transformer backbone for custom NER. | CC-BY-SA-4.0 | ~440MB | Slow on CPU | **Skip.** Requires fine-tuning on labeled legal NER data. CC-BY-SA license is restrictive for commercial. Transformer = slow CPU. |
| **OpenNyAI Legal NER** (`en_legal_ner_trf`) | spaCy transformer model for Indian legal NER (petitioner, respondent, court, statute, etc.). | MIT | ~500MB+ | Slow on CPU | **Skip.** Indian legal domain. Transformer-based = slow CPU. |

### 2. Alternative Keyword Extraction

| Library | What It Does | License | Model Size | CPU? | Verdict |
|---------|-------------|---------|------------|------|---------|
| **YAKE** | Unsupervised statistical keyword extraction. No training, no corpus, no external dependencies. Uses text features (casing, position, frequency, relatedness). | GPL-3.0 | No model (pure stats) | Very fast | **Strong supplement** to RAKE. Different algorithm = different keywords found. BUT GPL-3.0 license is problematic for commercial use. |
| **KeyBERT** | BERT embeddings + cosine similarity to find keyphrases most representative of a document. Benchmarks show higher accuracy than RAKE/YAKE. | MIT | ~80MB (all-MiniLM-L6-v2) to ~420MB (mpnet) | Moderate (seconds per doc) | **Strong supplement.** Semantic keyword extraction vs. RAKE's statistical approach. Can use lightweight `all-MiniLM-L6-v2` or even `potion-base-8M` (8MB) via Model2Vec for fast CPU. MIT license. |
| **PKE** (Python Keyphrase Extraction) | Toolkit with TopicRank, PositionRank, MultipartiteRank, SingleRank. Graph-based and statistical methods. | GPL-3.0 | No model (graph algorithms) | Fast | **Skip for commercial.** GPL-3.0 license. |

### 3. Medical NER Models

| Library | What It Does | License | Model Size | CPU? | Verdict |
|---------|-------------|---------|------------|------|---------|
| **scispaCy** (`en_core_sci_md`) | spaCy pipeline for biomedical text. 101K vocab, 98K word vectors. NER for chemicals, diseases, genes. Abbreviation detection. | Apache 2.0 | ~200MB (md), ~400MB (lg) | Yes, 33ms/abstract | **Strong supplement.** Catches medical terms in depositions (drug names, conditions, procedures). Apache 2.0. Runs alongside en_core_web_lg. |
| **scispaCy NER models** (`en_ner_bc5cdr_md`) | Specialized NER for chemicals and diseases. Trained on BC5CDR corpus (4409 chemicals, 5818 diseases). | Apache 2.0 | ~200MB | Yes | **Best medical NER option.** Directly identifies drug names and disease mentions in medical depositions. |
| **medspaCy** | Clinical NLP: negation detection, section detection, UMLS concept linking. Built on spaCy. | MIT | Varies | Yes | **Consider later.** More relevant for clinical notes than legal depositions. Adds negation detection ("no fracture") which could help Q&A accuracy. |

### 4. Name Deduplication and Coreference Resolution

| Library | What It Does | License | CPU? | Verdict |
|---------|-------------|---------|------|---------|
| **coreferee** | spaCy-based coreference resolution. Resolves "he/she/they" to named entities. English, French, German, Polish. Rule + neural hybrid. | MIT | Yes (lightweight) | **Strong supplement.** Resolves pronouns to names, improving name completeness. Integrates with existing spaCy pipeline. |
| **fastcoref** (F-coref) | Fast coreference using LingMess/distilled models. State-of-the-art accuracy. | MIT | ~500MB | Yes (with `device='cpu'`), but designed for GPU | **Alternative to coreferee** if higher accuracy needed. Heavier. |
| **thefuzz** (fuzzywuzzy) | Fuzzy string matching for name deduplication ("Dr. Smith" = "Smith, John" etc.). | MIT | No model | Yes | **Already doing this?** Check current `name_deduplicator.py`. Good for merging name variants. |

### 5. Alternative NER Frameworks

| Library | What It Does | License | Model Size | CPU? | Verdict |
|---------|-------------|---------|------------|------|---------|
| **GLiNER** | Zero-shot NER. Extract ANY entity type by specifying labels at inference time. 50-300M params. Bidirectional transformer. Near GPT-4o accuracy on benchmarks. | Apache 2.0 | ~200MB (small), ~600MB (medium), ~1.2GB (large) | Yes (designed for CPU) | **Strongest candidate.** Define custom entity types like "medical_condition", "legal_term", "case_name" without training. Replaces need for multiple specialized NER models. ONNX export available. Apache 2.0. |
| **GLiNER-BioMed** | GLiNER fine-tuned for biomedical entities. | Apache 2.0 | ~600MB-1.2GB | Yes | **Combines legal+medical NER** in one model if using custom labels. |
| **Flair** (`ner-fast`) | Character-level embeddings + LSTM-CRF NER. Context-sensitive word representations. | MIT | ~250MB (fast), ~1.2GB+ (large) | Moderate (fast variant OK) | **Supplement.** Different architecture catches entities spaCy misses. `ner-fast` variant is CPU-viable. MIT license. |
| **Stanza** (Stanford NLP) | Full NLP pipeline with NER, dependency parsing. Trained on multiple treebanks. | Apache 2.0 | ~200-400MB | Yes | **Alternative** to spaCy but not clearly better for this use case. |

### 6. Term Importance Scoring Alternatives to BM25

| Approach | What It Does | Verdict |
|----------|-------------|---------|
| **TF-IDF** (scikit-learn) | Simpler than BM25. No saturation or length normalization. | **BM25 is already better.** No reason to switch. |
| **BM25+** | BM25 variant that fixes the issue where very long documents can get negative IDF contribution. | **Check if current implementation uses BM25+ or vanilla BM25.** rank_bm25 supports BM25Okapi, BM25L, BM25Plus. |
| **TextRank** (via PKE or pytextrank) | Graph-based keyword importance. Nodes=words, edges=co-occurrence. PageRank-style scoring. | **Supplement** but GPL concern with PKE. `pytextrank` is MIT and integrates with spaCy. |

### Recommendations (Priority Order)

1. **GLiNER** (Apache 2.0) -- Biggest bang for effort. Zero-shot means you define "person", "medical_condition", "legal_term", "case_citation" etc. at runtime. Replaces need for Blackstone + scispaCy NER as separate models. ~200MB small model runs on CPU.

2. **scispaCy `en_ner_bc5cdr_md`** (Apache 2.0) -- If GLiNER doesn't catch medical terms well enough, add this as a targeted medical NER pass. Proven on drug/disease detection. ~200MB, fast CPU.

3. **KeyBERT with all-MiniLM-L6-v2** (MIT) -- Semantic keyword extraction to supplement RAKE. Different algorithm = different keywords discovered. ~80MB model. Use `potion-base-8M` (8MB) if speed is critical.

4. **coreferee** (MIT) -- Lightweight coreference resolution. Maps "he said" back to "Dr. Smith said." Improves name detection completeness. Plugs into existing spaCy pipeline.

5. **pytextrank** (MIT) -- Graph-based keyword scoring as a 5th algorithm alongside RAKE, BM25, NER, LLM. spaCy pipeline component. No new model download.

**Sources:**
- https://github.com/urchade/GLiNER
- https://allenai.github.io/scispacy/
- https://maartengr.github.io/KeyBERT/
- https://github.com/richardpaulhudson/coreferee
- https://github.com/LIAAD/yake (GPL -- not recommended for commercial)
- https://github.com/boudinfl/pke (GPL -- not recommended for commercial)
- https://github.com/flairNLP/flair
- https://github.com/ICLRandD/Blackstone
- https://pypi.org/project/fastcoref/
- https://spacy.io/universe/project/medspacy
- https://github.com/DerwenAI/pytextrank

---

## OCR Image Preprocessing Libraries

**Question:** What libraries improve OCR accuracy for scanned legal documents?

**Libraries Evaluated:**

| Library | License | Purpose | Decision |
|---------|---------|---------|----------|
| opencv-python-headless | Apache 2.0 | Image preprocessing | ✅ Chosen |
| deskew | MIT | Skew detection/correction | ✅ Chosen |
| scikit-image | BSD | Additional image processing | ✅ Chosen |
| EasyOCR | Apache 2.0 | Alternative OCR engine | ❌ Not needed |
| OCRmyPDF | MPL 2.0 | PDF OCR tool | ❌ CLI-focused |
| unpaper | GPL | Document cleanup | ❌ Not pip-installable |

**Decision:** opencv-python-headless + deskew + scikit-image. Preprocessing improves OCR accuracy 20-50% per Tesseract docs.

**Sources:**
- https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
- https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

---

## Phone Photo Preprocessing (Rotation + Crop)

**Question:** How to auto-rotate and crop phone photos of documents?

**Approaches Evaluated:**

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Tesseract OSD | Already have pytesseract; detects 0°/90°/180°/270° | ~90% accuracy | ✅ Chosen |
| OpenCV Hough Transform | Pure OpenCV | Complex tuning | ❌ Overkill |
| OpenCV Contour Detection | Already installed; handles perspective | Struggles with busy backgrounds | ✅ Chosen |
| Deep Learning (docTR) | Most robust | Heavy dependencies | ❌ Overkill |

**Decision:** Tesseract OSD for rotation, OpenCV contours for document detection. No new dependencies.

**Sources:**
- https://pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
- https://pyimagesearch.com/2022/01/31/correcting-text-orientation-with-tesseract-and-python/

---

## Gibberish Detection Libraries

**Question:** How to filter OCR errors and gibberish from vocabulary?

**Libraries Evaluated:**

| Library | Approach | Decision |
|---------|----------|----------|
| gibberish-detector | Markov model, requires training | ❌ Too complex |
| nostril | 99%+ accuracy, fast | ❌ Would work but overkill |
| pyspellchecker | Dictionary-based, simple API | ✅ Chosen |
| OCRfixr | BERT-based | ❌ Inactive project |

**Decision:** pyspellchecker. Word is gibberish if unknown AND has no spelling corrections. Simple, no ML dependencies.

**Sources:**
- https://pypi.org/project/pyspellchecker/
- https://github.com/casics/nostril

---

## Hallucination Detection

**Question:** How to detect when Q&A answers are fabricated?

**Decision:** LettuceDetect library. MIT license, ModernBERT model (~570MB), 79.22% F1 on RAGTruth benchmark. Provides span-level probabilities for color-coded display.

**Sources:**
- https://github.com/KRLabsOrg/LettuceDetect

---

## Hybrid Retrieval Weights

**Question:** How should BM25+ and FAISS scores be weighted for Q&A?

**Decision:** BM25+ weight 1.0 (primary), FAISS weight 0.5 (secondary).

**Why:** The embedding model (`all-MiniLM-L6-v2`) isn't trained on legal terminology. Semantic search alone returns "no information found." BM25+ provides reliable exact keyword matching for legal terms.

---

## Vector Store Choice

**Question:** ChromaDB vs FAISS?

**Decision:** FAISS

**Why:** File-based storage (no database config), simpler deployment for Windows installer, well-documented Python API.

---

## UI Framework Choice

**Question:** Which GUI framework for Windows desktop app?

**Decision:** CustomTkinter

**Why:** Modern dark theme out of box, no licensing concerns for commercial use, simpler than Qt.

---

## AI Backend Choice

**Question:** How to run AI models locally?

**Decision:** Ollama REST API

**Why:** Handles model management, quantization, GPU/CPU routing automatically. REST API is simple. Supports any GGUF model.

---

## NER Model Choice

**Question:** Which spaCy model for named entity recognition?

**Decision:** `en_core_web_lg` (large model)

**Why:** 4% better accuracy than medium model on legal entities. Acceptable download (~560MB). Runs on CPU.

---

## Few-Shot Prompting for Extraction

**Question:** How to prevent LLM from hallucinating example names from JSON schema?

**Decision:** Use 3 few-shot examples instead of rules/instructions.

**Why:** Google's Gemma documentation says "Show patterns to follow, not anti-patterns to avoid." Research shows 10-12% accuracy improvement over zero-shot. Negative instructions are ineffective.

---

## Query Transformation

**Question:** How to handle vague user questions like "What happened?"

**Decision:** LlamaIndex + Ollama to expand queries into 3-4 specific search variants.

**Why:** Vague questions don't match specific document text. Expanding improves retrieval recall.

---

## Diagram Tools

**Question:** Mermaid vs D2 for architecture diagrams?

**Decision:** D2 for manual diagrams, pydeps for auto-generated dependency graphs.

**Why:** D2 has cleaner syntax, CSS-like styling, command-line rendering. pydeps generates accurate module graphs from actual imports—always in sync with code.

**Sources:**
- https://d2lang.com
- https://github.com/thebjorn/pydeps