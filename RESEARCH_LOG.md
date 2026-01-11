# Research Log

> **Purpose:** Cache external research so it doesn't have to be repeated. Check here BEFORE searching.
>
> **Format:** Append new entries at the top. Only log actual research with external sources.

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