"""
Tooltip metadata for CasePrepd Advanced settings.

Each entry provides a human-readable label and tooltip explaining
what the setting does, what happens when you increase/decrease it,
and concrete examples where helpful.

Used by:
- src/ui/settings/advanced_registry.py: Reads labels/tooltips for UI widgets
"""

DESCRIPTIONS = {
    # =======================================================================
    # ML Training
    # =======================================================================
    "ml_min_samples": {
        "label": "ML minimum training samples",
        "tooltip": (
            "Minimum feedback entries before ML training activates.\n\n"
            "Default: 30\n\n"
            "Increase: More conservative - needs more data before personalizing.\n"
            "Decrease: Starts personalizing sooner but may be less stable.\n\n"
            "Example: At 30, you need roughly 15 thumbs-up and 15 thumbs-down\n"
            "ratings before the model begins learning your preferences."
        ),
    },
    "ml_ensemble_min_samples": {
        "label": "Ensemble model activation threshold",
        "tooltip": (
            "Minimum feedback entries before the ensemble model (combining\n"
            "Logistic Regression + Random Forest) activates.\n\n"
            "Default: 40\n\n"
            "Increase: More data required before ensemble kicks in.\n"
            "Decrease: Enables ensemble sooner (may overfit with too little data).\n\n"
            "The ensemble model is more accurate than a single model\n"
            "but needs more data to train reliably."
        ),
    },
    "ml_decay_half_life_days": {
        "label": "Feedback decay half-life (days)",
        "tooltip": (
            "How quickly old feedback loses influence. After this many days,\n"
            "feedback is weighted at 50% of its original value.\n\n"
            "Default: 1270 days (~3.5 years)\n\n"
            "Increase: Old feedback stays influential longer.\n"
            "Decrease: Adapts faster to changing preferences, but forgets\n"
            "earlier training sooner.\n\n"
            "With default, weight reaches the floor (~55%) after 3 years."
        ),
    },
    "ml_decay_weight_floor": {
        "label": "Minimum feedback weight",
        "tooltip": (
            "Minimum weight for old feedback, regardless of age.\n"
            "Prevents ancient feedback from being completely ignored.\n\n"
            "Default: 0.55 (55%)\n\n"
            "Increase: Old feedback retains more influence (more stable).\n"
            "Decrease: Old feedback loses more influence (more adaptive).\n\n"
            "At 0.55, feedback from years ago still has 55% of its\n"
            "original weight."
        ),
    },
    # =======================================================================
    # Quality Scoring
    # =======================================================================
    "score_multi_doc_boost": {
        "label": "Multi-document boost",
        "tooltip": (
            "Bonus points added to terms found in 2+ documents.\n"
            "Terms appearing across multiple files are more likely important.\n\n"
            "Default: 10 points\n\n"
            "Increase: Strongly favor terms found in multiple documents.\n"
            "Decrease: Treat multi-doc terms more equally with single-doc terms."
        ),
    },
    "score_high_conf_boost": {
        "label": "High confidence boost",
        "tooltip": (
            "Bonus points for terms where >80% of document sources have\n"
            "high extraction confidence.\n\n"
            "Default: 5 points\n\n"
            "Increase: More reward for consistently high-confidence extractions.\n"
            "Decrease: Less emphasis on extraction confidence."
        ),
    },
    "score_all_low_conf_penalty": {
        "label": "All low-confidence penalty",
        "tooltip": (
            "Penalty applied when ALL sources for a term have confidence\n"
            "below 60%. These terms may be OCR errors or extraction artifacts.\n\n"
            "Default: -10 points\n\n"
            "More negative: Stronger penalty for unreliable extractions.\n"
            "Less negative: More lenient toward low-confidence terms."
        ),
    },
    "score_single_source_penalty": {
        "label": "Single source penalty",
        "tooltip": (
            "Penalty for terms found by only one algorithm with low\n"
            "confidence (when the session has 3+ documents).\n\n"
            "Default: -10 points\n\n"
            "More negative: Stronger penalty for single-algorithm terms.\n"
            "Less negative: More lenient toward terms found by one method."
        ),
    },
    # =======================================================================
    # BM25 Algorithm
    # =======================================================================
    "bm25_min_score_threshold": {
        "label": "BM25 minimum score",
        "tooltip": (
            "Minimum BM25 score for a term to be included in results.\n"
            "Terms scoring below this are considered not distinctive enough.\n\n"
            "Default: 2.0\n\n"
            "Increase: Only keep highly distinctive terms (fewer results).\n"
            "Decrease: Include more terms (may include common ones)."
        ),
    },
    "bm25_k1": {
        "label": "BM25 k1 (term frequency saturation)",
        "tooltip": (
            "Controls how quickly term frequency saturates in BM25 scoring.\n"
            "Higher k1 gives more weight to terms appearing many times.\n\n"
            "Default: 1.5\n\n"
            "Increase: More distinction between terms appearing 5 vs 50 times.\n"
            "Decrease: Diminishing returns set in sooner for repeated terms.\n\n"
            "Standard range: 1.2-2.0. Values below 1.0 are unusual."
        ),
    },
    "bm25_b": {
        "label": "BM25 b (length normalization)",
        "tooltip": (
            "How much document length affects scoring. At 1.0, longer\n"
            "documents are penalized heavily. At 0.0, length is ignored.\n\n"
            "Default: 0.75\n\n"
            "Increase: Penalize long documents more (favor short docs).\n"
            "Decrease: Reduce length penalty (favor comprehensive docs).\n\n"
            "Standard range: 0.5-0.8."
        ),
    },
    "bm25_delta": {
        "label": "BM25+ delta",
        "tooltip": (
            "BM25+ improvement factor that prevents zero scores for terms\n"
            "in very long documents. Standard BM25 can score 0 for valid\n"
            "terms; BM25+ adds delta to fix this.\n\n"
            "Default: 1.0\n\n"
            "Increase: Stronger floor on term scores.\n"
            "Decrease: Allow scores closer to zero for long-document terms.\n\n"
            "Set to 0 to disable the BM25+ improvement (use standard BM25)."
        ),
    },
    # =======================================================================
    # Algorithm Weights
    # =======================================================================
    "vocab_weight_ner": {
        "label": "NER algorithm weight",
        "tooltip": (
            "Influence of the NER (Named Entity Recognition) algorithm on\n"
            "final vocabulary confidence scores. NER is best at finding\n"
            "person names, organizations, and locations.\n\n"
            "Default: 1.0\n\n"
            "Increase: NER findings weighted more heavily in merging.\n"
            "Decrease: NER findings have less influence on final scores."
        ),
    },
    "vocab_weight_rake": {
        "label": "RAKE algorithm weight",
        "tooltip": (
            "Influence of the RAKE algorithm on final vocabulary scores.\n"
            "RAKE is best at finding multi-word technical phrases and\n"
            "key terms based on statistical word patterns.\n\n"
            "Default: 0.7\n\n"
            "Increase: More influence from phrase-based extraction.\n"
            "Decrease: Less influence from RAKE results."
        ),
    },
    "vocab_weight_bm25": {
        "label": "BM25 algorithm weight",
        "tooltip": (
            "Influence of BM25 corpus analysis on final vocabulary scores.\n"
            "BM25 identifies terms that are distinctive to this document\n"
            "compared to your corpus of past transcripts.\n\n"
            "Default: 0.8\n\n"
            "Increase: More influence from corpus comparison.\n"
            "Decrease: Less influence from corpus-based scoring.\n\n"
            "Requires 5+ documents in your corpus to be effective."
        ),
    },
    # =======================================================================
    # Deduplication
    # =======================================================================
    "name_similarity_threshold": {
        "label": "Name similarity threshold",
        "tooltip": (
            "Minimum string similarity (0-1) to consider two names as\n"
            "the same person. Used for fuzzy name matching.\n\n"
            "Default: 0.85\n\n"
            "Increase: Stricter matching (fewer false merges, more duplicates).\n"
            "Decrease: More aggressive merging (fewer duplicates, risk of\n"
            "merging different people).\n\n"
            "Example: 'John Smith' vs 'Jon Smith' similarity ~ 0.89."
        ),
    },
    "text_similarity_threshold": {
        "label": "Text duplicate threshold",
        "tooltip": (
            "Minimum similarity to consider two text entries as duplicates.\n"
            "Used for detecting duplicate vocabulary terms.\n\n"
            "Default: 0.80\n\n"
            "Increase: Only merge very similar text (more duplicates survive).\n"
            "Decrease: More aggressive deduplication (risk of removing\n"
            "distinct but similar terms)."
        ),
    },
    "edit_distance_ratio_threshold": {
        "label": "Typo detection sensitivity",
        "tooltip": (
            "Maximum edit distance ratio for detecting typos. Higher values\n"
            "mean more aggressive typo detection (more corrections applied).\n\n"
            "Default: 0.35\n\n"
            "Increase: Detect more typos (may flag intentional variations).\n"
            "Decrease: Only catch obvious typos (fewer false corrections).\n\n"
            "Example: 'Comiskey' vs 'Comisely' edit ratio ~ 0.25."
        ),
    },
    # =======================================================================
    # Document Reading
    # =======================================================================
    "pdf_extraction_mode": {
        "label": "PDF extraction mode",
        "tooltip": (
            "Strategy for reading text from PDF files.\n\n"
            "Default: Hybrid (both extractors)\n\n"
            "Hybrid: Uses both PyMuPDF and pdfplumber, reconciling results\n"
            "with word-level voting. Most accurate but slightly slower.\n\n"
            "PyMuPDF only: Faster, good for clean PDFs.\n"
            "pdfplumber only: Better for complex table layouts."
        ),
    },
    "pdf_voting_enabled": {
        "label": "PDF word-level voting",
        "tooltip": (
            "When both PDF extractors are used (hybrid mode), enable\n"
            "word-level voting to reconcile differences between them.\n\n"
            "Default: Enabled\n\n"
            "Enabled: More accurate text for disputed words.\n"
            "Disabled: Faster processing, uses primary extractor output."
        ),
    },
    "ocr_dpi": {
        "label": "OCR scan resolution",
        "tooltip": (
            "DPI (dots per inch) for scanning images during OCR.\n"
            "Higher DPI produces better text recognition but is slower.\n\n"
            "Default: 300 DPI\n\n"
            "Higher: Better accuracy for small text or poor scans.\n"
            "Lower: Faster processing, acceptable for clear documents.\n\n"
            "300 DPI is the standard for most legal documents."
        ),
    },
    "ocr_confidence_threshold": {
        "label": "OCR confidence threshold",
        "tooltip": (
            "Minimum OCR confidence (%) to keep recognized text.\n"
            "Characters below this confidence are marked as uncertain.\n\n"
            "Default: 70%\n\n"
            "Increase: Only keep high-confidence text (more gaps).\n"
            "Decrease: Keep more text (may include garbled characters).\n\n"
            "Files below this threshold are pre-unchecked in the file list."
        ),
    },
    "ocr_denoise_strength": {
        "label": "OCR denoising strength",
        "tooltip": (
            "Image denoising intensity before OCR processing.\n"
            "Higher values smooth out more noise but may blur text.\n\n"
            "Default: 10\n\n"
            "Increase: Better for noisy/grainy scans (more smoothing).\n"
            "Decrease: Better for clean scans (preserves detail).\n\n"
            "Range: 1 (minimal) to 30 (heavy smoothing)."
        ),
    },
    "ocr_enable_clahe": {
        "label": "OCR contrast enhancement (CLAHE)",
        "tooltip": (
            "Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)\n"
            "to images before OCR. Improves recognition of faded or\n"
            "low-contrast text.\n\n"
            "Default: Enabled\n\n"
            "Enabled: Better for faded documents or uneven lighting.\n"
            "Disabled: Use for already high-contrast scans."
        ),
    },
    # =======================================================================
    # LLM Generation
    # =======================================================================
    "summary_temperature": {
        "label": "Summary temperature",
        "tooltip": (
            "Controls randomness in summary generation.\n"
            "Lower = more factual and consistent.\n"
            "Higher = more varied and creative.\n\n"
            "Default: 0.3\n\n"
            "Increase: More varied language (risk of less accuracy).\n"
            "Decrease: More deterministic output (may be repetitive).\n\n"
            "For legal documents, 0.1-0.4 is recommended."
        ),
    },
    "llm_top_p": {
        "label": "Top-p (nucleus sampling)",
        "tooltip": (
            "Controls the breadth of word choices during generation.\n"
            "At 0.9, the model considers the top 90% probability mass.\n\n"
            "Default: 0.9\n\n"
            "Increase: Broader vocabulary in output.\n"
            "Decrease: More focused, predictable language.\n\n"
            "Works in combination with temperature. Lower top_p with\n"
            "low temperature produces very consistent output."
        ),
    },
    "ollama_timeout_seconds": {
        "label": "LLM timeout (seconds)",
        "tooltip": (
            "Maximum time to wait for the LLM to generate a response.\n"
            "Long documents or complex prompts may need more time.\n\n"
            "Default: 600 seconds (10 minutes)\n\n"
            "Increase: Allow more time for long documents or slow hardware.\n"
            "Decrease: Fail faster if LLM is stuck (retry sooner).\n\n"
            "CPU-only systems may need 900+ seconds for large documents."
        ),
    },
    "summary_length_tolerance": {
        "label": "Summary length tolerance",
        "tooltip": (
            "How much the summary can exceed the target word count before\n"
            "being condensed. Expressed as a fraction of target length.\n\n"
            "Default: 0.20 (20% overage allowed)\n\n"
            "Increase: Allow longer summaries before condensation.\n"
            "Decrease: Stricter adherence to target length.\n\n"
            "Example: With 200 target words and 0.20 tolerance,\n"
            "summaries up to 240 words are accepted."
        ),
    },
    # =======================================================================
    # Chunking
    # =======================================================================
    "retrieval_chunk_size": {
        "label": "Retrieval chunk size (chars)",
        "tooltip": (
            "Characters per chunk for Q&A retrieval indexing.\n"
            "Smaller chunks = more precise retrieval but more fragments.\n\n"
            "Default: 500 characters\n\n"
            "Increase: Larger context per chunk (fewer, broader chunks).\n"
            "Decrease: More precise matching (more, smaller chunks).\n\n"
            "Research suggests 300-600 chars for best precision."
        ),
    },
    "retrieval_chunk_overlap": {
        "label": "Retrieval chunk overlap (chars)",
        "tooltip": (
            "Character overlap between consecutive retrieval chunks.\n"
            "Prevents information loss at chunk boundaries.\n\n"
            "Default: 50 characters\n\n"
            "Increase: Better boundary coverage (more redundancy).\n"
            "Decrease: Less redundancy (risk of split sentences).\n\n"
            "Typical: 10-20% of chunk size."
        ),
    },
    "unified_chunk_min_tokens": {
        "label": "LLM chunk minimum tokens",
        "tooltip": (
            "Minimum tokens per chunk sent to the LLM for processing.\n"
            "Prevents creating fragments too small to be useful.\n\n"
            "Default: 400 tokens\n\n"
            "Increase: Larger minimum chunks (fewer, bigger pieces).\n"
            "Decrease: Allow smaller chunks (more granular processing)."
        ),
    },
    "unified_chunk_target_tokens": {
        "label": "LLM chunk target tokens",
        "tooltip": (
            "Ideal token count per chunk for LLM processing.\n"
            "The chunker aims for this size when splitting documents.\n\n"
            "Default: 700 tokens\n\n"
            "Increase: Larger chunks (more context per LLM call).\n"
            "Decrease: Smaller chunks (more calls, less context each).\n\n"
            "Research: 500-800 tokens optimal for mixed queries."
        ),
    },
    "unified_chunk_max_tokens": {
        "label": "LLM chunk maximum tokens",
        "tooltip": (
            "Maximum tokens per chunk. Chunks exceeding this are split.\n\n"
            "Default: 1000 tokens\n\n"
            "Increase: Allow very large chunks (may hurt retrieval precision).\n"
            "Decrease: Enforce smaller chunks (better precision, more splits).\n\n"
            "Research: >1024 tokens hurts retrieval precision."
        ),
    },
    # =======================================================================
    # Q&A Retrieval
    # =======================================================================
    "qa_retrieval_k": {
        "label": "Chunks to retrieve",
        "tooltip": (
            "Number of document chunks retrieved for each Q&A question.\n"
            "More chunks = broader search but more noise.\n\n"
            "Default: 20 chunks\n\n"
            "Increase: Cast wider net (better recall, more noise).\n"
            "Decrease: Fewer, more relevant chunks (may miss info).\n\n"
            "These chunks are then reranked to find the best matches."
        ),
    },
    "qa_max_tokens": {
        "label": "Maximum answer length (tokens)",
        "tooltip": (
            "Maximum tokens for generated Q&A answers.\n"
            "Controls how long answers can be.\n\n"
            "Default: 750 tokens (~500 words)\n\n"
            "Increase: Allow longer, more detailed answers.\n"
            "Decrease: Shorter, more concise answers.\n\n"
            "Very long answers may include less relevant information."
        ),
    },
    "qa_temperature": {
        "label": "Q&A answer temperature",
        "tooltip": (
            "Creativity/randomness for Q&A answer generation.\n"
            "Lower values produce more factual, consistent answers.\n\n"
            "Default: 0.1\n\n"
            "Increase: More varied phrasing (risk of less accuracy).\n"
            "Decrease: More deterministic, factual answers.\n\n"
            "For legal Q&A, 0.0-0.2 is recommended."
        ),
    },
    "qa_similarity_threshold": {
        "label": "Chunk relevance threshold",
        "tooltip": (
            "Minimum relevance score (0-1) for a chunk to be considered\n"
            "relevant to the question.\n\n"
            "Default: 0.5\n\n"
            "Increase: Only use highly relevant chunks (may miss context).\n"
            "Decrease: Include marginally relevant chunks (more context\n"
            "but potentially more noise)."
        ),
    },
    "retrieval_enable_bm25": {
        "label": "Enable BM25+ retrieval",
        "tooltip": (
            "Use BM25+ lexical (keyword) search for Q&A chunk retrieval.\n"
            "Finds chunks containing exact query terms.\n\n"
            "Default: Enabled\n\n"
            "Enabled: Better for specific terminology lookups.\n"
            "Disabled: Rely solely on semantic (FAISS) search.\n\n"
            "Recommended: Keep enabled for legal/medical documents where\n"
            "exact terminology matters."
        ),
    },
    "retrieval_enable_faiss": {
        "label": "Enable FAISS retrieval",
        "tooltip": (
            "Use FAISS semantic (embedding) search for Q&A chunk retrieval.\n"
            "Finds conceptually related chunks even without exact word matches.\n\n"
            "Default: Enabled\n\n"
            "Enabled: Better for conceptual questions and paraphrasing.\n"
            "Disabled: Rely solely on lexical (BM25+) search.\n\n"
            "Recommended: Keep enabled for natural language questions."
        ),
    },
    "reranking_enabled": {
        "label": "Enable cross-encoder reranking",
        "tooltip": (
            "Re-rank retrieved chunks using a cross-encoder model for\n"
            "more accurate relevance ordering.\n\n"
            "Default: Enabled\n\n"
            "Enabled: More accurate chunk selection (slower, ~100ms).\n"
            "Disabled: Use initial retrieval scores (faster, less precise).\n\n"
            "Cross-encoders are significantly more accurate than\n"
            "bi-encoders for relevance scoring."
        ),
    },
    "reranker_top_k": {
        "label": "Chunks kept after reranking",
        "tooltip": (
            "Number of top chunks to keep after cross-encoder reranking.\n"
            "These are the chunks actually sent to the LLM for answering.\n\n"
            "Default: 5 chunks\n\n"
            "Increase: More context for the LLM (may exceed context window).\n"
            "Decrease: Fewer, most-relevant chunks only.\n\n"
            "From the initial 20 retrieved, only the top-K reranked are used."
        ),
    },
    "retrieval_min_score": {
        "label": "Chunk minimum relevance score",
        "tooltip": (
            "Minimum merged relevance score (0-1) for a chunk to be included\n"
            "in Q&A context. Chunks scoring below this are filtered out.\n\n"
            "Default: 0.10\n\n"
            "Increase: Stricter filtering, only highly relevant chunks.\n"
            "Decrease: Include more marginally relevant chunks.\n\n"
            "After BM25+ and FAISS scores are merged, this is the floor\n"
            "for including a chunk."
        ),
    },
    "retrieval_confidence_gate": {
        "label": "Retrieval confidence gate",
        "tooltip": (
            "Minimum best-chunk score needed to attempt answering a question.\n"
            "If no chunk scores above this, the question is treated as\n"
            "unanswerable for the current documents.\n\n"
            "Default: 0.15\n\n"
            "Increase: More conservative — refuses more questions.\n"
            "Decrease: Attempts answers even with weak evidence.\n\n"
            "This prevents the system from generating bogus answers when\n"
            "the document doesn't contain relevant information."
        ),
    },
    "retrieval_multi_algo_bonus": {
        "label": "Multi-algorithm agreement bonus",
        "tooltip": (
            "Extra score added when both BM25+ and FAISS find the same chunk.\n"
            "Agreement between algorithms signals higher confidence.\n\n"
            "Default: 0.1\n\n"
            "Increase: More reward for cross-algorithm agreement.\n"
            "Decrease: Less influence from algorithm agreement.\n\n"
            "Applied per additional algorithm beyond the first."
        ),
    },
    "faiss_relevance_floor": {
        "label": "FAISS semantic relevance floor",
        "tooltip": (
            "Minimum FAISS (semantic) score required from the best chunk.\n"
            "If no chunk reaches this score, the query has no semantic match\n"
            "and retrieval returns empty (question likely unanswerable).\n\n"
            "Default: 0.10\n\n"
            "Increase: Stricter semantic check — more questions refused.\n"
            "Decrease: More lenient — attempts answers with weaker matches.\n\n"
            "FAISS uses cosine similarity (0=unrelated, 1=identical meaning).\n"
            "Acts as a sanity check before BM25+ results are merged in."
        ),
    },
    "qa_citation_max_chars": {
        "label": "Citation excerpt length",
        "tooltip": (
            "Target character length for the focused citation excerpt.\n"
            "Uses embedding similarity to find the best window in the\n"
            "top retrieval chunk that matches the question.\n\n"
            "Default: 1250 characters (~250 words)\n\n"
            "Increase: Longer excerpt with more surrounding context.\n"
            "Decrease: Shorter, more focused excerpt."
        ),
    },
    # =======================================================================
    # Answer Quality
    # =======================================================================
    "hallucination_verification_enabled": {
        "label": "Hallucination verification",
        "tooltip": (
            "Run hallucination detection on Q&A answers to identify\n"
            "potentially unsupported claims.\n\n"
            "Default: Enabled\n\n"
            "Enabled: Answers are color-coded by reliability. Adds ~100-200ms.\n"
            "Disabled: Answers shown without reliability indicators.\n\n"
            "Uses LettuceDetect to compare answer spans against source text."
        ),
    },
    "answer_rejection_threshold": {
        "label": "Answer rejection threshold",
        "tooltip": (
            "Minimum reliability score (0-1) to show a generated answer.\n"
            "Below this threshold, the answer is rejected as unreliable.\n\n"
            "Default: 0.50 (50% reliability)\n\n"
            "Increase: Only show highly reliable answers (more rejections).\n"
            "Decrease: Show more answers even if uncertain (fewer rejections).\n\n"
            "Rejected answers display: 'Confidence too low, declining to show.'"
        ),
    },
    # =======================================================================
    # Q&A Export
    # =======================================================================
    "qa_export_confidence_floor": {
        "label": "Q&A export confidence floor",
        "tooltip": (
            "Minimum confidence score (0-1) for a Q&A answer to be\n"
            "included in exports. Answers below this threshold are\n"
            "excluded from Export All and combined HTML reports.\n\n"
            "Default: 0.80 (80%)\n\n"
            "Increase: Only export highly confident answers.\n"
            "Decrease: Include more answers even with lower confidence."
        ),
    },
}
