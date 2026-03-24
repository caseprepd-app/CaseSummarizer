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
    "topicrank_max_text_kb": {
        "label": "TopicRank max text size (KB)",
        "tooltip": (
            "Maximum text size in KB for TopicRank processing.\n"
            "Larger texts are truncated to this limit before graph analysis.\n\n"
            "Default: 1000 KB\n\n"
            "Increase: Process more text (slower, more memory).\n"
            "Decrease: Faster processing but may miss terms in long documents."
        ),
    },
    "score_topicrank_centrality_boost": {
        "label": "TopicRank centrality boost",
        "tooltip": (
            "Maximum boost for terms with high TopicRank centrality.\n"
            "TopicRank clusters phrases into topics and runs PageRank\n"
            "on the topic graph to find central concepts.\n\n"
            "Default: +8 points (scaled by centrality score 0-1)\n\n"
            "Higher: Favor terms that are central to document topics.\n"
            "Lower: Reduce influence of graph centrality on ranking."
        ),
    },
    "score_algo_confidence_boost": {
        "label": "Algorithm confidence boost",
        "tooltip": (
            "Maximum bonus points for terms with high algorithm confidence.\n"
            "Uses the best confidence score across YAKE, RAKE, and BM25.\n\n"
            "Default: 6 points (scaled by confidence 0-1)\n\n"
            "Higher: Favor terms where algorithms report high confidence.\n"
            "Lower: Reduce influence of algorithm confidence on ranking."
        ),
    },
    # =======================================================================
    # RAKE Algorithm
    # =======================================================================
    "rake_min_frequency": {
        "label": "RAKE minimum word frequency",
        "tooltip": (
            "Minimum word frequency for RAKE keyword extraction.\n"
            "Words appearing fewer times than this threshold are excluded\n"
            "from candidate keyphrases.\n\n"
            "Default: 3\n\n"
            "Increase: Only consider frequently used words (fewer results).\n"
            "Decrease: Include rarer words in keyphrases (more results)."
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
            "final vocabulary quality scores. NER is best at finding\n"
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
    "vocab_weight_topicrank": {
        "label": "TopicRank algorithm weight",
        "tooltip": (
            "Influence of TopicRank on final vocabulary scores.\n"
            "TopicRank clusters keyphrases into topics and uses PageRank\n"
            "to identify the most central concepts in the document.\n\n"
            "Default: 0.6\n\n"
            "Increase: More influence from graph-based topic analysis.\n"
            "Decrease: Less influence from TopicRank results."
        ),
    },
    "vocab_weight_medical_ner": {
        "label": "MedicalNER algorithm weight",
        "tooltip": (
            "Confidence weight for MedicalNER (scispaCy) algorithm results.\n"
            "MedicalNER uses biomedical NLP models to identify medical\n"
            "terminology, conditions, procedures, and anatomy terms.\n\n"
            "Default: 0.75\n\n"
            "Increase: More influence from medical entity recognition.\n"
            "Decrease: Less influence from MedicalNER results."
        ),
    },
    "vocab_weight_yake": {
        "label": "YAKE algorithm weight",
        "tooltip": (
            "Influence of the YAKE algorithm on final vocabulary scores.\n"
            "YAKE uses pure text statistics (casing, frequency, position)\n"
            "to extract keywords — no model or corpus needed.\n\n"
            "Default: 0.55\n\n"
            "Increase: More influence from statistical keyword extraction.\n"
            "Decrease: Less influence from YAKE results."
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
    # Chunking
    # =======================================================================
    "retrieval_chunk_size": {
        "label": "Retrieval chunk size (chars)",
        "tooltip": (
            "Characters per chunk for BM25+ and FAISS search indexing.\n"
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
        "label": "Chunk minimum tokens",
        "tooltip": (
            "Minimum tokens per chunk. Chunks smaller than this are\n"
            "merged with their neighbor.\n\n"
            "Default: 125 tokens\n\n"
            "Increase: Larger minimum chunks (fewer, bigger pieces).\n"
            "Decrease: Allow smaller chunks (more granular retrieval).\n\n"
            "Research (2025-2026): 200-word fixed chunks match or beat\n"
            "semantic chunking for retrieval tasks."
        ),
    },
    "unified_chunk_target_tokens": {
        "label": "Chunk target tokens",
        "tooltip": (
            "Ideal token count per chunk. The splitter aims for this\n"
            "size when splitting documents at sentence boundaries.\n\n"
            "Default: 225 tokens\n\n"
            "Increase: Larger chunks (more context per passage).\n"
            "Decrease: Smaller chunks (more precise retrieval).\n\n"
            "Research (2025-2026): 200-512 tokens optimal for retrieval."
        ),
    },
    "unified_chunk_max_tokens": {
        "label": "Chunk maximum tokens",
        "tooltip": (
            "Maximum tokens per chunk. Chunks exceeding this are split.\n\n"
            "Default: 325 tokens\n\n"
            "Increase: Allow larger chunks (more context, less precision).\n"
            "Decrease: Enforce smaller chunks (better precision, more splits).\n\n"
            "Research: Keeping chunks under 512 tokens maximizes precision."
        ),
    },
    "unified_chunk_overlap_tokens": {
        "label": "Chunk overlap tokens",
        "tooltip": (
            "Number of tokens from the end of each chunk to repeat at\n"
            "the start of the next chunk. Prevents information loss\n"
            "at chunk boundaries.\n\n"
            "Default: 35 tokens (~15% of target)\n\n"
            "Increase: Better boundary coverage (more redundancy).\n"
            "Decrease: Less redundancy (risk of split context).\n\n"
            "Typical: 10-20% of target chunk size."
        ),
    },
    # =======================================================================
    # Search Retrieval
    # =======================================================================
    "semantic_retrieval_k": {
        "label": "Chunks to retrieve",
        "tooltip": (
            "Number of document chunks retrieved for each search query.\n"
            "More chunks = broader search but more noise.\n\n"
            "Default: 20 chunks\n\n"
            "Increase: Cast wider net (better recall, more noise).\n"
            "Decrease: Fewer, more relevant chunks (may miss info).\n\n"
            "These chunks are then reranked to find the best matches."
        ),
    },
    "semantic_max_tokens": {
        "label": "Context window reserve (tokens)",
        "tooltip": (
            "Tokens reserved for output when budgeting the context window.\n"
            "The remaining space is used for retrieved document chunks.\n\n"
            "Default: 750 tokens\n\n"
            "Increase: Smaller context window for chunks (fewer passages).\n"
            "Decrease: Larger context window for chunks (more passages).\n\n"
            "Most users won't need to change this."
        ),
    },
    "semantic_similarity_threshold": {
        "label": "Chunk relevance threshold",
        "tooltip": (
            "Minimum relevance score (0-1) for a chunk to be considered\n"
            "relevant to the search query.\n\n"
            "Default: 0.5\n\n"
            "Increase: Only use highly relevant chunks (may miss context).\n"
            "Decrease: Include marginally relevant chunks (more context\n"
            "but potentially more noise)."
        ),
    },
    "retrieval_enable_bm25": {
        "label": "Enable BM25+ retrieval",
        "tooltip": (
            "Use BM25+ lexical (keyword) search for chunk retrieval.\n"
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
            "Use FAISS semantic (embedding) search for chunk retrieval.\n"
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
            "These are the chunks used for building the citation excerpt.\n\n"
            "Default: 5 chunks\n\n"
            "Increase: More passages considered for citation (broader context).\n"
            "Decrease: Fewer, most-relevant chunks only.\n\n"
            "From the initial 20 retrieved, only the top-K reranked are used."
        ),
    },
    "retrieval_min_score": {
        "label": "Chunk minimum relevance score",
        "tooltip": (
            "Minimum merged relevance score (0-1) for a chunk to be included\n"
            "in search context. Chunks scoring below this are filtered out.\n\n"
            "Default: 0.10\n\n"
            "Increase: Stricter filtering, only highly relevant chunks.\n"
            "Decrease: Include more marginally relevant chunks.\n\n"
            "After BM25+ and FAISS scores are merged, this is the floor\n"
            "for including a chunk."
        ),
    },
    "retrieval_relevance_gate": {
        "label": "Retrieval relevance gate",
        "tooltip": (
            "Minimum best-chunk relevance score needed to attempt answering\n"
            "a query. If no chunk scores above this, the query is treated as\n"
            "unanswerable for the current documents.\n\n"
            "Default: 0.50\n\n"
            "Increase: More conservative — refuses more queries.\n"
            "Decrease: Attempts answers even with weak evidence.\n\n"
            "This prevents the system from showing bogus answers when\n"
            "the document doesn't contain relevant information."
        ),
    },
    "faiss_relevance_floor": {
        "label": "FAISS semantic relevance floor",
        "tooltip": (
            "Minimum FAISS (semantic) score required from the best chunk.\n"
            "If no chunk reaches this score, the query has no semantic match\n"
            "and retrieval returns empty (question likely unanswerable).\n\n"
            "Default: 0.25\n\n"
            "Increase: Stricter semantic check — more questions refused.\n"
            "Decrease: More lenient — attempts answers with weaker matches.\n\n"
            "FAISS uses cosine similarity (0=unrelated, 1=identical meaning).\n"
            "Acts as a sanity check before BM25+ results are merged in."
        ),
    },
    "semantic_citation_max_chars": {
        "label": "Citation excerpt length",
        "tooltip": (
            "Target character length for the focused citation excerpt.\n"
            "Uses embedding similarity to find the best window in the\n"
            "top retrieval chunk that matches the query.\n\n"
            "Default: 1250 characters (~250 words)\n\n"
            "Increase: Longer excerpt with more surrounding context.\n"
            "Decrease: Shorter, more focused excerpt."
        ),
    },
    # =======================================================================
    # Export
    # =======================================================================
    "semantic_export_relevance_floor": {
        "label": "Search relevance floor",
        "tooltip": (
            "Minimum retrieval relevance (0-1) for a semantic search\n"
            "result to be included in exports. This measures how relevant\n"
            "the retrieved document chunks were to the question.\n\n"
            "Default: 0.51 (51%)\n\n"
            "Increase: Require stronger document matches.\n"
            "Decrease: Allow weaker matches in exports."
        ),
    },
}
