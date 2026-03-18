"""
Default configuration values for CasePrepd Advanced settings.

Single source of truth for tunable parameters that affect output quality
or processing behavior. Each entry includes value, type, range constraints,
and category for UI grouping.

Used by:
- src/config.py: Imports default values for module-level constants
- src/ui/settings/advanced_registry.py: Reads metadata for widget generation
"""

DEFAULTS = {
    # =======================================================================
    # ML Training
    # =======================================================================
    "ml_min_samples": {
        "value": 30,
        "min": 5,
        "max": 100,
        "type": "int",
        "category": "ML Training",
    },
    "ml_ensemble_min_samples": {
        "value": 40,
        "min": 20,
        "max": 150,
        "type": "int",
        "category": "ML Training",
    },
    "ml_decay_half_life_days": {
        "value": 1270,
        "min": 365,
        "max": 3650,
        "step": 10,
        "type": "int",
        "category": "ML Training",
    },
    "ml_decay_weight_floor": {
        "value": 0.55,
        "min": 0.1,
        "max": 0.9,
        "step": 0.05,
        "type": "float",
        "category": "ML Training",
    },
    # =======================================================================
    # Quality Scoring
    # =======================================================================
    "score_multi_doc_boost": {
        "value": 10,
        "min": 0,
        "max": 30,
        "type": "int",
        "category": "Quality Scoring",
    },
    "score_high_conf_boost": {
        "value": 5,
        "min": 0,
        "max": 20,
        "type": "int",
        "category": "Quality Scoring",
    },
    "score_all_low_conf_penalty": {
        "value": -10,
        "min": -30,
        "max": 0,
        "type": "int",
        "category": "Quality Scoring",
    },
    "score_single_source_penalty": {
        "value": -10,
        "min": -30,
        "max": 0,
        "type": "int",
        "category": "Quality Scoring",
    },
    "topicrank_max_text_kb": {
        "value": 1000,
        "min": 200,
        "max": 5000,
        "step": 100,
        "type": "int",
        "category": "Quality Scoring",
    },
    "score_topicrank_centrality_boost": {
        "value": 8,
        "min": 0,
        "max": 15,
        "type": "int",
        "category": "Quality Scoring",
    },
    "score_algo_confidence_boost": {
        "value": 6,
        "min": 0,
        "max": 15,
        "type": "int",
        "category": "Quality Scoring",
    },
    # =======================================================================
    # RAKE Algorithm
    # =======================================================================
    "rake_min_frequency": {
        "value": 3,
        "min": 1,
        "max": 10,
        "type": "int",
        "category": "RAKE Algorithm",
    },
    # =======================================================================
    # BM25 Algorithm
    # =======================================================================
    "bm25_min_score_threshold": {
        "value": 2.0,
        "min": 0.5,
        "max": 5.0,
        "step": 0.1,
        "type": "float",
        "category": "BM25 Algorithm",
    },
    "bm25_k1": {
        "value": 1.5,
        "min": 0.5,
        "max": 3.0,
        "step": 0.1,
        "type": "float",
        "category": "BM25 Algorithm",
    },
    "bm25_b": {
        "value": 0.75,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "type": "float",
        "category": "BM25 Algorithm",
    },
    "bm25_delta": {
        "value": 1.0,
        "min": 0.0,
        "max": 2.0,
        "step": 0.1,
        "type": "float",
        "category": "BM25 Algorithm",
    },
    # =======================================================================
    # Algorithm Weights
    # =======================================================================
    "vocab_weight_ner": {
        "value": 1.0,
        "min": 0.0,
        "max": 2.0,
        "step": 0.1,
        "type": "float",
        "category": "Algorithm Weights",
    },
    "vocab_weight_rake": {
        "value": 0.7,
        "min": 0.0,
        "max": 2.0,
        "step": 0.1,
        "type": "float",
        "category": "Algorithm Weights",
    },
    "vocab_weight_bm25": {
        "value": 0.8,
        "min": 0.0,
        "max": 2.0,
        "step": 0.1,
        "type": "float",
        "category": "Algorithm Weights",
    },
    "vocab_weight_topicrank": {
        "value": 0.6,
        "min": 0.0,
        "max": 2.0,
        "step": 0.1,
        "type": "float",
        "category": "Algorithm Weights",
    },
    "vocab_weight_medical_ner": {
        "value": 0.75,
        "min": 0.0,
        "max": 2.0,
        "step": 0.1,
        "type": "float",
        "category": "Algorithm Weights",
    },
    "vocab_weight_yake": {
        "value": 0.55,
        "min": 0.0,
        "max": 2.0,
        "step": 0.05,
        "type": "float",
        "category": "Algorithm Weights",
    },
    # =======================================================================
    # Deduplication
    # =======================================================================
    "name_similarity_threshold": {
        "value": 0.85,
        "min": 0.5,
        "max": 1.0,
        "step": 0.05,
        "type": "float",
        "category": "Deduplication",
    },
    "text_similarity_threshold": {
        "value": 0.80,
        "min": 0.5,
        "max": 1.0,
        "step": 0.05,
        "type": "float",
        "category": "Deduplication",
    },
    "edit_distance_ratio_threshold": {
        "value": 0.35,
        "min": 0.1,
        "max": 0.7,
        "step": 0.05,
        "type": "float",
        "category": "Deduplication",
    },
    # =======================================================================
    # Document Reading
    # =======================================================================
    "pdf_extraction_mode": {
        "value": "hybrid",
        "options": [
            ("Hybrid (both extractors)", "hybrid"),
            ("PyMuPDF only", "pymupdf_only"),
            ("pdfplumber only", "pdfplumber_only"),
        ],
        "type": "dropdown",
        "category": "Document Reading",
    },
    "pdf_voting_enabled": {
        "value": True,
        "type": "bool",
        "category": "Document Reading",
    },
    "ocr_dpi": {
        "value": 300,
        "options": [
            ("150 DPI (fast, lower quality)", 150),
            ("200 DPI", 200),
            ("300 DPI (recommended)", 300),
            ("400 DPI", 400),
            ("600 DPI (slow, highest quality)", 600),
        ],
        "type": "dropdown_int",
        "category": "Document Reading",
    },
    "ocr_confidence_threshold": {
        "value": 70,
        "min": 30,
        "max": 95,
        "step": 5,
        "type": "int",
        "category": "Document Reading",
    },
    "ocr_denoise_strength": {
        "value": 10,
        "min": 1,
        "max": 30,
        "type": "int",
        "category": "Document Reading",
    },
    "ocr_enable_clahe": {
        "value": True,
        "type": "bool",
        "category": "Document Reading",
    },
    # =======================================================================
    # Chunking
    # =======================================================================
    "retrieval_chunk_size": {
        "value": 500,
        "min": 200,
        "max": 1000,
        "step": 50,
        "type": "int",
        "category": "Chunking",
    },
    "retrieval_chunk_overlap": {
        "value": 50,
        "min": 0,
        "max": 200,
        "step": 10,
        "type": "int",
        "category": "Chunking",
    },
    # Chunk sizes tuned for meaningful paragraph-length excerpts:
    # 150-token target ≈ one solid paragraph (~100-160 words)
    # Balances excerpt readability with search precision
    "unified_chunk_min_tokens": {
        "value": 75,
        "min": 20,
        "max": 150,
        "step": 10,
        "type": "int",
        "category": "Chunking",
    },
    "unified_chunk_target_tokens": {
        "value": 150,
        "min": 30,
        "max": 300,
        "step": 5,
        "type": "int",
        "category": "Chunking",
    },
    "unified_chunk_max_tokens": {
        "value": 225,
        "min": 50,
        "max": 400,
        "step": 10,
        "type": "int",
        "category": "Chunking",
    },
    # ~15% of target_tokens; Firecrawl 2026 recommends 10-20% overlap for RAG
    "unified_chunk_overlap_tokens": {
        "value": 22,
        "min": 0,
        "max": 60,
        "step": 5,
        "type": "int",
        "category": "Chunking",
    },
    # =======================================================================
    # Search Retrieval
    # =======================================================================
    "semantic_retrieval_k": {
        "value": 20,
        "min": 5,
        "max": 50,
        "type": "int",
        "category": "Search Retrieval",
    },
    "semantic_max_tokens": {
        "value": 750,
        "min": 200,
        "max": 2000,
        "step": 50,
        "type": "int",
        "category": "Search Retrieval",
    },
    "semantic_similarity_threshold": {
        "value": 0.5,
        "min": 0.1,
        "max": 0.9,
        "step": 0.05,
        "type": "float",
        "category": "Search Retrieval",
    },
    "retrieval_enable_bm25": {
        "value": True,
        "type": "bool",
        "category": "Search Retrieval",
    },
    "retrieval_enable_faiss": {
        "value": True,
        "type": "bool",
        "category": "Search Retrieval",
    },
    "reranking_enabled": {
        "value": True,
        "type": "bool",
        "category": "Search Retrieval",
    },
    "reranker_top_k": {
        "value": 5,
        "min": 3,
        "max": 20,
        "type": "int",
        "category": "Search Retrieval",
    },
    "retrieval_min_score": {
        "value": 0.10,
        "min": 0.0,
        "max": 0.5,
        "step": 0.01,
        "type": "float",
        "category": "Search Retrieval",
    },
    "retrieval_relevance_gate": {
        "value": 0.50,
        "min": 0.05,
        "max": 0.8,
        "step": 0.01,
        "type": "float",
        "category": "Search Retrieval",
    },
    "faiss_relevance_floor": {
        "value": 0.25,
        "min": 0.0,
        "max": 0.5,
        "step": 0.01,
        "type": "float",
        "category": "Search Retrieval",
    },
    "semantic_citation_max_chars": {
        "value": 1250,
        "min": 250,
        "max": 5000,
        "step": 250,
        "type": "int",
        "category": "Search Retrieval",
    },
    # =======================================================================
    # Export
    # =======================================================================
    "semantic_export_relevance_floor": {
        "value": 0.51,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "type": "float",
        "category": "Export",
    },
}


def get_default(key: str):
    """
    Get the default value for a setting key.

    Args:
        key: Setting key from DEFAULTS dict.

    Returns:
        The default value.

    Raises:
        KeyError: If key not found in DEFAULTS.
    """
    return DEFAULTS[key]["value"]


def get_categories() -> list[str]:
    """
    Get ordered list of unique categories.

    Returns:
        List of category names in definition order.
    """
    seen = []
    for entry in DEFAULTS.values():
        cat = entry["category"]
        if cat not in seen:
            seen.append(cat)
    return seen
