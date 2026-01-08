"""
LocalScribe Configuration Module
Centralized configuration for the application.
"""

import os
from pathlib import Path

import yaml

# Debug Mode Configuration
# TEMP: Hard-coded for Session 77 debugging and feedback collection
DEBUG_MODE = True  # os.environ.get('DEBUG', 'false').lower() == 'true'

# Application Paths
APP_NAME = "LocalScribe"
APPDATA_DIR = Path(os.environ.get("APPDATA", os.path.expanduser("~/.config"))) / APP_NAME
MODELS_DIR = APPDATA_DIR / "models"
CACHE_DIR = APPDATA_DIR / "cache"
LOGS_DIR = APPDATA_DIR / "logs"
CONFIG_DIR = APPDATA_DIR / "config"

# Data directory for ML training data
DATA_DIR = APPDATA_DIR / "data"

# Ensure directories exist
for directory in [APPDATA_DIR, MODELS_DIR, CACHE_DIR, LOGS_DIR, CONFIG_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Processing Metrics CSV (for future ML prediction of processing time)
PROCESSING_METRICS_CSV = DATA_DIR / "processing_metrics.csv"

# Feedback and ML Configuration (Session 25, updated Session 55)
FEEDBACK_DIR = DATA_DIR / "feedback"
MODELS_ML_DIR = DATA_DIR / "models"  # ML models (separate from Ollama models)
VOCAB_MODEL_PATH = MODELS_ML_DIR / "vocab_meta_learner.pkl"

# Two-file feedback system (Session 55)
# - Default feedback ships with app (developer's training data)
# - User feedback is collected during normal use
DEFAULT_FEEDBACK_CSV = Path(__file__).parent.parent / "config" / "default_feedback.csv"
USER_FEEDBACK_CSV = FEEDBACK_DIR / "user_feedback.csv"

# Legacy path - kept for backward compatibility detection only
VOCAB_FEEDBACK_CSV = FEEDBACK_DIR / "vocab_feedback.csv"

# Default model path (bundled with app for reset functionality)
# This model is trained by the developer and shipped with the app
# Users can reset to this model if they accidentally train in a bad direction
DEFAULT_VOCAB_MODEL_PATH = Path(__file__).parent.parent / "config" / "default_vocab_model.pkl"

# ML Training Thresholds (Session 84)
# Don't train until we have enough samples to matter.
ML_MIN_SAMPLES = 30  # Minimum samples before ML training starts
ML_ENSEMBLE_MIN_SAMPLES = 40  # Minimum samples to enable ensemble (LR + RF)
ML_RETRAIN_THRESHOLD = 1  # Retrain on ANY new user feedback (was 10)

# Graduated RF Weight in Ensemble (Session 84)
# RF starts with low weight and increases as samples grow.
# Below 200 samples: fixed weight blend. At 200+: confidence-weighted blend.
# Thresholds: (min_samples, rf_weight) - finds first threshold where count < min
ML_RF_WEIGHT_THRESHOLDS = [
    (40, 0.0),  # < 40 samples: LR only (RF not trained)
    (60, 0.10),  # 40-59 samples: 10% RF
    (100, 0.20),  # 60-99 samples: 20% RF
    (150, 0.30),  # 100-149 samples: 30% RF
    (200, 0.40),  # 150-199 samples: 40% RF
    # 200+: confidence_weighted_blend (dynamic based on model confidence)
]

# ML Time Decay Configuration (Session 47)
# Older feedback is weighted less to adapt to changing user preferences
# - Half-life: Tuned so weight reaches floor at 3 years
# - Floor: Minimum weight (old feedback still matters, just less)
# Rationale: Most early feedback flags universal false positives (common words)
# which should persist. Reporters change courthouses ~every few years.
#
# Decay curve:
#   Today: 1.00 → 1 year: 0.82 → 2 years: 0.67 → 3 years: 0.55 (floor)
ML_DECAY_HALF_LIFE_DAYS = 1270  # ~3.5 years - tuned so weight hits floor at 3 years
ML_DECAY_WEIGHT_FLOOR = 0.55  # Old feedback retains 55% weight minimum

# Graduated ML Weight (Session 84)
# ML influence on final score increases with user's training corpus size.
# Formula: score = base_score * (1 - ml_weight) + ml_probability * 100 * ml_weight
# Thresholds: (min_samples, ml_weight) - finds first threshold where count < min
#
# Conservative ramp: pure rules until 30 samples, then gradual handover.
# ML caps at 80% - rules always have 20% say as a safety net.
ML_WEIGHT_THRESHOLDS = [
    (30, 0.0),  # 0-29 samples: pure rules (no ML)
    (41, 0.40),  # 30-40 samples: 40% ML
    (61, 0.50),  # 41-60 samples: 50% ML
    (101, 0.60),  # 61-100 samples: 60% ML
    (float("inf"), 0.80),  # 100+ samples: 80% ML (cap)
]

# Source-Based Training Weights (Session 76)
# User feedback is weighted higher than shipped default data from the FIRST observation.
# This personalizes the model immediately while keeping defaults as a stable baseline.
# Default data is never deleted, just gradually de-emphasized as user adds more data.
#
# Thresholds: (min_user_samples, default_weight, user_weight)
# Influence with 30 defaults: 1 user@1.5x = ~5%, 10 user@2x = ~40%, 100 user@3.5x = ~94%
ML_SOURCE_WEIGHTS = [
    (1, 1.0, 1.0),  # 0 user samples: only default data exists
    (3, 1.0, 1.5),  # 1-2 samples: user 1.5x (early boost)
    (10, 1.0, 2.0),  # 3-9 samples: user 2x
    (25, 0.95, 2.5),  # 10-24 samples: user 2.5x, default starts dropping
    (50, 0.9, 3.0),  # 25-49 samples: user 3x
    (100, 0.8, 3.5),  # 50-99 samples: user 3.5x
    (200, 0.7, 4.0),  # 100-199 samples: user 4x
    (float("inf"), 0.6, 5.0),  # 200+ samples: user 5x, default 0.6x
]

# Rule-Based Quality Score: TermSources Adjustments (Session 79)
# These adjustments are applied BEFORE ML blending, based on document source quality.
# All values are additive to the base score (50 points).
SCORE_MULTI_DOC_BOOST = 10  # Bonus for terms found in 2+ documents
SCORE_HIGH_CONF_BOOST = 5  # Bonus if high_conf_doc_ratio > 0.8
SCORE_ALL_LOW_CONF_PENALTY = -10  # Penalty if ALL sources have confidence < 0.60
SCORE_SINGLE_SOURCE_PENALTY = -10  # Penalty for single low-conf source (conditional)
SCORE_SINGLE_SOURCE_MIN_DOCS = 3  # Only apply single-source penalty when session has 3+ docs
SCORE_SINGLE_SOURCE_CONF_THRESHOLD = 0.70  # Confidence threshold for single-source penalty

# Ensure ML directories exist
for ml_dir in [FEEDBACK_DIR, MODELS_ML_DIR]:
    ml_dir.mkdir(parents=True, exist_ok=True)

# BM25 Corpus Configuration (Session 26)
# User's corpus of previous transcripts for BM25-based term importance
CORPUS_DIR = APPDATA_DIR / "corpus"
CORPUS_MIN_DOCUMENTS = 5  # Minimum docs before BM25 activates
BM25_ENABLED = True  # User can disable in settings
BM25_MIN_SCORE_THRESHOLD = 2.0  # Minimum BM25 score to include term
BM25_WEIGHT = 0.8  # Legacy - kept for backward compatibility

# BM25+ Algorithm Parameters (unified across vocabulary and retrieval)
# Using BM25+ parameters which are strictly better than standard BM25
# (delta parameter prevents zero scores for very long documents)
BM25_K1 = 1.5  # Term frequency saturation (higher = more weight on repeated terms)
BM25_B = 0.75  # Length normalization (0 = no normalization, 1 = full normalization)
BM25_DELTA = 1.0  # BM25+ improvement factor (prevents zero scores)

# Corpus Familiarity Filtering Configuration (Session 68)
# Filters terms that appear too frequently across the user's corpus.
# Terms above threshold are removed (user already knows them).
# Terms below threshold get corpus_familiarity_score as ML feature.
CORPUS_FAMILIARITY_THRESHOLD = 0.75  # Filter terms in 75%+ of corpus docs
CORPUS_FAMILIARITY_MIN_DOCS = 10  # Alternative: filter if in 10+ docs
CORPUS_FAMILIARITY_EXEMPT_PERSONS = True  # Exempt person names from filtering

# Vocabulary Extraction Algorithm Weights (Session 47)
# Centralized weights for multi-algorithm vocabulary extraction
# Higher weight = more influence on final confidence score
# These weights are used by ResultMerger to combine algorithm results
VOCAB_ALGORITHM_WEIGHTS = {
    "NER": 1.0,  # Primary - spaCy NER, most precise for names/entities
    "RAKE": 0.7,  # Secondary - good for multi-word technical phrases
    "BM25": 0.8,  # Corpus-based term importance (requires 5+ docs)
}

# Ensure corpus directory exists
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

# File Processing Limits
MAX_FILE_SIZE_MB = 500
LARGE_FILE_WARNING_MB = 100
MIN_LINE_LENGTH = 15
MIN_DICTIONARY_CONFIDENCE = 60  # Percentage

# PDF Extraction Configuration (Session 79)
# Hybrid extraction uses both PyMuPDF and pdfplumber, reconciling with word-level voting
PDF_EXTRACTION_MODE = "hybrid"  # "hybrid", "pymupdf_only", "pdfplumber_only"
PDF_VOTING_ENABLED = True  # Enable word-level voting when both extractors succeed

# OCR Configuration
OCR_DPI = 300
OCR_CONFIDENCE_THRESHOLD = 70  # Files below this are pre-unchecked

# OCR Image Preprocessing Configuration
# Preprocessing can improve OCR accuracy by 20-50% for scanned documents
# Reference: https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
OCR_PREPROCESSING_ENABLED = True  # Enable image preprocessing before OCR
OCR_DENOISE_STRENGTH = 10  # Denoising strength (1-30, higher = more smoothing)
OCR_ENABLE_CLAHE = True  # Enable CLAHE contrast enhancement

# AI Model Configuration
OLLAMA_API_BASE = "http://localhost:11434"  # Default Ollama API endpoint
OLLAMA_MODEL_NAME = "gemma3:1b"  # Default model for the application
OLLAMA_MODEL_FALLBACK = "gemma3:1b"  # Fallback if the primary model fails
OLLAMA_TIMEOUT_SECONDS = 600  # 10 minutes for long summaries
QUEUE_TIMEOUT_SECONDS = 2.0  # Timeout for multiprocessing queue operations

# Context Window Configuration
# Session 64: Now dynamically set based on GPU VRAM via user preferences.
# This constant is only used as a fallback if user preferences are unavailable.
# Actual context size is determined by: user_preferences.get_effective_context_size()
# which auto-detects optimal size based on VRAM (4K-64K range).
OLLAMA_CONTEXT_WINDOW = 4000  # Fallback default (conservative, CPU-safe)

# --- New Model Configuration System ---
MODEL_CONFIG_FILE = Path(__file__).parent.parent / "config" / "models.yaml"
MODEL_CONFIGS = {}


def load_model_configs():
    """Loads model configurations from config/models.yaml."""
    global MODEL_CONFIGS
    try:
        with open(MODEL_CONFIG_FILE) as f:
            data = yaml.safe_load(f)
            MODEL_CONFIGS = data.get("models", {})
        if DEBUG_MODE and MODEL_CONFIGS:
            from src.logging_config import debug_log

            debug_log(
                f"[Config] Loaded {len(MODEL_CONFIGS)} model configurations from {MODEL_CONFIG_FILE}"
            )
    except FileNotFoundError:
        if DEBUG_MODE:
            from src.logging_config import debug_log

            debug_log(
                f"[Config] WARNING: Model config file not found at {MODEL_CONFIG_FILE}. Using fallback values."
            )
        MODEL_CONFIGS = {}
    except Exception as e:
        from src.logging_config import debug_log

        debug_log(f"[Config] ERROR: Failed to load or parse model config file: {e}")
        MODEL_CONFIGS = {}


def get_model_config(model_name: str) -> dict:
    """
    Returns the configuration for a specific model, with fallbacks.

    Args:
        model_name: The name of the model (e.g., 'gemma3:1b').

    Returns:
        A dictionary containing the model's configuration.
    """
    if not MODEL_CONFIGS:
        load_model_configs()

    # 1. Try to find the exact model name
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]

    # 2. Fallback for base names (e.g., user has 'gemma3:1b-instruct', config has 'gemma3:1b')
    base_name = model_name.split(":")[0]
    for name, config in MODEL_CONFIGS.items():
        if name.startswith(base_name):
            if DEBUG_MODE:
                from src.logging_config import debug_log

                debug_log(
                    f"[Config] Found partial match for '{model_name}': using config for '{name}'."
                )
            return config

    # 3. Fallback to the default model name if no match found
    if OLLAMA_MODEL_NAME in MODEL_CONFIGS:
        if DEBUG_MODE:
            from src.logging_config import debug_log

            debug_log(
                f"[Config] WARNING: Model '{model_name}' not found. Falling back to default model '{OLLAMA_MODEL_NAME}'."
            )
        return MODEL_CONFIGS[OLLAMA_MODEL_NAME]

    # 4. Absolute fallback if config is empty or default is missing
    if DEBUG_MODE:
        from src.logging_config import debug_log

        debug_log(
            "[Config] WARNING: No model configurations found. Using hard-coded fallback values."
        )
    return {
        "context_window": 4096,
        "max_input_tokens": 2048,
    }


# Note: Model configs loaded lazily by get_model_config() to avoid circular import
# --- End New Model Configuration System ---


# Default Processing Settings
DEFAULT_SUMMARY_WORDS = 200
MIN_SUMMARY_WORDS = 100
MAX_SUMMARY_WORDS = 500

# Summary Length Enforcement Settings
# When a generated summary exceeds target by more than TOLERANCE, it will be condensed
SUMMARY_LENGTH_TOLERANCE = 0.20  # 20% overage allowed (200 words → accepts up to 240)
SUMMARY_MAX_CONDENSE_ATTEMPTS = 3  # Maximum condensation attempts before returning best effort

# Data Files
GOOGLE_FREQ_LIST = Path(__file__).parent.parent / "data" / "frequency" / "google_word_freq.txt"
LEGAL_KEYWORDS_NY = Path(__file__).parent.parent / "data" / "keywords" / "legal_keywords_ny.txt"
LEGAL_KEYWORDS_CA = Path(__file__).parent.parent / "data" / "keywords" / "legal_keywords_ca.txt"

# New: Vocabulary Extractor Data Files
LEGAL_EXCLUDE_LIST_PATH = Path(__file__).parent.parent / "config" / "legal_exclude.txt"
MEDICAL_TERMS_LIST_PATH = Path(__file__).parent.parent / "config" / "medical_terms.txt"
# User-specific vocabulary exclusions (stored in AppData, user can add via right-click)
USER_VOCAB_EXCLUDE_PATH = CONFIG_DIR / "user_vocab_exclude.txt"

# Vocabulary Extraction Rarity Settings
# Path to Google word frequency dataset (word\tfrequency_count format)
# Moved to data/frequency/ in Session 34 for better organization
GOOGLE_WORD_FREQUENCY_FILE = (
    Path(__file__).parent.parent / "data" / "frequency" / "Word_rarity-count_1w.txt"
)
# Words with rank >= threshold are considered rare
# Higher threshold = more aggressive filtering (fewer terms extracted)
# 150000 = bottom 55% of vocabulary (original, too permissive)
# 180000 = bottom 46% of vocabulary (balanced)
# 200000 = bottom 40% of vocabulary (aggressive)
# Examples: "medical" (rank 501) FILTERED, "adenocarcinoma" (rank >180K) EXTRACTED
# Set to -1 to disable frequency-based filtering (use WordNet only)
VOCABULARY_RARITY_THRESHOLD = 180000
# Vocabulary sort method: "quality_score" (ML-boosted, default) or "rarity" (rare words first)
# "quality_score": Sorts by Quality Score descending - ML predictions push good terms up
# "rarity": Sorts by word frequency (rare words first) - useful before ML model is trained
VOCABULARY_SORT_METHOD = "quality_score"

# Legacy setting for backwards compatibility (deprecated - use VOCABULARY_SORT_METHOD)
VOCABULARY_SORT_BY_RARITY = VOCABULARY_SORT_METHOD == "rarity"

# Minimum occurrences for term extraction (filters single-occurrence OCR errors/typos)
# Set to 1 to disable (extract all terms regardless of frequency)
# Set to 2 to require terms appear at least twice (recommended - filters OCR errors)
# Set to 3+ for very conservative filtering
# Note: PERSON entities are exempt (party names may appear once but are important)
VOCABULARY_MIN_OCCURRENCES = 2

# Phrase Component Rarity Filtering (Session 53)
# Filters multi-word phrases where ALL component words are too common.
# Example: "the same", "left side" - high RAKE scores but no vocabulary value.
#
# KEY INSIGHT: If a phrase has even ONE rare word, it might be worth keeping.
# We only filter when ALL words are common.
#
# Commonality scores are 0.0-1.0 (log-scaled from Google word frequency):
# RANK-BASED SCORING (Session 58):
# Score = rank / total_words (percentile position)
#   0.0 = most common word ("the", rank 1)
#   0.5 = median word (top 50%)
#   1.0 = rarest word in dataset
#
# This answers: "What percentage of English words are more common than this?"
# Court reporters know common English; they need only specialized terms.
#
# PHRASE MAX threshold: Filter if even the RAREST word is in top X%
#   0.50 = balanced (filter if all words in top 50%)
#   0.30 = aggressive (filter if all words in top 30%)
#
# PHRASE MEAN threshold: Filter if AVERAGE word is in top X%
#   0.40 = balanced
#   0.30 = aggressive
#
# SINGLE WORD threshold: Filter if word is in top X%
#   0.50 = filter top 50% of English vocabulary
#
# Person names are exempt (names like "John Smith" use common words but are valuable)
PHRASE_MAX_COMMONALITY_THRESHOLD = 0.50  # Filter if rarest word in top 50%
PHRASE_MEAN_COMMONALITY_THRESHOLD = 0.40  # Filter if average word in top 40%

# Single-word rarity threshold (Session 58 - rank-based)
# Filter words in the top X% as "too common for vocabulary prep"
# Examples with 0.50 threshold:
#   "age" (rank 579, score 0.0017) < 0.50 -> FILTERED (top 0.17%)
#   "cervical" (rank ~50000, score 0.15) < 0.50 -> FILTERED (top 15%)
#   "radiculopathy" (rank ~250000, score 0.75) > 0.50 -> KEPT (bottom 25%)
SINGLE_WORD_COMMONALITY_THRESHOLD = 0.50  # Filter top 50% of vocabulary

# GUI Display Limits for Vocabulary Table
# Based on tkinter Treeview performance testing:
# - < 100 rows: Excellent performance
# - 100-200 rows: Generally acceptable
# - 200+ rows: Performance degrades, especially with text wrapping
# Default: 50 rows (conservative for responsiveness)
# Maximum ceiling: 200 rows (hard limit to prevent GUI freezing)
VOCABULARY_DISPLAY_LIMIT = 50  # User-configurable default (conservative)
VOCABULARY_DISPLAY_MAX = 200  # Hard ceiling - cannot exceed this

# Vocabulary Display Pagination (Session 16 - GUI responsiveness)
# Controls async batch insertion to prevent GUI freezing during large loads
VOCABULARY_ROWS_PER_PAGE = 50  # Initial rows shown; "Load More" adds more
VOCABULARY_BATCH_INSERT_SIZE = 20  # Rows inserted per async batch
VOCABULARY_BATCH_INSERT_DELAY_MS = 10  # Delay between batches (ms)

# spaCy Model Download Timeouts (Session 15)
# Controls timeout behavior during automatic spaCy model downloads
SPACY_DOWNLOAD_TIMEOUT_SEC = 600  # Overall timeout: 10 minutes
SPACY_SOCKET_TIMEOUT_SEC = 10  # Socket timeout per request
SPACY_THREAD_TIMEOUT_SEC = 15  # Thread termination timeout

# Document Chunking (Session 20 - hierarchical summarization)
# Overlap fraction prevents context loss at chunk boundaries
CHUNK_OVERLAP_FRACTION = 0.1  # 10% overlap between chunks

# System Monitor Color Thresholds (CPU and RAM)
# Used for color-coded status indicators in the system monitor widget
# Applied independently to both CPU and RAM percentages
SYSTEM_MONITOR_THRESHOLD_GREEN = 75  # 0-74%: Green (healthy)
SYSTEM_MONITOR_THRESHOLD_YELLOW = 85  # 75-84%: Yellow (elevated)
SYSTEM_MONITOR_THRESHOLD_CRITICAL = 90  # 90%+: Red with "!" indicator

# Vocabulary Extraction Performance Settings
# Max text size in KB for spaCy NLP processing
# spaCy processes ~10-20K words/sec; 200KB ≈ 35K words ≈ 2-3 seconds
# Larger documents are truncated (still captures most named entities from early pages)
VOCABULARY_MAX_TEXT_KB = 200  # 200KB max for NLP processing (200,000 characters)

# spaCy batch processing - higher values process faster with more memory
# Testing shows: batch_size=4 (baseline), 8 (~17% faster), 16 (~25% faster but +100MB RAM)
# Default: 8 for optimal balance on 8-16GB systems
VOCABULARY_BATCH_SIZE = 8

# Parallel Processing Configuration
# Controls concurrent document extraction for multi-file workflows
#
# User Override Options:
# - USER_PICKS_MAX_WORKER_COUNT: If True, use USER_DEFINED_MAX_WORKER_COUNT
#   instead of auto-detection. Default: False (auto-detect based on CPU)
# - USER_DEFINED_MAX_WORKER_COUNT: Manual worker count when override enabled.
#   Range: 1-8. Default: 2 (conservative for most systems)
#
# Auto-detection (when USER_PICKS_MAX_WORKER_COUNT=False):
# - Uses min(cpu_count, 4) - caps at 4 for memory safety
# - Memory profile: Each document can use 200-500MB during processing
# - With 4 workers: ~2.1GB peak memory usage (safe for 8GB systems)

# User override settings (change these to customize)
USER_PICKS_MAX_WORKER_COUNT = False  # Set to True to use manual worker count
USER_DEFINED_MAX_WORKER_COUNT = 2  # Manual count when override enabled (1-8)

# Enforce bounds on user-defined count (1 minimum, 8 maximum)
_user_workers = max(1, min(8, USER_DEFINED_MAX_WORKER_COUNT))

# Compute actual max workers based on settings
PARALLEL_MAX_WORKERS = _user_workers if USER_PICKS_MAX_WORKER_COUNT else min(os.cpu_count() or 4, 4)

# AI Prompt Templates
PROMPTS_DIR = Path(__file__).parent.parent / "config" / "prompts"
USER_PROMPTS_DIR = APPDATA_DIR / "prompts"  # User-created prompts survive app updates

# Ensure user prompts directory exists
USER_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
LEGAL_KEYWORDS_FEDERAL = (
    Path(__file__).parent.parent / "data" / "keywords" / "legal_keywords_federal.txt"
)

# License Configuration
LICENSE_FILE = CONFIG_DIR / "license.dat"
LICENSE_API_BASE_URL = "https://api.localscribe.example.com"  # Placeholder - will be updated
LICENSE_CACHE_HOURS = 24

# Logging Configuration
LOG_FILE = LOGS_DIR / "processing.log"
LOG_FORMAT = "[%(levelname)s %(asctime)s] %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

# Debug Mode Default File (for streamlined testing)
DEBUG_DEFAULT_FILE = Path(__file__).parent.parent / "tests" / "sample_docs" / "test_complaint.pdf"

# ============================================================================
# Q&A / Vector Search Configuration (Session 24 - RAG-based Q&A)
# ============================================================================

# Vector Store Settings
# Stores FAISS indexes as files in user's AppData directory
VECTOR_STORE_DIR = APPDATA_DIR / "vector_stores"
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Q&A Retrieval Settings
# Set to None to retrieve ALL chunks (searches entire document corpus)
# Set to a number to limit retrieval to top-K chunks
QA_RETRIEVAL_K = None  # None = all chunks, or integer for top-K
QA_MAX_TOKENS = 300  # Maximum tokens for generated answer
QA_TEMPERATURE = 0.1  # Low temperature for factual, consistent answers
QA_SIMILARITY_THRESHOLD = 0.5  # Minimum relevance score for chunks

# Q&A Context Window
# Session 67: Now dynamically set to match LLM context window based on GPU VRAM.
# See qa_retriever._get_effective_qa_context_window() for the dynamic logic.
# This constant is a FALLBACK value used if user preferences are unavailable.
QA_CONTEXT_WINDOW = 4096  # Fallback tokens for RAG context

# Chat History Settings
QA_CONVERSATION_CONTEXT_PAIRS = 3  # Include last N Q&A pairs in follow-up questions

# ============================================================================
# Hybrid Retrieval Configuration (Session 31 - BM25+ Integration)
# ============================================================================
# Multi-algorithm retrieval for Q&A - mirrors vocabulary extraction architecture

# Algorithm weights for result merging
# Higher weight = more influence on final relevance score
# BM25+ is primary (lexical/keyword matching - reliable for legal terminology)
# FAISS is secondary (semantic/embedding matching - can find related concepts)
RETRIEVAL_ALGORITHM_WEIGHTS = {
    "BM25+": 1.0,  # Primary - exact term matching, reliable for legal docs
    "FAISS": 0.5,  # Secondary - semantic search, complements BM25+
}

# Algorithm enable/disable flags
RETRIEVAL_ENABLE_BM25 = True  # BM25+ lexical search (recommended: always on)
RETRIEVAL_ENABLE_FAISS = True  # FAISS semantic search (can disable for speed)

# Chunking settings for retrieval (smaller chunks = more precise retrieval)
RETRIEVAL_CHUNK_SIZE = 500  # Characters per chunk
RETRIEVAL_CHUNK_OVERLAP = 50  # Overlap between chunks

# Minimum relevance score threshold for merged results
# Lower than before since BM25+ scores are more reliable
RETRIEVAL_MIN_SCORE = 0.1  # Minimum combined score to include chunk

# Multi-algorithm bonus: extra score when multiple algorithms find the same chunk
# This reflects higher confidence when both BM25+ and FAISS agree
RETRIEVAL_MULTI_ALGO_BONUS = 0.1

# ============================================================================
# Query Transformation Configuration (Session 44 - LlamaIndex Integration)
# ============================================================================
# Uses LlamaIndex + Ollama to expand vague queries into specific search terms
# Example: "What happened?" → ["What happened?", "plaintiff injuries", "defendant actions"]

# Enable/disable query transformation (disable for faster retrieval)
QUERY_TRANSFORM_ENABLED = True

# Number of query variants to generate (1-5)
# More variants = broader search but slower
QUERY_TRANSFORM_VARIANTS = 3

# Maximum time to wait for LLM to generate query variants
QUERY_TRANSFORM_TIMEOUT = 30.0

# ============================================================================
# Unified Semantic Chunking Configuration (Session 45, Session 67)
# ============================================================================
# Single chunking pass for all downstream consumers (LLM extraction + Q&A indexing)
# Uses semantic chunking with token enforcement via tiktoken
#
# Session 67: Based on RAG research (2024-2025), chunk sizes are FIXED and do NOT
# scale with context window. What scales is how many chunks fit in the context.
#
# Research findings:
# - Optimal chunk size: 400-1024 tokens regardless of context window
# - Chroma research: 200-400 tokens for best precision
# - arXiv study: 512-1024 tokens for analytical queries
# - Key insight: Larger context = more chunks retrieved, not bigger chunks
#
# See gpu_detector.get_optimal_chunk_sizes() for the research-based values.

# Token limits for chunk sizing (research-based fixed values)
UNIFIED_CHUNK_MIN_TOKENS = 400  # Minimum to prevent fragmentation
UNIFIED_CHUNK_TARGET_TOKENS = 700  # Optimal for mixed queries (research: 500-800)
UNIFIED_CHUNK_MAX_TOKENS = 1000  # Upper bound (>1024 hurts retrieval precision)

# tiktoken encoding for token counting
# cl100k_base is compatible with most modern models (GPT-3.5+, Claude, Llama)
UNIFIED_CHUNK_ENCODING = "cl100k_base"

# ============================================================================
# Hallucination Verification Configuration (Session 60)
# ============================================================================
# Uses LettuceDetect to verify Q&A answers against source documents
# Identifies potentially hallucinated spans with color-coded reliability

# Enable/disable hallucination verification for Q&A answers
# When enabled, adds ~100-200ms per answer
HALLUCINATION_VERIFICATION_ENABLED = True

# Model to use for verification (smaller = faster, larger = more accurate)
# Options: "KRLabsOrg/lettucedect-base-modernbert-en-v1" (~150MB, recommended)
#          "KRLabsOrg/tinylettuce-17m-v1" (~35MB, faster but less accurate)
HALLUCINATION_MODEL = "KRLabsOrg/lettucedect-base-modernbert-en-v1"

# Bundled model configuration for Windows installer
# Models are stored in PROJECT_ROOT/models/ and shipped with the installer
# This prevents network calls at runtime for privacy and offline use
# LOG-001: Renamed to avoid conflict with MODELS_DIR defined earlier
BUNDLED_MODELS_DIR = Path(__file__).parent.parent / "models"
HALLUCINATION_MODEL_LOCAL_PATH = BUNDLED_MODELS_DIR / "lettucedect-base-modernbert-en-v1"

# HuggingFace cache directory (used if bundled model not found)
# Falls back to downloading if bundled model is missing (dev mode)
HF_CACHE_DIR = BUNDLED_MODELS_DIR / ".hf_cache"

# Prevent network calls when bundled model exists
# Set to True for production/installer builds, False for development
HALLUCINATION_LOCAL_ONLY = HALLUCINATION_MODEL_LOCAL_PATH.exists()


# ============================================================================
# Vocabulary Table Column Configuration (Session 83 - moved from core)
# ============================================================================
# Single source of truth for column definitions used by both
# GUI (dynamic_output.py) and HTML export (html_builder.py).
#
# This module defines:
# - Column metadata (width, visibility, hideability)
# - Sort behavior (which columns trigger warnings)
# - Numeric vs string sorting
# - Display-to-data key mappings

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnDefinition:
    """
    Definition of a vocabulary table column.

    Attributes:
        name: Display name shown in column header
        data_key: Key in vocabulary data dict (may differ from name)
        width: Default width in pixels
        max_chars: Max characters before truncation
        default_visible: Whether shown by default
        can_hide: Whether user can hide this column (False = required)
        triggers_sort_warning: Whether sorting by this column shows a warning
        is_numeric: Whether column uses numeric sorting (vs string)
    """

    name: str
    data_key: str
    width: int
    max_chars: int
    default_visible: bool
    can_hide: bool
    triggers_sort_warning: bool
    is_numeric: bool


# All column definitions in display order
# Note: Only "Score" has triggers_sort_warning=False because it's the quality
# ranking - sorting by Score shows best results first (intended behavior).
# All other columns trigger a warning since non-Score sorts show lower-quality first.
COLUMN_DEFINITIONS = [
    # Basic columns (default visible)
    ColumnDefinition("Term", "Term", 180, 30, True, False, False, False),
    ColumnDefinition("Score", "Quality Score", 55, 5, True, True, False, True),
    ColumnDefinition("Is Person", "Is Person", 65, 4, True, True, True, False),
    ColumnDefinition("Found By", "Found By", 120, 20, True, True, True, False),
    # TermSources columns (default visible)
    ColumnDefinition("# Docs", "# Docs", 55, 4, True, True, True, True),
    ColumnDefinition("Count", "Count", 60, 6, True, True, True, True),
    ColumnDefinition("Median Conf", "Median Conf", 80, 5, True, True, True, False),
    # Algorithm detail columns (default hidden)
    ColumnDefinition("NER", "NER", 45, 4, False, True, True, False),
    ColumnDefinition("RAKE", "RAKE", 50, 4, False, True, True, False),
    ColumnDefinition("BM25", "BM25", 50, 4, False, True, True, False),
    ColumnDefinition("Algo Count", "Algo Count", 55, 3, False, True, True, True),
    # Additional columns (default hidden)
    ColumnDefinition("Freq Rank", "Freq Rank", 80, 10, False, True, True, True),
    # Feedback columns (default visible) - Keep/Skip don't need sort warning
    # as they're action columns, not data columns
    ColumnDefinition("Keep", "Keep", 45, 3, True, True, False, False),
    ColumnDefinition("Skip", "Skip", 45, 3, True, True, False, False),
]

# ============================================================================
# Convenience lookups (derived from COLUMN_DEFINITIONS)
# ============================================================================

# Column names in display order
COLUMN_NAMES = tuple(c.name for c in COLUMN_DEFINITIONS)

# Columns that cannot be hidden (Term is required)
PROTECTED_COLUMNS = frozenset(c.name for c in COLUMN_DEFINITIONS if not c.can_hide)

# Columns that trigger sort warning (all except Score, Keep, Skip)
SORT_WARNING_COLUMNS = frozenset(c.name for c in COLUMN_DEFINITIONS if c.triggers_sort_warning)

# Columns that use numeric sorting
NUMERIC_COLUMNS = frozenset(c.name for c in COLUMN_DEFINITIONS if c.is_numeric)

# Display name to data key mapping (for columns where they differ)
DISPLAY_TO_DATA_KEY = {c.name: c.data_key for c in COLUMN_DEFINITIONS if c.name != c.data_key}


def get_column_by_name(name: str) -> ColumnDefinition | None:
    """
    Get column definition by display name.

    Args:
        name: Column display name (e.g., "Score", "Term")

    Returns:
        ColumnDefinition if found, None otherwise
    """
    for col in COLUMN_DEFINITIONS:
        if col.name == name:
            return col
    return None


def get_data_key(display_name: str) -> str:
    """
    Get the data dictionary key for a column.

    Args:
        display_name: Column display name (e.g., "Score")

    Returns:
        Data key (e.g., "Quality Score" for "Score", or same as input)
    """
    return DISPLAY_TO_DATA_KEY.get(display_name, display_name)


def build_column_registry() -> dict[str, dict]:
    """
    Build COLUMN_REGISTRY dict for backward compatibility with dynamic_output.py.

    Returns:
        Dict mapping column name to {width, max_chars, default, can_hide}
    """
    return {
        c.name: {
            "width": c.width,
            "max_chars": c.max_chars,
            "default": c.default_visible,
            "can_hide": c.can_hide,
        }
        for c in COLUMN_DEFINITIONS
    }
