"""
CasePrepd Configuration Module
Centralized configuration for the application.
"""

import logging
import os
import sys
from pathlib import Path

import yaml

from src.config_defaults import get_default as _d

logger = logging.getLogger(__name__)

# Debug Mode — controls feedback file routing (developer vs user data)
DEBUG_MODE = os.environ.get("DEBUG", "false").lower() == "true"

# Base directory for bundled files (works in both dev and PyInstaller frozen mode)
# In dev: src/config.py -> parent.parent = project root
# In frozen (onedir): sys._MEIPASS = dist/CasePrepd/_internal/
if getattr(sys, "frozen", False):
    BUNDLED_BASE_DIR = Path(sys._MEIPASS)
else:
    BUNDLED_BASE_DIR = Path(__file__).parent.parent
BUNDLED_CONFIG_DIR = BUNDLED_BASE_DIR / "config"

# Application Name (loaded from config/app_name.txt for easy rebranding)
# This file contains just the app name on a single line
_APP_NAME_FILE = BUNDLED_CONFIG_DIR / "app_name.txt"
if _APP_NAME_FILE.exists():
    APP_NAME = _APP_NAME_FILE.read_text(encoding="utf-8").strip()
else:
    APP_NAME = "CasePrepd"  # Fallback default
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

# Feedback and ML Configuration
FEEDBACK_DIR = DATA_DIR / "feedback"
MODELS_ML_DIR = DATA_DIR / "models"  # ML models (separate from Ollama models)
VOCAB_MODEL_PATH = MODELS_ML_DIR / "vocab_meta_learner.pkl"

# Two-file feedback system:
# - Default feedback ships with app (developer's training data)
# - User feedback is collected during normal use
DEFAULT_FEEDBACK_CSV = BUNDLED_CONFIG_DIR / "default_feedback.csv"
USER_FEEDBACK_CSV = FEEDBACK_DIR / "user_feedback.csv"

# Legacy path - kept for backward compatibility detection only
VOCAB_FEEDBACK_CSV = FEEDBACK_DIR / "vocab_feedback.csv"

# ML Training Thresholds
# Don't train until we have enough samples to matter.
ML_MIN_SAMPLES = _d("ml_min_samples")  # Minimum samples before ML training starts
ML_ENSEMBLE_MIN_SAMPLES = _d("ml_ensemble_min_samples")  # Minimum samples to enable ensemble
ML_RETRAIN_THRESHOLD = 1  # Retrain on ANY new user feedback (was 10)

# Graduated RF Weight in Ensemble
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

# ML Time Decay Configuration
# Older feedback is weighted less to adapt to changing user preferences
# - Half-life: Tuned so weight reaches floor at 3 years
# - Floor: Minimum weight (old feedback still matters, just less)
# Rationale: Most early feedback flags universal false positives (common words)
# which should persist. Reporters change courthouses ~every few years.
#
# Decay curve:
#   Today: 1.00 → 1 year: 0.82 → 2 years: 0.67 → 3 years: 0.55 (floor)
ML_DECAY_HALF_LIFE_DAYS = _d("ml_decay_half_life_days")  # ~3.5 years
ML_DECAY_WEIGHT_FLOOR = _d("ml_decay_weight_floor")  # Old feedback retains 55% weight minimum

# Graduated ML Weight
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

# Source-Based Training Weights
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

# Count Bin Configuration
# Centralized definition of occurrence count bins for ML features and deduplication.
# Rationale: count=1 could be OCR error, higher counts are progressively more reliable.
# Extra granularity above 7 occurrences ensures high-frequency legitimate names
# are scored differently from names with only 7 occurrences.
#
# Used by:
# - feedback_manager.py: Deduplication key (term, count_bin)
# - preference_learner.py: One-hot encoded features for ML model
COUNT_BIN_NAMES = (
    "bin_1",  # Single occurrence - may be OCR error
    "bin_2",  # Two occurrences
    "bin_3",  # Three occurrences
    "bin_4_6",  # 4-6 occurrences - moderate confidence
    "bin_7_20",  # 7-20 occurrences - mentioned multiple times
    "bin_21_50",  # 21-50 occurrences - appears throughout document
    "bin_51_plus",  # 51+ occurrences - major figure in transcript
)


def get_count_bin(count: int) -> str:
    """
    Get count bin name for a given occurrence count.

    Args:
        count: Term occurrence count (occurrences)

    Returns:
        Bin name: one of COUNT_BIN_NAMES
    """
    if count <= 0:
        return "bin_1"
    if count == 1:
        return "bin_1"
    if count == 2:
        return "bin_2"
    if count == 3:
        return "bin_3"
    if 4 <= count <= 6:
        return "bin_4_6"
    if 7 <= count <= 20:
        return "bin_7_20"
    if 21 <= count <= 50:
        return "bin_21_50"
    return "bin_51_plus"


def get_count_bin_features(count: int) -> tuple[float, float, float, float, float, float, float]:
    """
    Get one-hot encoded count bin features for ML model.

    Args:
        count: Term occurrence count (occurrences)

    Returns:
        Tuple of 7 floats: (bin_1, bin_2, bin_3, bin_4_6, bin_7_20, bin_21_50, bin_51_plus)
        One value will be 1.0, rest will be 0.0
    """
    bin_name = get_count_bin(count)
    return (
        1.0 if bin_name == "bin_1" else 0.0,
        1.0 if bin_name == "bin_2" else 0.0,
        1.0 if bin_name == "bin_3" else 0.0,
        1.0 if bin_name == "bin_4_6" else 0.0,
        1.0 if bin_name == "bin_7_20" else 0.0,
        1.0 if bin_name == "bin_21_50" else 0.0,
        1.0 if bin_name == "bin_51_plus" else 0.0,
    )


# Rule-Based Quality Score: TermSources Adjustments
# These adjustments are applied BEFORE ML blending, based on document source quality.
# All values are additive to the base score (50 points).
SCORE_MULTI_DOC_BOOST = _d("score_multi_doc_boost")
SCORE_HIGH_CONF_BOOST = _d("score_high_conf_boost")
SCORE_ALL_LOW_CONF_PENALTY = _d("score_all_low_conf_penalty")
SCORE_SINGLE_SOURCE_PENALTY = _d("score_single_source_penalty")
SCORE_TEXTRANK_CENTRALITY_BOOST = _d("score_textrank_centrality_boost")
SCORE_SINGLE_SOURCE_MIN_DOCS = 3  # Only apply single-source penalty when session has 3+ docs
SCORE_SINGLE_SOURCE_CONF_THRESHOLD = 0.70  # Confidence threshold for single-source penalty

# Ensure ML directories exist
for ml_dir in [FEEDBACK_DIR, MODELS_ML_DIR]:
    ml_dir.mkdir(parents=True, exist_ok=True)

# BM25 Corpus Configuration
# User's corpus of previous transcripts for BM25-based term importance
CORPUS_DIR = APPDATA_DIR / "corpus"
CORPUS_MIN_DOCUMENTS = 5  # Minimum docs before BM25 activates
BM25_ENABLED = True  # User can disable in settings
BM25_MIN_SCORE_THRESHOLD = _d("bm25_min_score_threshold")
BM25_WEIGHT = 0.8  # Legacy - kept for backward compatibility

# BM25+ Algorithm Parameters (unified across vocabulary and retrieval)
# Using BM25+ parameters which are strictly better than standard BM25
# (delta parameter prevents zero scores for very long documents)
BM25_K1 = _d("bm25_k1")  # Term frequency saturation
BM25_B = _d("bm25_b")  # Length normalization
BM25_DELTA = _d("bm25_delta")  # BM25+ improvement factor

# Vocabulary Extraction Algorithm Weights
# Centralized weights for multi-algorithm vocabulary extraction
# Higher weight = more influence on final confidence score
# These weights are used by AlgorithmScoreMerger to combine algorithm results
VOCAB_ALGORITHM_WEIGHTS = {
    "NER": _d("vocab_weight_ner"),
    "RAKE": _d("vocab_weight_rake"),
    "BM25": _d("vocab_weight_bm25"),
    "TextRank": 0.6,
    "MedicalNER": _d("vocab_weight_medical_ner"),
    "GLiNER": 0.75,
}

# GLiNER Zero-Shot NER Configuration
GLINER_DEFAULT_LABELS = [
    "anatomical body part",
    "medical procedure",
    "medical condition",
    "medication",
    "chemical compound",
    "specialized scientific term",
    "foreign phrase",
]
GLINER_MAX_LABELS = 20
GLINER_LABELS_FILE = CONFIG_DIR / "gliner_labels.txt"
GLINER_DEFAULT_LABELS_FILE = BUNDLED_CONFIG_DIR / "gliner_labels.txt"


def load_gliner_labels() -> list[str]:
    """
    Load GLiNER labels from user's config file, falling back to defaults.

    Reads labels from CONFIG_DIR/gliner_labels.txt. If the file doesn't
    exist, copies the shipped default file. Validates labels and warns
    if the maximum is exceeded.

    Returns:
        List of validated label strings (1-20 labels)
    """
    import shutil

    # Copy default file to user config if it doesn't exist yet
    if not GLINER_LABELS_FILE.exists() and GLINER_DEFAULT_LABELS_FILE.exists():
        shutil.copy2(GLINER_DEFAULT_LABELS_FILE, GLINER_LABELS_FILE)
        logger.debug("Copied default GLiNER labels to %s", GLINER_LABELS_FILE)

    # Read from user file
    labels = []
    source = GLINER_LABELS_FILE if GLINER_LABELS_FILE.exists() else GLINER_DEFAULT_LABELS_FILE
    try:
        with open(source, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                # Validate: 2-50 chars, must contain letters
                if len(line) < 2 or len(line) > 50:
                    logger.debug("Skipping invalid GLiNER label (length): '%s'", line)
                    continue
                if not any(c.isalpha() for c in line):
                    logger.debug("Skipping invalid GLiNER label (no letters): '%s'", line)
                    continue
                # Skip duplicates (case-insensitive)
                if line.lower() in {l.lower() for l in labels}:
                    continue
                labels.append(line)
    except Exception as e:
        logger.warning("Failed to read GLiNER labels file: %s", e)

    # Warn and truncate if over maximum
    if len(labels) > GLINER_MAX_LABELS:
        logger.warning(
            "GLiNER labels file has %d labels (maximum is %d). Only the first %d will be used.",
            len(labels),
            GLINER_MAX_LABELS,
            GLINER_MAX_LABELS,
        )
        labels = labels[:GLINER_MAX_LABELS]

    # Fall back to defaults if empty
    if not labels:
        logger.debug("No valid labels found, using defaults")
        return list(GLINER_DEFAULT_LABELS)

    return labels


# Similarity Thresholds (consolidated from scattered definitions)
# Used for name deduplication, fuzzy matching, and text similarity
NAME_SIMILARITY_THRESHOLD = _d("name_similarity_threshold")
TEXT_SIMILARITY_THRESHOLD = _d("text_similarity_threshold")
EDIT_DISTANCE_RATIO_THRESHOLD = _d("edit_distance_ratio_threshold")
GIBBERISH_SIMILARITY_THRESHOLD = 0.80  # Min similarity to dictionary word (not gibberish)

# Ensure corpus directory exists
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

# File Processing Limits
MAX_FILE_SIZE_MB = 500
LARGE_FILE_WARNING_MB = 100
MIN_LINE_LENGTH = 15
MIN_DICTIONARY_CONFIDENCE = 60  # Percentage

# PDF Extraction Configuration
# Hybrid extraction uses both PyMuPDF and pdfplumber, reconciling with word-level voting
PDF_EXTRACTION_MODE = _d("pdf_extraction_mode")
PDF_VOTING_ENABLED = _d("pdf_voting_enabled")

# OCR Configuration
OCR_DPI = _d("ocr_dpi")
OCR_CONFIDENCE_THRESHOLD = _d("ocr_confidence_threshold")

# OCR Image Preprocessing Configuration
# Preprocessing can improve OCR accuracy by 20-50% for scanned documents
# Reference: https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
OCR_PREPROCESSING_ENABLED = True  # Enable image preprocessing before OCR
OCR_DENOISE_STRENGTH = _d("ocr_denoise_strength")
OCR_ENABLE_CLAHE = _d("ocr_enable_clahe")

# AI Model Configuration
OLLAMA_API_BASE = "http://localhost:11434"  # Default Ollama API endpoint
OLLAMA_MODEL_NAME = ""  # No default — user selects via Ollama GUI, app persists last choice
OLLAMA_MODEL_FALLBACK = ""  # No fallback — user must select a model
OLLAMA_TIMEOUT_SECONDS = _d("ollama_timeout_seconds")
QUEUE_TIMEOUT_SECONDS = 2.0  # Timeout for multiprocessing queue operations

GPU_DETECTION_TIMEOUT = 15  # Seconds for GPU/VRAM detection via WMI

# Ollama Streaming Heartbeat Timeouts
# Max silence (seconds) between streaming chunks before assuming Ollama crashed.
# The longest legitimate gap is time-to-first-token (prompt eval + cold model load).
OLLAMA_STREAM_READ_TIMEOUT_GPU = 300  # 5 min max silence between chunks (GPU)
OLLAMA_STREAM_READ_TIMEOUT_CPU = 900  # 15 min max silence between chunks (CPU-only)

# Ollama Model Keep-Alive
# Controls how long Ollama keeps a model loaded in VRAM after the last request.
OLLAMA_KEEP_ALIVE_ACTIVE = "-1"  # Keep model in VRAM indefinitely during processing
OLLAMA_KEEP_ALIVE_UNLOAD = 0  # Unload immediately when processing complete

# Context Window Configuration
# Dynamically set based on GPU VRAM via user preferences.
# This constant is only used as a fallback if user preferences are unavailable.
# Actual context size is determined by: user_preferences.get_effective_context_size()
# which auto-detects optimal size based on VRAM (4K-64K range).
OLLAMA_CONTEXT_WINDOW = 4000  # Fallback default (conservative, CPU-safe)

# --- New Model Configuration System ---
MODEL_CONFIG_FILE = BUNDLED_CONFIG_DIR / "models.yaml"
MODEL_CONFIGS = {}


def load_model_configs():
    """Loads model configurations from config/models.yaml."""
    global MODEL_CONFIGS
    try:
        with open(MODEL_CONFIG_FILE, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            MODEL_CONFIGS = data.get("models", {})
        if MODEL_CONFIGS:
            logger.debug(
                "Loaded %d model configurations from %s",
                len(MODEL_CONFIGS),
                MODEL_CONFIG_FILE,
            )
    except FileNotFoundError:
        logger.warning(
            "Model config file not found at %s. Using fallback values.",
            MODEL_CONFIG_FILE,
        )
        MODEL_CONFIGS = {}
    except Exception as e:
        logger.error("Failed to load or parse model config file: %s", e)
        MODEL_CONFIGS = {}


# --- End New Model Configuration System ---


# Default Processing Settings
DEFAULT_SUMMARY_WORDS = 200
MIN_SUMMARY_WORDS = 100
MAX_SUMMARY_WORDS = 500

# Summary Generation Parameters (defaults for PromptConfig)
SUMMARY_WORD_COUNT_TOLERANCE = 20  # Absolute word tolerance (+/- from target)
SUMMARY_SLIDER_INCREMENT = 50  # UI slider step size
SUMMARY_TEMPERATURE = _d("summary_temperature")

# LLM Generation Parameters
LLM_TOP_P = _d("llm_top_p")
LLM_TOKENS_PER_WORD = 1.5  # Estimate for token budget calculation
LLM_TOKEN_BUFFER_MULTIPLIER = 1.3  # Safety buffer to prevent mid-sentence cutoffs

# LLM Vocabulary Extractor Settings
# Used by llm_extractor.py for vocabulary extraction
LLM_EXTRACTOR_MAX_TOKENS = 1000  # Maximum tokens for LLM response

# Summary Length Enforcement Settings
# When a generated summary exceeds target by more than TOLERANCE, it will be condensed
SUMMARY_LENGTH_TOLERANCE = _d("summary_length_tolerance")
SUMMARY_MAX_CONDENSE_ATTEMPTS = 3  # Maximum condensation attempts before returning best effort

# Vocabulary Extractor Data Files
LEGAL_EXCLUDE_LIST_PATH = BUNDLED_CONFIG_DIR / "legal_exclude.txt"
MEDICAL_TERMS_LIST_PATH = BUNDLED_CONFIG_DIR / "medical_terms.txt"
# User-specific vocabulary exclusions (stored in AppData, user can add via right-click)
USER_VOCAB_EXCLUDE_PATH = CONFIG_DIR / "user_vocab_exclude.txt"

# Vocabulary Extraction Rarity Settings
# Path to Google word frequency dataset (word\tfrequency_count format)
# Moved to data/frequency/ for better organization
GOOGLE_WORD_FREQUENCY_FILE = BUNDLED_BASE_DIR / "data" / "frequency" / "Word_rarity-count_1w.txt"
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

# Phrase Component Rarity Filtering
# Filters multi-word phrases where ALL component words are too common.
# Example: "the same", "left side" - high RAKE scores but no vocabulary value.
#
# KEY INSIGHT: If a phrase has even ONE rare word, it might be worth keeping.
# We only filter when ALL words are common.
#
# Commonality scores are 0.0-1.0 (log-scaled from Google word frequency):
# RANK-BASED SCORING:
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

# Single-word rarity threshold (rank-based)
# Filter words in the top X% as "too common for vocabulary prep"
# Examples with 0.50 threshold:
#   "age" (rank 579, score 0.0017) < 0.50 -> FILTERED (top 0.17%)
#   "cervical" (rank ~50000, score 0.15) < 0.50 -> FILTERED (top 15%)
#   "radiculopathy" (rank ~250000, score 0.75) > 0.50 -> KEPT (bottom 25%)
SINGLE_WORD_COMMONALITY_THRESHOLD = 0.50  # Filter top 50% of vocabulary

# Non-NER Rarity Passthrough Thresholds
# Passes RAKE/BM25-found terms through rarity filtering when they're sufficiently rare.
# Words not in the Google frequency dataset get this score instead of 0.0:
NON_NER_UNKNOWN_WORD_RARITY = 0.85  # Treat unknown words as rare (0.85 out of 1.0)
# Single-word passthrough: keep if rarity >= threshold
NON_NER_SINGLE_PASSTHROUGH_THRESHOLD = 0.80
# Multi-word passthrough: keep if max_rarity >= threshold AND mean_rarity >= threshold
NON_NER_PHRASE_MAX_PASSTHROUGH_THRESHOLD = 0.85
NON_NER_PHRASE_MEAN_PASSTHROUGH_THRESHOLD = 0.65

# Floor for adjusted mean rarity calculation
# Words with rarity score below this are excluded from the mean calculation.
# This prevents common filler words ("of", "the", "and") from dragging down
# the mean rarity of phrases that contain genuinely rare words.
# 0.10 = exclude words in the top 10% most common English words
NON_NER_PHRASE_COMMON_WORD_FLOOR = 0.10

# Person Title Prefixes (title-aware name synthesis)
# Used by name_deduplicator.py to merge "Dr. Jones" + "James Jones" → "James Jones (Dr.)"
# Split into generic (discardable in conflict) and role (kept as separate entries)
PERSON_TITLE_PREFIXES_GENERIC = ["mr.", "ms.", "mrs."]
PERSON_TITLE_PREFIXES_ROLE = [
    "dr.",
    "hon.",
    "judge",
    "justice",
    "senator",
    "nurse",
    "officer",
    "trooper",
    "investigator",
    "referee",
]
PERSON_TITLE_PREFIXES = PERSON_TITLE_PREFIXES_GENERIC + PERSON_TITLE_PREFIXES_ROLE

# Transcript Section Keywords (header artifact removal)
# Used by artifact_filter.py to detect "Smith - Direct" style artifacts
TRANSCRIPT_SECTION_KEYWORDS = {
    "direct",
    "cross",
    "redirect",
    "recross",
    "examination",
    "voir",
    "dire",
    "rebuttal",
    "continued",
    "resumed",
    "recalled",
}

# Stop Word Threshold for ML feature extraction
# Words in the top N most common are "stop words" for starts_with/ends_with features
STOP_WORD_THRESHOLD = 1000

# Artifact Filter: Common word threshold for detecting name+common-word artifacts
# Used to identify patterns like "Luigi Napolitano Patient" where "Patient" is common
# Words with rank below this threshold are considered "common" for artifact detection
ARTIFACT_FILTER_COMMON_WORD_THRESHOLD = 200000

# Artifact Filter: Maximum edit distance for fuzzy matching
# Used when checking if terms are fuzzy matches of canonical terms
ARTIFACT_FILTER_FUZZY_MAX_EDIT_DISTANCE = 2

# Index Page Remover Configuration
# Thresholds for detecting index/concordance pages in legal transcripts
INDEX_MIN_INDEX_LINES = 10  # Require 10+ index-like lines to detect
INDEX_MIN_DENSITY_PERCENT = 30  # At least 30% of lines must match patterns
INDEX_DETECTION_WINDOW_SIZE = 50  # Lines to analyze per window
INDEX_MIN_TEXT_LENGTH = 1000  # Minimum chars before checking for index
INDEX_TAIL_CHECK_FRACTION = 10  # Check 1/10 of document tail
INDEX_MAX_CHECK_LENGTH = 20000  # Maximum chars to check in tail
INDEX_ESTIMATED_CHARS_PER_LINE = 60  # For estimating line count from chars
INDEX_PAGE_REF_DIVISOR = 2  # Divide page refs by this for density calc
INDEX_CHAR_WINDOW_SIZE = 2000  # Character window size for char-based detection

# Semantic Chunker Embedding Model
# Used by UnifiedChunker for semantic text splitting
SEMANTIC_CHUNKER_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# GUI Display Limits for Vocabulary Table
# Based on tkinter Treeview performance testing:
# - < 100 rows: Excellent performance
# - 100-200 rows: Generally acceptable
# - 200+ rows: Performance degrades, especially with text wrapping
# Default: 50 rows (conservative for responsiveness)
# Maximum ceiling: 200 rows (hard limit to prevent GUI freezing)
VOCABULARY_DISPLAY_LIMIT = 50  # User-configurable default (conservative)
VOCABULARY_DISPLAY_MAX = 200  # Hard ceiling - cannot exceed this

# Vocabulary Display Pagination (GUI responsiveness)
# Controls async batch insertion to prevent GUI freezing during large loads
VOCABULARY_ROWS_PER_PAGE = 50  # Initial rows shown; "Load More" adds more
VOCABULARY_BATCH_INSERT_SIZE = 20  # Rows inserted per async batch
VOCABULARY_BATCH_INSERT_DELAY_MS = 10  # Delay between batches (ms)

# Vocabulary Indicator Pattern Defaults
# Negative indicators: common transcript procedural terms that court reporters
# already know and don't need in a prep vocabulary list.
DEFAULT_NEGATIVE_INDICATORS = ["proceedings", "Direct", "Cross", "Redirect", "Recross"]
DEFAULT_POSITIVE_INDICATORS: list[str] = []

# spaCy Model Download Timeouts
# Controls timeout behavior during automatic spaCy model downloads
SPACY_DOWNLOAD_TIMEOUT_SEC = 3600  # Overall timeout: 1 hour (slow connections)
SPACY_SOCKET_TIMEOUT_SEC = 10  # Socket timeout per request
SPACY_THREAD_TIMEOUT_SEC = 15  # Thread termination timeout

# Document Chunking (hierarchical summarization)
# Overlap fraction prevents context loss at chunk boundaries
CHUNK_OVERLAP_FRACTION = 0.1  # 10% overlap between chunks

# System Monitor Color Thresholds (CPU and RAM)
# Used for color-coded status indicators in the system monitor widget
# Applied independently to both CPU and RAM percentages
SYSTEM_MONITOR_THRESHOLD_GREEN = 75  # 0-74%: Green (healthy)
SYSTEM_MONITOR_THRESHOLD_YELLOW = 85  # 75-84%: Yellow (elevated)
SYSTEM_MONITOR_THRESHOLD_CRITICAL = 90  # 90%+: Red with "!" indicator

# Vocabulary Extraction Performance Settings
# Max text size in KB for vocabulary orchestrator (safety net for degenerate inputs)
# Individual algorithms handle their own limits internally
VOCABULARY_MAX_TEXT_KB = 10000  # 10MB (~2,500 pages) - safety net only

# TextRank-specific limit (O(n^2) on vocabulary size)
TEXTRANK_MAX_TEXT_KB = 1000  # 1MB (~250 pages) - balances coverage vs performance

# RAKE minimum phrase frequency — phrases appearing fewer times are filtered
RAKE_MIN_FREQUENCY = 3

# spaCy batch processing - higher values process faster with more memory
# Testing shows: batch_size=4 (baseline), 8 (~17% faster), 16 (~25% faster but +100MB RAM)
# Lower values yield the GIL more often, keeping GUI responsive during NER
VOCABULARY_BATCH_SIZE = 16

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
PROMPTS_DIR = BUNDLED_CONFIG_DIR / "prompts"
USER_PROMPTS_DIR = APPDATA_DIR / "prompts"  # User-created prompts survive app updates

# Ensure user prompts directory exists
USER_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
# License Configuration
LICENSE_FILE = CONFIG_DIR / "license.dat"
LICENSE_API_BASE_URL = "https://api.localscribe.example.com"  # Placeholder - will be updated
LICENSE_CACHE_HOURS = 24

# Logging Configuration
LOG_FILE = LOGS_DIR / "processing.log"
LOG_FORMAT = "[%(levelname)s %(asctime)s] %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

# Debug Mode Default File (for streamlined testing)
DEBUG_DEFAULT_FILE = BUNDLED_BASE_DIR / "tests" / "sample_docs" / "test_complaint.pdf"

# ============================================================================
# UI Timing Constants
# ============================================================================
# Queue polling interval in milliseconds (~30 FPS equivalent for UI updates)
QUEUE_POLL_INTERVAL_MS = 33
# Debounce delay for resize events to prevent excessive redraws
RESIZE_DEBOUNCE_MS = 100
# Maximum characters to show in error messages before truncation
ERROR_DISPLAY_MAX_CHARS = 200

# ============================================================================
# Q&A / Vector Search Configuration (RAG-based Q&A)
# ============================================================================

# Vector Store Settings
# Stores FAISS indexes as files in user's AppData directory
VECTOR_STORE_DIR = APPDATA_DIR / "vector_stores"
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Q&A Retrieval Settings
# Set to None to retrieve ALL chunks (searches entire document corpus)
# Set to a number to limit retrieval to top-K chunks
QA_RETRIEVAL_K = _d("qa_retrieval_k")
QA_MAX_TOKENS = _d("qa_max_tokens")
QA_TEMPERATURE = _d("qa_temperature")
QA_SIMILARITY_THRESHOLD = _d("qa_similarity_threshold")

# Q&A Context Window
# Dynamically set to match LLM context window based on GPU VRAM.
# See qa_retriever._get_effective_qa_context_window() for the dynamic logic.
# This constant is a FALLBACK value used if user preferences are unavailable.
QA_CONTEXT_WINDOW = 4096  # Fallback tokens for RAG context

# Chat History Settings
QA_CONVERSATION_CONTEXT_PAIRS = 3  # Include last N Q&A pairs in follow-up questions

# ============================================================================
# Hybrid Retrieval Configuration (BM25+ Integration)
# ============================================================================
# Multi-algorithm retrieval for Q&A - mirrors vocabulary extraction architecture

# Algorithm weights for weighted RRF merging (scale each algorithm's rank contribution)
# Higher weight = more influence on final ranking
# Semantic search weighted higher: reporters ask exploratory questions about
# documents they haven't memorized, so meaning-based retrieval matters more
# than exact keyword matching.
RETRIEVAL_ALGORITHM_WEIGHTS = {
    "FAISS": 1.0,  # Semantic search — weighted higher for exploratory queries
    "BM25+": 0.9,  # Lexical matching — still valuable for name/term queries
}

# Algorithm enable/disable flags
RETRIEVAL_ENABLE_BM25 = _d("retrieval_enable_bm25")
RETRIEVAL_ENABLE_FAISS = _d("retrieval_enable_faiss")

# Chunking settings for retrieval (smaller chunks = more precise retrieval)
RETRIEVAL_CHUNK_SIZE = _d("retrieval_chunk_size")
RETRIEVAL_CHUNK_OVERLAP = _d("retrieval_chunk_overlap")

# Minimum relevance score threshold for merged results
RETRIEVAL_MIN_SCORE = _d("retrieval_min_score")

# Minimum best-chunk score to attempt answering a question
# Below this, the question is treated as unanswerable for this document
RETRIEVAL_CONFIDENCE_GATE = _d("retrieval_confidence_gate")

# Multi-algorithm bonus: legacy setting, kept for backward compatibility
# RRF fusion naturally rewards multi-algorithm consensus without explicit bonus
RETRIEVAL_MULTI_ALGO_BONUS = _d("retrieval_multi_algo_bonus")

# Reciprocal Rank Fusion constant (standard value from literature)
# Higher k = less advantage for top-ranked items, more uniform blending
RRF_K = 60

# FAISS semantic relevance floor: if the best FAISS chunk scores below this,
# there's no semantic match and the question is likely unanswerable
FAISS_RELEVANCE_FLOOR = _d("faiss_relevance_floor")

# Target characters for focused citation excerpt — embedding-selected window
QA_CITATION_MAX_CHARS = _d("qa_citation_max_chars")

# ============================================================================
# Unified Semantic Chunking Configuration
# ============================================================================
# Single chunking pass for all downstream consumers (LLM extraction + Q&A indexing)
# Uses semantic chunking with token enforcement via tiktoken
#
# Based on RAG research (2024-2025), chunk sizes are FIXED and do NOT
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
UNIFIED_CHUNK_MIN_TOKENS = _d("unified_chunk_min_tokens")
UNIFIED_CHUNK_TARGET_TOKENS = _d("unified_chunk_target_tokens")
UNIFIED_CHUNK_MAX_TOKENS = _d("unified_chunk_max_tokens")

# tiktoken encoding for token counting
# cl100k_base is compatible with most modern models (GPT-3.5+, Claude, Llama)
UNIFIED_CHUNK_ENCODING = "cl100k_base"

# ============================================================================
# Hallucination Verification Configuration
# ============================================================================
# Uses LettuceDetect to verify Q&A answers against source documents
# Identifies potentially hallucinated spans with color-coded reliability

# Enable/disable hallucination verification for Q&A answers
# When enabled, adds ~100-200ms per answer
HALLUCINATION_VERIFICATION_ENABLED = _d("hallucination_verification_enabled")

# Hallucination detection model (TinyLettuce 68M)
# 68M params, 75% F1 on RAGTruth — good balance of accuracy and size
# Uses lettucedetect library for span-level hallucination detection
HALLUCINATION_MODEL = "KRLabsOrg/tinylettuce-ettin-68m-en"

# Span classification thresholds for color-coding answer text
# LettuceDetect returns probability of hallucination (0.0-1.0)
# LOWER scores = more reliable, HIGHER scores = less reliable
HALLUCINATION_THRESHOLDS = {
    "verified": 0.30,  # < 0.30 = green (verified, strongly supported)
    "uncertain": 0.50,  # 0.30 - 0.50 = yellow (uncertain, borderline)
    "suspicious": 0.70,  # 0.50 - 0.70 = orange (suspicious, likely unsupported)
    "unreliable": 0.85,  # 0.70 - 0.85 = red (unreliable, probably hallucinated)
    # >= 0.85 = strikethrough (hallucinated, very high confidence)
}

# Overall answer rejection threshold - reject if reliability below this
ANSWER_REJECTION_THRESHOLD = _d("answer_rejection_threshold")

# Q&A Export: minimum thresholds to include in exports (both must be met)
QA_EXPORT_CONFIDENCE_FLOOR = _d("qa_export_confidence_floor")
QA_EXPORT_VERIFICATION_FLOOR = _d("qa_export_verification_floor")

# Message shown when answer confidence is too low
HALLUCINATION_REJECTION_MESSAGE = (
    "Confidence in answer too low after verification step, declining to show answer..."
)

# Bundled model configuration for Windows installer
# Models are stored in PROJECT_ROOT/models/ and shipped with the installer
# This prevents network calls at runtime for privacy and offline use
# Renamed to avoid conflict with MODELS_DIR defined earlier
BUNDLED_MODELS_DIR = BUNDLED_BASE_DIR / "models"
HALLUCINATION_MODEL_LOCAL_PATH = BUNDLED_MODELS_DIR / "tinylettuce-ettin-68m-en"

# GLiNER zero-shot NER model (209M params, Apache 2.0)
GLINER_MODEL_NAME = "urchade/gliner_medium-v2.1"
GLINER_MODEL_LOCAL_PATH = BUNDLED_MODELS_DIR / "gliner_medium-v2.1"

# Embedding model for FAISS semantic search
# nomic-embed-text-v1.5 (137M params, 768 dims, 8192-token context, 270MB)
# Downsized from modernbert-embed-large (1.58GB) — research shows small embeddings
# + strong cross-encoder reranker >= large embeddings + reranker. Saves ~1.3GB.
# Uses GPU when available via torch.cuda.is_available(), falls back to CPU.
EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_MODEL_LOCAL_PATH = BUNDLED_MODELS_DIR / "embeddings" / "nomic-embed-text-v1.5"

# Semantic chunker embedding model (used by UnifiedChunker for text splitting)
# all-MiniLM-L6-v2 (22MB) — lightweight model for detecting semantic boundaries
SEMANTIC_CHUNKER_MODEL_LOCAL_PATH = BUNDLED_MODELS_DIR / "embeddings" / "all-MiniLM-L6-v2"

# HuggingFace cache directory (used if bundled model not found)
# Falls back to downloading if bundled model is missing (dev mode)
HF_CACHE_DIR = BUNDLED_MODELS_DIR / ".hf_cache"

# Prevent network calls when bundled model exists
# Set to True for production/installer builds, False for development
HALLUCINATION_LOCAL_ONLY = HALLUCINATION_MODEL_LOCAL_PATH.exists()

# ============================================================================
# Cross-Encoder Reranking Configuration
# ============================================================================
# Uses Alibaba-NLP/gte-reranker-modernbert-base to rerank candidate chunks after
# hybrid retrieval. 149M params, 8192-token context (sees full chunks without truncation).
# Cross-encoders process query+document pairs together for more accurate relevance.

RERANKING_ENABLED = _d("reranking_enabled")
RERANKER_MODEL_NAME = (
    "Alibaba-NLP/gte-reranker-modernbert-base"  # ~300MB model, downloads on first use
)
RERANKER_MODEL_LOCAL_PATH = BUNDLED_MODELS_DIR / "gte-reranker-modernbert-base"

# Coreference resolution model (fastcoref F-coref)
# Resolves pronouns to named entities for improved search/retrieval accuracy
# F-coref: 78.5% F1 on OntoNotes, 16x faster than LingMess, distilled model
COREF_MODEL_NAME = "biu-nlp/f-coref"
COREF_MODEL_LOCAL_PATH = BUNDLED_MODELS_DIR / "coref" / "f-coref"

# Bundled spaCy models for Windows installer (no runtime downloads)
SPACY_MODELS_DIR = BUNDLED_MODELS_DIR / "spacy"
SPACY_EN_CORE_WEB_LG_PATH = SPACY_MODELS_DIR / "en_core_web_lg"
SPACY_EN_CORE_WEB_SM_PATH = SPACY_MODELS_DIR / "en_core_web_sm"
SPACY_EN_NER_BC5CDR_MD_PATH = SPACY_MODELS_DIR / "en_ner_bc5cdr_md"

# Bundled NLTK data for Windows installer (no runtime downloads)
NLTK_DATA_DIR = BUNDLED_MODELS_DIR / "nltk_data"

# Register bundled NLTK data path before any NLTK consumer
if NLTK_DATA_DIR.exists():
    import nltk

    nltk.data.path.insert(0, str(NLTK_DATA_DIR))

# Bundled Tesseract OCR binary for Windows installer (no system install needed)
TESSERACT_BUNDLED_DIR = BUNDLED_MODELS_DIR / "tesseract"
TESSERACT_BUNDLED_EXE = TESSERACT_BUNDLED_DIR / "tesseract.exe"

# Bundled Poppler utilities for PDF-to-image conversion
POPPLER_BUNDLED_DIR = BUNDLED_MODELS_DIR / "poppler"

# Register bundled Tesseract tessdata path before any OCR consumer
if TESSERACT_BUNDLED_DIR.exists():
    os.environ["TESSDATA_PREFIX"] = str(TESSERACT_BUNDLED_DIR / "tessdata")

RERANKER_MAX_LENGTH = 8192
RERANKER_TOP_K = _d("reranker_top_k")


# ============================================================================
# Vocabulary Table Column Configuration (moved from core)
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
    ColumnDefinition("OCR Confidence", "OCR Confidence", 80, 5, True, True, True, False),
    # Algorithm detail columns (default hidden)
    ColumnDefinition("NER", "NER", 45, 4, False, True, True, False),
    ColumnDefinition("RAKE", "RAKE", 50, 4, False, True, True, False),
    ColumnDefinition("BM25", "BM25", 50, 4, False, True, True, False),
    ColumnDefinition("Algo Count", "Algo Count", 55, 3, False, True, True, True),
    # Additional columns (default hidden)
    ColumnDefinition("Google Rarity Rank", "Google Rarity Rank", 80, 10, False, True, True, True),
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


def scale_column_widths(factor: float) -> None:
    """
    Rebuild COLUMN_DEFINITIONS with scaled widths for UI scaling.

    ColumnDefinition is frozen, so we must rebuild the list.
    Also rebuilds all derived constants (COLUMN_NAMES, etc.).

    Args:
        factor: Scale multiplier (e.g. 1.25 for 125%)
    """
    global COLUMN_DEFINITIONS, COLUMN_NAMES, PROTECTED_COLUMNS
    global SORT_WARNING_COLUMNS, NUMERIC_COLUMNS, DISPLAY_TO_DATA_KEY

    if factor == 1.0:
        return

    COLUMN_DEFINITIONS = [
        ColumnDefinition(
            c.name,
            c.data_key,
            int(c.width * factor),
            c.max_chars,
            c.default_visible,
            c.can_hide,
            c.triggers_sort_warning,
            c.is_numeric,
        )
        for c in COLUMN_DEFINITIONS
    ]

    # Rebuild derived constants
    COLUMN_NAMES = tuple(c.name for c in COLUMN_DEFINITIONS)
    PROTECTED_COLUMNS = frozenset(c.name for c in COLUMN_DEFINITIONS if not c.can_hide)
    SORT_WARNING_COLUMNS = frozenset(c.name for c in COLUMN_DEFINITIONS if c.triggers_sort_warning)
    NUMERIC_COLUMNS = frozenset(c.name for c in COLUMN_DEFINITIONS if c.is_numeric)
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
