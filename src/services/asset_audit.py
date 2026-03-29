"""
Asset Provenance Audit.

Logs the actual resolved path for every bundled model, dataset, and data file
at startup. This makes it easy to verify that the installed app is loading
everything from its own directory — not from system-installed packages,
pip site-packages, or user home directories.

Run automatically at startup via main.py. Check the log for lines tagged
[ASSET AUDIT] to verify all assets load from the expected base directory.
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_BUNDLED_TAG = "BUNDLED"
_SYSTEM_TAG = "SYSTEM"
_MISSING_TAG = "MISSING"


def _classify(path: Path, base_dir: Path) -> str:
    """Classify a path as bundled, system, or missing."""
    if not path.exists():
        return _MISSING_TAG
    try:
        path.resolve().relative_to(base_dir.resolve())
        return _BUNDLED_TAG
    except ValueError:
        return _SYSTEM_TAG


def _log_asset(name: str, path: Path, base_dir: Path) -> str:
    """Log one asset's provenance. Returns the classification tag."""
    tag = _classify(path, base_dir)
    resolved = path.resolve() if path.exists() else path
    if tag == _BUNDLED_TAG:
        logger.info("[ASSET AUDIT] %s: %s -> %s", _BUNDLED_TAG, name, resolved)
    elif tag == _SYSTEM_TAG:
        logger.warning(
            "[ASSET AUDIT] %s: %s -> %s (expected under %s)",
            _SYSTEM_TAG,
            name,
            resolved,
            base_dir,
        )
    else:
        logger.warning("[ASSET AUDIT] %s: %s -> %s", _MISSING_TAG, name, path)
    return tag


def _check_nltk_paths(base_dir: Path) -> None:
    """Log which NLTK data paths are active and where corpora resolve."""
    try:
        import nltk
    except ImportError:
        logger.warning("[ASSET AUDIT] NLTK not installed")
        return

    logger.info("[ASSET AUDIT] NLTK data search path:")
    for i, p in enumerate(nltk.data.path):
        exists = Path(p).exists() if p else False
        logger.info("[ASSET AUDIT]   [%d] %s (exists=%s)", i, p, exists)

    for corpus_name in ("words", "wordnet", "omw-1.4", "stopwords"):
        try:
            found = nltk.data.find(f"corpora/{corpus_name}")
            tag = _classify(Path(str(found)), base_dir)
            logger.info(
                "[ASSET AUDIT] NLTK corpus '%s': %s -> %s",
                corpus_name,
                tag,
                found,
            )
            if tag == _SYSTEM_TAG:
                logger.warning(
                    "[ASSET AUDIT] NLTK corpus '%s' loaded from SYSTEM, not bundled directory!",
                    corpus_name,
                )
        except LookupError:
            logger.warning(
                "[ASSET AUDIT] NLTK corpus '%s': MISSING",
                corpus_name,
            )


def _check_spacy_model(name: str, bundled_path: Path, base_dir: Path) -> None:
    """Log where a spaCy model would load from."""
    _log_asset(f"spaCy model '{name}'", bundled_path, base_dir)

    if not bundled_path.exists():
        try:
            import spacy

            info = spacy.util.get_package_path(name)
            logger.warning(
                "[ASSET AUDIT] spaCy '%s' would fall back to pip package: %s",
                name,
                info,
            )
        except Exception:
            pass


def run_asset_audit() -> None:
    """
    Log the provenance of every asset the app depends on.

    Call this once at startup after logging is configured.
    """
    from src.config import (
        BUNDLED_BASE_DIR,
        BUNDLED_CONFIG_DIR,
        BUNDLED_MODELS_DIR,
        EMBEDDING_MODEL_LOCAL_PATH,
        GOOGLE_WORD_FREQUENCY_FILE,
        NLTK_DATA_DIR,
        POPPLER_BUNDLED_DIR,
        RERANKER_MODEL_LOCAL_PATH,
        SPACY_EN_CORE_WEB_LG_PATH,
        SPACY_EN_CORE_WEB_SM_PATH,
        SPACY_EN_NER_BC5CDR_MD_PATH,
        TESSERACT_BUNDLED_EXE,
    )

    base = BUNDLED_BASE_DIR
    is_frozen = getattr(sys, "frozen", False)

    logger.info(
        "[ASSET AUDIT] === Asset Provenance Report === (frozen=%s, base=%s)",
        is_frozen,
        base.resolve(),
    )

    # -- ML Models --
    _log_asset("Embedding model (nomic-embed-text-v1.5)", EMBEDDING_MODEL_LOCAL_PATH, base)
    safetensors = EMBEDDING_MODEL_LOCAL_PATH / "model.safetensors"
    if safetensors.exists():
        size_mb = safetensors.stat().st_size / (1024 * 1024)
        logger.info("[ASSET AUDIT]   model.safetensors: %.1f MB", size_mb)
    else:
        logger.warning("[ASSET AUDIT]   model.safetensors: MISSING")

    _log_asset("Reranker model (gte-reranker-modernbert-base)", RERANKER_MODEL_LOCAL_PATH, base)
    reranker_st = RERANKER_MODEL_LOCAL_PATH / "model.safetensors"
    if reranker_st.exists():
        size_mb = reranker_st.stat().st_size / (1024 * 1024)
        logger.info("[ASSET AUDIT]   model.safetensors: %.1f MB", size_mb)
    else:
        logger.warning("[ASSET AUDIT]   model.safetensors: MISSING")

    # -- spaCy Models --
    _check_spacy_model("en_core_web_lg", SPACY_EN_CORE_WEB_LG_PATH, base)
    _check_spacy_model("en_core_web_sm", SPACY_EN_CORE_WEB_SM_PATH, base)
    _check_spacy_model("en_ner_bc5cdr_md", SPACY_EN_NER_BC5CDR_MD_PATH, base)

    # -- NLTK Data --
    _log_asset("NLTK data directory", NLTK_DATA_DIR, base)
    _check_nltk_paths(base)

    # -- Tiktoken Cache --
    tiktoken_cache = BUNDLED_MODELS_DIR / "tiktoken_cache"
    _log_asset("Tiktoken cache", tiktoken_cache, base)
    tiktoken_env = os.environ.get("TIKTOKEN_CACHE_DIR", "<not set>")
    logger.info("[ASSET AUDIT] TIKTOKEN_CACHE_DIR env: %s", tiktoken_env)

    # -- OCR Binaries --
    _log_asset("Tesseract binary", TESSERACT_BUNDLED_EXE, base)
    _log_asset("Poppler directory", POPPLER_BUNDLED_DIR, base)
    tessdata_env = os.environ.get("TESSDATA_PREFIX", "<not set>")
    logger.info("[ASSET AUDIT] TESSDATA_PREFIX env: %s", tessdata_env)

    # -- Data Files --
    _log_asset("Google word frequency dataset", GOOGLE_WORD_FREQUENCY_FILE, base)

    names_dir = base / "data" / "names"
    _log_asset("Forenames dataset", names_dir / "international_forenames.csv", base)
    _log_asset("Surnames dataset", names_dir / "international_surnames.csv", base)

    # -- Config Files --
    for config_name in [
        "app_name.txt",
        "categories.json",
        "default_feedback.csv",
        "models.yaml",
        "silly_messages.txt",
        "transcript_patterns.json",
        "vocab_exclude_patterns.txt",
    ]:
        _log_asset(f"Config: {config_name}", BUNDLED_CONFIG_DIR / config_name, base)

    # -- HuggingFace Env Vars --
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME", "TRANSFORMERS_CACHE"):
        val = os.environ.get(var, "<not set>")
        logger.info("[ASSET AUDIT] %s = %s", var, val)

    logger.info("[ASSET AUDIT] === End Asset Provenance Report ===")
