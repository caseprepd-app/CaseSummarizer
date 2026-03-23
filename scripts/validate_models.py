"""
Pre-build validation: verify all bundled assets exist and are complete.

Run this before PyInstaller to catch missing or truncated files
that would cause runtime failures on end-user machines.

Usage:
    python scripts/validate_models.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"

# Minimum safetensors size (MB) to detect truncated files
MIN_SAFETENSORS_MB = 100  # Both models are 500MB+

# ── ML Models ──────────────────────────────────────────────────
REQUIRED_MODELS = {
    "embeddings/nomic-embed-text-v1.5": {
        "files": [
            "config.json",
            "config_sentence_transformers.json",
            "configuration_hf_nomic_bert.py",
            "modeling_hf_nomic_bert.py",
            "model.safetensors",
            "modules.json",
            "sentence_bert_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "1_Pooling/config.json",
        ],
        "safetensors": "model.safetensors",
    },
    "gte-reranker-modernbert-base": {
        "files": [
            "config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
        "safetensors": "model.safetensors",
    },
}

# ── spaCy Models ───────────────────────────────────────────────
REQUIRED_SPACY = [
    "spacy/en_core_web_lg",
    "spacy/en_core_web_sm",
    "spacy/en_ner_bc5cdr_md",
]

# ── NLTK Data ──────────────────────────────────────────────────
REQUIRED_NLTK = [
    "nltk_data/corpora/words",
    "nltk_data/corpora/wordnet",
    "nltk_data/corpora/omw-1.4",
    "nltk_data/corpora/stopwords",
]

# ── OCR Binaries ───────────────────────────────────────────────
REQUIRED_OCR = {
    "tesseract/tesseract.exe": "Tesseract OCR binary",
    "poppler/pdftoppm.exe": "Poppler PDF converter",
}

# ── Tiktoken Cache ─────────────────────────────────────────────
REQUIRED_TIKTOKEN = ["tiktoken_cache"]

# ── Data Files ─────────────────────────────────────────────────
REQUIRED_DATA = [
    "frequency/Word_rarity-count_1w.txt",
    "names/international_forenames.csv",
    "names/international_surnames.csv",
]

# ── Config Files ───────────────────────────────────────────────
REQUIRED_CONFIG = [
    "app_name.txt",
    "categories.json",
    "default_feedback.csv",
    "models.yaml",
    "silly_messages.txt",
    "transcript_patterns.json",
    "vocab_exclude_patterns.txt",
]


def _check_exists(path: Path, label: str) -> list[str]:
    """Return error list if path missing or empty."""
    if not path.exists():
        return [f"MISSING: {label} -> {path}"]
    if path.is_file() and path.stat().st_size == 0:
        return [f"EMPTY: {label} -> {path}"]
    if path.is_dir() and not any(path.iterdir()):
        return [f"EMPTY DIR: {label} -> {path}"]
    return []


def validate_ml_models() -> list[str]:
    """Check ML model directories for missing or truncated files."""
    errors = []
    for name, spec in REQUIRED_MODELS.items():
        model_dir = MODELS_DIR / name
        if not model_dir.exists():
            errors.append(f"MISSING: ML model {name} -> {model_dir}")
            continue
        for filename in spec["files"]:
            errors.extend(_check_exists(model_dir / filename, f"{name}/{filename}"))
        safetensors = model_dir / spec["safetensors"]
        if safetensors.exists():
            size_mb = safetensors.stat().st_size / (1024 * 1024)
            if size_mb < MIN_SAFETENSORS_MB:
                errors.append(
                    f"TRUNCATED: {safetensors} "
                    f"({size_mb:.1f} MB, expected > {MIN_SAFETENSORS_MB} MB)"
                )
    return errors


def validate_spacy() -> list[str]:
    """Check spaCy model directories exist and have meta.json."""
    errors = []
    for rel_path in REQUIRED_SPACY:
        model_dir = MODELS_DIR / rel_path
        errors.extend(_check_exists(model_dir, f"spaCy: {rel_path}"))
        if model_dir.exists():
            meta = model_dir / "meta.json"
            if not meta.exists():
                errors.append(f"MISSING: {rel_path}/meta.json (incomplete spaCy model)")
    return errors


def validate_nltk() -> list[str]:
    """Check NLTK corpus directories exist."""
    errors = []
    for rel_path in REQUIRED_NLTK:
        errors.extend(_check_exists(MODELS_DIR / rel_path, f"NLTK: {rel_path}"))
    return errors


def validate_ocr() -> list[str]:
    """Check OCR binaries exist."""
    errors = []
    for rel_path, label in REQUIRED_OCR.items():
        errors.extend(_check_exists(MODELS_DIR / rel_path, label))
    return errors


def validate_tiktoken() -> list[str]:
    """Check tiktoken cache directory exists."""
    errors = []
    for rel_path in REQUIRED_TIKTOKEN:
        errors.extend(_check_exists(MODELS_DIR / rel_path, f"Tiktoken: {rel_path}"))
    return errors


def validate_data() -> list[str]:
    """Check required data files exist."""
    errors = []
    for rel_path in REQUIRED_DATA:
        errors.extend(_check_exists(DATA_DIR / rel_path, f"Data: {rel_path}"))
    return errors


def validate_config() -> list[str]:
    """Check required config files exist."""
    errors = []
    for filename in REQUIRED_CONFIG:
        errors.extend(_check_exists(CONFIG_DIR / filename, f"Config: {filename}"))
    return errors


def main():
    """Run all validations and report results."""
    print("Validating bundled assets for CasePrepd build\n")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Models dir:   {MODELS_DIR}")
    print(f"  Data dir:     {DATA_DIR}")
    print(f"  Config dir:   {CONFIG_DIR}\n")

    sections = [
        ("ML Models", validate_ml_models),
        ("spaCy Models", validate_spacy),
        ("NLTK Data", validate_nltk),
        ("OCR Binaries", validate_ocr),
        ("Tiktoken Cache", validate_tiktoken),
        ("Data Files", validate_data),
        ("Config Files", validate_config),
    ]

    all_errors = []
    for section_name, validator in sections:
        errors = validator()
        status = "FAIL" if errors else "OK"
        print(f"  [{status}] {section_name}")
        for err in errors:
            print(f"         {err}")
        all_errors.extend(errors)

    print()
    if all_errors:
        print(f"FAILED: {len(all_errors)} issue(s) found.")
        print("Fix these before building the installer.")
        sys.exit(1)
    else:
        print("All bundled assets validated successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
