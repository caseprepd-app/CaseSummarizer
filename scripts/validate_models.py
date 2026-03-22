"""
Pre-build validation: verify all required model files exist and are complete.

Run this before PyInstaller to catch missing or truncated model files
that would cause runtime failures in the installed application.

Usage:
    python scripts/validate_models.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Minimum file sizes (bytes) to detect truncated files.
# Actual sizes are much larger; these are conservative lower bounds.
MIN_SAFETENSORS_MB = 100  # Both models are 500MB+

REQUIRED_MODELS = {
    "embeddings/nomic-embed-text-v1.5": {
        "files": [
            "config.json",
            "config_sentence_transformers.json",
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

REQUIRED_DIRS = [
    "nltk_data",
    "spacy",
    "tiktoken_cache",
]


def validate_model(name: str, spec: dict) -> list[str]:
    """Check a single model directory for missing or truncated files."""
    errors = []
    model_dir = MODELS_DIR / name

    if not model_dir.exists():
        errors.append(f"MISSING: {model_dir}")
        return errors

    for filename in spec["files"]:
        filepath = model_dir / filename
        if not filepath.exists():
            errors.append(f"MISSING: {filepath}")
        elif filepath.stat().st_size == 0:
            errors.append(f"EMPTY: {filepath}")

    safetensors = model_dir / spec["safetensors"]
    if safetensors.exists():
        size_mb = safetensors.stat().st_size / (1024 * 1024)
        if size_mb < MIN_SAFETENSORS_MB:
            errors.append(
                f"TRUNCATED: {safetensors} ({size_mb:.1f} MB, expected > {MIN_SAFETENSORS_MB} MB)"
            )

    return errors


def validate_directories() -> list[str]:
    """Check that required support directories exist."""
    errors = []
    for dirname in REQUIRED_DIRS:
        dirpath = MODELS_DIR / dirname
        if not dirpath.exists():
            errors.append(f"MISSING DIR: {dirpath}")
        elif not any(dirpath.iterdir()):
            errors.append(f"EMPTY DIR: {dirpath}")
    return errors


def main():
    """Run all validations and report results."""
    print(f"Validating models in: {MODELS_DIR}\n")
    all_errors = []

    for name, spec in REQUIRED_MODELS.items():
        errors = validate_model(name, spec)
        status = "FAIL" if errors else "OK"
        print(f"  [{status}] {name}")
        for err in errors:
            print(f"         {err}")
        all_errors.extend(errors)

    dir_errors = validate_directories()
    for dirname in REQUIRED_DIRS:
        dirpath = MODELS_DIR / dirname
        has_error = any(dirname in e for e in dir_errors)
        status = "FAIL" if has_error else "OK"
        print(f"  [{status}] {dirname}")
    all_errors.extend(dir_errors)

    print()
    if all_errors:
        print(f"FAILED: {len(all_errors)} issue(s) found.")
        print("Fix these before building the installer.")
        sys.exit(1)
    else:
        print("All models validated successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
