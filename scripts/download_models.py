"""
Download all bundled models for the Windows installer.

Downloads spaCy, NLTK, and HuggingFace models into the project's models/
directory. Run this once before building the installer.

Usage:
    python scripts/download_models.py

Models downloaded:
    models/spacy/en_core_web_lg/          (~560 MB)
    models/spacy/en_core_web_sm/          (~12 MB)
    models/spacy/en_ner_bc5cdr_md/        (~100 MB)
    models/nltk_data/corpora/words/
    models/nltk_data/corpora/wordnet/
    models/nltk_data/corpora/omw-1.4/
    models/tinylettuce-ettin-68m-en/      (hallucination detector)
    models/gliner_medium-v2.1/            (zero-shot NER)
    models/embeddings/nomic-embed-text-v1.5/  (FAISS embeddings)
    models/gte-reranker-modernbert-base/  (cross-encoder reranker)
    models/coref/f-coref/                 (coreference resolution)
"""

import shutil
import subprocess
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# spaCy models to bundle
SPACY_MODELS = ["en_core_web_lg", "en_core_web_sm", "en_ner_bc5cdr_md"]
SPACY_DIR = MODELS_DIR / "spacy"

# NLTK corpora to bundle
NLTK_CORPORA = ["words", "wordnet", "omw-1.4"]
NLTK_DIR = MODELS_DIR / "nltk_data"

# HuggingFace models: (repo_id, local_subdir, ignore_patterns)
HF_MODELS = [
    ("tinylettuce/ettin-68m-en", "tinylettuce-ettin-68m-en", None),
    ("urchade/gliner_medium-v2.1", "gliner_medium-v2.1", None),
    (
        "nomic-ai/nomic-embed-text-v1.5",
        "embeddings/nomic-embed-text-v1.5",
        ["onnx/*", "onnx/**"],
    ),  # Skip ONNX variants (~1.6GB)
    ("Alibaba-NLP/gte-reranker-modernbert-base", "gte-reranker-modernbert-base", None),
    ("biu-nlp/f-coref", "coref/f-coref", None),
]


def download_spacy_models() -> dict[str, bool]:
    """
    Download spaCy models and copy to bundled directory.

    Downloads each model via `python -m spacy download`, then locates the
    installed package in site-packages and copies it to models/spacy/.

    Returns:
        Dict mapping model name to success status.
    """
    results = {}
    SPACY_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in SPACY_MODELS:
        target = SPACY_DIR / model_name
        if target.exists():
            print(f"  [SKIP] {model_name} (already exists)")
            results[model_name] = True
            continue

        print(f"  Bundling {model_name}...")
        try:
            import importlib

            # Try importing already-installed model first
            try:
                mod = importlib.import_module(model_name)
            except ImportError:
                # Not installed yet — download via spacy (works for official models)
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", model_name],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                mod = importlib.import_module(model_name)

            source = Path(mod.__file__).parent

            # Copy to bundled directory
            shutil.copytree(source, target)
            print(f"  [OK] {model_name} -> {target}")
            results[model_name] = True
        except Exception as e:
            print(f"  [FAILED] {model_name}: {e}")
            results[model_name] = False

    return results


def download_nltk_data() -> dict[str, bool]:
    """
    Download NLTK corpora directly into bundled directory.

    Uses nltk.download() with a custom download_dir to place data
    directly into models/nltk_data/.

    Returns:
        Dict mapping corpus name to success status.
    """
    import nltk

    results = {}
    NLTK_DIR.mkdir(parents=True, exist_ok=True)

    for corpus_name in NLTK_CORPORA:
        print(f"  Downloading {corpus_name}...")
        try:
            nltk.download(corpus_name, download_dir=str(NLTK_DIR), quiet=True)
            print(f"  [OK] {corpus_name}")
            results[corpus_name] = True
        except Exception as e:
            print(f"  [FAILED] {corpus_name}: {e}")
            results[corpus_name] = False

    return results


def download_huggingface_models() -> dict[str, bool]:
    """
    Download HuggingFace models via snapshot_download.

    Skips models that already exist in the target directory.

    Returns:
        Dict mapping model repo_id to success status.
    """
    from huggingface_hub import snapshot_download

    results = {}

    for repo_id, local_subdir, ignore in HF_MODELS:
        target = MODELS_DIR / local_subdir

        if target.exists() and any(target.iterdir()):
            print(f"  [SKIP] {repo_id} (already exists)")
            results[repo_id] = True
            continue

        target.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {repo_id}...")
        try:
            kwargs = {
                "repo_id": repo_id,
                "local_dir": str(target),
                "local_dir_use_symlinks": False,
            }
            if ignore:
                kwargs["ignore_patterns"] = ignore
            snapshot_download(**kwargs)
            print(f"  [OK] {repo_id}")
            results[repo_id] = True
        except Exception as e:
            print(f"  [FAILED] {repo_id}: {e}")
            results[repo_id] = False

    return results


def main():
    """Download all models for the Windows installer."""
    print("=" * 60)
    print("CasePrepd Model Downloader")
    print("=" * 60)

    # 1. spaCy models
    print("\n--- spaCy Models ---")
    spacy_results = download_spacy_models()

    # 2. NLTK data
    print("\n--- NLTK Data ---")
    nltk_results = download_nltk_data()

    # 3. HuggingFace models
    print("\n--- HuggingFace Models ---")
    hf_results = download_huggingface_models()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_results = {**spacy_results, **nltk_results, **hf_results}
    passed = sum(1 for v in all_results.values() if v)
    failed = sum(1 for v in all_results.values() if not v)

    for name, ok in all_results.items():
        status = "[OK]" if ok else "[FAILED]"
        print(f"  {status} {name}")

    print(f"\n  {passed} succeeded, {failed} failed")

    if failed:
        print("\n[WARNING] Some downloads failed. Re-run the script.")
        sys.exit(1)
    else:
        total_mb = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file()) / (
            1024 * 1024
        )
        print(f"\n  Total size: {total_mb:.0f} MB")
        print("  All models ready for bundling.")


if __name__ == "__main__":
    main()
