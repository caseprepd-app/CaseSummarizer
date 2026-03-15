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
    models/nltk_data/corpora/omw-1.4/     (pruned to English-only metadata)
    models/nltk_data/corpora/stopwords/      (rake-nltk dependency)
    models/nltk_data/tokenizers/punkt_tab/   (rake-nltk sentence tokenizer)
    models/embeddings/nomic-embed-text-v1.5/  (FAISS embeddings)
    models/gte-reranker-modernbert-base/  (cross-encoder reranker)
    models/tesseract/                     (OCR engine + eng.traineddata)
    models/poppler/                       (PDF-to-image conversion)
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
NLTK_CORPORA = ["words", "wordnet", "omw-1.4", "stopwords", "punkt_tab"]
NLTK_DIR = MODELS_DIR / "nltk_data"

# HuggingFace models: (repo_id, local_subdir, ignore_patterns)
HF_MODELS = [
    (
        "nomic-ai/nomic-embed-text-v1.5",
        "embeddings/nomic-embed-text-v1.5",
        ["onnx/*", "onnx/**"],
    ),  # Skip ONNX variants (~1.6GB)
    (
        "sentence-transformers/all-MiniLM-L6-v2",
        "embeddings/all-MiniLM-L6-v2",
        [
            "onnx/*",
            "onnx/**",
            "openvino/*",
            "openvino/**",
            "rust_model.ot",
            "tf_model.h5",
            "pytorch_model.bin",
            "train_script.py",
            "data_config.json",
        ],
    ),  # Keep only safetensors + tokenizer (~87MB vs ~932MB)
    ("Alibaba-NLP/gte-reranker-modernbert-base", "gte-reranker-modernbert-base", None),
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

            package_dir = Path(mod.__file__).parent

            # spaCy packages nest the actual model data in a versioned
            # subdirectory (e.g. en_core_web_lg/en_core_web_lg-3.7.1/).
            # spacy.load() expects config.cfg at the top level, so copy
            # the inner data directory, not the Python package wrapper.
            data_dirs = [
                d for d in package_dir.iterdir() if d.is_dir() and (d / "config.cfg").exists()
            ]
            if data_dirs:
                source = data_dirs[0]
            else:
                source = package_dir

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

    # Extract any .zip files that lack a corresponding extracted directory.
    # PyInstaller frozen bundles on Windows can't reliably read NLTK zips,
    # so we need the extracted directories to exist.
    import zipfile

    for zip_path in NLTK_DIR.rglob("*.zip"):
        extracted_dir = zip_path.with_suffix("")
        if not extracted_dir.is_dir():
            print(f"  Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(zip_path.parent)

    # Delete .zip files after extraction (saves ~42 MB)
    for zip_path in NLTK_DIR.rglob("*.zip"):
        zip_path.unlink()
        print(f"  Deleted {zip_path.name} (extracted dir exists)")

    # Prune omw-1.4 non-English language data (saves ~91 MB).
    # The app only uses English WordNet — multilingual data is unnecessary.
    omw_dir = NLTK_DIR / "corpora" / "omw-1.4"
    if omw_dir.is_dir():
        import shutil

        for entry in omw_dir.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
                print(f"  Pruned omw-1.4/{entry.name} (non-English)")

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


TESSERACT_DIR = MODELS_DIR / "tesseract"
TESSERACT_SYSTEM_DIR = Path("C:/Program Files/Tesseract-OCR")
POPPLER_DIR = MODELS_DIR / "poppler"
POPPLER_GITHUB_API = "https://api.github.com/repos/oschwartz10612/poppler-windows/releases/latest"


def download_tesseract() -> dict[str, bool]:
    """
    Copy Tesseract binaries from system install to bundled directory.

    Copies tesseract.exe, all DLLs, and tessdata/eng.traineddata from
    C:\\Program Files\\Tesseract-OCR\\ to models/tesseract/.

    Returns:
        Dict mapping 'tesseract' to success status.
    """
    if (TESSERACT_DIR / "tesseract.exe").exists():
        print("  [SKIP] tesseract (already exists)")
        return {"tesseract": True}

    if not TESSERACT_SYSTEM_DIR.exists():
        print(
            f"  [FAILED] tesseract: System install not found at {TESSERACT_SYSTEM_DIR}\n"
            f"           Install Tesseract first: https://github.com/UB-Mannheim/tesseract/wiki"
        )
        return {"tesseract": False}

    print("  Copying Tesseract from system install...")
    try:
        TESSERACT_DIR.mkdir(parents=True, exist_ok=True)
        tessdata_dir = TESSERACT_DIR / "tessdata"
        tessdata_dir.mkdir(exist_ok=True)

        # Copy tesseract.exe and all DLLs
        for f in TESSERACT_SYSTEM_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in (".exe", ".dll"):
                shutil.copy2(f, TESSERACT_DIR / f.name)

        # Copy eng.traineddata
        src_tessdata = TESSERACT_SYSTEM_DIR / "tessdata" / "eng.traineddata"
        if src_tessdata.exists():
            shutil.copy2(src_tessdata, tessdata_dir / "eng.traineddata")
        else:
            print("  [WARNING] eng.traineddata not found in system tessdata")

        print(f"  [OK] tesseract -> {TESSERACT_DIR}")
        return {"tesseract": True}
    except Exception as e:
        print(f"  [FAILED] tesseract: {e}")
        return {"tesseract": False}


def download_poppler() -> dict[str, bool]:
    """
    Download Poppler Windows binaries from GitHub releases.

    Downloads the latest release zip from oschwartz10612/poppler-windows
    and extracts Library/bin/ contents to models/poppler/.

    Returns:
        Dict mapping 'poppler' to success status.
    """
    if (POPPLER_DIR / "pdftoppm.exe").exists():
        print("  [SKIP] poppler (already exists)")
        return {"poppler": True}

    print("  Downloading Poppler from GitHub...")
    try:
        import io
        import json
        import zipfile
        from urllib.request import Request, urlopen

        # Get latest release info
        req = Request(POPPLER_GITHUB_API, headers={"User-Agent": "CasePrepd-Downloader"})
        with urlopen(req, timeout=30) as resp:
            release = json.loads(resp.read())

        # Find the zip asset
        zip_url = None
        for asset in release.get("assets", []):
            if asset["name"].endswith(".zip"):
                zip_url = asset["browser_download_url"]
                break

        if not zip_url:
            print("  [FAILED] poppler: No .zip asset found in latest release")
            return {"poppler": False}

        print(f"  Downloading {zip_url}...")
        req = Request(zip_url, headers={"User-Agent": "CasePrepd-Downloader"})
        with urlopen(req, timeout=120) as resp:
            zip_data = resp.read()

        POPPLER_DIR.mkdir(parents=True, exist_ok=True)

        # Extract Library/bin/ contents
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            for member in zf.namelist():
                # Match paths like Release-XX.YY.Z-0/Library/bin/filename.ext
                parts = member.replace("\\", "/").split("/")
                if "Library" in parts and "bin" in parts:
                    bin_idx = parts.index("bin")
                    if bin_idx + 1 < len(parts) and parts[-1]:
                        filename = parts[-1]
                        with zf.open(member) as src:
                            (POPPLER_DIR / filename).write_bytes(src.read())

        # Verify key files
        if not (POPPLER_DIR / "pdftoppm.exe").exists():
            print("  [FAILED] poppler: pdftoppm.exe not found after extraction")
            return {"poppler": False}

        print(f"  [OK] poppler -> {POPPLER_DIR}")
        return {"poppler": True}
    except Exception as e:
        print(f"  [FAILED] poppler: {e}")
        return {"poppler": False}


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

    # 4. Tesseract OCR
    print("\n--- Tesseract OCR ---")
    tesseract_results = download_tesseract()

    # 5. Poppler (PDF-to-image)
    print("\n--- Poppler ---")
    poppler_results = download_poppler()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_results = {
        **spacy_results,
        **nltk_results,
        **hf_results,
        **tesseract_results,
        **poppler_results,
    }
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
