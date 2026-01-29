"""
Download LettuceDetect hallucination verification models from HuggingFace.

This script downloads all three model variants to the project's models/ directory
for bundling with the Windows installer. Run this once before building the installer.

Models:
    Standard: KRLabsOrg/lettucedect-base-modernbert-en-v1 (~150MB, 76% F1)
    Fast:     KRLabsOrg/tinylettuce-ettin-68m-en (~68M params, 75% F1)
    Fastest:  KRLabsOrg/tinylettuce-ettin-17m-en (~17M params, 69% F1)

Usage:
    python scripts/download_hallucination_model.py
    python scripts/download_hallucination_model.py --standard-only
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# Target directory: project_root/models/
script_dir = Path(__file__).parent
project_root = script_dir.parent
models_dir = project_root / "models"

# All model variants
MODELS = [
    {
        "id": "KRLabsOrg/lettucedect-base-modernbert-en-v1",
        "dir": "lettucedect-base-modernbert-en-v1",
        "label": "Standard (150M params)",
        "size": "~150MB",
    },
    {
        "id": "KRLabsOrg/tinylettuce-ettin-68m-en",
        "dir": "tinylettuce-ettin-68m-en",
        "label": "Fast (68M params)",
        "size": "~130MB",
    },
    {
        "id": "KRLabsOrg/tinylettuce-ettin-17m-en",
        "dir": "tinylettuce-ettin-17m-en",
        "label": "Fastest (17M params)",
        "size": "~35MB",
    },
]


def download_model(model_info: dict) -> bool:
    """
    Download a single model from HuggingFace.

    Args:
        model_info: Dict with 'id', 'dir', 'label', 'size' keys.

    Returns:
        True if download succeeded, False otherwise.
    """
    target_dir = models_dir / model_info["dir"]

    print(f"\n  Model:  {model_info['id']}")
    print(f"  Target: {target_dir}")
    print(f"  Size:   {model_info['size']}")

    try:
        snapshot_download(
            repo_id=model_info["id"],
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        print(f"  [OK] {model_info['label']} downloaded")
        return True
    except Exception as e:
        print(f"  [FAILED] {e}")
        return False


def verify_model(model_info: dict) -> bool:
    """
    Verify that key model files exist after download.

    Args:
        model_info: Dict with 'dir' key.

    Returns:
        True if all required files are present.
    """
    target_dir = models_dir / model_info["dir"]
    required_files = ["config.json"]  # All models have at least this

    # Check for model weights (safetensors or pytorch)
    has_weights = (target_dir / "model.safetensors").exists() or (
        target_dir / "pytorch_model.bin"
    ).exists()

    all_present = has_weights
    for filename in required_files:
        if not (target_dir / filename).exists():
            all_present = False

    return all_present


def main():
    """Download all hallucination detection models for installer bundling."""
    standard_only = "--standard-only" in sys.argv

    print("=" * 60)
    print("Downloading LettuceDetect Hallucination Models")
    print("=" * 60)

    models_dir.mkdir(parents=True, exist_ok=True)

    models_to_download = [MODELS[0]] if standard_only else MODELS

    print(f"\nDownloading {len(models_to_download)} model(s)...")

    results = []
    for model in models_to_download:
        print(f"\n--- {model['label']} ---")
        success = download_model(model)
        results.append((model, success))

    # Verification
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    all_ok = True
    for model, downloaded in results:
        if downloaded and verify_model(model):
            target = models_dir / model["dir"]
            total_mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / (
                1024 * 1024
            )
            print(f"  [OK] {model['label']}: {total_mb:.1f} MB")
        else:
            print(f"  [MISSING] {model['label']}")
            all_ok = False

    if all_ok:
        print("\nAll models ready for bundling with the installer.")
    else:
        print("\n[WARNING] Some models failed. Re-run the download.")
        sys.exit(1)


if __name__ == "__main__":
    main()
