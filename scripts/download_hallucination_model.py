"""
Download TinyLettuce hallucination verification model from HuggingFace.

Downloads the model to the project's models/ directory for bundling
with the Windows installer. Run this once before building the installer.

Model:
    KRLabsOrg/tinylettuce-ettin-68m-en (68M params, 75% F1 on RAGTruth)

Usage:
    python scripts/download_hallucination_model.py
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# Target directory: project_root/models/
script_dir = Path(__file__).parent
project_root = script_dir.parent
models_dir = project_root / "models"

MODEL_ID = "KRLabsOrg/tinylettuce-ettin-68m-en"
MODEL_DIR = "tinylettuce-ettin-68m-en"


def download_model() -> bool:
    """
    Download hallucination detection model from HuggingFace.

    Returns:
        True if download succeeded, False otherwise.
    """
    target_dir = models_dir / MODEL_DIR

    print(f"\n  Model:  {MODEL_ID}")
    print(f"  Target: {target_dir}")

    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        print("  [OK] Model downloaded")
        return True
    except Exception as e:
        print(f"  [FAILED] {e}")
        return False


def verify_model() -> bool:
    """
    Verify that key model files exist after download.

    Returns:
        True if config.json and model weights are present.
    """
    target_dir = models_dir / MODEL_DIR

    has_config = (target_dir / "config.json").exists()
    has_weights = (target_dir / "model.safetensors").exists() or (
        target_dir / "pytorch_model.bin"
    ).exists()

    return has_config and has_weights


def main():
    """Download hallucination detection model for installer bundling."""
    print("=" * 60)
    print("Downloading TinyLettuce Hallucination Model")
    print("=" * 60)

    models_dir.mkdir(parents=True, exist_ok=True)

    success = download_model()

    # Verification
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    if success and verify_model():
        target = models_dir / MODEL_DIR
        total_mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / (1024 * 1024)
        print(f"  [OK] Hallucination model: {total_mb:.1f} MB")
        print("\nModel ready for bundling with the installer.")
    else:
        print("  [MISSING] Hallucination model")
        print("\n[WARNING] Download failed. Re-run the script.")
        sys.exit(1)


if __name__ == "__main__":
    main()
