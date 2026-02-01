"""
Download LingMess coreference resolution model from HuggingFace.

Downloads the model to the project's models/coref/lingmess/ directory
for bundling with the Windows installer. Run this once before building.

Model:
    biu-nlp/lingmess-large (LingMess, 81.4 F1 on OntoNotes)

Usage:
    python scripts/download_coref_model.py
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# Target directory: project_root/models/coref/lingmess/
script_dir = Path(__file__).parent
project_root = script_dir.parent
target_dir = project_root / "models" / "coref" / "lingmess"

MODEL_ID = "biu-nlp/lingmess-large"


def download_model() -> bool:
    """
    Download LingMess coref model from HuggingFace.

    Returns:
        True if download succeeded, False otherwise.
    """
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
    has_config = (target_dir / "config.json").exists()
    has_weights = (target_dir / "model.safetensors").exists() or (
        target_dir / "pytorch_model.bin"
    ).exists()

    return has_config and has_weights


def main():
    """Download LingMess coref model for installer bundling."""
    print("=" * 60)
    print("Downloading LingMess Coreference Model")
    print("=" * 60)

    target_dir.parent.mkdir(parents=True, exist_ok=True)

    success = download_model()

    # Verification
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    if success and verify_model():
        total_mb = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file()) / (
            1024 * 1024
        )
        print(f"  [OK] Coref model: {total_mb:.1f} MB")
        print("\nModel ready for bundling with the installer.")
    else:
        print("  [MISSING] Coref model")
        print("\n[WARNING] Download failed. Re-run the script.")
        sys.exit(1)


if __name__ == "__main__":
    main()
