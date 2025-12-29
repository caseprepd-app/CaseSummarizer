"""
Download LettuceDetect hallucination verification model from HuggingFace.

This script downloads the model to the project's models/ directory for bundling
with the Windows installer. Run this once before building the installer.

Model: KRLabsOrg/lettucedect-base-modernbert-en-v1 (~150MB)

Usage:
    python scripts/download_hallucination_model.py
"""

from huggingface_hub import snapshot_download
from pathlib import Path
import sys

# Target directory: project_root/models/lettucedect-base-modernbert-en-v1/
script_dir = Path(__file__).parent
project_root = script_dir.parent
models_dir = project_root / "models"
target_dir = models_dir / "lettucedect-base-modernbert-en-v1"

# HuggingFace model identifier
MODEL_ID = "KRLabsOrg/lettucedect-base-modernbert-en-v1"

print("=" * 60)
print("Downloading LettuceDetect Hallucination Model")
print("=" * 60)
print(f"\nModel: {MODEL_ID}")
print(f"Target: {target_dir}")
print(f"Size: ~150MB")
print()

# Create models directory if needed
models_dir.mkdir(parents=True, exist_ok=True)

# Download the model
print("Downloading model files...")
print("(This may take a few minutes on first download)")
print()

try:
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False
    )
    print("\n[OK] Model downloaded successfully!")
except Exception as e:
    print(f"\n[FAILED] Error downloading model: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Download complete!")
print("=" * 60)
print(f"\nModel saved to: {target_dir}")
print("\nVerification:")

# Verify key files exist
required_files = ["config.json", "model.safetensors", "tokenizer.json"]
all_present = True
for filename in required_files:
    filepath = target_dir / filename
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  [OK] {filename} ({size_mb:.1f} MB)")
    else:
        print(f"  [MISSING] {filename}")
        all_present = False

if all_present:
    print("\nThe model is ready for bundling with the installer.")
    print("The hallucination verifier will automatically use this local copy.")
else:
    print("\n[WARNING] Some model files are missing. Re-run the download.")
    sys.exit(1)
