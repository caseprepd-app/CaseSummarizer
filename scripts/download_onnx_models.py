"""
Download Phi-3 Mini ONNX models from HuggingFace.
This script downloads both DirectML and CPU-optimized versions.
"""

from huggingface_hub import snapshot_download
from pathlib import Path
import sys

# Get models directory
models_dir = Path.home() / "AppData" / "Roaming" / "CasePrepd" / "models"
models_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Downloading Phi-3 Mini ONNX Models")
print("=" * 60)

# Download DirectML version (for integrated GPU acceleration)
print("\n[1/2] Downloading DirectML version (GPU-accelerated)...")
print("      This may take several minutes (~2-3 GB)")
print("      Target:", models_dir / "phi-3-mini-onnx-directml")

try:
    snapshot_download(
        repo_id="microsoft/Phi-3-mini-4k-instruct-onnx",
        allow_patterns=["directml/*"],
        local_dir=str(models_dir / "phi-3-mini-onnx-directml"),
        local_dir_use_symlinks=False
    )
    print("[OK] DirectML version downloaded successfully!")
except Exception as e:
    print(f"[FAILED] Error downloading DirectML version: {e}")
    sys.exit(1)

# Download CPU version (fallback for systems without DirectX 12)
print("\n[2/2] Downloading CPU version (fallback)...")
print("      This may take several minutes (~2-3 GB)")
print("      Target:", models_dir / "phi-3-mini-onnx-cpu")

try:
    snapshot_download(
        repo_id="microsoft/Phi-3-mini-4k-instruct-onnx",
        allow_patterns=["cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*"],
        local_dir=str(models_dir / "phi-3-mini-onnx-cpu"),
        local_dir_use_symlinks=False
    )
    print("[OK] CPU version downloaded successfully!")
except Exception as e:
    print(f"[FAILED] Error downloading CPU version: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Download complete!")
print("=" * 60)
print(f"\nModels saved to: {models_dir}")
print("\nNext steps:")
print("1. Create ONNX-based ModelManager")
print("2. Update configuration to use ONNX models")
print("3. Test performance with DirectML")
