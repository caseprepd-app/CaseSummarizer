"""
Download nomic-embed-text-v1.5 embedding model from HuggingFace.

Downloads the embedding model to the project's models/ directory
for bundling with the Windows installer. Run this once before building.

Model:
    nomic-ai/nomic-embed-text-v1.5 (137M params, 768 dims, 8192-token context)

Usage:
    python scripts/download_embedding_model.py
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# Target directory: project_root/models/embeddings/
script_dir = Path(__file__).parent
project_root = script_dir.parent
models_dir = project_root / "models" / "embeddings"

MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
MODEL_DIR = "nomic-embed-text-v1.5"


def download_model() -> bool:
    """
    Download embedding model from HuggingFace.

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
            ignore_patterns=["onnx/*", "onnx/**"],  # Skip ONNX variants (~1.6GB)
        )
        print("  [OK] Model downloaded")
        return True
    except Exception as e:
        print(f"  [FAILED] {e}")
        return False


def _bundle_custom_code(target_dir: Path) -> None:
    """
    Bundle custom model code and patch config for offline loading.

    The nomic model uses trust_remote_code with custom Python files
    hosted in the nomic-ai/nomic-bert-2048 repo. Without these files,
    offline loading fails. This downloads them and rewrites auto_map
    entries in config.json to point to local modules.
    """
    import json
    import shutil

    from huggingface_hub import hf_hub_download

    code_repo = "nomic-ai/nomic-bert-2048"
    code_files = ["configuration_hf_nomic_bert.py", "modeling_hf_nomic_bert.py"]

    for filename in code_files:
        src = hf_hub_download(code_repo, filename)
        shutil.copy2(src, target_dir / filename)
        print(f"  [OK] Bundled {filename}")

    config_path = target_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "auto_map" in config:
        for key, value in config["auto_map"].items():
            if "--" in value:
                config["auto_map"][key] = value.split("--", 1)[1]
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print("  [OK] Patched config.json auto_map for offline use")


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
    """Download embedding model for installer bundling."""
    print("=" * 60)
    print("Downloading Nomic Embed Text v1.5")
    print("=" * 60)

    models_dir.mkdir(parents=True, exist_ok=True)

    success = download_model()

    if success:
        _bundle_custom_code(models_dir / MODEL_DIR)

    # Verification
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    if success and verify_model():
        target = models_dir / MODEL_DIR
        total_mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / (1024 * 1024)
        print(f"  [OK] Embedding model: {total_mb:.1f} MB")
        print("\nModel ready for bundling with the installer.")
    else:
        print("  [MISSING] Embedding model")
        print("\n[WARNING] Download failed. Re-run the script.")
        sys.exit(1)


if __name__ == "__main__":
    main()
