"""
Download gte-reranker-modernbert-base from HuggingFace.

Downloads the cross-encoder reranker model to the project's models/ directory
for bundling with the Windows installer. Run this once before building.

Model:
    Alibaba-NLP/gte-reranker-modernbert-base (~300MB, 149M params, 8192-token context)

Usage:
    python scripts/download_reranker_model.py
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# Target directory: project_root/models/
script_dir = Path(__file__).parent
project_root = script_dir.parent
models_dir = project_root / "models"

MODEL_ID = "Alibaba-NLP/gte-reranker-modernbert-base"
MODEL_DIR = "gte-reranker-modernbert-base"

# TODO: Convert reranker to ONNX-only (ship no PyTorch weights)
#
# WHY: The reranker is the slowest stage in our 3-stage RAG pipeline (hybrid
# retrieval → reranking → LLM answer). On CPU-only machines (our target users
# are stenographers without GPUs), ONNX Runtime gives 2-3x inference speedup
# over PyTorch by converting the dynamic computation graph to an optimized
# static graph. Same model, same accuracy, just faster. Also saves ~300MB in
# the bundled installer by deleting PyTorch weights after ONNX export.
#
# WHY WE'RE WAITING: The HuggingFace ONNX ecosystem is mid-migration (Mar 2026).
# The old `optimum` package (1.x) had ModernBERT O3 optimization (PR #2208,
# merged June 2025, released in 1.25+). But `optimum` 2.x removed its
# onnxruntime/ code, delegating to the new `optimum-onnx` package — and
# `optimum-onnx` is only at v0.1.0 (Dec 2024), which does NOT include
# ModernBERT optimization support yet.
#
# Why we can't install either path today:
#   - optimum 1.27.0 (old, has ModernBERT O3): requires transformers<4.54,
#     which would DOWNGRADE our transformers 4.57.3 → 4.53.3. Too risky.
#   - optimum 2.1.0 + optimum-onnx 0.1.0 (new): no ModernBERT O3 optimization.
#     Basic ONNX export may work but without O3 the speedup is marginal.
#   - sentence-transformers 5.1.2 imports `from optimum.onnxruntime` (old path).
#     sentence-transformers 5.2.3 still uses the old path too. The main branch
#     (5.3.0.dev0) switched to `optimum-onnx`, but isn't released yet.
#
# WHAT TO CHECK PERIODICALLY:
#   - optimum-onnx on PyPI: needs a release with ModernBERT optimization config
#     (currently 0.1.0, need 0.2+ or similar)
#   - sentence-transformers: needs a release that depends on optimum-onnx instead
#     of old optimum (main branch already does, awaiting 5.3.0 release)
#   - Once both align, the install is safe: dry-run showed only 3-4 NEW packages,
#     zero upgrades/downgrades to existing deps
#
# WHEN READY — implementation steps:
#   1. pip install sentence-transformers[onnx]  (should pull optimum-onnx with
#      ModernBERT support; optimum-onnx is build-time only, not needed at runtime)
#   2. After download, load with CrossEncoder(target_dir, backend="onnx")
#   3. Call export_optimized_onnx_model(model, "O3", target_dir)
#   4. Verify onnx/model_O3.onnx exists, delete model.safetensors to save ~300MB
#   5. Update cross_encoder_reranker.py _load_model() to pass:
#        backend="onnx", model_kwargs={"file_name": "onnx/model_O3.onnx"}
#      max_length=8192 works the same in ONNX mode (confirmed in sbert docs)
#   6. Update verify_model() in this script to check for onnx/model_O3.onnx
#      instead of model.safetensors / pytorch_model.bin
#   7. Update tests that assert on model weight filenames
#   8. onnxruntime is already installed (1.23.2) and collected in caseprepd.spec
#
# DOCS & REFERENCES:
#   - sbert ONNX efficiency guide: https://sbert.net/docs/cross_encoder/usage/efficiency.html
#   - optimum ModernBERT optimization fix: https://github.com/huggingface/optimum/pull/2208
#   - optimum ModernBERT export issue: https://github.com/huggingface/optimum/issues/2177


def download_model() -> bool:
    """
    Download reranker model from HuggingFace.

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
    """Download reranker model for installer bundling."""
    print("=" * 60)
    print("Downloading GTE Reranker (ModernBERT)")
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
        print(f"  [OK] Reranker model: {total_mb:.1f} MB")
        print("\nModel ready for bundling with the installer.")
    else:
        print("  [MISSING] Reranker model")
        print("\n[WARNING] Download failed. Re-run the script.")
        sys.exit(1)


if __name__ == "__main__":
    main()
