"""
Quick test script to verify AI model loading and text generation.
Tests the ModelManager without launching the full GUI.
"""

import os
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    os.system("chcp 65001 > nul 2>&1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.ai import ModelManager


def test_model_loading():
    """Test basic model loading and text generation."""
    print("=" * 60)
    print("LocalScribe Model Test")
    print("=" * 60)

    # Initialize model manager
    print("\n[1/4] Initializing ModelManager...")
    manager = ModelManager()

    # Check available models
    print("\n[2/4] Checking available models...")
    models = manager.get_available_models()

    for _model_key, model_info in models.items():
        status = "✓ Available" if model_info["available"] else "✗ Not found"
        print(f"  {model_info['name']}: {status}")
        if model_info["available"]:
            print(f"    Path: {model_info['path']}")
            print(f"    Size: {model_info['size_gb']} GB")

    # Find first available model
    available_model = None
    for model_key, model_info in models.items():
        if model_info["available"]:
            available_model = model_key
            break

    if not available_model:
        print("\n❌ ERROR: No models found!")
        print("   Please download a model to the models directory")
        return False

    # Load the model
    print(f"\n[3/4] Loading {models[available_model]['name']} model...")
    print("   (This may take 30-60 seconds...)")

    success = manager.load_model(available_model, verbose=False)

    if not success:
        print("\n❌ ERROR: Failed to load model!")
        return False

    print("   ✓ Model loaded successfully!")

    # Test text generation
    print("\n[4/4] Testing text generation...")
    test_prompt = "Summarize this in one sentence: The plaintiff filed a complaint alleging breach of contract."

    print(f"   Prompt: {test_prompt}")
    print("   Response: ", end="", flush=True)

    try:
        # Generate with streaming
        full_response = ""
        for token in manager.generate_text(
            prompt=test_prompt, max_tokens=50, temperature=0.3, stream=True
        ):
            print(token, end="", flush=True)
            full_response += token

        print("\n")

        if full_response.strip():
            print("\n✓ Text generation successful!")
            print("\n" + "=" * 60)
            print("MODEL TEST PASSED ✓")
            print("=" * 60)
            return True
        else:
            print("\n❌ ERROR: Model generated empty response!")
            return False

    except Exception as e:
        print(f"\n\n❌ ERROR during generation: {e!s}")
        return False
    finally:
        # Cleanup
        manager.unload_model()
        print("\nModel unloaded.")


if __name__ == "__main__":
    try:
        success = test_model_loading()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e!s}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
