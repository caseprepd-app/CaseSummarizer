#!/usr/bin/env python3
"""
Test script to verify the complete Ollama integration workflow.
This simulates what the GUI does when a user clicks "Generate Summaries".
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import PROMPTS_DIR  # noqa: E402
from src.core.ai.ollama_model_manager import OllamaModelManager  # noqa: E402
from src.core.prompting import PromptTemplateManager  # noqa: E402


def test_ollama_connection():
    """Test 1: Can we connect to Ollama?"""
    print("\n" + "=" * 60)
    print("TEST 1: Ollama Connection")
    print("=" * 60)

    manager = OllamaModelManager()

    if not manager.is_connected:
        print("[FAIL] Cannot connect to Ollama")
        print(f"       Trying to reach: {manager.api_base}")
        return False

    print("[PASS] Connected to Ollama")
    print(f"       API Base: {manager.api_base}")
    return True


def test_model_availability():
    """Test 2: Can we get available models?"""
    print("\n" + "=" * 60)
    print("TEST 2: Model Availability")
    print("=" * 60)

    manager = OllamaModelManager()
    models = manager.get_available_models()

    if not models:
        print("[FAIL] No models available")
        return False

    print(f"[PASS] Found {len(models)} model(s):")
    for model_name in models:
        print(f"       - {model_name}")

    return True


def test_prompt_templates():
    """Test 3: Can we load prompt templates?"""
    print("\n" + "=" * 60)
    print("TEST 3: Prompt Templates")
    print("=" * 60)

    manager = PromptTemplateManager(PROMPTS_DIR)

    available_models = manager.get_available_models()
    print(f"Template models available: {available_models}")

    if "phi-3-mini" not in available_models:
        print("[FAIL] phi-3-mini templates not found")
        return False

    presets = manager.get_available_presets("phi-3-mini")

    if not presets:
        print("[FAIL] No presets found for phi-3-mini")
        return False

    print(f"[PASS] Found {len(presets)} preset(s):")
    for preset in presets:
        print(f"       - {preset['name']} ({preset['id']})")

    return True


def test_summary_generation():
    """Test 4: Can we generate a summary?"""
    print("\n" + "=" * 60)
    print("TEST 4: Summary Generation")
    print("=" * 60)

    # Read test document (moved to tests/sample_docs/ in Session 34)
    test_doc = Path(__file__).parent.parent / "sample_docs" / "test_simple_case.txt"
    if not test_doc.exists():
        print("[FAIL] Test document not found")
        return False

    with open(test_doc, encoding="utf-8") as f:
        case_text = f.read()

    print(f"Test document loaded: {len(case_text)} chars")

    # Initialize manager
    manager = OllamaModelManager()

    if not manager.is_model_loaded():
        print("[FAIL] Ollama not accessible")
        return False

    try:
        # Generate summary (small, for quick testing)
        print("Generating summary (this may take 30-60 seconds on CPU)...")
        summary = manager.generate_summary(
            case_text=case_text, max_words=100, preset_id="factual-summary"
        )

        if not summary:
            print("[FAIL] Empty summary returned")
            return False

        word_count = len(summary.split())
        print("[PASS] Summary generated successfully!")
        print(f"       Length: {word_count} words, {len(summary)} chars")
        print("\nSummary preview (first 200 chars):")
        print(f"       {summary[:200]}...")

        return True

    except Exception as e:
        print(f"[FAIL] Summary generation error: {e!s}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "[TEST] OLLAMA INTEGRATION TEST SUITE" + "\n")

    tests = [
        test_ollama_connection,
        test_model_availability,
        test_prompt_templates,
        test_summary_generation,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception:
            print(f"\n[ERROR] Unexpected exception in {test.__name__}:")
            import traceback

            traceback.print_exc()
            results.append((test.__name__, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}  {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed! Ollama integration is working correctly.")
        return 0
    else:
        print(f"\n[FAILED] {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
