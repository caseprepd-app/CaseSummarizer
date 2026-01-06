"""
Quick test script to verify prompt template system works.
"""

from src.config import PROMPTS_DIR
from src.core.prompting import PromptTemplateManager


def test_template_system():
    """Test that both presets can be loaded and formatted."""
    print("=" * 60)
    print("Testing Prompt Template System")
    print("=" * 60)

    manager = PromptTemplateManager(PROMPTS_DIR)

    # Test 1: Discover models
    print("\n1. Discovering available models...")
    models = manager.get_available_models()
    print(f"   Found models: {models}")
    assert "phi-3-mini" in models, "phi-3-mini model directory not found!"

    # Test 2: Discover presets
    print("\n2. Discovering presets for phi-3-mini...")
    presets = manager.get_available_presets("phi-3-mini")
    print(f"   Found {len(presets)} presets:")
    for preset in presets:
        print(f"     - {preset['name']} (id: {preset['id']})")

    assert len(presets) == 2, f"Expected 2 presets, found {len(presets)}"

    # Test 3: Load and format both templates
    print("\n3. Testing template loading and formatting...")
    sample_text = "This is a sample legal document for testing purposes."

    for preset in presets:
        print(f"\n   Testing preset: {preset['name']}")
        try:
            # Load template
            template = manager.load_template("phi-3-mini", preset["id"])
            print(f"     [OK] Template loaded ({len(template)} chars)")

            # Validate template
            manager.validate_template(template)
            print("     [OK] Template validated (has required tokens)")

            # Format template
            formatted = manager.format_template(
                template=template,
                min_words=180,
                max_words=200,
                max_words_range=220,
                case_text=sample_text,
            )
            print(f"     [OK] Template formatted ({len(formatted)} chars)")

            # Check formatted output has required elements
            assert "<|system|>" in formatted
            assert "<|user|>" in formatted
            assert "<|assistant|>" in formatted
            assert sample_text in formatted
            assert "180" in formatted  # min_words
            assert "200" in formatted  # max_words
            assert "220" in formatted  # max_words_range
            print("     [OK] Formatted prompt has all required elements")

        except Exception as e:
            print(f"     [FAIL] FAILED: {e}")
            raise

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_template_system()
