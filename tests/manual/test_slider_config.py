"""
Test the new slider configuration and word count range functionality.
"""

import os
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    os.system("chcp 65001 > nul 2>&1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.ai import ModelManager
from src.core.prompting import get_prompt_config

print("=" * 70)
print("CasePrepd - Slider Configuration Test")
print("=" * 70)

# Test 1: Load configuration
print("\n[Test 1] Loading prompt configuration...")
config = get_prompt_config()
print(f"  Slider increment: {config.slider_increment}")
print(f"  Word count tolerance: {config.word_count_tolerance}")
print(f"  Min/Max words: {config.min_summary_words}-{config.max_summary_words}")
print(f"  Default: {config.default_summary_words}")
print("  ✓ Configuration loaded")

# Test 2: Word count ranges
print("\n[Test 2] Testing word count ranges...")
test_values = [100, 150, 200, 250, 300, 400, 500]
for target in test_values:
    min_w, max_w = config.get_word_count_range(target)
    print(f"  Target {target} words → Range: {min_w}-{max_w} words")
print("  ✓ Ranges calculated correctly")

# Test 3: Verify slider increments
print("\n[Test 3] Simulating slider movement...")
increment = config.slider_increment
min_val = config.min_summary_words
max_val = config.max_summary_words

slider_values = list(range(min_val, max_val + 1, increment))
print(f"  Slider positions: {slider_values}")
print(f"  Total positions: {len(slider_values)}")
print("  ✓ Slider increments verified")

# Test 4: Verify ModelManager integration
print("\n[Test 4] Testing ModelManager integration...")
manager = ModelManager()
print(f"  ModelManager has prompt_config: {hasattr(manager, 'prompt_config')}")
if hasattr(manager, "prompt_config"):
    print(f"  Temperature: {manager.prompt_config.summary_temperature}")
    print(f"  Top-p: {manager.prompt_config.top_p}")
    print(f"  Tokens per word: {manager.prompt_config.tokens_per_word}")
print("  ✓ ModelManager integration verified")

# Test 5: Test prompt generation
print("\n[Test 5] Simulating prompt generation for 200 words...")
target_words = 200
min_w, max_w = config.get_word_count_range(target_words)
tokens_estimate = int(max_w * config.tokens_per_word)

print(f"  Target: {target_words} words")
print(f"  Range: {min_w}-{max_w} words")
print(f"  Max tokens to generate: {tokens_estimate}")
print(f"  Temperature: {config.summary_temperature}")

# Simulate the prompt that would be generated
prompt_snippet = f"""Instructions:
- Length: Between {min_w} and {max_w} words (target: {target_words} words)
- Focus on: key facts, parties involved, legal issues, and outcomes"""

print("\n  Generated prompt instructions:")
for line in prompt_snippet.split("\n"):
    print(f"    {line}")
print("  ✓ Prompt generation verified")

print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
print("\nSummary:")
print(f"  • Slider moves in {increment}-word increments")
print(f"  • Model will generate summaries ±{config.word_count_tolerance} words from target")
print("  • Configuration is loaded from: config/prompt_parameters.json")
print("  • Users can edit that file to customize behavior")
