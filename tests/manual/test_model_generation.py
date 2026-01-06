"""
Simple test to verify model can generate text.

This bypasses all UI complexity and tests the core model directly.
If this fails, the issue is with the model or model_manager.
If this works, the issue is in the UI/worker integration.
"""

import time

from src.core.ai import ModelManager

print("=" * 60)
print("Direct Model Generation Test")
print("=" * 60)

# Initialize model manager
print("\n1. Initializing ModelManager...")
mm = ModelManager()

# Check if model is available
print("\n2. Checking model availability...")
models = mm.get_available_models()
print(f"   Standard model available: {models['standard']['available']}")

if not models["standard"]["available"]:
    print("   ERROR: Model not downloaded!")
    exit(1)

# Load model
print("\n3. Loading standard model...")
load_start = time.time()
success = mm.load_model("standard")
load_time = time.time() - load_start

if not success:
    print("   ERROR: Model failed to load!")
    exit(1)

print(f"   SUCCESS: Model loaded in {load_time:.1f} seconds")

# Test 1: Simple non-streaming generation
print("\n4. Test 1: Simple non-streaming generation")
print("   Prompt: 'Write a 10 word sentence about cats.'")
print("   Generating...")

gen_start = time.time()
try:
    result = mm.generate_text(
        prompt="Write a 10 word sentence about cats.", max_tokens=50, stream=False
    )
    gen_time = time.time() - gen_start
    print(f"   SUCCESS (took {gen_time:.1f} seconds):")
    print(f"   Output: {result}")
except Exception as e:
    print(f"   ERROR: {e}")
    exit(1)

# Test 2: Streaming generation
print("\n5. Test 2: Streaming generation")
print("   Prompt: 'Count from 1 to 5 with commas.'")
print("   Streaming...")

gen_start = time.time()
try:
    tokens = []
    for token in mm.generate_text(
        prompt="Count from 1 to 5 with commas.", max_tokens=30, stream=True
    ):
        tokens.append(token)
        print(f"   Token: '{token}'", end="", flush=True)

    gen_time = time.time() - gen_start
    print(f"\n   SUCCESS (took {gen_time:.1f} seconds):")
    print(f"   Full output: {''.join(tokens)}")
    print(f"   Total tokens: {len(tokens)}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Test 3: Summary generation with small input
print("\n6. Test 3: Summary generation (500 words input)")
test_text = """This is a test legal document. """ * 100  # ~500 words
print(f"   Input: {len(test_text.split())} words")
print("   Generating summary...")

gen_start = time.time()
try:
    tokens = []
    first_token_time = None

    for token in mm.generate_summary(case_text=test_text, max_words=50, stream=True):
        if first_token_time is None:
            first_token_time = time.time() - gen_start
            print(f"   First token after {first_token_time:.1f} seconds!")

        tokens.append(token)

    gen_time = time.time() - gen_start
    full_text = "".join(tokens)

    print(f"   SUCCESS (took {gen_time:.1f} seconds total):")
    print(f"   Summary: {full_text[:200]}...")
    print(f"   Total tokens: {len(tokens)}")
    print(f"   Word count: {len(full_text.split())}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("All tests PASSED!")
print("Model is working correctly.")
print("If UI still doesn't work, the issue is in UI/worker code.")
print("=" * 60)
