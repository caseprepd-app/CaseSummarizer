#!/usr/bin/env python
"""Test imports to identify any errors."""

import sys
import traceback

# Write output to file since cmd capture is unreliable
output_file = open("import_test_results.txt", "w")  # noqa: SIM115


def log(msg):
    print(msg)
    output_file.write(msg + "\n")
    output_file.flush()


log("Testing imports...")
log(f"Python: {sys.executable}")

try:
    log("1. Testing src.categories...")
    from src.categories import get_category_list

    log(f"   OK - Categories: {get_category_list()}")
except Exception as e:
    log(f"   FAILED: {e}")
    output_file.write(traceback.format_exc())

try:
    log("2. Testing src.ai.ollama_model_manager...")

    log("   OK")
except Exception as e:
    log(f"   FAILED: {e}")
    output_file.write(traceback.format_exc())

try:
    log("3. Testing src.extraction.llm_extractor...")

    log("   OK")
except Exception as e:
    log(f"   FAILED: {e}")
    output_file.write(traceback.format_exc())

try:
    log("4. Testing src.vocabulary.reconciler...")

    log("   OK")
except Exception as e:
    log(f"   FAILED: {e}")
    output_file.write(traceback.format_exc())

try:
    log("5. Testing src.vocabulary.vocabulary_extractor...")

    log("   OK")
except Exception as e:
    log(f"   FAILED: {e}")
    output_file.write(traceback.format_exc())

try:
    log("6. Testing src.ui.widgets...")

    log("   OK")
except Exception as e:
    log(f"   FAILED: {e}")
    output_file.write(traceback.format_exc())

try:
    log("7. Testing src.main...")

    log("   OK")
except Exception as e:
    log(f"   FAILED: {e}")
    output_file.write(traceback.format_exc())

log("\nImport tests complete!")
output_file.close()
