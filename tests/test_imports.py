#!/usr/bin/env python
"""Test imports to identify any errors."""
import sys
import traceback

# Write output to file since cmd capture is unreliable
output_file = open("import_test_results.txt", "w")

def log(msg):
    print(msg)
    output_file.write(msg + "\n")
    output_file.flush()

log("Testing imports...")
log(f"Python: {sys.executable}")

try:
    log("1. Testing src.categories...")
    from src.categories import get_category_list, normalize_category
    log(f"   OK - Categories: {get_category_list()}")
except Exception as e:
    log(f"   FAILED: {e}")
    output_file.write(traceback.format_exc())

try:
    log("2. Testing src.ai.ollama_model_manager...")
    from src.core.ai.ollama_model_manager import OllamaModelManager
    log("   OK")
except Exception as e:
    log(f"   FAILED: {e}")
    output_file.write(traceback.format_exc())

try:
    log("3. Testing src.extraction.llm_extractor...")
    from src.core.extraction.llm_extractor import LLMVocabExtractor, LLMTerm
    log("   OK")
except Exception as e:
    log(f"   FAILED: {e}")
    output_file.write(traceback.format_exc())

try:
    log("4. Testing src.vocabulary.reconciler...")
    from src.core.vocabulary.reconciler import VocabularyReconciler
    log("   OK")
except Exception as e:
    log(f"   FAILED: {e}")
    output_file.write(traceback.format_exc())

try:
    log("5. Testing src.vocabulary.vocabulary_extractor...")
    from src.core.vocabulary.vocabulary_extractor import VocabularyExtractor
    log("   OK")
except Exception as e:
    log(f"   FAILED: {e}")
    output_file.write(traceback.format_exc())

try:
    log("6. Testing src.ui.widgets...")
    from src.ui.widgets import OutputOptionsWidget
    log("   OK")
except Exception as e:
    log(f"   FAILED: {e}")
    output_file.write(traceback.format_exc())

try:
    log("7. Testing src.main...")
    from src.main import main
    log("   OK")
except Exception as e:
    log(f"   FAILED: {e}")
    output_file.write(traceback.format_exc())

log("\nImport tests complete!")
output_file.close()
