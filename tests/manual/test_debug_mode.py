"""
Debug Mode Test
Tests the AI model with a simulated case document using debug mode.
"""

import sys
import os
from pathlib import Path
import time

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Enable debug mode
os.environ['DEBUG'] = 'true'

from src.core.ai import ModelManager
from src.core.utils.logger import debug

# Sample case text for testing
SAMPLE_CASE = """
IN THE SUPREME COURT OF THE STATE OF NEW YORK
COUNTY OF NEW YORK

JOHN DOE,
    Plaintiff,

v.

ACME CORPORATION,
    Defendant.

Index No. 123456/2024

COMPLAINT

Plaintiff John Doe, by and through his attorneys, alleges as follows:

PARTIES

1. Plaintiff John Doe is a resident of New York County, New York.

2. Defendant ACME Corporation is a Delaware corporation with its principal place of
business located in New York, New York.

JURISDICTION AND VENUE

3. This Court has jurisdiction over this matter pursuant to CPLR § 301.

4. Venue is proper in New York County pursuant to CPLR § 503.

FACTUAL ALLEGATIONS

5. On or about January 15, 2024, Plaintiff and Defendant entered into a written
contract (the "Contract") whereby Defendant agreed to provide consulting services
to Plaintiff for a period of six months in exchange for payment of $50,000.

6. Plaintiff performed all obligations under the Contract, including timely payment
of the full $50,000 fee.

7. Defendant failed to provide the consulting services as required under the Contract.

8. Despite repeated demands, Defendant has refused to perform its obligations or
return the payment.

FIRST CAUSE OF ACTION
(Breach of Contract)

9. Plaintiff repeats and realleges each allegation contained in paragraphs 1-8 as
if fully set forth herein.

10. The Contract constitutes a valid and enforceable agreement.

11. Plaintiff performed all conditions precedent to Defendant's obligations under
the Contract.

12. Defendant breached the Contract by failing to provide the agreed-upon services.

13. As a direct and proximate result of Defendant's breach, Plaintiff has been
damaged in an amount of not less than $50,000, plus interest and costs.

WHEREFORE, Plaintiff demands judgment against Defendant as follows:

A. Compensatory damages in the amount of $50,000;
B. Pre-judgment and post-judgment interest;
C. Costs and disbursements of this action; and
D. Such other and further relief as the Court deems just and proper.

Dated: March 1, 2024
       New York, New York

                                    _________________________
                                    Attorney for Plaintiff
                                    123 Legal Ave
                                    New York, NY 10001
                                    (212) 555-0100
"""

def test_debug_mode():
    """Test model with debug mode enabled."""
    print("=" * 70)
    print("LocalScribe Debug Mode Test")
    print("=" * 70)

    # Verify debug mode is enabled
    from src.config import DEBUG_MODE
    print(f"\n[✓] Debug mode: {DEBUG_MODE}")

    if not DEBUG_MODE:
        print("   WARNING: Debug mode not enabled!")
        return False

    # Initialize model manager
    print("\n" + "=" * 70)
    print("STEP 1: Initialize Model Manager")
    print("=" * 70)

    start_time = time.time()
    manager = ModelManager()
    elapsed = time.time() - start_time
    debug(f"ModelManager initialization took {elapsed*1000:.1f} ms")

    # Check available models
    print("\n" + "=" * 70)
    print("STEP 2: Check Available Models")
    print("=" * 70)

    start_time = time.time()
    models = manager.get_available_models()
    elapsed = time.time() - start_time
    debug(f"Model check took {elapsed*1000:.1f} ms")

    available_model = None
    for model_key, model_info in models.items():
        if model_info['available']:
            available_model = model_key
            print(f"\n[✓] Found: {model_info['name']}")
            print(f"    Path: {model_info['path']}")
            print(f"    Size: {model_info['size_gb']} GB")
            break

    if not available_model:
        print("\n[✗] ERROR: No models available!")
        return False

    # Load model
    print("\n" + "=" * 70)
    print("STEP 3: Load Model")
    print("=" * 70)
    print("(This may take 30-60 seconds...)\n")

    start_time = time.time()
    success = manager.load_model(available_model, verbose=False)
    elapsed = time.time() - start_time

    if not success:
        print(f"\n[✗] ERROR: Model loading failed after {elapsed:.1f}s")
        return False

    debug(f"Model loading took {elapsed:.1f} seconds")
    print(f"\n[✓] Model loaded in {elapsed:.1f} seconds")

    # Generate summary
    print("\n" + "=" * 70)
    print("STEP 4: Generate Case Summary")
    print("=" * 70)
    print(f"\nInput document length: {len(SAMPLE_CASE)} characters\n")

    print("Generating summary (200 words)...\n")
    print("-" * 70)

    start_time = time.time()
    try:
        summary = ""
        for token in manager.generate_summary(
            case_text=SAMPLE_CASE,
            max_words=200,
            stream=True
        ):
            print(token, end="", flush=True)
            summary += token

        elapsed = time.time() - start_time
        print("\n" + "-" * 70)

        if summary.strip():
            word_count = len(summary.split())
            debug(f"Summary generation took {elapsed:.2f} seconds")
            print(f"\n[✓] Summary generated successfully!")
            print(f"    Words: {word_count}")
            print(f"    Characters: {len(summary)}")
            print(f"    Time: {elapsed:.2f}s")
            print(f"    Speed: {word_count/elapsed:.1f} words/second")

            # Cleanup
            print("\n" + "=" * 70)
            print("STEP 5: Cleanup")
            print("=" * 70)
            manager.unload_model()
            debug("Model unloaded successfully")

            print("\n" + "=" * 70)
            print("DEBUG MODE TEST PASSED ✓")
            print("=" * 70)
            return True
        else:
            print("\n[✗] ERROR: Empty summary generated")
            return False

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n\n[✗] ERROR after {elapsed:.2f}s: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        manager.unload_model()

if __name__ == "__main__":
    try:
        success = test_debug_mode()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[✗] UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
