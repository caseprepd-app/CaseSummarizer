"""
Test script to verify UI initialization works correctly.
This tests that all widgets are created and the tuple unpacking is correct.
"""

import sys

try:
    print("[TEST] Importing modules...")
    from src.ui.main_window import MainWindow

    print("[TEST] Creating MainWindow instance...")
    # This will fail at __init__ if tuple unpacking is wrong
    app = MainWindow()

    print("[TEST] Checking cancel button exists...")
    assert hasattr(app, "cancel_btn"), "MainWindow missing cancel_btn attribute"

    print("[TEST] Checking output_options has lock/unlock methods...")
    assert hasattr(app.output_options, "lock_controls"), (
        "OutputOptionsWidget missing lock_controls method"
    )
    assert hasattr(app.output_options, "unlock_controls"), (
        "OutputOptionsWidget missing unlock_controls method"
    )

    print("[TEST] All checks passed!")
    print("[TEST] The UI startup is working correctly.")

    # Clean exit without showing window
    app.quit()
    sys.exit(0)

except Exception as e:
    print("[TEST FAILED] Error during UI initialization:")
    print(f"  {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
