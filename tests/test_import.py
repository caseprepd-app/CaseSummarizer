import sys
import traceback

try:
    from src.ui.dynamic_output import DynamicOutputWidget
    print("SUCCESS: DynamicOutputWidget imported successfully")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
