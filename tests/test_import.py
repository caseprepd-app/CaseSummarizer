import traceback

try:
    print("SUCCESS: DynamicOutputWidget imported successfully")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
