# Development Scripts

Utility scripts for setting up and testing CasePrepd.

## Scripts

| Script | Purpose |
|--------|---------|
| `check_spacy.py` | Verify spaCy installation and model availability |
| `download_onnx_models.py` | Download ONNX models (legacy - now using Ollama) |

## Usage

Run from project root:

```bash
# Check spaCy installation
python scripts/check_spacy.py

# Download ONNX models (legacy)
python scripts/download_onnx_models.py
```

## Notes

- `download_onnx_models.py` is kept for reference but CasePrepd now uses Ollama for inference
- These scripts are not part of the main application; they're development/setup utilities
