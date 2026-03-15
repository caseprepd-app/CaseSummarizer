# CasePrepd

100% offline, private document processing for court reporters. Extracts names, vocabulary, key sentences, and supports semantic search across legal documents without any data leaving your computer.

## Requirements

- Python 3.12+
- ~4GB disk space for spaCy model + embedding models

## Installation

```bash
# Clone and enter directory
cd CasePrepd

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for testing/linting

# Download spaCy model
python -m spacy download en_core_web_lg
```

## Running

Entry point: `src/main.py`

**Important:** The system PATH Python may be 3.13, but the project requires
3.12 via `.venv`. Always activate the venv first, or run directly with
`.venv/Scripts/python`.

```bash
# Activate first (Windows)
.venv\Scripts\activate

# Normal mode
python src/main.py

# Debug mode (developer training data)
python src/main.py --debug

# Or run directly without activating
.venv/Scripts/python src/main.py --debug
```

### Debug Mode and the Developer Dataset

CasePrepd uses a two-dataset ML system for vocabulary filtering. When you rate terms with thumbs up/down, the feedback trains a model that learns which extracted terms are useful vs. noise.

**Two separate CSVs:**
- `config/default_feedback.csv` — Developer dataset. Ships with the app as a baseline.
- `%APPDATA%/CasePrepd/feedback/user_feedback.csv` — End-user dataset. Built up as the user rates terms.

**Running with `--debug`** routes all feedback to the developer CSV instead of the user CSV. The window title shows `"CasePrepd [DEBUG]"` so you know which mode you're in. Use this to build the baseline training data — rate common junk terms (transcript artifacts, OCR errors, filler phrases) as thumbs-down so new users get a useful model out of the box.

**At training time, both CSVs are combined with source-based weighting.** The developer baseline stabilizes the model early on, then gradually yields to the user's own preferences:

| User Samples | Developer Weight | User Weight | Effect |
|---|---|---|---|
| 0 | 1.0x | — | Only developer data exists |
| 1-2 | 1.0x | 1.5x | User feedback immediately valued higher |
| 10-24 | 0.95x | 2.5x | Developer data starts fading |
| 50-99 | 0.8x | 3.5x | User strongly dominates |
| 200+ | 0.6x | 5.0x | Developer data is just a baseline floor |

The developer dataset is never deleted — it just gets increasingly outweighed as the user adds their own ratings.

### Logs

- **stdout/stderr:** `%APPDATA%/CasePrepd/logs/main_log_YYYYMMDD_HHMMSS.txt`
- **Structured log:** `%APPDATA%/CasePrepd/logs/caseprepd.log` (rotating)
- **Crash log:** `%APPDATA%/CasePrepd/crash.log` (import failures only)

## Tests

**Important:** Always activate the virtual environment before running tests.
The project uses Python 3.12 in `.venv`, but your system PATH may point to a
different Python (e.g. 3.13). Running tests with the wrong Python causes
misleading failures — packages like `nupunkt` and `tkinterdnd2` will appear
missing even though they're installed in the venv.

```bash
# Activate first (Windows)
.venv\Scripts\activate

# Verify you're using the venv Python
python --version   # Should show 3.12.x

# Quick tests (skip slow integration tests)
python -m pytest tests/ -v -m "not slow"

# All tests
python -m pytest tests/ -v

# Always exclude (hangs on Windows due to GUI/spaCy timeouts)
# tests/test_gui_workflow.py
```

## Architecture

Four output tabs: **Document** (file preview), **Vocabulary** (extracted terms with ML filtering), **Search** (semantic + BM25 retrieval), **Key Excerpts** (representative passages via K-means clustering).

Heavy work (extraction, semantic indexing, vocabulary extraction) runs in a separate subprocess. The GUI communicates with it via `multiprocessing.Queue` and polls every 33ms.

Run `pydeps src -o deps.svg` to visualize the full module dependency graph.
