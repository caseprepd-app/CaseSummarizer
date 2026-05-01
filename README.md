# CasePrepd

**100% offline, private document processing for court reporters.** Extracts names, vocabulary, key excerpts, and supports semantic search across legal documents without any data leaving your computer.

**[Download the latest release](https://github.com/caseprepd-app/CaseSummarizer/releases/latest)** | [Website](https://caseprepd-app.github.io/CaseSummarizer)

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

## Headless / CLI mode

Run the full pipeline from the command line without launching the GUI. Useful for scripting, batch jobs, or piping outputs into other tools.

```bash
# All three outputs (vocabulary, key excerpts, combined) for every document in docs/
python -m src.cli --input docs/ --output out/

# Specific files, vocabulary only
python -m src.cli --input case1.pdf case2.pdf --output out/ --only vocab

# Run a search and emit machine-readable JSON
python -m src.cli --input docs/ --output out/ --query "knee injury" --format json

# Both human (.docx/.txt) and machine (.json) outputs
python -m src.cli --input docs/ --output out/ --format both
```

**Arguments:**

| Flag | Description |
|---|---|
| `--input` | One or more files, or a directory (recursively scanned for supported types). Required. |
| `--output` | Directory where outputs are written. Created if missing. Required. |
| `--only` | Repeatable. Limit outputs to `vocab`, `excerpts`, and/or `combined`. Default: all three. |
| `--query` | Run a single semantic search with this phrase. Default: no search. |
| `--format` | `human` (default), `json`, or `both`. |
| `--verbose` / `-v` | Enable debug logging. |

**Supported input types:** `.pdf`, `.docx`, `.txt`, `.rtf`, `.png`, `.jpg`, `.jpeg`.

**Output files:**

| File | When | Format |
|---|---|---|
| `vocabulary.docx` | `--only vocab` (or default) | human |
| `key_excerpts.txt` | `--only excerpts` (or default) | human |
| `combined.docx` | `--only combined` (or default) | human |
| `vocabulary.json` | `--format json` or `both` | machine |
| `excerpts.json` | `--format json` or `both` | machine |
| `search.json` | `--format json`/`both` and `--query` given | machine |

The CLI runs the same pipeline as the GUI (extraction → preprocessing → vocabulary → semantic indexing → key-excerpt extraction) entirely in-process — no subprocess, no Tk. First run is slow because embedding models load from disk; subsequent runs reuse the cached models.

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
