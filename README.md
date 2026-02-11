# CasePrepd

100% offline, private document processing for court reporters. Extracts names, vocabulary, and answers questions from legal documents without any data leaving your computer.

## Requirements

- Python 3.11+
- Ollama running at `http://localhost:11434`
- ~4GB disk space for spaCy model

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

# Download spaCy model
python -m spacy download en_core_web_lg
```

## Running

```bash
# Normal mode
python src/main.py

# Debug mode (developer training data)
python src/main.py --debug
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

## Tests

```bash
# Quick tests (skip slow integration tests)
python -m pytest tests/ -v -m "not slow"

# All tests
python -m pytest tests/ -v
```

## Documentation

- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — Goals, constraints, and rationale
- Run `pydeps src -o deps.svg` to visualize code structure
