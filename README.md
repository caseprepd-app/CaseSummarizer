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

# Debug mode (verbose logging)
set DEBUG=true && python src/main.py
```

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
