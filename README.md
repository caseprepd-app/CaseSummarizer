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

## For Claude Code Sessions

The virtual environment is at `.venv`. Claude Code's bash tool runs in a Linux shell, so use PowerShell to run commands on Windows:

```bash
# Run the app (normal mode)
powershell -Command "& '.venv\Scripts\python.exe' src/main.py"

# Run the app with DEBUG=true
powershell -Command "$env:DEBUG='true'; & '.venv\Scripts\python.exe' src/main.py"

# Run any Python command
powershell -Command "& '.venv\Scripts\python.exe' -c \"print('hello')\""

# Run tests
powershell -Command "& '.venv\Scripts\python.exe' -m pytest tests/ -v"

# Check import violations
powershell -Command "& '.venv\Scripts\python.exe' find_violations.py --quick"
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
