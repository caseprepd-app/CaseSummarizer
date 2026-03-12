# Migration Plan: Directory Rename + Python 3.12 + Dependency Cleanup

**Created:** 2026-03-12
**Status:** NOT STARTED — do these steps in order

## Overview

Three changes bundled together since all require a fresh venv:
1. Rename directory to remove spaces (fixes Claude Code grep approval prompts)
2. Upgrade Python from 3.11 to 3.12 (~10-15% perf improvement)
3. Remove 7 dead dependencies left over from Ollama removal

## Pre-Flight Checklist

- [ ] All changes committed and pushed (`git status` clean)
- [ ] Note: git and GitHub are unaffected by the rename

---

## Step 1: Rename Directories

Close everything using the project first (Claude Code, terminals, VS Code, etc).

Rename these directories (no spaces):
```
Dropbox\Not Work\Data Science\CaseSummarizer
→
Dropbox\NotWork\DataScience\CaseSummarizer
```

You may need to rename in two steps:
1. `Not Work` → `NotWork`
2. `Data Science` → `DataScience`

**Rollback:** Rename back to original paths. Old .venv will work again.

---

## Step 2: Install Python 3.12

Download Python 3.12 from python.org if not already installed.
During install: check "Add to PATH" or note the install location.

Verify:
```
py -3.12 --version
```

---

## Step 3: Create Fresh venv

```bash
cd Dropbox\NotWork\DataScience\CaseSummarizer
py -3.12 -m venv .venv
.venv\Scripts\activate
```

Verify:
```bash
python --version
# Should say Python 3.12.x
```

---

## Step 4: Clean Up requirements.txt Before Installing

**Remove these 7 dead runtime dependencies:**
- `langchain-experimental` — no imports in active code
- `einops` — no direct usage
- `requests` — only transitive (installed automatically by other packages)
- `pandas` — only in deprecated code
- `scispacy` — only in deprecated code
- `gliner` — only in deprecated algorithm
- `keybert` — only in deprecated algorithm

**Move these to a new `requirements-dev.txt`:**
- `pytest`
- `ruff`
- `black`

**Consider removing (optional, off by default):**
- `fastcoref` — unmaintained since May 2023, coreference is optional and off by default

---

## Step 5: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for testing/linting

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data (if not bundled in models/)
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('words'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## Step 6: Verify

```bash
# Quick import check
python -c "import src.config; print('Config OK')"

# Run violation checker
python find_violations.py --quick

# Run tests (exclude GUI workflow tests)
python -m pytest tests/ --ignore=tests/test_gui_workflow.py -x -q

# Launch the app
start .venv/Scripts/pythonw src/main.py --debug
```

---

## Step 7: Update Claude Code Memory

After opening Claude Code in the new directory, tell it:
> "Read MIGRATION_PLAN.md — we just completed the directory rename and Python upgrade."

Claude Code will have a fresh memory folder (keyed to the new path). It should:
- Re-read CLAUDE.md and this migration plan
- The old memory was at:
  `~/.claude/projects/C--Users-noahc-Dropbox-Not-Work-Data-Science-CaseSummarizer/memory/`
- Copy relevant memories to the new project key if needed

**Key facts to preserve:**
- venv is now Python 3.12 (not 3.11), system Python is 3.13
- 7 dependencies were removed (see Step 4)
- Everything else about the project is unchanged

---

## Step 8: Fix Bundled Paths (Separate Task)

After migration is stable, fix the 15 files that use `Path(__file__).parent...` chains
instead of the centralized `BUNDLED_CONFIG_DIR` / `BUNDLED_BASE_DIR` from `src/config.py`.
This is a separate task — don't mix it with the migration.

See the pathing audit results for the full list of files.

---

## Rollback

If anything goes wrong:
1. Rename directories back to original (with spaces)
2. Old `.venv` (Python 3.11) will work again immediately
3. `requirements.txt` is in git — revert with `git checkout requirements.txt`

---

## Dependency Compatibility Notes (Python 3.12)

| Package | 3.12 Support |
|---------|-------------|
| spacy | Yes (since v3.7) |
| faiss-cpu | Yes |
| sentence-transformers | Yes |
| torch (PyTorch) | Yes |
| customtkinter | Yes (needs setuptools installed) |
| nupunkt | Yes |
| lettucedetect | Yes |
| tkinterdnd2 | Likely (pure Python wrapper) |
| fastcoref | RISK — unmaintained, no 3.12 wheels (but optional/off) |
| scispacy | Removing — only in deprecated code |
