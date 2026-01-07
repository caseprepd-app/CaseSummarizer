# LocalScribe Refactoring Checklist

> **How to use:** Work through this checklist in order. Each phase builds on the previous.
> Check off items as you complete them. Run `python find_violations.py` after each phase.

---

## Phase 0: Setup (30 minutes)

- [ ] Save `PIPELINE.md` to project root
- [ ] Save `find_violations.py` to project root
- [ ] Run initial violation scan:
  ```powershell
  cd "C:\Users\noahc\Dropbox\Not Work\Data Science\CaseSummarizer"
  .venv\Scripts\python.exe find_violations.py > violations_initial.txt
  ```
- [ ] Count initial violations: `_______` (write this down)
- [ ] Commit current state:
  ```powershell
  git add -A
  git commit -m "Before refactoring: baseline"
  ```

---

## Phase 1: Fix Core→UI Violations (1-2 hours)

**Goal:** No core module should import anything from `src/ui/`

### Step 1.1: Find the violations
```powershell
.venv\Scripts\python.exe find_violations.py | findstr "CORE IMPORTING UI" -A 100
```

### Step 1.2: For each violation, choose a fix:

| Pattern | Fix |
|---------|-----|
| Core imports UI constant | Move constant to `src/config.py` |
| Core imports UI utility | Move utility to `src/utils/` |
| Core needs to call UI | Pass callback function as parameter instead |
| Core imports queue message type | Move message types to `src/core/` or `src/services/` |

### Step 1.3: Common fixes you'll likely need

**Move QueueMessage to services or core:**
```python
# If src/core/ imports QueueMessage from src/ui/queue_messages.py
# Move queue_messages.py to src/services/queue_messages.py
# Update all imports
```

**Move theme constants:**
```python
# If src/core/ imports colors from src/ui/theme.py
# Move the colors to src/config.py
# Or create src/core/constants.py
```

### Step 1.4: Verify
```powershell
.venv\Scripts\python.exe find_violations.py | findstr "CORE IMPORTING UI"
# Should return nothing
```

- [ ] All Core→UI violations fixed
- [ ] Tests still pass: `.venv\Scripts\python.exe -m pytest tests/ -x -q`
- [ ] Commit: `git commit -am "Phase 1: Remove core→ui imports"`

---

## Phase 2: Fix UI→Core Violations (2-4 hours)

**Goal:** UI only imports from `src/services/`, never directly from `src/core/`

### Step 2.1: Find the violations
```powershell
.venv\Scripts\python.exe find_violations.py | findstr "UI IMPORTING CORE" -A 100
```

### Step 2.2: Categorize each violation

For each UI file importing from core, decide:

| If UI imports... | Fix by... |
|------------------|-----------|
| `src/core/vocabulary/...` | Use `VocabularyService` instead |
| `src/core/qa/...` | Use `QAService` instead |
| `src/core/extraction/...` | Use `DocumentService` instead |
| `src/core/config/loader` | This is OK (shared utility) - add to allowed list |
| `src/core/ai/...` | Add method to appropriate service |
| A type/dataclass for type hints | Move type to `src/services/types.py` or allow it |

### Step 2.3: Expand services as needed

Your services may need new methods. Pattern:

```python
# In src/services/vocabulary_service.py
from src.core.vocabulary.some_module import some_function

class VocabularyService:
    def new_method_ui_needs(self, ...):
        """Wrap the core function for UI access."""
        return some_function(...)
```

Then in UI:
```python
# Before (violation):
from src.core.vocabulary.some_module import some_function

# After (clean):
from src.services.vocabulary_service import VocabularyService
service = VocabularyService()
service.new_method_ui_needs(...)
```

### Step 2.4: Handle workers specially

Workers (`src/ui/workers.py`) do heavy lifting and may have many core imports.
Options:
1. **Move workers to services layer** - they're really orchestration, not UI
2. **Accept workers as exception** - add `src/ui/workers.py` to allowed list
3. **Create service methods** - workers call services, services call core

Recommended: Option 1 or 2. Workers are borderline.

### Step 2.5: Verify
```powershell
.venv\Scripts\python.exe find_violations.py | findstr "UI IMPORTING CORE"
# Should return nothing (or only allowed exceptions)
```

- [ ] All UI→Core violations fixed (or documented exceptions)
- [ ] Tests still pass
- [ ] Commit: `git commit -am "Phase 2: Route UI through services"`

---

## Phase 3: Fix Parallel Module Cross-Imports (2-4 hours)

**Goal:** vocabulary, qa, and summarization modules are independent

### Step 3.1: Find the violations
```powershell
.venv\Scripts\python.exe find_violations.py | findstr "PARALLEL MODULE" -A 100
```

### Step 3.2: For each cross-import, find the shared need

Common patterns:

| If vocabulary imports from qa... | The shared thing is... | Move it to... |
|----------------------------------|------------------------|---------------|
| Chunking logic | Text chunking | `src/core/chunking/` (already exists) |
| Embedding model | Embeddings | `src/core/ai/embeddings.py` |
| Text utilities | String helpers | `src/utils/text_utils.py` |
| Ollama calls | LLM interface | `src/core/ai/` |

### Step 3.3: Extract shared code

When two Stage 5 modules need the same thing:

```python
# BEFORE: vocabulary imports from qa
# src/core/vocabulary/extractor.py
from src.core.qa.some_module import shared_function  # VIOLATION

# AFTER: both import from shared location
# 1. Create src/core/shared/text_processing.py (or appropriate location)
# 2. Move shared_function there
# 3. Both vocabulary and qa import from shared location
```

### Step 3.4: Verify
```powershell
.venv\Scripts\python.exe find_violations.py | findstr "PARALLEL MODULE"
# Should return nothing
```

- [ ] All parallel cross-imports fixed
- [ ] Tests still pass
- [ ] Commit: `git commit -am "Phase 3: Isolate parallel modules"`

---

## Phase 4: Verify & Document (1 hour)

### Step 4.1: Final violation check
```powershell
.venv\Scripts\python.exe find_violations.py
```

Expected output:
```
✓ No violations found! Your imports are clean.
```

### Step 4.2: Generate clean dependency graph
```powershell
pydeps src --max-bacon 2 -o deps_clean.svg
```

Compare to original - should be much simpler.

### Step 4.3: Update documentation

- [ ] Update README.md with new structure
- [ ] Delete old ARCHITECTURE.md (replaced by PIPELINE.md + pydeps)
- [ ] Commit: `git commit -am "Phase 4: Clean architecture verified"`

---

## Phase 5: Ongoing Enforcement

### Add to your workflow:

**Before each commit:**
```powershell
.venv\Scripts\python.exe find_violations.py
```

**Add to CI (if you have it):**
```yaml
- name: Check import violations
  run: python find_violations.py
```

**In CLAUDE.md, add:**
```markdown
## Import Rules
Before adding any import, check PIPELINE.md.
Run `python find_violations.py` before committing.
```

---

## Quick Reference: What Goes Where

| I need to... | Put it in... |
|--------------|--------------|
| Load a PDF/DOCX | `src/core/extraction/` |
| Clean text encoding | `src/core/sanitization/` |
| Remove headers/footers | `src/core/preprocessing/` |
| Split text into chunks | `src/core/chunking/` |
| Extract names/terms | `src/core/vocabulary/` |
| Build search index | `src/core/vector_store/` |
| Answer questions | `src/core/qa/` |
| Generate summaries | `src/core/summarization/` |
| Call Ollama | `src/core/ai/` |
| Format for export | `src/core/export/` |
| Expose to UI | `src/services/` |
| Display to user | `src/ui/` |
| Store a constant | `src/config.py` |
| Utility function | `src/utils/` |

---

## Estimated Total Time

| Phase | Time |
|-------|------|
| Phase 0: Setup | 30 min |
| Phase 1: Core→UI | 1-2 hours |
| Phase 2: UI→Core | 2-4 hours |
| Phase 3: Parallel modules | 2-4 hours |
| Phase 4: Verify | 1 hour |
| **Total** | **6-12 hours** |

Work in focused sessions. Commit after each phase. Take breaks.

---

## If You Get Stuck

Ask Claude Code:
```
I'm refactoring LocalScribe. I have this import violation:

[paste the violation]

The file needs [what it's trying to do].
What's the cleanest way to fix this following PIPELINE.md?
```
