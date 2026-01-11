# AI Coding Partner Instructions

You are helping a non-developer who designs through conversation. They read logs and diagrams, not implementation code.

## Session Start
1. **Activate the virtual environment first:**
   ```
   .venv\Scripts\activate
   ```
2. Read PROJECT_OVERVIEW.md and README.md
3. Give a 2-3 sentence status summary
4. Ask for time budget
5. Proceed

## Before Any Code Change
1. Describe approach in plain English
2. If adding a new pip dependency, mention it explicitly
3. Wait for approval

## Documentation Rules

### README.md
- Setup instructions (install, run, test)
- Brief description of what the project is

### PROJECT_OVERVIEW.md
- Business goals and constraints
- Only update when goals/constraints change
- Ask before modifying

### RESEARCH_LOG.md
**DO:**
- Log library/tool comparisons with decision rationale
- Log algorithm research with sources
- Check here BEFORE doing web searches

**DON'T:**
- Log implementation decisions
- Log bug fixes
- Log session summaries
- Add entries without external research

### TODO Comments
- Use `# TODO:` comments in code for incomplete work
- Delete them when the work is done
- Run `grep -r "TODO" src/` to see all outstanding items

## Code Style
- Max 200 lines per file (prefer 100-150)
- Max 30 lines per function (prefer 15-20)
- Every function has a docstring showing inputs/outputs
- No silent failures—always log or raise
- Validate inputs at function start (fail fast)

## Runtime Logging
Every significant function logs what it's doing:
```
[1] Starting extraction
[2] Processing file: case_001.pdf
  [2.1] Extracted 847 lines
```

## Import Rules (ENFORCED)

**Before adding ANY import from `src.*`, check these rules:**

| If you're in... | You CAN import from... | You CANNOT import from... |
|-----------------|------------------------|---------------------------|
| `src/ui/` | `src/services/`, `src/config.py` | `src/core/*` (except `src/core/config/`) |
| `src/core/vocabulary/` | `src/core/extraction/`, `src/core/sanitization/`, `src/core/preprocessing/`, `src/core/chunking/`, `src/core/ai/`, `src/config.py` | `src/core/qa/`, `src/core/summarization/`, `src/ui/` |
| `src/core/qa/` | Same as vocabulary | `src/core/vocabulary/`, `src/core/summarization/`, `src/ui/` |
| `src/core/summarization/` | Same as vocabulary | `src/core/vocabulary/`, `src/core/qa/`, `src/ui/` |
| `src/core/*` (any) | Earlier pipeline stages, `src/config.py`, `src/core/config/` | `src/ui/` |
| `src/services/` | `src/core/*`, `src/config.py` | `src/ui/` |

**If you need something from a forbidden module:**
1. STOP - don't add the import
2. Ask: "I need X from Y, but that violates PIPELINE.md. How should I restructure?"

**After editing any Python file:**
```
python find_violations.py --quick
```
If it fails, fix the violation before proceeding. Do not leave violations for later.
See `PIPELINE.md` for the full architecture diagram.

## Error Messages
Plain English, reference step numbers, suggest fixes:
```python
raise ExtractionError(
    f"Step 2.1 failed: Could not extract text from '{filename}'\n"
    f"Possible causes:\n"
    f"- File may be password protected\n"
    f"- PDF may be image-only (needs OCR)"
)
```

## When Stuck
1. Stop coding
2. Summarize what was tried
3. Ask: Is this a PROJECT problem (goals wrong) or IMPLEMENTATION problem (approach wrong)?
4. Propose fresh approach
5. Get approval
