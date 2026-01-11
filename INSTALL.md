# Enforcement Installation Guide

## Files to Install

| File | Destination | Purpose |
|------|-------------|---------|
| `find_violations.py` | Project root | Import violation checker |
| `PIPELINE.md` | Project root | Architecture reference |
| `pre-commit` | `.git/hooks/pre-commit` | Block bad commits |
| `settings.json` | Replace existing | Add Claude Code hook |
| `CLAUDE_ADDITIONS.md` | Append to CLAUDE.md | Tell Claude the rules |

## Step-by-Step

### 1. Install the violation checker
```powershell
# From project root
# (Already downloaded find_violations.py and PIPELINE.md)
```

### 2. Install git pre-commit hook
```powershell
copy pre-commit .git\hooks\pre-commit
```

Test it:
```powershell
git commit --allow-empty -m "test hook"
# Should see "Checking for import violations..."
```

### 3. Update Claude Code settings
```powershell
# Backup current settings
copy "%USERPROFILE%\.claude\settings.json" "%USERPROFILE%\.claude\settings.json.bak"

# Replace with new settings (or manually add the hook)
copy settings.json "%USERPROFILE%\.claude\settings.json"
```

Or manually add this to the "hooks" section of your existing settings.json:
```json
{
  "type": "command",
  "command": "py find_violations.py --quick 2>nul || echo [violations check skipped]"
}
```

### 4. Update CLAUDE.md
Open your project's CLAUDE.md and append the contents of CLAUDE_ADDITIONS.md.

### 5. Verify everything works

```powershell
# Full scan - see current violations
python find_violations.py

# Quick check - what hooks use
python find_violations.py --quick

# Check specific file
python find_violations.py src/ui/main_window.py
```

## What Happens Now

1. **When Claude Code edits Python:** The hook runs `--quick` mode. If there's a violation, you'll see it immediately.

2. **When you try to commit:** Git pre-commit runs full check. Violations block the commit.

3. **When Claude plans an import:** It checks the rules in CLAUDE.md and asks if unsure.

## Troubleshooting

**Hook doesn't run:**
- Make sure you're in the project directory
- Check that find_violations.py exists in project root

**Too many violations to fix now:**
- Comment out the git hook temporarily: `# python find_violations.py --quick`
- Fix violations in phases (see REFACTORING_CHECKLIST.md)
- Re-enable hook when clean

**False positive (legitimate exception):**
- Add to `ALLOWED_CORE_IN_UI` in find_violations.py
- Document why in PIPELINE.md
