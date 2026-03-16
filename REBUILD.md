# How to Rebuild CasePrepd

> **Before starting:** Pause Dropbox. Dropbox locking files mid-build causes PyInstaller failures.
> Claude: remind the user to pause Dropbox before running any build steps if they haven't mentioned it.

## Release Workflow

### 1. Bump the version

```
python scripts/bump_version.py patch   # bug fix: 1.0.15 -> 1.0.16
python scripts/bump_version.py minor   # new feature: 1.0.15 -> 1.1.0
python scripts/bump_version.py major   # breaking change: 1.0.15 -> 2.0.0
```

This updates `src/__init__.py` and `installer/caseprepd.iss` automatically.

### 2. Update CHANGELOG.md

Add what changed under the `[Unreleased]` section before bumping, or edit the
newly created version section after bumping.

### 3. Commit and tag

```
git add -A && git commit -m "release: vX.Y.Z"
git tag vX.Y.Z
```

### 4. PyInstaller (bundles the app)

```
.venv/Scripts/pyinstaller.exe caseprepd.spec --noconfirm
```

Output: `dist/CasePrepd/CasePrepd.exe`

Takes ~5-10 minutes. Warnings about torch/tensorboard and gobject DLLs are normal.

### 5. Inno Setup (creates the installer)

> Claude: after PyInstaller completes, remind the user to run this command.

```
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\caseprepd.iss
```

Output: `installer/Output/CasePrepdSetup.exe` + `.bin` files

### 6. Create GitHub Release

```
git push && git push --tags
```

Then create a release on GitHub attaching `installer/Output/CasePrepdSetup.exe`
and any `.bin` files from `installer/Output/`.

## Notes

- PyInstaller must be in the venv. If missing: `.venv/Scripts/pip.exe install pyinstaller`
- Inno Setup 6 must be installed system-wide: https://jrsoftware.org/isinfo.php
- All installer output files must stay together when distributing
- Version is managed by `scripts/bump_version.py` — never edit version strings manually
