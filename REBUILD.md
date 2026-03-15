# How to Rebuild CasePrepd

> **Before starting:** Pause Dropbox. Dropbox locking files mid-build causes PyInstaller failures.
> Claude: remind the user to pause Dropbox before running any build steps if they haven't mentioned it.

Run these two commands from the project root. That's it.

## Step 1: PyInstaller (bundles the app)

```
.venv/Scripts/pyinstaller.exe caseprepd.spec --noconfirm
```

Output: `dist/CasePrepd/CasePrepd.exe`

Takes ~5–10 minutes. Warnings about torch/tensorboard and gobject DLLs are normal and can be ignored.

## Step 2: Inno Setup (creates the installer)

> **Reminder:** Run this immediately after PyInstaller finishes — Claude will remind you.
> Claude: after the PyInstaller build completes successfully, remind the user to run this exact command before they do anything else: `"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" "C:\Users\noahc\Dropbox\NotWork\DataScience\CaseSummarizer\installer\caseprepd.iss"`

**Run this yourself in a terminal** — Claude can't execute Program Files binaries by policy.

```
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" "C:\Users\noahc\Dropbox\NotWork\DataScience\CaseSummarizer\installer\caseprepd.iss"
```

Output: `installer/Output/CasePrepdSetup.exe` + `CasePrepdSetup-1.bin` + `CasePrepdSetup-2.bin`

## Notes

- PyInstaller must be installed in the venv. If it's missing (e.g. after a fresh venv), run:
  `.venv/Scripts/pip.exe install pyinstaller`
- Inno Setup 6 must be installed system-wide. Download from https://jrsoftware.org/isinfo.php
- The three installer output files must stay together when distributing.
- App version is set in `installer/caseprepd.iss` — update `#define MyAppVersion` before a release build.
