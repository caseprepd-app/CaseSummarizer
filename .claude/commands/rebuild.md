# CasePrepd Release Build

Execute the full release cycle for CasePrepd. Arguments: $ARGUMENTS

## Pre-flight
- Verify we're on the `main` branch with a clean working tree
- Remind the user: **Pause Dropbox before building** (PyInstaller writes thousands of temp files)
- Activate venv: `.venv\Scripts\activate`

## Step 1: Version Bump
- If $ARGUMENTS includes "patch", "minor", or "major", use that. Otherwise ask the user.
- Run: `.venv/Scripts/python scripts/bump_version.py <patch|minor|major>`
- Confirm the new version number with the user.

## Step 2: Changelog
- Ask the user what changed, or offer to auto-populate from `git log` since the last tag.
- Update CHANGELOG.md under the new version's `## [x.y.z]` heading.
- Use Keep a Changelog format: ### Added, ### Fixed, ### Changed, ### Removed, ### Infrastructure

## Step 3: Commit and Tag
- Stage version files + CHANGELOG.md
- Commit: `release: vX.Y.Z — <short summary>`
- Tag: `git tag vX.Y.Z`

## Step 4: PyInstaller Build
- Run: `.venv/Scripts/pyinstaller.exe caseprepd.spec --noconfirm`
- Expected output: `dist/CasePrepd/` directory with `CasePrepd.exe`
- If it fails, diagnose and fix. Common issues: missing data packages in spec file.

## Step 5: Inno Setup Installer
- Run ISCC directly — do NOT ask the user to do this manually.
- Must run from inside the installer directory: `cd installer && "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" caseprepd.iss`
- Running ISCC from the project root with a relative path fails ("file not found") because ISCC resolves paths relative to its own CWD, not the .iss file location.
- Expected output: `installer/Output/CasePrepdSetup.exe`
- Verify the installer was created and report its file size.
- If ISCC.exe is not found or the command fails, THEN ask the user to run it manually as a fallback.

## Step 6: Push and Create GitHub Release
- Push commits and tags: `git push && git push --tags`
- Create release: `gh release create vX.Y.Z installer/Output/CasePrepd_Setup.exe README.md --title "CasePrepd vX.Y.Z" --notes-file -` (pipe changelog section)
- Verify the release was created: `gh release view vX.Y.Z`

## Error Handling
- If any step fails, stop and diagnose. Do NOT proceed to later steps.
- If PyInstaller fails, the version bump commit is still valid — don't revert it.
- If `gh release create` fails with auth issues, walk the user through `gh auth login`.
