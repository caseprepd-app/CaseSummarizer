"""Bump the CasePrepd version number in all locations.

Usage:
    python scripts/bump_version.py patch   # 1.0.15 -> 1.0.16
    python scripts/bump_version.py minor   # 1.0.15 -> 1.1.0
    python scripts/bump_version.py major   # 1.0.15 -> 2.0.0
    python scripts/bump_version.py 1.2.3   # set explicit version

Updates:
    - src/__init__.py  (__version__)
    - installer/caseprepd.iss  (#define MyAppVersion)
    - CHANGELOG.md  (adds new Unreleased section if needed)
"""

import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INIT_FILE = ROOT / "src" / "__init__.py"
ISS_FILE = ROOT / "installer" / "caseprepd.iss"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"


def get_current_version():
    """Read current version from src/__init__.py."""
    text = INIT_FILE.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        print("ERROR: Could not find __version__ in src/__init__.py")
        sys.exit(1)
    return match.group(1)


def bump(current, part):
    """Compute the next version string."""
    parts = current.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        print(f"ERROR: Current version '{current}' is not valid semver")
        sys.exit(1)

    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "major":
        return f"{major + 1}.0.0"

    print(f"ERROR: Unknown bump type '{part}'")
    sys.exit(1)


def update_init(new_version):
    """Update __version__ in src/__init__.py."""
    text = INIT_FILE.read_text(encoding="utf-8")
    updated = re.sub(
        r'__version__\s*=\s*"[^"]+"',
        f'__version__ = "{new_version}"',
        text,
    )
    INIT_FILE.write_text(updated, encoding="utf-8")


def update_iss(new_version):
    """Update #define MyAppVersion in the Inno Setup script."""
    text = ISS_FILE.read_text(encoding="utf-8")
    updated = re.sub(
        r'#define MyAppVersion ".*?"',
        f'#define MyAppVersion "{new_version}"',
        text,
    )
    ISS_FILE.write_text(updated, encoding="utf-8")


def update_changelog(new_version):
    """Add a dated release header to CHANGELOG.md."""
    if not CHANGELOG_FILE.exists():
        print("  CHANGELOG.md not found, skipping")
        return

    text = CHANGELOG_FILE.read_text(encoding="utf-8")
    today = datetime.now().strftime("%Y-%m-%d")

    # Replace [Unreleased] with the new version + date
    if "[Unreleased]" in text:
        new_header = f"## [{new_version}] - {today}"
        unreleased_section = f"## [Unreleased]\n\n{new_header}"
        text = text.replace("## [Unreleased]", unreleased_section)
        CHANGELOG_FILE.write_text(text, encoding="utf-8")
        print(f"  CHANGELOG.md: [Unreleased] -> [{new_version}] - {today}")
    else:
        print("  CHANGELOG.md: No [Unreleased] section found, skipping")


def main():
    """Entry point."""
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    arg = sys.argv[1]
    current = get_current_version()

    # Explicit version or bump keyword?
    if re.match(r"^\d+\.\d+\.\d+$", arg):
        new_version = arg
    elif arg in ("patch", "minor", "major"):
        new_version = bump(current, arg)
    else:
        print(f"ERROR: '{arg}' is not patch/minor/major or a version number")
        sys.exit(1)

    print(f"Bumping version: {current} -> {new_version}")
    print()

    update_init(new_version)
    print(f"  src/__init__.py: {new_version}")

    update_iss(new_version)
    print(f"  installer/caseprepd.iss: {new_version}")

    update_changelog(new_version)

    print()
    print(f"Version bumped to {new_version}.")
    print("Next steps:")
    print("  1. Update CHANGELOG.md with what changed")
    print(f"  2. Commit: git add -A && git commit -m 'release: v{new_version}'")
    print(f"  3. Tag:    git tag v{new_version}")
    print("  4. Build:  see REBUILD.md")


if __name__ == "__main__":
    main()
