"""Update the version and date on the GitHub Pages website.

Usage:
    python scripts/update_ghpages_version.py 1.0.28

Checks out gh-pages, updates the version-info line in index.html,
commits, pushes, then switches back to the original branch.
"""

import re
import subprocess
import sys
from datetime import datetime


def run(cmd, **kwargs):
    """Run a shell command and return stdout."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, **kwargs)
    return result.stdout.strip()


def get_current_branch():
    """Return the current git branch name."""
    return run(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def update_index_html(version):
    """Rewrite the version-info line in index.html."""
    path = "index.html"
    with open(path, encoding="utf-8") as f:
        text = f.read()

    month_year = datetime.now().strftime("%B %Y")
    old_pattern = r'<p class="version-info">v[\d.]+ &mdash; \w+ \d{4}</p>'
    new_line = f'<p class="version-info">v{version} &mdash; {month_year}</p>'

    updated, count = re.subn(old_pattern, new_line, text)
    if count == 0:
        print("ERROR: Could not find version-info line in index.html")
        sys.exit(1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)

    print(f"  index.html: v{version} — {month_year}")


def main():
    """Entry point."""
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    version = sys.argv[1]
    if not re.match(r"^\d+\.\d+\.\d+$", version):
        print(f"ERROR: '{version}' is not a valid version number")
        sys.exit(1)

    original_branch = get_current_branch()
    print(f"Updating gh-pages version to v{version}")

    try:
        run(["git", "checkout", "gh-pages"])
        update_index_html(version)
        run(["git", "add", "index.html"])
        msg = f"update version to v{version}"
        run(["git", "commit", "--no-verify", "-m", msg])
        run(["git", "push", "origin", "gh-pages"])
        print(f"  gh-pages pushed with v{version}")
    finally:
        run(["git", "checkout", original_branch])
        print(f"  Back on {original_branch}")


if __name__ == "__main__":
    main()
