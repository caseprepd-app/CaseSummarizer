#!/usr/bin/env python3
"""
find_violations.py - Detect import violations in CasePrepd

Run from project root:
    python find_violations.py

Or with verbose output:
    python find_violations.py --verbose
"""

import ast
import sys
from collections import defaultdict
from pathlib import Path

# =============================================================================
# RULES: Define what's allowed and forbidden
# =============================================================================

# Stage 5 modules that should NEVER import each other
# Note: qa/vector_store/retrieval form a single Q&A subsystem (see QA_SUBSYSTEM)
PARALLEL_MODULES = {
    "src.core.vocabulary",
    "src.core.summarization",
}

# Q&A Subsystem - these modules work together and CAN import each other
# qa/vector_store/retrieval are ONE logical unit in the pipeline
QA_SUBSYSTEM = {
    "src.core.qa",
    "src.core.vector_store",
    "src.core.retrieval",
}

# UI should NEVER import directly from core (must use services)
UI_MODULE = "src.ui"
CORE_MODULE = "src.core"
SERVICES_MODULE = "src.services"

# No core module should ever import from UI
FORBIDDEN_FOR_CORE = {"src.ui"}

# =============================================================================
# ANALYSIS
# =============================================================================


def get_imports(filepath: Path) -> list[tuple[str, int]]:
    """Extract all imports from a Python file with line numbers."""
    try:
        content = filepath.read_text(encoding="utf-8")
        tree = ast.parse(content)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"  Warning: Could not parse {filepath}: {e}")
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((node.module, node.lineno))

    return imports


def module_from_path(filepath: Path, project_root: Path) -> str:
    """Convert file path to module name."""
    relative = filepath.relative_to(project_root)
    parts = relative.with_suffix("").parts
    return ".".join(parts)


def get_module_prefix(module: str) -> str:
    """Get the top-level package prefix for comparison."""
    parts = module.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])  # e.g., "src.core.vocabulary"
    return module


def check_parallel_violation(source_module: str, imported_module: str) -> bool:
    """Check if this is a forbidden cross-import between parallel Stage 5 modules."""
    source_prefix = get_module_prefix(source_module)
    import_prefix = get_module_prefix(imported_module)

    # Allow imports within the Q&A subsystem (qa/vector_store/retrieval)
    if source_prefix in QA_SUBSYSTEM and import_prefix in QA_SUBSYSTEM:
        return False  # Q&A subsystem modules can import each other

    # Check parallel modules (vocabulary, summarization)
    all_stage5 = PARALLEL_MODULES | QA_SUBSYSTEM
    if source_prefix in all_stage5 and import_prefix in all_stage5:
        if source_prefix != import_prefix:
            # Exception: Q&A subsystem can't import vocabulary/summarization and vice versa
            return True
    return False


def check_ui_core_violation(source_module: str, imported_module: str) -> bool:
    """Check if UI is importing directly from core (should use services)."""
    if source_module.startswith(UI_MODULE):
        if imported_module.startswith(CORE_MODULE):
            # Allow imports from core.config (shared utility)
            if imported_module.startswith("src.core.config"):
                return False
            return True
    return False


def check_core_ui_violation(source_module: str, imported_module: str) -> bool:
    """Check if any core module imports from UI (never allowed)."""
    if source_module.startswith(CORE_MODULE):
        if imported_module.startswith(UI_MODULE):
            return True
    return False


def check_services_ui_violation(source_module: str, imported_module: str) -> bool:
    """Check if services imports from UI (should be the other way around)."""
    if source_module.startswith(SERVICES_MODULE):
        if imported_module.startswith(UI_MODULE):
            return True
    return False


def find_all_violations(project_root: Path, verbose: bool = False) -> dict:
    """Scan all Python files and find import violations."""

    violations = {
        "parallel_crossimport": [],  # Stage 5 modules importing each other
        "ui_imports_core": [],  # UI bypassing services
        "core_imports_ui": [],  # Core depending on UI
        "services_imports_ui": [],  # Services depending on UI
    }

    src_dir = project_root / "src"
    if not src_dir.exists():
        print(f"Error: {src_dir} not found. Run from project root.")
        sys.exit(1)

    py_files = list(src_dir.rglob("*.py"))

    for filepath in py_files:
        source_module = module_from_path(filepath, project_root)
        imports = get_imports(filepath)

        if verbose:
            print(f"Checking: {source_module}")

        for imported_module, lineno in imports:
            # Skip non-project imports
            if not imported_module.startswith("src"):
                continue

            # Check each violation type
            if check_parallel_violation(source_module, imported_module):
                violations["parallel_crossimport"].append(
                    {
                        "file": str(filepath),
                        "line": lineno,
                        "source": source_module,
                        "imports": imported_module,
                        "reason": "Stage 5 parallel modules cannot import each other",
                    }
                )

            if check_ui_core_violation(source_module, imported_module):
                violations["ui_imports_core"].append(
                    {
                        "file": str(filepath),
                        "line": lineno,
                        "source": source_module,
                        "imports": imported_module,
                        "reason": "UI must import from services, not core directly",
                    }
                )

            if check_core_ui_violation(source_module, imported_module):
                violations["core_imports_ui"].append(
                    {
                        "file": str(filepath),
                        "line": lineno,
                        "source": source_module,
                        "imports": imported_module,
                        "reason": "Core modules cannot import from UI",
                    }
                )

            if check_services_ui_violation(source_module, imported_module):
                violations["services_imports_ui"].append(
                    {
                        "file": str(filepath),
                        "line": lineno,
                        "source": source_module,
                        "imports": imported_module,
                        "reason": "Services cannot import from UI",
                    }
                )

    return violations


def print_violations(violations: dict) -> int:
    """Print violations in a readable format. Returns total count."""
    total = 0

    categories = [
        (
            "parallel_crossimport",
            "PARALLEL MODULE CROSS-IMPORTS",
            "These Stage 5 modules should be independent (vocabulary, qa, summarization)",
        ),
        (
            "ui_imports_core",
            "UI IMPORTING CORE DIRECTLY",
            "UI should only import from src/services/, not src/core/",
        ),
        ("core_imports_ui", "CORE IMPORTING UI", "Core modules should never depend on UI"),
        ("services_imports_ui", "SERVICES IMPORTING UI", "Services should never depend on UI"),
    ]

    for key, title, description in categories:
        items = violations[key]
        if items:
            print(f"\n{'=' * 70}")
            print(f"[X] {title} ({len(items)} violations)")
            print(f"   {description}")
            print("=" * 70)

            for v in items:
                print(f"\n  File: {v['file']}:{v['line']}")
                print(f"  Import: {v['imports']}")

            total += len(items)

    return total


def print_summary(violations: dict) -> None:
    """Print a summary of what needs fixing."""
    print("\n" + "=" * 70)
    print("SUMMARY: FILES TO FIX")
    print("=" * 70)

    # Group by file
    files = defaultdict(list)
    for category, items in violations.items():
        for v in items:
            files[v["file"]].append(v)

    if not files:
        print("\n[OK] No violations found! Your imports are clean.")
        return

    # Sort by violation count (worst first)
    sorted_files = sorted(files.items(), key=lambda x: -len(x[1]))

    print(f"\n{len(sorted_files)} files need refactoring:\n")

    for filepath, file_violations in sorted_files:
        print(f"  [{len(file_violations):2d} violations] {filepath}")

    print(f"\nTotal violations: {sum(len(v) for v in files.values())}")
    print("\nStart with the files that have the most violations.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    project_root = Path.cwd()

    print("CasePrepd Import Violation Finder")
    print("=" * 70)
    print(f"Scanning: {project_root / 'src'}")

    violations = find_all_violations(project_root, verbose)
    total = print_violations(violations)
    print_summary(violations)

    # Exit with error code if violations found (useful for CI)
    sys.exit(1 if total > 0 else 0)
