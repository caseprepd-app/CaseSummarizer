# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for CasePrepd v1.0.
Bundles the full Python environment, app source, config, data, and models
into dist/CasePrepd/ as a standalone Windows application.
"""

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
)

block_cipher = None

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(".")

# ── Data bundles ───────────────────────────────────────────────────────
# Format: (source_glob_or_dir, dest_dir_in_bundle)
added_data = [
    (os.path.join("config", "*"), "config"),
    (os.path.join("config", "extraction_prompts", "*"), os.path.join("config", "extraction_prompts")),
    (os.path.join("config", "prompts", "*"), os.path.join("config", "prompts")),
    (os.path.join("assets", "icon.ico"), "assets"),
]

# Data subdirectories — only include if they contain files
import glob
for subdir in ["frequency", "keywords", "names"]:
    pattern = os.path.join("data", subdir, "*")
    if glob.glob(pattern):
        added_data.append((pattern, os.path.join("data", subdir)))

# Models — include each subdirectory, but skip .hf_cache
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
for entry in os.listdir(MODELS_DIR):
    full_path = os.path.join(MODELS_DIR, entry)
    if entry == ".hf_cache":
        continue
    if os.path.isdir(full_path):
        added_data.append((full_path, os.path.join("models", entry)))
    else:
        added_data.append((full_path, "models"))

# ── Hidden imports ─────────────────────────────────────────────────────
# Modules loaded lazily at runtime that PyInstaller can't detect
hidden_imports = [
    # Vocabulary algorithms (lazy-loaded via create_default_algorithms)
    "src.core.vocabulary.algorithms.ner_algorithm",
    "src.core.vocabulary.algorithms.rake_algorithm",
    "src.core.vocabulary.algorithms.textrank_algorithm",
    "src.core.vocabulary.algorithms.bm25_algorithm",
    "src.core.vocabulary.algorithms.gliner_algorithm",
    "src.core.vocabulary.algorithms.scispacy_algorithm",
    # Drag-and-drop support
    "tkinterdnd2",
    # Tiktoken encoding registry
    "tiktoken_ext.openai_public",
    "tiktoken_ext",
    # Scikit-learn internals often missed
    "sklearn.utils._typedefs",
    "sklearn.utils._cython_blas",
    "sklearn.neighbors._typedefs",
    "sklearn.neighbors._partition_nodes",
    "sklearn.tree._utils",
    # ONNX Runtime
    "onnxruntime",
]

# ── Collect submodules for large packages ──────────────────────────────
collected_submodules = []
packages_to_collect = [
    "spacy",
    "thinc",
    "sentence_transformers",
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_text_splitters",
    "customtkinter",
    "onnxruntime",
    "transformers",
    "huggingface_hub",
    "tokenizers",
    "safetensors",
    "gliner",
    "torch",
    "cymem",
    "murmurhash",
    "preshed",
    "blis",
    "srsly",
    "catalogue",
    "confection",
    "pydantic",
]

for pkg in packages_to_collect:
    try:
        collected_submodules.extend(collect_submodules(pkg))
    except Exception:
        print(f"Warning: Could not collect submodules for {pkg}")

hidden_imports.extend(collected_submodules)

# ── Collect data files for packages that bundle data ───────────────────
pkg_data = []
data_packages = [
    "spacy",
    "thinc",
    "customtkinter",
    "sentence_transformers",
    "transformers",
    "langchain",
    "langchain_core",
    "langchain_community",
    "certifi",
    "gliner",
    "nupunkt",
    "spellchecker",      # Word frequency dictionaries for gibberish filter
    "docx",              # default.docx template for Word export
    "lettucedetect",     # Prompt templates for hallucination verifier
    "tkinterdnd2",       # TCL scripts + native DLLs for drag-and-drop
    "fpdf",              # sRGB ICC color profile for PDF export
]

for pkg in data_packages:
    try:
        pkg_data.extend(collect_data_files(pkg))
    except Exception:
        print(f"Warning: Could not collect data files for {pkg}")

added_data.extend(pkg_data)

# ── Collect dynamic libraries ──────────────────────────────────────────
added_binaries = []
binary_packages = ["onnxruntime", "torch", "tokenizers", "thinc", "sklearn"]

for pkg in binary_packages:
    try:
        added_binaries.extend(collect_dynamic_libs(pkg))
    except Exception:
        print(f"Warning: Could not collect dynamic libs for {pkg}")

# ── Excludes (dev-only packages) ───────────────────────────────────────
excludes = [
    "matplotlib",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "ruff",
    "black",
    "mypy",
    "pyflakes",
    "pylint",
    "sphinx",
    "tkinter.test",
]

# ── Prevent PyInstaller's NLTK hook from bundling system-wide data ─────
# PyInstaller has a built-in hook (hook-nltk.py) that calls nltk.data.path
# and bundles EVERYTHING it finds (often 3+ GB of corpora from %APPDATA%).
# We only need words, wordnet, omw-1.4 — already in models/nltk_data/.
# Clearing nltk.data.path before Analysis prevents the hook from finding
# the system-wide directory.
import nltk
nltk.data.path.clear()

# ── Analysis ───────────────────────────────────────────────────────────
a = Analysis(
    [os.path.join("src", "main.py")],
    pathex=[PROJECT_ROOT],
    binaries=added_binaries,
    datas=added_data,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CasePrepd",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=os.path.join("assets", "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="CasePrepd",
)
