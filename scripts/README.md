# Development Scripts

Utility scripts for setting up, diagnosing, and building CasePrepd.

Run all scripts from the project root with the venv activated.

## Model Download Scripts (run before building installer)

| Script | Purpose |
|--------|---------|
| `download_models.py` | Download all bundled models in one shot (spaCy, NLTK, embedding, reranker) — start here |
| `download_embedding_model.py` | Download nomic-embed-text-v1.5 embedding model to `models/` |
| `download_reranker_model.py` | Download gte-reranker-modernbert-base cross-encoder to `models/` |

## Diagnostic Scripts

| Script | Purpose |
|--------|---------|
| `check_spacy.py` | Verify spaCy installation and model (en_core_web_lg) availability |
| `diagnose_ml.py` | Analyze feedback data and ML model feature importances for the vocabulary preference learner |
| `diagnose_semantic_metadata.py` | Debug Q&A metadata flow (source_summary, citation fields) |

## Dev Utilities

| Script | Purpose |
|--------|---------|
| `generate_default_feedback.py` | Build `config/default_feedback.csv` — the developer baseline training data for vocabulary ML |
| `run_debug.py` | Launch the app in debug mode (routes feedback to developer CSV, shows `[DEBUG]` in title) |

## Dead Scripts (features removed)

These scripts remain in the repo but their target features have been removed. Do not run them.

| Script | Was for |
|--------|---------|
| `download_coref_model.py` | LingMess coreference resolution (removed Mar 2026) |
| `download_hallucination_model.py` | TinyLettuce hallucination detection (removed Mar 2026) |
| `download_onnx_models.py` | ONNX hallucination detection models (removed Mar 2026) |
