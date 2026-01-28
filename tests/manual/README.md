# Manual Integration Tests

This directory contains **manual integration tests** that are separate from the automated pytest suite.

## Purpose

These tests validate:
- Ollama model integration and connectivity
- Prompt template loading and formatting
- Model generation (streaming and non-streaming)
- End-to-end summarization workflows

## Requirements

- **Ollama** must be running locally (`ollama serve`)
- A model must be available (e.g., `phi-3-mini`)
- Test files may require specific test documents (e.g., `test_simple_case.txt`)

## Usage

Run individual tests from the project root:

```bash
# Quick smoke test
python tests/manual/test_model_quick.py

# Full Ollama workflow test
python tests/manual/test_ollama_workflow.py

# Prompt template tests
python tests/manual/test_prompts.py

# Slider configuration tests
python tests/manual/test_slider_config.py
```

## Test Files

| File | Purpose |
|------|---------|
| `test_debug_mode.py` | Integration test for model pipeline with verbose logging output |
| `test_model_generation.py` | Unit tests for ModelManager text generation methods |
| `test_model_quick.py` | Fast smoke test for basic model connectivity |
| `test_ollama_workflow.py` | Comprehensive Ollama integration test (4 workflows) |
| `test_prompts.py` | PromptTemplateManager loading and validation |
| `test_slider_config.py` | Slider configuration and word count range tests |

## Note

These are **not** part of the automated `pytest` test suite (224 tests in `tests/`).
They require manual execution and external dependencies (Ollama running).
