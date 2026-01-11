"""
CasePrepd AI Module
Handles AI model loading and text generation via Ollama.

Architecture:
=============
This module uses Ollama as the primary AI backend for several key reasons:

1. **Cross-Platform Stability**: Works identically on Windows, macOS, and Linux
   without DLL or environment-specific issues.

2. **No External Dependencies**: Ollama is a standalone service; no Python packages
   for the actual model inference (requests library only for API calls).

3. **Commercial Viability**: MIT-licensed, safe for commercial distribution
   with no license conflicts.

4. **Flexibility**: Easy to swap models at runtime, pull new models via UI,
   and support multiple backends if needed in the future.

Deprecated Managers:
- ONNXModelManager: Removed due to Phi-3 token corruption bug (ONNX tokenizer mismatch)
- LlamaCppModelManager: Legacy implementation kept for reference only
"""

# Primary AI Model Manager: Ollama-based
# - REST API integration with local Ollama service
# - Supports any model available on https://ollama.ai/library
from .ollama_model_manager import OllamaModelManager

# DEFAULT: Use Ollama for all AI operations
ModelManager = OllamaModelManager

__all__ = ["ModelManager", "OllamaModelManager"]
