"""
Prompting Package for LocalScribe - Unified API for Prompt Management.

This is the main entry point for all prompt-related functionality. Import
everything prompting-related from this package:

    from src.core.prompting import (
        # Template management
        PromptTemplateManager,
        SKELETON_FILENAME, README_FILENAME, GENERIC_FALLBACK_TEMPLATE,
        # Focus extraction
        FocusExtractor, AIFocusExtractor,
        # Prompt adapters
        PromptAdapter, MultiDocPromptAdapter,
        # Configuration
        PromptConfig, get_prompt_config, PROMPT_PARAMS_FILE,
    )

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  src.core.prompting (this package) - Unified Prompting API  │
    ├─────────────────────────────────────────────────────────────┤
    │  PromptTemplateManager (loads/validates prompt templates)   │
    │            ↓                                                │
    │  AIFocusExtractor (extracts focus areas from templates)     │
    │            ↓                                                │
    │  MultiDocPromptAdapter (creates stage-specific prompts)     │
    │            ↓                                                │
    │  PromptConfig (loads prompt parameters from config)         │
    └─────────────────────────────────────────────────────────────┘

Components:
- Template Management: PromptTemplateManager handles loading, validating,
  and caching prompt templates from built-in and user directories.
- Focus Extraction: AIFocusExtractor uses Ollama to extract user's focus
  areas from templates to thread through the summarization pipeline.
- Prompt Adapters: MultiDocPromptAdapter creates chunk/document/meta prompts
  that incorporate the user's focus areas at every stage.
- Configuration: PromptConfig loads prompt parameters (word counts, etc.)
  from config/prompt_parameters.json.

Created in Session 33 by consolidating orphan files from src/ root:
- prompt_adapters.py → adapters.py
- prompt_focus_extractor.py → focus_extractor.py
- prompt_template_manager.py → template_manager.py
- prompt_config.py → config.py
"""

# Template management
# Prompt adapters
from src.core.prompting.adapters import (
    MultiDocPromptAdapter,
    PromptAdapter,
)

# Configuration
from src.core.prompting.config import (
    PROMPT_PARAMS_FILE,
    PromptConfig,
    get_prompt_config,
)

# Focus extraction
from src.core.prompting.focus_extractor import (
    AIFocusExtractor,
    FocusExtractor,
)
from src.core.prompting.template_manager import (
    GENERIC_FALLBACK_TEMPLATE,
    README_FILENAME,
    SKELETON_FILENAME,
    USER_README_CONTENT,
    USER_SKELETON_TEMPLATE,
    PromptTemplateManager,
)

__all__ = [
    "GENERIC_FALLBACK_TEMPLATE",
    "PROMPT_PARAMS_FILE",
    "README_FILENAME",
    "SKELETON_FILENAME",
    "USER_README_CONTENT",
    "USER_SKELETON_TEMPLATE",
    "AIFocusExtractor",
    # Focus extraction
    "FocusExtractor",
    "MultiDocPromptAdapter",
    # Prompt adapters
    "PromptAdapter",
    # Configuration
    "PromptConfig",
    # Template management
    "PromptTemplateManager",
    "get_prompt_config",
]
