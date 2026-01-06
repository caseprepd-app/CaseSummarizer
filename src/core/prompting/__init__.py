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
from src.core.prompting.template_manager import (
    PromptTemplateManager,
    SKELETON_FILENAME,
    README_FILENAME,
    USER_SKELETON_TEMPLATE,
    USER_README_CONTENT,
    GENERIC_FALLBACK_TEMPLATE,
)

# Focus extraction
from src.core.prompting.focus_extractor import (
    FocusExtractor,
    AIFocusExtractor,
)

# Prompt adapters
from src.core.prompting.adapters import (
    PromptAdapter,
    MultiDocPromptAdapter,
)

# Configuration
from src.core.prompting.config import (
    PromptConfig,
    get_prompt_config,
    PROMPT_PARAMS_FILE,
)

__all__ = [
    # Template management
    "PromptTemplateManager",
    "SKELETON_FILENAME",
    "README_FILENAME",
    "USER_SKELETON_TEMPLATE",
    "USER_README_CONTENT",
    "GENERIC_FALLBACK_TEMPLATE",
    # Focus extraction
    "FocusExtractor",
    "AIFocusExtractor",
    # Prompt adapters
    "PromptAdapter",
    "MultiDocPromptAdapter",
    # Configuration
    "PromptConfig",
    "get_prompt_config",
    "PROMPT_PARAMS_FILE",
]
