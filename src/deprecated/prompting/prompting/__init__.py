"""
Prompting Package for CasePrepd - Unified API for Prompt Management.

This is the main entry point for all prompt-related functionality. Import
everything prompting-related from this package:

    from src.core.prompting import (
        # Template management
        PromptTemplateManager,
        SKELETON_FILENAME, README_FILENAME, GENERIC_FALLBACK_TEMPLATE,
        # Focus extraction
        FocusExtractor, AIFocusExtractor,
        # Prompt adapters
        StagePromptBuilder, MultiDocStagePromptBuilder,
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
    │  MultiDocStagePromptBuilder (creates stage-specific prompts)     │
    │            ↓                                                │
    │  PromptConfig (loads prompt parameters from config)         │
    └─────────────────────────────────────────────────────────────┘

Components:
- Template Management: PromptTemplateManager handles loading, validating,
  and caching prompt templates from built-in and user directories.
- Focus Extraction: AIFocusExtractor uses Ollama to extract user's focus
  areas from templates to thread through the summarization pipeline.
- Prompt Adapters: MultiDocStagePromptBuilder creates chunk/document/meta prompts
  that incorporate the user's focus areas at every stage.
- Configuration: PromptConfig loads prompt parameters (word counts, etc.)
  from config/prompt_parameters.json.

Consolidated from individual files into this package:
- adapters.py (prompt adapters)
- focus_extractor.py (focus area extraction)
- template_manager.py (template loading/validation)
- config.py (prompt parameters)
"""

# Template management
# Prompt adapters
from src.core.prompting.adapters import (
    MultiDocStagePromptBuilder,
    StagePromptBuilder,
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
    "MultiDocStagePromptBuilder",
    # Prompt adapters
    "StagePromptBuilder",
    # Configuration
    "PromptConfig",
    # Template management
    "PromptTemplateManager",
    "get_prompt_config",
]
