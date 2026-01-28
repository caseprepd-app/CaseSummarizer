"""
Multi-Document Prompt Adapters

Generates stage-specific prompts that incorporate user's focus areas:
1. Chunk prompts: Generic + focus emphasis
2. Document prompts: Generic + focus emphasis
3. Meta-summary: Structured with user's focus instructions

Design Principles:
1. Abstract base class for future implementations
2. Focus areas preserved at every stage of the pipeline
3. All methods accept preset_id + model_name for flexibility
4. Graceful fallback if template loading fails

Moved from src/prompt_adapters.py to src/prompting/adapters.py in Session 33.

Usage:
    from src.core.prompting import MultiDocPromptAdapter

    adapter = MultiDocPromptAdapter(template_manager, model_manager)

    # Generate chunk prompt with focus emphasis
    prompt = adapter.create_chunk_prompt(
        preset_id="injuries-focus",
        model_name="phi-3-mini",
        global_context="[Document overview...]",
        local_context="[Previous section...]",
        chunk_text="The plaintiff reported...",
        max_words=75
    )
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.core.prompting.focus_extractor import AIFocusExtractor, FocusExtractor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.core.ai.ollama_model_manager import OllamaModelManager
    from src.core.prompting.template_manager import PromptTemplateManager


class PromptAdapter(ABC):
    """
    Abstract interface for generating stage-specific prompts.

    This ABC allows customizing prompt generation without changing callers:
    - MultiDocPromptAdapter (current): Thread-through focus emphasis
    - VerbatimPromptAdapter (future): Use original template as-is
    - DebugPromptAdapter (future): Add debug markers to prompts

    All implementations must provide methods for:
    - Chunk summarization prompts
    - Final document summary prompts
    - Meta-summary prompts
    """

    @abstractmethod
    def create_chunk_prompt(
        self,
        preset_id: str,
        model_name: str,
        global_context: str,
        local_context: str,
        chunk_text: str,
        max_words: int,
    ) -> str:
        """Create prompt for summarizing a single chunk."""
        pass

    @abstractmethod
    def create_document_final_prompt(
        self, preset_id: str, model_name: str, chunk_summaries: str, filename: str, max_words: int
    ) -> str:
        """Create prompt for final document summary from chunk summaries."""
        pass

    @abstractmethod
    def create_meta_summary_prompt(
        self,
        preset_id: str,
        model_name: str,
        formatted_summaries: str,
        max_words: int,
        doc_count: int,
    ) -> str:
        """Create prompt for meta-summary from individual document summaries."""
        pass


class MultiDocPromptAdapter(PromptAdapter):
    """
    Adapts user prompts for multi-document pipeline stages.

    Threads the user's focus areas through the entire summarization pipeline:
    - Chunk prompts include focus emphasis to capture relevant details
    - Document prompts preserve information related to focus areas
    - Meta-summary uses extracted instructions to prioritize user's interests

    The focus areas are extracted once (cached) from the user's selected
    prompt template using FocusExtractor.
    """

    def __init__(
        self,
        template_manager: "PromptTemplateManager",
        model_manager: "OllamaModelManager",
        focus_extractor: FocusExtractor | None = None,
    ):
        """
        Initialize the prompt adapter.

        Args:
            template_manager: PromptTemplateManager for loading templates
            model_manager: OllamaModelManager for AI focus extraction
            focus_extractor: Optional custom FocusExtractor. If None,
                           uses AIFocusExtractor (default behavior).
        """
        self.template_manager = template_manager
        self.model_manager = model_manager

        # Use provided extractor or create default
        self.focus_extractor = focus_extractor or AIFocusExtractor(model_manager)

        # Cache focus results per preset (avoids repeated extraction)
        self._focus_cache: dict[str, dict] = {}

    def get_focus_for_preset(self, preset_id: str, model_name: str) -> dict:
        """
        Get or compute focus areas for a preset.

        Uses internal cache to avoid repeated template loading and extraction.
        The underlying FocusExtractor also caches by content hash.

        Args:
            preset_id: Template identifier
            model_name: Model name for template lookup

        Returns:
            Dict with 'emphasis' and 'instructions'
        """
        cache_key = f"{model_name}/{preset_id}"

        if cache_key in self._focus_cache:
            return self._focus_cache[cache_key]

        try:
            # Load template content
            template = self.template_manager.load_template(model_name, preset_id)

            # Extract focus areas
            focus = self.focus_extractor.extract_focus(template, preset_id)
            self._focus_cache[cache_key] = focus

            logger.debug("Cached focus for '%s'", cache_key)
            return focus

        except Exception as e:
            logger.error("Failed to get focus for '%s': %s", preset_id, e)
            # Return generic fallback
            return {
                "emphasis": "key facts, parties, timeline, outcomes",
                "instructions": """1. Synthesize the overall case narrative
2. Identify key parties and their roles
3. Highlight important findings""",
            }

    def create_chunk_prompt(
        self,
        preset_id: str,
        model_name: str,
        global_context: str,
        local_context: str,
        chunk_text: str,
        max_words: int,
    ) -> str:
        """
        Create chunk summarization prompt with focus emphasis.

        Includes the user's focus areas so relevant details are captured
        even in early chunks of the document.

        Args:
            preset_id: Template identifier for focus extraction
            model_name: Model name for template lookup
            global_context: Summary of document so far
            local_context: Summary of previous section
            chunk_text: Current chunk text to summarize
            max_words: Target word count for summary

        Returns:
            Formatted prompt string
        """
        focus = self.get_focus_for_preset(preset_id, model_name)

        return f"""<|system|>
You are a legal case summarizer analyzing sections of a long document. Your summaries will be combined with others to create a comprehensive case overview.
<|end|>
<|user|>
DOCUMENT CONTEXT:
{global_context}

PREVIOUS SECTION:
{local_context}

Summarize this section in approximately {max_words} words.

Focus on key facts, developments, and decisions.
Pay particular attention to: {focus["emphasis"]}

Preserve any information related to these focus areas, even if it seems minor - it may be important for the overall case analysis.

SECTION TEXT:
{chunk_text}
<|end|>
<|assistant|>
"""

    def create_document_final_prompt(
        self, preset_id: str, model_name: str, chunk_summaries: str, filename: str, max_words: int
    ) -> str:
        """
        Create final document summary prompt with focus emphasis.

        Combines chunk summaries into a single document summary while
        preserving information related to the user's focus areas.

        Args:
            preset_id: Template identifier for focus extraction
            model_name: Model name for template lookup
            chunk_summaries: Combined chunk summaries text
            filename: Document filename for context
            max_words: Target word count for summary

        Returns:
            Formatted prompt string
        """
        focus = self.get_focus_for_preset(preset_id, model_name)

        return f"""<|system|>
You are creating a comprehensive summary of a legal document from its section summaries.
<|end|>
<|user|>
Create a {max_words}-word summary of "{filename}" from these section summaries.

Pay particular attention to: {focus["emphasis"]}

Preserve any information related to these focus areas, even if it seems minor.
Present the information in a logical, chronological order where possible.

SECTION SUMMARIES:
{chunk_summaries}
<|end|>
<|assistant|>
"""

    def create_meta_summary_prompt(
        self,
        preset_id: str,
        model_name: str,
        formatted_summaries: str,
        max_words: int,
        doc_count: int,
    ) -> str:
        """
        Create meta-summary prompt using user's focus instructions.

        Uses the extracted instructions from the user's template to guide
        what information should be emphasized in the final meta-summary.

        Args:
            preset_id: Template identifier for focus extraction
            model_name: Model name for template lookup
            formatted_summaries: Combined document summaries
            max_words: Target word count for meta-summary
            doc_count: Number of documents being summarized

        Returns:
            Formatted prompt string
        """
        focus = self.get_focus_for_preset(preset_id, model_name)
        min_words = max(100, int(max_words * 0.7))

        return f"""<|system|>
You are a legal document analyst reviewing summaries of {doc_count} documents from a single case. Your job is to create a comprehensive meta-summary that synthesizes the key information.
<|end|>
<|user|>
Individual document summaries:

{formatted_summaries}

Create a comprehensive meta-summary ({min_words}-{max_words} words) that:
{focus["instructions"]}

Present the information in a logical, chronological order where appropriate.
Synthesize information across documents rather than summarizing each document separately.
<|end|>
<|assistant|>
"""

    def clear_cache(self):
        """Clear the internal focus cache."""
        self._focus_cache.clear()
        logger.debug("Focus cache cleared")
