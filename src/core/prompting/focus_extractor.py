"""
Focus Area Extractor for Multi-Document Summarization

Extracts user's focus areas from prompt templates to guide summarization.
Uses AI-assisted extraction via Ollama with caching by template content hash.

Design Principles:
1. ALL templates use AI extraction (no hardcoded mappings)
2. Cache by content hash - if template file changes, re-extract
3. Graceful fallback if AI extraction fails
4. Abstract base class for future implementations (keyword-based, hybrid, etc.)

Usage:
    from src.core.prompting import AIFocusExtractor

    extractor = AIFocusExtractor(model_manager)
    focus = extractor.extract_focus(template_content, "my-template")

    # focus = {
    #     'emphasis': "injuries, timeline, damages...",
    #     'instructions': "1. Identify parties\n2. Note claims..."
    # }
"""

import logging
from abc import ABC, abstractmethod
from hashlib import md5
from typing import TYPE_CHECKING, ClassVar

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.core.ai.ollama_model_manager import OllamaModelManager


class FocusExtractor(ABC):
    """
    Abstract interface for extracting focus areas from templates.

    This ABC allows swapping implementations without changing callers:
    - AIFocusExtractor (current): Uses Ollama to extract
    - KeywordFocusExtractor (future): Simple keyword/regex parsing
    - HybridFocusExtractor (future): AI + keyword validation

    All implementations must return a dict with 'emphasis' and 'instructions'.
    """

    @abstractmethod
    def extract_focus(self, template: str, preset_id: str) -> dict:
        """
        Extract focus areas from a prompt template.

        Args:
            template: Full prompt template content
            preset_id: Template identifier (for logging/debugging)

        Returns:
            Dict with:
                - 'emphasis': String for intermediate prompts
                  (e.g., "injuries, timeline, damages")
                - 'instructions': Numbered list for meta-summary prompt
                  (e.g., "1. Identify parties\n2. Note claims...")
        """
        pass


class AIFocusExtractor(FocusExtractor):
    """
    Extracts focus areas from ANY prompt template using AI.

    Design Decision: ALL templates use AI extraction (cached per template hash).
    This is because:
    1. Built-in templates are placeholders that will change over time
    2. Users can create their own custom templates
    3. The meta-summary needs to preserve what the USER is seeking
    4. One unified code path is simpler and more maintainable

    Cache ensures we only call the AI once per unique template content.
    Cache is keyed by content HASH, not preset_id, so edits trigger re-extraction.
    """

    # Class-level cache for AI-extracted focus (shared across instances)
    # Keyed by template content hash
    _cache: ClassVar[dict[str, dict]] = {}

    def __init__(self, model_manager: "OllamaModelManager"):
        """
        Initialize extractor with model manager for AI-assisted extraction.

        Args:
            model_manager: OllamaModelManager for focus extraction.
                          Required - this class uses AI for all templates.

        Raises:
            ValueError: If model_manager is None
        """
        if model_manager is None:
            raise ValueError("AIFocusExtractor requires a model_manager")
        self.model_manager = model_manager

    def extract_focus(self, template: str, preset_id: str) -> dict:
        """
        Extract focus areas from a prompt template using AI.

        Uses cached results if the same template was seen before.
        The cache is keyed by template CONTENT hash, not preset_id,
        so if a template file is updated, new focus is extracted.

        Args:
            template: Full prompt template content
            preset_id: Template identifier (for logging only)

        Returns:
            Dict with 'emphasis' (string) and 'instructions' (string)
        """
        # Cache by content hash (not preset_id) so edits trigger re-extraction
        template_hash = md5(template.encode()).hexdigest()[:8]

        if template_hash in self._cache:
            logger.debug("Using cached focus for '%s' (hash: %s)", preset_id, template_hash)
            return self._cache[template_hash]

        # Extract focus via AI
        logger.debug("Extracting focus from '%s' via AI...", preset_id)
        focus = self._extract_via_ai(template, preset_id)
        self._cache[template_hash] = focus

        # Log extracted emphasis (truncated for readability)
        emphasis = str(focus.get("emphasis", ""))
        emphasis_preview = emphasis[:60] + "..." if len(emphasis) > 60 else emphasis
        logger.debug("Extracted emphasis: %s", emphasis_preview)

        return focus

    def _extract_via_ai(self, template: str, preset_id: str) -> dict:
        """
        Use Ollama to extract focus areas from a template.

        Sends a structured prompt asking the AI to identify the key focus areas
        from the template. The response is parsed into emphasis/instructions format.

        Args:
            template: Full prompt template content
            preset_id: Template identifier (for logging)

        Returns:
            Dict with 'emphasis' and 'instructions'
        """
        # Truncate template to avoid exceeding context window
        template_excerpt = template[:2000]
        if len(template) > 2000:
            template_excerpt += "\n[...template truncated...]"

        prompt = f"""<|system|>
You are analyzing a legal document summarization template to extract its key focus areas.
<|end|>
<|user|>
Analyze this legal document summarization template and extract the main focus areas the user wants to capture.

TEMPLATE:
---
{template_excerpt}
---

Identify the 3-5 key things this template wants the summarizer to focus on when summarizing documents.

Respond in this EXACT format (no other text):
EMPHASIS: <comma-separated list of focus areas, e.g., "injuries, medical treatment, timeline, damages">
INSTRUCTIONS:
1. <first instruction for creating meta-summary>
2. <second instruction>
3. <third instruction>
4. <optional fourth instruction>
5. <optional fifth instruction>
<|end|>
<|assistant|>
"""

        try:
            response = self.model_manager.generate_text(prompt, max_tokens=400)
            return self._parse_ai_response(response)
        except Exception as e:
            logger.error("AI extraction failed for '%s': %s", preset_id, e, exc_info=True)
            logger.debug("Using generic fallback due to error")
            return self._generic_fallback()

    def _parse_ai_response(self, response: str) -> dict:
        """
        Parse AI response into emphasis/instructions dict.

        Looks for EMPHASIS: and INSTRUCTIONS: markers in the response.
        Falls back to defaults if parsing fails.

        Args:
            response: Raw AI response text

        Returns:
            Dict with 'emphasis' and 'instructions'
        """
        # Default values in case parsing fails
        emphasis = "key facts, parties involved, timeline, outcomes"
        instructions = """1. Synthesize the overall case narrative
2. Identify all key parties and their roles
3. Highlight primary claims and defenses
4. Note significant evidence and outcomes"""

        try:
            lines = response.strip().split("\n")
            instructions_lines = []
            in_instructions = False

            for line in lines:
                line_stripped = line.strip()

                if line_stripped.upper().startswith("EMPHASIS:"):
                    # Extract everything after "EMPHASIS:"
                    emphasis = line_stripped.split(":", 1)[1].strip()
                    in_instructions = False

                elif line_stripped.upper().startswith("INSTRUCTIONS:"):
                    in_instructions = True
                    # Check if instructions start on same line
                    rest = line_stripped.split(":", 1)[1].strip()
                    if rest:
                        instructions_lines.append(rest)

                elif in_instructions and line_stripped:
                    instructions_lines.append(line_stripped)

            if instructions_lines:
                instructions = "\n".join(instructions_lines)

        except Exception as e:
            logger.debug("Error parsing AI response: %s, using defaults", e)

        return {"emphasis": emphasis, "instructions": instructions}

    def _generic_fallback(self) -> dict:
        """
        Generic focus when AI extraction fails.

        Returns a reasonable default that works for most legal documents.

        Returns:
            Dict with generic 'emphasis' and 'instructions'
        """
        return {
            "emphasis": "key facts, parties involved, timeline, outcomes, and significant evidence",
            "instructions": """1. Synthesize the overall case narrative and timeline
2. Identify all key parties and their relationships
3. Highlight primary claims, defenses, and legal issues
4. Note significant evidence, testimony, and outcomes""",
        }

    @classmethod
    def clear_cache(cls):
        """
        Clear the focus extraction cache.

        Useful for testing or when templates are known to have changed.
        """
        cls._cache.clear()
        logger.debug("Cache cleared")
