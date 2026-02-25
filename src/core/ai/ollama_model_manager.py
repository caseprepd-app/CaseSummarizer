"""
Ollama Model Manager for CasePrepd
Handles loading and managing models through Ollama's REST API.

This is the next-generation model manager optimized for commercial use:
- Ollama REST API (no DLL or version conflicts)
- MIT-licensed (safe for commercial distribution)
- Multiple model options (Mistral, Llama 2, Neural-Chat)
- Simple installation and deployment

Structured Output Support (Ollama v0.5+):
- generate_structured() method for JSON schema-constrained output
- Used by vocabulary extraction for reliable term extraction
- Falls back to regex JSON parsing if needed
"""

import json
import logging
import re
import time
from typing import Any

import requests

from src.config import (
    OLLAMA_API_BASE,
    OLLAMA_CONTEXT_WINDOW,
    OLLAMA_MODEL_NAME,
    OLLAMA_TIMEOUT_SECONDS,
    PROMPTS_DIR,
    USER_PROMPTS_DIR,
)
from src.core.prompting import PromptTemplateManager, get_prompt_config

from .prompt_formatter import wrap_prompt_for_model
from .summary_post_processor import SummaryPostProcessor

logger = logging.getLogger(__name__)


def _get_context_window() -> int:
    """
    Get the effective context window size from user preferences.

    Uses dynamic context sizing based on GPU VRAM detection.

    Returns:
        int: Context window size (num_ctx) for Ollama API calls.
    """
    try:
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        return prefs.get_effective_context_size()
    except Exception as e:
        # Fallback to config default if preferences unavailable
        logger.debug("Could not load user preferences for context size: %s", e)
        return OLLAMA_CONTEXT_WINDOW


class OllamaModelManager:
    """
    Manages Ollama-based AI models for case summarization.

    Uses Ollama REST API for hardware-accelerated inference.
    No version conflicts, commercial-safe, cross-platform.
    """

    def __init__(self):
        """Initialize the Ollama model manager."""
        self.api_base = OLLAMA_API_BASE

        # Read saved model preference, fall back to config default
        try:
            from src.user_preferences import get_user_preferences

            prefs = get_user_preferences()
            saved_model = prefs.get("ollama_model", "")
            self.model_name = saved_model if saved_model else OLLAMA_MODEL_NAME
        except Exception as e:
            logger.debug("Could not load saved model preference: %s", e)
            self.model_name = OLLAMA_MODEL_NAME

        self.current_model_name = self.model_name  # For compatibility with worker code
        self.timeout = OLLAMA_TIMEOUT_SECONDS

        self.is_connected = False
        self.prompt_config = get_prompt_config()
        self.prompt_template_manager = PromptTemplateManager(PROMPTS_DIR, USER_PROMPTS_DIR)

        # Post-processor for summary length enforcement (dependency injection)
        self.post_processor = SummaryPostProcessor(
            generate_text_fn=self._generate_text_for_post_processor,
            prompt_template_manager=self.prompt_template_manager,
            model_id=self.model_name,
        )

        # Test connection on initialization
        self._check_connection()

    def _generate_text_for_post_processor(self, prompt: str, max_tokens: int) -> str:
        """
        Wrapper for generate_text used by SummaryPostProcessor.

        This provides a clean interface matching the expected signature.

        Args:
            prompt: The prompt to generate from
            max_tokens: Maximum tokens to generate

        Returns:
            str: Generated text
        """
        return self.generate_text(prompt=prompt, max_tokens=max_tokens)

    def _check_connection(self) -> bool:
        """
        Check if Ollama is running and accessible.

        Returns:
            bool: True if Ollama is accessible, False otherwise
        """
        from src.config import OLLAMA_CONNECTION_TIMEOUT

        try:
            response = requests.get(f"{self.api_base}/api/tags", timeout=OLLAMA_CONNECTION_TIMEOUT)
            self.is_connected = response.status_code == 200
            if self.is_connected:
                logger.debug("Successfully connected to Ollama")
            else:
                logger.debug("Ollama returned status %s", response.status_code)
        except requests.exceptions.ConnectionError:
            logger.debug("Could not connect to Ollama at %s", self.api_base)
            self.is_connected = False
        except Exception as e:
            logger.debug("Connection check failed: %s", e)
            self.is_connected = False

        return self.is_connected

    def get_available_models(self) -> dict:
        """
        Get list of available models from Ollama.

        Returns:
            dict: Model names mapped to availability and metadata
        """
        from src.config import OLLAMA_API_TIMEOUT

        if not self.is_connected:
            self._check_connection()

        models = {}

        if self.is_connected:
            try:
                response = requests.get(f"{self.api_base}/api/tags", timeout=OLLAMA_API_TIMEOUT)
                if response.status_code == 200:
                    data = response.json()
                    for model in data.get("models", []):
                        model_name = model["name"]
                        models[model_name] = {
                            "name": model_name,
                            "available": True,
                            "size": model.get("size", 0),
                            "modified": model.get("modified_at", ""),
                            "description": f"Size: {self._format_size(model.get('size', 0))}",
                        }
                    logger.debug("Found %s available models: %s", len(models), list(models.keys()))
                else:
                    logger.debug("Failed to get models: %s", response.status_code)
            except Exception as e:
                logger.debug("Error fetching available models: %s", e)
        else:
            logger.debug("Ollama not connected - cannot get available models")

        return models

    def _format_size(self, size_bytes: int) -> str:
        """Format bytes to human-readable size."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    def load_model(self, model_name: str | None = None) -> bool:
        """
        Load a model via Ollama (pulls if not already available).

        Args:
            model_name: Model to load (defaults to configured model)

        Returns:
            bool: True if model is available/loaded, False otherwise
        """
        if model_name is None:
            model_name = self.model_name

        self.model_name = model_name
        self.current_model_name = model_name  # Keep in sync for compatibility

        if not self.is_connected:
            self._check_connection()

        if not self.is_connected:
            logger.debug("Cannot load model: Ollama not running at %s", self.api_base)
            return False

        try:
            logger.debug("Loading model: %s", model_name)

            # Check if model is available
            available_models = self.get_available_models()
            if model_name not in available_models:
                logger.debug("Model %s not found, attempting pull...", model_name)

                # Ollama doesn't have explicit "pull" via REST API in older versions
                # So we attempt to use it and let it auto-pull
                # This is handled by generate call

            logger.debug("Model ready: %s", model_name)
            return True

        except Exception as e:
            logger.debug("Failed to load model %s: %s", model_name, e)
            return False

    @property
    def has_model(self) -> bool:
        """Check if a model has been selected (non-empty name)."""
        return bool(self.model_name)

    def is_model_loaded(self) -> bool:
        """Check if a model is available and connection is active."""
        if not self.has_model:
            return False
        if not self.is_connected:
            self._check_connection()
        return self.is_connected

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        """
        Generate text using official Ollama Python library with streaming.

        Uses the official ollama library instead of raw requests for better
        reliability and streaming support. Streaming prevents apparent hangs
        on long generations.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter

        Returns:
            str: Generated text

        Raises:
            RuntimeError: If Ollama is not available
        """
        import ollama

        if not self.is_model_loaded():
            raise RuntimeError(
                f"Ollama not available at {self.api_base}. "
                "Please ensure Ollama is running: https://ollama.ai"
            )

        if temperature is None:
            temperature = self.prompt_config.summary_temperature
        if top_p is None:
            top_p = self.prompt_config.top_p

        logger.debug(
            "Generating text (max_tokens=%s, temp=%s, top_p=%s)", max_tokens, temperature, top_p
        )
        logger.debug("Starting text generation (streaming)")
        logger.debug("Model: %s", self.model_name)
        logger.debug("Max tokens: %s", max_tokens)
        logger.debug("Prompt length: %s chars", len(prompt))
        logger.debug("Temperature: %s, Top P: %s", temperature, top_p)

        try:
            # Wrap prompt for model-specific format compatibility (Phase 2.7)
            wrapped_prompt = wrap_prompt_for_model(self.model_name, prompt)
            logger.debug("Wrapped prompt length: %s chars", len(wrapped_prompt))

            # Check if prompt may exceed context window
            # Use dynamic context size based on GPU/VRAM
            from src.core.qa.token_budget import count_tokens

            estimated_tokens = count_tokens(wrapped_prompt)
            context_window = _get_context_window()
            if estimated_tokens > context_window - 300:  # Leave room for output
                logger.warning(
                    "Prompt (%s estimated tokens) may be truncated. Context window is %s tokens.",
                    estimated_tokens,
                    context_window,
                )
                logger.debug("Prompt may exceed context window!")

            logger.debug("Using context window: %s tokens", context_window)

            logger.debug("===== ORIGINAL PROMPT START =====")
            logger.debug("%s", prompt)
            logger.debug("===== ORIGINAL PROMPT END =====")
            logger.debug("===== WRAPPED PROMPT START =====")
            logger.debug("%s", wrapped_prompt)
            logger.debug("===== WRAPPED PROMPT END =====")

            # Use official ollama library with streaming for reliability
            start_time = time.time()
            generated_chunks = []

            # Stream the response - this prevents hanging on long generations
            stream = ollama.generate(
                model=self.model_name,
                prompt=wrapped_prompt,
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                    "num_ctx": context_window,
                },
                stream=True,
            )

            # Collect streamed chunks
            for chunk in stream:
                if "response" in chunk:
                    generated_chunks.append(chunk["response"])

            generated_text = "".join(generated_chunks)
            elapsed = time.time() - start_time

            logger.debug("Generation complete in %.2fs", elapsed)
            logger.debug("Output length: %s chars", len(generated_text))
            logger.debug("Output preview (first 100 chars): %s", generated_text[:100])

            return generated_text.strip()

        except ollama.ResponseError as e:
            logger.debug("Response error: %s", e)
            raise RuntimeError(f"Ollama error: {e!s}") from e
        except Exception as e:
            logger.debug("Text generation failed: %s", e)
            raise RuntimeError(f"Text generation failed: {e!s}") from e

    def generate_summary(
        self, case_text: str, max_words: int = 200, preset_id: str = "factual-summary"
    ) -> str:
        """
        Generate a case summary from document text via Ollama.

        Includes recursive length enforcement: if the generated summary exceeds
        the target length by more than the configured tolerance, it will be
        condensed by the SummaryPostProcessor.

        Args:
            case_text: The cleaned case document text
            max_words: Target summary length in words (100-500)
            preset_id: Template preset to use

        Returns:
            str: Complete summary text (within target length or best effort)
        """
        # Get word count range from config
        min_words, max_words_range = self.prompt_config.get_word_count_range(max_words)

        # Load and format prompt template
        model_id = "phi-3-mini"  # Use phi-3 templates with Ollama

        try:
            template = self.prompt_template_manager.load_template(model_id, preset_id)
            prompt = self.prompt_template_manager.format_template(
                template=template,
                min_words=min_words,
                max_words=max_words,
                max_words_range=max_words_range,
                case_text=case_text,
            )
        except FileNotFoundError:
            logger.debug("Template not found: %s. Using factual-summary fallback.", preset_id)
            # Fallback to factual-summary
            template = self.prompt_template_manager.load_template(model_id, "factual-summary")
            prompt = self.prompt_template_manager.format_template(
                template=template,
                min_words=min_words,
                max_words=max_words,
                max_words_range=max_words_range,
                case_text=case_text,
            )

        # Estimate tokens and generate
        tokens_per_word = self.prompt_config.tokens_per_word
        buffer_multiplier = self.prompt_config.token_buffer_multiplier
        max_tokens = int(max_words_range * tokens_per_word * buffer_multiplier)

        summary = self.generate_text(prompt=prompt, max_tokens=max_tokens)

        # Delegate length enforcement to post-processor
        summary = self.post_processor.enforce_length(summary, max_words)

        return summary

    def unload_model(self):
        """Unload the current model (Ollama keeps models in memory)."""
        logger.debug("Unloading model: %s", self.model_name)
        # Ollama handles unloading automatically
        # This is just for API compatibility

    def health_check(self) -> dict:
        """
        Get health information about Ollama connection and models.

        Returns:
            dict: Health status and available models
        """
        status = {
            "connected": self.is_connected,
            "api_base": self.api_base,
            "model": self.model_name,
            "available_models": [],
        }

        if self.is_connected:
            models = self.get_available_models()
            status["available_models"] = list(models.keys())

        return status

    def generate_structured(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> dict[str, Any] | None:
        """
        Generate structured JSON output using Ollama's format mode.

        Uses the official ollama library with format="json" for reliable
        JSON output. Falls back to regex JSON extraction if needed.

        This is the primary method for LLM-based extraction.

        Args:
            prompt: The prompt including JSON schema instructions
            max_tokens: Maximum tokens to generate (default 1000)
            temperature: Sampling temperature (default 0.0 for deterministic)

        Returns:
            Parsed JSON as dict, or None if parsing fails

        Raises:
            RuntimeError: If Ollama is not available
        """
        import ollama

        if not self.is_model_loaded():
            raise RuntimeError(
                f"Ollama not available at {self.api_base}. "
                "Please ensure Ollama is running: https://ollama.ai"
            )

        logger.debug("Starting structured generation")
        logger.debug("Model: %s", self.model_name)
        logger.debug("Max tokens: %s, Temperature: %s", max_tokens, temperature)
        logger.debug("Prompt length: %s chars", len(prompt))

        try:
            # Use dynamic context size based on GPU/VRAM
            context_window = _get_context_window()
            logger.debug("Using context window: %s tokens", context_window)

            logger.debug("===== PROMPT START =====")
            logger.debug("%s", prompt[:500] + "..." if len(prompt) > 500 else prompt)
            logger.debug("===== PROMPT END =====")

            # Use official ollama library with JSON format
            start_time = time.time()

            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                format="json",  # Ollama structured output mode
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": context_window,
                },
                stream=False,  # JSON format works better without streaming
            )

            generated_text = response.get("response", "").strip()
            elapsed = time.time() - start_time

            logger.debug("Complete in %.2fs", elapsed)
            logger.debug("Response length: %s chars", len(generated_text))
            logger.debug("Response preview: %s...", generated_text[:200])

            # Try to parse the JSON
            parsed = self._parse_json_response(generated_text)

            if parsed is not None:
                logger.debug("Successfully parsed JSON with %s keys", len(parsed))
            else:
                logger.debug("Failed to parse JSON response")

            return parsed

        except ollama.ResponseError as e:
            logger.debug("Ollama error: %s", e)
            return None
        except Exception as e:
            logger.debug("Error: %s", e)
            return None

    def _parse_json_response(self, text: str) -> dict[str, Any] | None:
        """
        Parse JSON from LLM response with fallback strategies.

        Tries multiple strategies:
        0. Merge duplicate "terms" arrays (common LLM error)
        1. Direct JSON parsing
        2. Extract JSON object from surrounding text
        3. Extract JSON array from surrounding text

        Args:
            text: Raw response text from LLM

        Returns:
            Parsed JSON as dict/list, or None if all parsing fails
        """
        if not text:
            return None

        # Strategy 0: Fix duplicate "terms" keys (common LLM output error)
        # Model sometimes outputs: {"terms": [...], "terms": [...]} which is invalid JSON
        # Python's json.loads keeps only the LAST duplicate key, losing earlier terms
        if text.count('"terms"') > 1:
            try:
                # Extract all "terms" arrays and merge them
                all_terms = []
                for match in re.finditer(r'"terms"\s*:\s*\[([^\]]*)\]', text):
                    array_content = match.group(1)
                    # Parse individual term objects from the array content
                    for term_match in re.finditer(r"\{[^{}]+\}", array_content):
                        try:
                            term_obj = json.loads(term_match.group())
                            all_terms.append(term_obj)
                        except json.JSONDecodeError:
                            continue
                if all_terms:
                    logger.debug("Merged %s terms from duplicate arrays", len(all_terms))
                    return {"terms": all_terms}
            except Exception as e:
                logger.debug("Duplicate terms merge failed: %s", e)

        # Strategy 1: Direct parse (ideal case)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find JSON object in text (common with chatty models)
        try:
            # Look for {...} pattern
            match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass

        # Strategy 3: Try to find the largest valid JSON block
        try:
            # Find all { and } positions
            start_positions = [m.start() for m in re.finditer(r"\{", text)]
            end_positions = [m.end() for m in re.finditer(r"\}", text)]

            # Try from each start position
            for start in start_positions:
                for end in reversed(end_positions):
                    if end > start:
                        try:
                            candidate = text[start:end]
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.debug("JSON block extraction failed: %s", e)

        # Strategy 4: If it's a JSON array, wrap in dict
        try:
            match = re.search(r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]", text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return {"items": parsed}
                return parsed
        except json.JSONDecodeError:
            pass

        logger.debug("All JSON parsing strategies failed for: %s...", text[:100])
        return None
