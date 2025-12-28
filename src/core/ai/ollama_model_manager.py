"""
Ollama Model Manager for LocalScribe
Handles loading and managing models through Ollama's REST API.

This is the next-generation model manager optimized for commercial use:
- Ollama REST API (no DLL or version conflicts)
- MIT-licensed (safe for commercial distribution)
- Multiple model options (Mistral, Llama 2, Neural-Chat)
- Simple installation and deployment

Structured Output Support (Ollama v0.5+):
- generate_structured() method for JSON schema-constrained output
- Used by Case Briefing Generator for reliable extraction
- Falls back to regex JSON parsing if needed
"""

import json
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
from src.logging_config import debug, debug_log, warning
from src.core.prompting import get_prompt_config, PromptTemplateManager
from .prompt_formatter import wrap_prompt_for_model
from .summary_post_processor import SummaryPostProcessor


class OllamaModelManager:
    """
    Manages Ollama-based AI models for case summarization.

    Uses Ollama REST API for hardware-accelerated inference.
    No version conflicts, commercial-safe, cross-platform.
    """

    def __init__(self):
        """Initialize the Ollama model manager."""
        self.api_base = OLLAMA_API_BASE
        self.model_name = OLLAMA_MODEL_NAME
        self.current_model_name = OLLAMA_MODEL_NAME  # For compatibility with worker code
        self.timeout = OLLAMA_TIMEOUT_SECONDS
        self.is_connected = False
        self.prompt_config = get_prompt_config()
        self.prompt_template_manager = PromptTemplateManager(PROMPTS_DIR, USER_PROMPTS_DIR)

        # Post-processor for summary length enforcement (dependency injection)
        self.post_processor = SummaryPostProcessor(
            generate_text_fn=self._generate_text_for_post_processor,
            prompt_template_manager=self.prompt_template_manager
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
        try:
            response = requests.get(
                f"{self.api_base}/api/tags",
                timeout=5
            )
            self.is_connected = response.status_code == 200
            if self.is_connected:
                debug("Successfully connected to Ollama")
                debug_log("[OLLAMA] Connection successful")
            else:
                debug(f"Ollama returned status {response.status_code}")
                debug_log(f"[OLLAMA] Connection failed: Status {response.status_code}")
        except requests.exceptions.ConnectionError:
            debug(f"Could not connect to Ollama at {self.api_base}")
            debug_log(f"[OLLAMA] Connection error: Cannot reach {self.api_base}")
            self.is_connected = False
        except Exception as e:
            debug(f"Connection check failed: {str(e)}")
            debug_log(f"[OLLAMA] Connection error: {str(e)}")
            self.is_connected = False

        return self.is_connected

    def get_available_models(self) -> dict:
        """
        Get list of available models from Ollama.

        Returns:
            dict: Model names mapped to availability and metadata
        """
        if not self.is_connected:
            self._check_connection()

        models = {}

        if self.is_connected:
            try:
                response = requests.get(
                    f"{self.api_base}/api/tags",
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    for model in data.get('models', []):
                        model_name = model['name']
                        models[model_name] = {
                            'name': model_name,
                            'available': True,
                            'size': model.get('size', 0),
                            'modified': model.get('modified_at', ''),
                            'description': f"Size: {self._format_size(model.get('size', 0))}"
                        }
                    debug(f"Found {len(models)} available models")
                    debug_log(f"[OLLAMA] Found {len(models)} models: {list(models.keys())}")
                else:
                    debug(f"Failed to get models: {response.status_code}")
            except Exception as e:
                debug(f"Error fetching available models: {str(e)}")
                debug_log(f"[OLLAMA] Error fetching models: {str(e)}")
        else:
            debug("Ollama not connected - cannot get available models")
            debug_log("[OLLAMA] Not connected - cannot list models")

        return models

    def _format_size(self, size_bytes: int) -> str:
        """Format bytes to human-readable size."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    def load_model(self, model_name: str = None) -> bool:
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
            debug(f"Cannot load model: Ollama not running at {self.api_base}")
            debug_log("[OLLAMA LOAD] Failed: Ollama not accessible")
            return False

        try:
            debug(f"Loading model: {model_name}")
            debug_log(f"[OLLAMA LOAD] Loading model: {model_name}")

            # Check if model is available
            available_models = self.get_available_models()
            if model_name not in available_models:
                debug(f"Model {model_name} not found. Attempting to pull...")
                debug_log("[OLLAMA LOAD] Model not found, attempting pull...")

                # Ollama doesn't have explicit "pull" via REST API in older versions
                # So we attempt to use it and let it auto-pull
                # This is handled by generate call

            debug(f"Model {model_name} is ready")
            debug_log(f"[OLLAMA LOAD] Model ready: {model_name}")
            return True

        except Exception as e:
            debug(f"Failed to load model {model_name}: {str(e)}")
            debug_log(f"[OLLAMA LOAD] Error: {str(e)}")
            return False

    def is_model_loaded(self) -> bool:
        """Check if a model is available and connection is active."""
        if not self.is_connected:
            self._check_connection()
        return self.is_connected

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = None,
        top_p: float = None
    ) -> str:
        """
        Generate text using Ollama REST API.

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
        if not self.is_model_loaded():
            raise RuntimeError(
                f"Ollama not available at {self.api_base}. "
                "Please ensure Ollama is running: https://ollama.ai"
            )

        if temperature is None:
            temperature = self.prompt_config.summary_temperature
        if top_p is None:
            top_p = self.prompt_config.top_p

        debug(f"Generating text (max_tokens={max_tokens}, temp={temperature}, top_p={top_p})")
        debug_log("\n[OLLAMA GENERATE] Starting text generation")
        debug_log(f"[OLLAMA GENERATE] Model: {self.model_name}")
        debug_log(f"[OLLAMA GENERATE] Max tokens: {max_tokens}")
        debug_log(f"[OLLAMA GENERATE] Prompt length: {len(prompt)} chars")
        debug_log(f"[OLLAMA GENERATE] Temperature: {temperature}, Top P: {top_p}")

        try:
            # Wrap prompt for model-specific format compatibility (Phase 2.7)
            wrapped_prompt = wrap_prompt_for_model(self.model_name, prompt)
            debug_log(f"[OLLAMA GENERATE] Wrapped prompt length: {len(wrapped_prompt)} chars")

            # Check if prompt may exceed context window (1 token ≈ 4 chars)
            estimated_tokens = len(wrapped_prompt) // 4
            context_window = OLLAMA_CONTEXT_WINDOW
            if estimated_tokens > context_window - 300:  # Leave room for output
                warning(
                    f"Prompt ({estimated_tokens} estimated tokens) may be truncated. "
                    f"Context window is {context_window} tokens."
                )
                debug_log(f"[OLLAMA GENERATE] WARNING: Prompt may exceed context window!")

            # Build request payload with explicit context window
            payload = {
                "model": self.model_name,
                "prompt": wrapped_prompt,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False,  # Non-streaming to avoid UTF-8 issues
                "num_predict": max_tokens,
                "options": {
                    "num_ctx": context_window,  # Explicit context window for CPU performance
                },
            }

            debug_log("[OLLAMA GENERATE] ===== ORIGINAL PROMPT START =====")
            debug_log(prompt)
            debug_log("[OLLAMA GENERATE] ===== ORIGINAL PROMPT END =====")
            debug_log("[OLLAMA GENERATE] ===== WRAPPED PROMPT START =====")
            debug_log(wrapped_prompt)
            debug_log("[OLLAMA GENERATE] ===== WRAPPED PROMPT END =====")

            # Make request to Ollama
            start_time = time.time()
            response = requests.post(
                f"{self.api_base}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code != 200:
                raise RuntimeError(f"Ollama returned status {response.status_code}: {response.text}")

            # Parse response
            result = response.json()
            generated_text = result.get('response', '')
            tokens_used = result.get('eval_count', 0)
            elapsed = time.time() - start_time

            debug_log(f"[OLLAMA GENERATE] Generation complete: {tokens_used} tokens in {elapsed:.2f}s")
            debug_log(f"[OLLAMA GENERATE] Output length: {len(generated_text)} chars")
            debug_log(f"[OLLAMA GENERATE] Output preview (first 100 chars): {generated_text[:100]}")

            return generated_text.strip()

        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                f"Generation timeout after {self.timeout} seconds. "
                "Try reducing summary length or increasing OLLAMA_TIMEOUT_SECONDS in config."
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.api_base}. "
                "Is Ollama running? Start with: ollama serve"
            ) from e
        except Exception as e:
            debug(f"Text generation failed: {str(e)}")
            debug_log(f"[OLLAMA GENERATE] Error: {str(e)}")
            raise RuntimeError(f"Text generation failed: {str(e)}") from e

    def generate_summary(
        self,
        case_text: str,
        max_words: int = 200,
        preset_id: str = "factual-summary"
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
                case_text=case_text
            )
        except FileNotFoundError:
            debug(f"Template not found: {preset_id}. Using factual-summary fallback.")
            # Fallback to factual-summary
            template = self.prompt_template_manager.load_template(model_id, "factual-summary")
            prompt = self.prompt_template_manager.format_template(
                template=template,
                min_words=min_words,
                max_words=max_words,
                max_words_range=max_words_range,
                case_text=case_text
            )

        # Estimate tokens and generate
        tokens_per_word = self.prompt_config.tokens_per_word
        buffer_multiplier = self.prompt_config.token_buffer_multiplier
        max_tokens = int(max_words_range * tokens_per_word * buffer_multiplier)

        summary = self.generate_text(
            prompt=prompt,
            max_tokens=max_tokens
        )

        # Delegate length enforcement to post-processor
        summary = self.post_processor.enforce_length(summary, max_words)

        return summary

    def unload_model(self):
        """Unload the current model (Ollama keeps models in memory)."""
        debug(f"Unloading model: {self.model_name}")
        # Ollama handles unloading automatically
        # This is just for API compatibility

    def health_check(self) -> dict:
        """
        Get health information about Ollama connection and models.

        Returns:
            dict: Health status and available models
        """
        status = {
            'connected': self.is_connected,
            'api_base': self.api_base,
            'model': self.model_name,
            'available_models': []
        }

        if self.is_connected:
            models = self.get_available_models()
            status['available_models'] = list(models.keys())

        return status

    def generate_structured(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> dict[str, Any] | None:
        """
        Generate structured JSON output using Ollama's format mode.

        Uses temperature=0 for deterministic extraction and format="json"
        to constrain output to valid JSON. Falls back to regex JSON
        extraction if the response contains extra text.

        This is the primary method for Case Briefing extraction.

        Args:
            prompt: The prompt including JSON schema instructions
            max_tokens: Maximum tokens to generate (default 1000)
            temperature: Sampling temperature (default 0.0 for deterministic)

        Returns:
            Parsed JSON as dict, or None if parsing fails

        Raises:
            RuntimeError: If Ollama is not available
        """
        if not self.is_model_loaded():
            raise RuntimeError(
                f"Ollama not available at {self.api_base}. "
                "Please ensure Ollama is running: https://ollama.ai"
            )

        debug_log("\n[OLLAMA STRUCTURED] Starting structured generation")
        debug_log(f"[OLLAMA STRUCTURED] Model: {self.model_name}")
        debug_log(f"[OLLAMA STRUCTURED] Max tokens: {max_tokens}, Temperature: {temperature}")
        debug_log(f"[OLLAMA STRUCTURED] Prompt length: {len(prompt)} chars")

        try:
            # Build request payload with JSON format mode
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
                "num_predict": max_tokens,
                "format": "json",  # Ollama v0.5+ structured output mode
                "options": {
                    "num_ctx": OLLAMA_CONTEXT_WINDOW,
                },
            }

            debug_log("[OLLAMA STRUCTURED] ===== PROMPT START =====")
            debug_log(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            debug_log("[OLLAMA STRUCTURED] ===== PROMPT END =====")

            # Make request to Ollama
            start_time = time.time()
            response = requests.post(
                f"{self.api_base}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code != 200:
                debug_log(f"[OLLAMA STRUCTURED] Error: Status {response.status_code}")
                return None

            # Parse response
            result = response.json()
            generated_text = result.get('response', '').strip()
            tokens_used = result.get('eval_count', 0)
            elapsed = time.time() - start_time

            debug_log(f"[OLLAMA STRUCTURED] Complete: {tokens_used} tokens in {elapsed:.2f}s")
            debug_log(f"[OLLAMA STRUCTURED] Response length: {len(generated_text)} chars")
            debug_log(f"[OLLAMA STRUCTURED] Response preview: {generated_text[:200]}...")

            # Try to parse the JSON
            parsed = self._parse_json_response(generated_text)

            if parsed is not None:
                debug_log(f"[OLLAMA STRUCTURED] Successfully parsed JSON with {len(parsed)} keys")
            else:
                debug_log("[OLLAMA STRUCTURED] Failed to parse JSON response")

            return parsed

        except requests.exceptions.Timeout:
            debug_log(f"[OLLAMA STRUCTURED] Timeout after {self.timeout}s")
            return None
        except requests.exceptions.ConnectionError:
            debug_log(f"[OLLAMA STRUCTURED] Connection error to {self.api_base}")
            return None
        except Exception as e:
            debug_log(f"[OLLAMA STRUCTURED] Error: {str(e)}")
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
                    for term_match in re.finditer(r'\{[^{}]+\}', array_content):
                        try:
                            term_obj = json.loads(term_match.group())
                            all_terms.append(term_obj)
                        except json.JSONDecodeError:
                            continue
                if all_terms:
                    debug_log(f"[OLLAMA STRUCTURED] Merged {len(all_terms)} terms from duplicate arrays")
                    return {"terms": all_terms}
            except Exception as e:
                debug_log(f"[OLLAMA STRUCTURED] Duplicate terms merge failed: {e}")

        # Strategy 1: Direct parse (ideal case)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find JSON object in text (common with chatty models)
        try:
            # Look for {...} pattern
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass

        # Strategy 3: Try to find the largest valid JSON block
        try:
            # Find all { and } positions
            start_positions = [m.start() for m in re.finditer(r'\{', text)]
            end_positions = [m.end() for m in re.finditer(r'\}', text)]

            # Try from each start position
            for start in start_positions:
                for end in reversed(end_positions):
                    if end > start:
                        try:
                            candidate = text[start:end]
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            continue
        except Exception:
            pass

        # Strategy 4: If it's a JSON array
        try:
            match = re.search(r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass

        debug_log(f"[OLLAMA STRUCTURED] All JSON parsing strategies failed for: {text[:100]}...")
        return None
