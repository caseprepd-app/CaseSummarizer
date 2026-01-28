"""
AI Service for CasePrepd.

Provides a clean interface for AI/Ollama operations and GPU detection.
Wraps OllamaModelManager, GPU detector, and prompt template components.

Session 83: Created to provide services layer access to AI operations,
enforcing the pipeline architecture (UI -> Services -> Core).

Usage:
    from src.services import AIService

    service = AIService()
    manager = service.get_ollama_manager()
    gpu_text = service.get_gpu_status_text()
"""

import logging

logger = logging.getLogger(__name__)

# Module-level singleton instance
_ai_service_instance = None


class AIService:
    """
    Service layer for AI/Ollama operations.

    Provides access to LLM management, GPU detection, and prompt templates.

    This is a singleton class - all calls to AIService() return the same instance,
    ensuring consistent state for the Ollama model manager across the application.
    """

    def __new__(cls):
        """Ensure only one instance of AIService exists."""
        global _ai_service_instance
        if _ai_service_instance is None:
            _ai_service_instance = super().__new__(cls)
            _ai_service_instance._initialized = False
        return _ai_service_instance

    def __init__(self):
        """Initialize the AI service (only runs once due to singleton)."""
        if self._initialized:
            return
        self._initialized = True
        self._ollama_manager = None

    def get_ollama_manager(self):
        """
        Get the Ollama model manager instance.

        Lazy-loaded singleton for managing Ollama LLM operations.

        Returns:
            OllamaModelManager instance.
        """
        if self._ollama_manager is None:
            from src.core.ai import OllamaModelManager

            self._ollama_manager = OllamaModelManager()

        return self._ollama_manager

    def get_gpu_status_text(self) -> str:
        """
        Get human-readable GPU status text.

        Returns:
            Status string like "NVIDIA RTX 3080 (10GB VRAM)" or "No dedicated GPU".
        """
        from src.core.utils.gpu_detector import get_gpu_status_text

        return get_gpu_status_text()

    def has_dedicated_gpu(self) -> bool:
        """
        Check if system has a dedicated GPU.

        Returns:
            True if dedicated GPU detected, False otherwise.
        """
        from src.core.utils.gpu_detector import has_dedicated_gpu

        return has_dedicated_gpu()

    def get_optimal_context_size(self) -> int:
        """
        Get optimal context window size based on GPU VRAM.

        Returns:
            Context size in tokens (e.g., 4096, 8192, 16384).
        """
        from src.core.utils.gpu_detector import get_optimal_context_size

        return get_optimal_context_size()

    def get_vram_gb(self) -> float:
        """
        Get GPU VRAM in gigabytes.

        Returns:
            VRAM size in GB, or 0.0 if no GPU detected.
        """
        from src.core.utils.gpu_detector import get_vram_gb

        return get_vram_gb()

    def get_gpu_info(self) -> dict:
        """
        Get detailed GPU information.

        Returns:
            Dict with has_gpu, vram_gb, gpu_name, etc.
        """
        from src.core.utils.gpu_detector import get_gpu_info

        return get_gpu_info()

    def get_prompt_template_manager(self):
        """
        Get the prompt template manager.

        Returns:
            PromptTemplateManager for loading and managing prompts.
        """
        from src.config import PROMPTS_DIR, USER_PROMPTS_DIR
        from src.core.prompting.template_manager import PromptTemplateManager

        return PromptTemplateManager(PROMPTS_DIR, USER_PROMPTS_DIR)

    def check_ollama_connection(self) -> bool:
        """
        Check if Ollama is running and accessible.

        Returns:
            True if Ollama API is reachable, False otherwise.
        """
        manager = self.get_ollama_manager()
        is_connected = manager.check_connection()

        status = "connected" if is_connected else "not reachable"
        logger.debug("Ollama %s", status)

        return is_connected

    def get_available_models(self) -> list[dict]:
        """
        Get list of available Ollama models.

        Returns:
            List of model dicts with name, size, etc.
        """
        manager = self.get_ollama_manager()
        return manager.get_available_models()

    def get_current_model(self) -> str:
        """
        Get the currently selected Ollama model name.

        Returns:
            Model name string (e.g., "gemma3:1b").
        """
        manager = self.get_ollama_manager()
        return manager.get_current_model()

    def set_current_model(self, model_name: str) -> bool:
        """
        Set the current Ollama model.

        Args:
            model_name: Name of model to use.

        Returns:
            True if model exists and was set, False otherwise.
        """
        manager = self.get_ollama_manager()
        success = manager.set_current_model(model_name)

        if success:
            logger.debug("Model set to: %s", model_name)
        else:
            logger.debug("Failed to set model: %s", model_name)

        return success
