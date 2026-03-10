"""
AI Service for CasePrepd.

Provides GPU detection and embedding model management.
Ollama LLM integration has been removed.

Usage:
    from src.services import AIService

    service = AIService()
    gpu_text = service.get_gpu_status_text()
"""

import logging
import threading

logger = logging.getLogger(__name__)

# Module-level singleton instance with lock for thread safety
_ai_service_instance = None
_ai_service_lock = threading.Lock()


class AIService:
    """
    Service layer for AI operations (GPU detection, embeddings).

    This is a singleton class - all calls to AIService() return the same instance.
    """

    def __new__(cls):
        """Ensure only one instance of AIService exists (thread-safe)."""
        global _ai_service_instance
        if _ai_service_instance is None:
            with _ai_service_lock:
                if _ai_service_instance is None:
                    _ai_service_instance = super().__new__(cls)
                    _ai_service_instance._initialized = False
        return _ai_service_instance

    def __init__(self):
        """Initialize the AI service (only runs once due to singleton)."""
        if self._initialized:
            return
        self._initialized = True

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

    @classmethod
    def reset_singleton(cls) -> None:
        """
        Clear the cached AIService instance.

        Next call to AIService() will create a fresh instance.
        Intended for test isolation -- not for production use.
        """
        global _ai_service_instance
        with _ai_service_lock:
            _ai_service_instance = None
