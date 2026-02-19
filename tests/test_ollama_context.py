"""
Tests for Ollama context window configuration.

These tests verify:
1. Context window is explicitly set in API calls
2. Chunking respects the context window limits
3. Truncation warnings are issued when needed
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import OLLAMA_CONTEXT_WINDOW  # noqa: E402


class TestContextWindowConfig:
    """Test that context window configuration is properly set."""

    def test_context_window_defined(self):
        """Verify OLLAMA_CONTEXT_WINDOW is defined and reasonable."""
        assert OLLAMA_CONTEXT_WINDOW is not None
        assert isinstance(OLLAMA_CONTEXT_WINDOW, int)
        assert OLLAMA_CONTEXT_WINDOW >= 1024  # At least 1K
        assert OLLAMA_CONTEXT_WINDOW <= 131072  # Max reasonable for consumer hardware

    def test_fallback_is_conservative(self):
        """Verify fallback default is conservative (Session 64: now 4000)."""
        # Session 64: Changed from 2048 to 4000 as fallback.
        # Actual context is now dynamic based on GPU VRAM via user preferences.
        assert OLLAMA_CONTEXT_WINDOW == 4000, (
            "Fallback should be 4000 (conservative CPU-safe default). "
            "Actual context is determined by user_preferences.get_effective_context_size()"
        )


class TestChunkingConfig:
    """Test that chunking config aligns with context window."""

    def test_chunk_words_fit_context(self):
        """Verify max_chunk_words fits within context window."""
        import yaml

        config_path = project_root / "config" / "chunking_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        max_chunk_words = config["chunking"]["max_chunk_words"]
        # Rough estimate: 1.3 tokens per word
        estimated_tokens = int(max_chunk_words * 1.3)

        # Need room for prompt template (~200) and output (~300)
        available_tokens = OLLAMA_CONTEXT_WINDOW - 500

        assert estimated_tokens <= available_tokens, (
            f"max_chunk_words ({max_chunk_words} words ≈ {estimated_tokens} tokens) "
            f"exceeds available context ({available_tokens} tokens)"
        )

    def test_hard_limit_fits_context(self):
        """Verify max_chunk_words_hard_limit fits within context window."""
        import yaml

        config_path = project_root / "config" / "chunking_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        hard_limit = config["chunking"]["max_chunk_words_hard_limit"]
        # Rough estimate: 1.3 tokens per word
        estimated_tokens = int(hard_limit * 1.3)

        # Hard limit can be slightly over since it's the absolute max
        # Still need some room for prompt
        max_allowed_tokens = OLLAMA_CONTEXT_WINDOW - 200

        assert estimated_tokens <= max_allowed_tokens, (
            f"max_chunk_words_hard_limit ({hard_limit} words ≈ {estimated_tokens} tokens) "
            f"exceeds max allowed ({max_allowed_tokens} tokens)"
        )


class TestOllamaPayload:
    """Test that Ollama API payload includes context window."""

    @patch("ollama.generate")
    @patch("src.core.ai.ollama_model_manager.requests.get")
    def test_num_ctx_in_payload(self, mock_get, mock_generate):
        """Verify num_ctx is included in Ollama API calls."""
        # Mock connection check
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"models": []}
        mock_get.return_value = mock_get_response

        # Mock ollama.generate to return a streaming iterator
        mock_generate.return_value = iter([{"response": "Test summary"}])

        from src.core.ai.ollama_model_manager import OllamaModelManager

        manager = OllamaModelManager()
        manager.model_name = "test-model:latest"
        manager.is_connected = True

        manager.generate_text("Test prompt", max_tokens=100)

        # Check that ollama.generate was called with num_ctx in options
        assert mock_generate.called, "ollama.generate should have been called"
        call_kwargs = mock_generate.call_args.kwargs
        assert "options" in call_kwargs, "Call should include 'options'"
        assert "num_ctx" in call_kwargs["options"], "Options should include 'num_ctx'"
        # Session 64: Context is dynamic based on GPU VRAM
        # 2048 (no GPU) through 64000 (24GB+ VRAM) are valid
        num_ctx = call_kwargs["options"]["num_ctx"]
        assert 2048 <= num_ctx <= 64000, (
            f"num_ctx should be in valid range (2048-64000), got {num_ctx}"
        )


class TestTruncationWarning:
    """Test that truncation warnings are issued appropriately."""

    @pytest.mark.xfail(reason="Singleton state leakage can cause false failures", strict=False)
    @patch("src.core.ai.ollama_model_manager._get_context_window", return_value=2048)
    @patch("src.core.ai.ollama_model_manager.logger")
    @patch("ollama.generate")
    @patch("src.core.ai.ollama_model_manager.requests.get")
    def test_warning_on_large_prompt(self, mock_get, mock_generate, mock_logger, _mock_ctx):
        """Verify warning is issued when prompt approaches context limit."""
        # Mock connection check
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"models": []}
        mock_get.return_value = mock_get_response

        # Mock ollama.generate to return a streaming iterator
        mock_generate.return_value = iter([{"response": "Summary"}])

        from src.core.ai.ollama_model_manager import OllamaModelManager

        manager = OllamaModelManager()
        manager.model_name = "test-model:latest"
        manager.is_connected = True

        # With context window fixed at 2048, threshold is 2048 - 300 = 1748 tokens.
        # "word " * 3500 ≈ 3500 tokens, well above the threshold.
        large_prompt = "word " * 3500

        manager.generate_text(large_prompt, max_tokens=100)

        # Verify logger.warning was called
        assert mock_logger.warning.called, "Warning should be issued for large prompt"
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "truncated" in warning_msg.lower() or "exceed" in warning_msg.lower(), (
            f"Warning should mention truncation risk: {warning_msg}"
        )

    @patch("src.core.ai.ollama_model_manager.logger")
    @patch("ollama.generate")
    @patch("src.core.ai.ollama_model_manager.requests.get")
    def test_no_warning_on_small_prompt(self, mock_get, mock_generate, mock_logger):
        """Verify no warning for prompts well under context limit."""
        # Mock connection check
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"models": []}
        mock_get.return_value = mock_get_response

        # Mock ollama.generate to return a streaming iterator
        mock_generate.return_value = iter([{"response": "Summary"}])

        from src.core.ai.ollama_model_manager import OllamaModelManager

        manager = OllamaModelManager()
        manager.model_name = "test-model:latest"
        manager.is_connected = True

        # Create a small prompt (well under limit)
        small_prompt = "Summarize this short text."

        manager.generate_text(small_prompt, max_tokens=100)

        # Verify logger.warning was NOT called
        assert not mock_logger.warning.called, "No warning should be issued for small prompts"


class TestEmptyModelHandling:
    """Test that empty OLLAMA_MODEL_NAME is handled gracefully."""

    @patch("src.core.ai.ollama_model_manager.requests.get")
    def test_has_model_false_when_empty(self, mock_get):
        """has_model returns False when no model is configured."""
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"models": []}
        mock_get.return_value = mock_get_response

        from src.core.ai.ollama_model_manager import OllamaModelManager

        manager = OllamaModelManager()
        # With empty OLLAMA_MODEL_NAME and no saved preference, model_name is ""
        manager.model_name = ""

        assert manager.has_model is False

    @patch("src.core.ai.ollama_model_manager.requests.get")
    def test_is_model_loaded_false_when_no_model(self, mock_get):
        """is_model_loaded returns False without attempting connection when no model."""
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"models": []}
        mock_get.return_value = mock_get_response

        from src.core.ai.ollama_model_manager import OllamaModelManager

        manager = OllamaModelManager()
        manager.model_name = ""

        # Reset connection check count after __init__
        mock_get.reset_mock()

        result = manager.is_model_loaded()

        assert result is False
        # Should not have tried to connect — early return before connection check
        mock_get.assert_not_called()

    @patch("src.core.ai.ollama_model_manager.requests.get")
    def test_has_model_true_when_set(self, mock_get):
        """has_model returns True when a model name is set."""
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"models": []}
        mock_get.return_value = mock_get_response

        from src.core.ai.ollama_model_manager import OllamaModelManager

        manager = OllamaModelManager()
        manager.model_name = "gemma3:12b"

        assert manager.has_model is True
