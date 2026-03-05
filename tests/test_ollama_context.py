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

    @patch("src.core.ai.ollama_model_manager._create_ollama_client")
    def test_num_ctx_in_payload(self, mock_create_client):
        """Verify num_ctx is included in Ollama API calls."""
        # Mock ollama client with generate returning a streaming iterator
        mock_client = MagicMock()
        mock_client.generate.return_value = iter([{"response": "Test summary"}])
        mock_client.list.return_value = MagicMock(models=[])
        mock_create_client.return_value = mock_client

        from src.core.ai.ollama_model_manager import OllamaModelManager

        manager = OllamaModelManager()
        manager.model_name = "test-model:latest"
        manager.is_connected = True

        manager.generate_text("Test prompt", max_tokens=100)

        # Check that client.generate was called with num_ctx in options
        assert mock_client.generate.called, "client.generate should have been called"
        call_kwargs = mock_client.generate.call_args.kwargs
        assert "options" in call_kwargs, "Call should include 'options'"
        assert "num_ctx" in call_kwargs["options"], "Options should include 'num_ctx'"
        # Context is dynamic based on GPU VRAM
        # 2048 (no GPU) through 64000 (24GB+ VRAM) are valid
        num_ctx = call_kwargs["options"]["num_ctx"]
        assert 2048 <= num_ctx <= 64000, (
            f"num_ctx should be in valid range (2048-64000), got {num_ctx}"
        )


class TestTruncationWarning:
    """Test that truncation warnings are issued appropriately."""

    @patch("src.core.ai.ollama_model_manager._get_context_window", return_value=2048)
    @patch("src.core.ai.ollama_model_manager.logger")
    @patch("src.core.ai.ollama_model_manager._create_ollama_client")
    def test_warning_on_large_prompt(self, mock_create_client, mock_logger, _mock_ctx):
        """Verify warning is issued when prompt approaches context limit."""
        # Mock ollama client with generate returning a streaming iterator
        mock_client = MagicMock()
        mock_client.generate.return_value = iter([{"response": "Summary"}])
        mock_client.list.return_value = MagicMock(models=[])
        mock_create_client.return_value = mock_client

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
    @patch("src.core.ai.ollama_model_manager._create_ollama_client")
    def test_no_warning_on_small_prompt(self, mock_create_client, mock_logger):
        """Verify no warning for prompts well under context limit."""
        # Mock ollama client with generate returning a streaming iterator
        mock_client = MagicMock()
        mock_client.generate.return_value = iter([{"response": "Summary"}])
        mock_client.list.return_value = MagicMock(models=[])
        mock_create_client.return_value = mock_client

        from src.core.ai.ollama_model_manager import OllamaModelManager

        manager = OllamaModelManager()
        manager.model_name = "test-model:latest"
        manager.is_connected = True

        # Create a small prompt (well under limit)
        small_prompt = "Summarize this short text."

        manager.generate_text(small_prompt, max_tokens=100)

        # Verify logger.warning was NOT called
        assert not mock_logger.warning.called, "No warning should be issued for small prompts"


class TestStreamReadTimeoutConstants:
    """Test that heartbeat timeout constants are defined correctly."""

    def test_gpu_timeout_is_300(self):
        """GPU read timeout should be 300 seconds (5 minutes)."""
        from src.config import OLLAMA_STREAM_READ_TIMEOUT_GPU

        assert OLLAMA_STREAM_READ_TIMEOUT_GPU == 300

    def test_cpu_timeout_is_900(self):
        """CPU read timeout should be 900 seconds (15 minutes)."""
        from src.config import OLLAMA_STREAM_READ_TIMEOUT_CPU

        assert OLLAMA_STREAM_READ_TIMEOUT_CPU == 900


class TestCreateOllamaClient:
    """Test GPU-aware client creation with correct httpx timeouts."""

    @patch("src.core.utils.gpu_detector.has_dedicated_gpu", return_value=True)
    def test_gpu_detected_uses_gpu_timeout(self, _mock_gpu):
        """When GPU is detected, read timeout should be 300s (5 min)."""
        from src.core.ai.ollama_model_manager import _create_ollama_client

        client = _create_ollama_client()
        assert client._client.timeout.read == 300.0

    @patch("src.core.utils.gpu_detector.has_dedicated_gpu", return_value=False)
    def test_no_gpu_uses_cpu_timeout(self, _mock_gpu):
        """When no GPU detected, read timeout should be 900s (15 min)."""
        from src.core.ai.ollama_model_manager import _create_ollama_client

        client = _create_ollama_client()
        assert client._client.timeout.read == 900.0

    @patch("src.core.utils.gpu_detector.has_dedicated_gpu", return_value=True)
    def test_connect_write_pool_timeouts(self, _mock_gpu):
        """Connect, write, and pool timeouts should all be 15 seconds."""
        from src.core.ai.ollama_model_manager import _create_ollama_client

        client = _create_ollama_client()
        timeout = client._client.timeout
        assert timeout.connect == 15.0
        assert timeout.write == 15.0
        assert timeout.pool == 15.0


class TestReadTimeoutHandling:
    """Test that httpx.ReadTimeout is caught and re-raised as RuntimeError."""

    @patch("src.core.ai.ollama_model_manager._create_ollama_client")
    def test_generate_text_catches_read_timeout(self, mock_create_client):
        """generate_text raises RuntimeError with clear message on ReadTimeout."""
        import httpx

        # Mock client that raises ReadTimeout on generate
        mock_client = MagicMock()
        mock_client.generate.side_effect = httpx.ReadTimeout("read timed out")
        mock_client._client.timeout.read = 300.0
        mock_client.list.return_value = MagicMock(models=[])
        mock_create_client.return_value = mock_client

        from src.core.ai.ollama_model_manager import OllamaModelManager

        manager = OllamaModelManager()
        manager.model_name = "test-model:latest"
        manager.is_connected = True

        with pytest.raises(RuntimeError, match="Ollama stopped responding after 5 minutes"):
            manager.generate_text("Test prompt", max_tokens=100)

    @patch("src.core.ai.ollama_model_manager._create_ollama_client")
    def test_generate_structured_catches_read_timeout(self, mock_create_client):
        """generate_structured raises RuntimeError with clear message on ReadTimeout."""
        import httpx

        # Mock client that raises ReadTimeout on generate
        mock_client = MagicMock()
        mock_client.generate.side_effect = httpx.ReadTimeout("read timed out")
        mock_client._client.timeout.read = 900.0
        mock_client.list.return_value = MagicMock(models=[])
        mock_create_client.return_value = mock_client

        from src.core.ai.ollama_model_manager import OllamaModelManager

        manager = OllamaModelManager()
        manager.model_name = "test-model:latest"
        manager.is_connected = True

        with pytest.raises(RuntimeError, match="Ollama stopped responding after 15 minutes"):
            manager.generate_structured("Return JSON", max_tokens=100)

    @patch("src.core.ai.ollama_model_manager._create_ollama_client")
    def test_timeout_message_includes_crash_hint(self, mock_create_client):
        """Error message should suggest crash or OOM as possible cause."""
        import httpx

        mock_client = MagicMock()
        mock_client.generate.side_effect = httpx.ReadTimeout("read timed out")
        mock_client._client.timeout.read = 300.0
        mock_client.list.return_value = MagicMock(models=[])
        mock_create_client.return_value = mock_client

        from src.core.ai.ollama_model_manager import OllamaModelManager

        manager = OllamaModelManager()
        manager.model_name = "test-model:latest"
        manager.is_connected = True

        with pytest.raises(RuntimeError, match="crashed or run out of memory"):
            manager.generate_text("Test prompt", max_tokens=100)


class TestEmptyModelHandling:
    """Test that empty OLLAMA_MODEL_NAME is handled gracefully."""

    @patch("src.core.ai.ollama_model_manager._create_ollama_client")
    def test_has_model_false_when_empty(self, mock_create_client):
        """has_model returns False when no model is configured."""
        mock_client = MagicMock()
        mock_client.list.return_value = MagicMock(models=[])
        mock_create_client.return_value = mock_client

        from src.core.ai.ollama_model_manager import OllamaModelManager

        manager = OllamaModelManager()
        # With empty OLLAMA_MODEL_NAME and no saved preference, model_name is ""
        manager.model_name = ""

        assert manager.has_model is False

    @patch("src.core.ai.ollama_model_manager._create_ollama_client")
    def test_is_model_loaded_false_when_no_model(self, mock_create_client):
        """is_model_loaded returns False without attempting connection when no model."""
        mock_client = MagicMock()
        mock_client.list.return_value = MagicMock(models=[])
        mock_create_client.return_value = mock_client

        from src.core.ai.ollama_model_manager import OllamaModelManager

        manager = OllamaModelManager()
        manager.model_name = ""

        # Reset list call count after __init__
        mock_client.list.reset_mock()

        result = manager.is_model_loaded()

        assert result is False
        # Should not have tried to connect — early return before connection check
        mock_client.list.assert_not_called()

    @patch("src.core.ai.ollama_model_manager._create_ollama_client")
    def test_has_model_true_when_set(self, mock_create_client):
        """has_model returns True when a model name is set."""
        mock_client = MagicMock()
        mock_client.list.return_value = MagicMock(models=[])
        mock_create_client.return_value = mock_client

        from src.core.ai.ollama_model_manager import OllamaModelManager

        manager = OllamaModelManager()
        manager.model_name = "gemma3:12b"

        assert manager.has_model is True


# ---------------------------------------------------------------------------
# unload_model — actual VRAM release via keep_alive=0
# ---------------------------------------------------------------------------


class TestUnloadModel:
    """unload_model() sends keep_alive=0 to free VRAM."""

    def _make_manager(self, mock_create_client):
        mock_client = MagicMock()
        mock_client.list.return_value = MagicMock(models=[])
        mock_create_client.return_value = mock_client

        from src.core.ai.ollama_model_manager import OllamaModelManager

        manager = OllamaModelManager()
        manager.model_name = "gemma3:12b"
        manager.is_connected = True
        return manager, mock_client

    @patch("src.core.ai.ollama_model_manager._create_ollama_client")
    def test_unload_calls_generate_with_keep_alive_zero(self, mock_create_client):
        """unload_model sends keep_alive=0 to Ollama to free VRAM."""
        manager, mock_client = self._make_manager(mock_create_client)

        manager.unload_model()

        mock_client.generate.assert_called_once()
        call_kwargs = mock_client.generate.call_args.kwargs
        assert call_kwargs.get("keep_alive") == 0
        assert call_kwargs.get("model") == "gemma3:12b"

    @patch("src.core.ai.ollama_model_manager._create_ollama_client")
    def test_unload_skips_when_no_model(self, mock_create_client):
        """unload_model does nothing when model_name is empty."""
        manager, mock_client = self._make_manager(mock_create_client)
        manager.model_name = ""
        mock_client.generate.reset_mock()

        manager.unload_model()

        mock_client.generate.assert_not_called()

    @patch("src.core.ai.ollama_model_manager._create_ollama_client")
    def test_unload_skips_when_not_connected(self, mock_create_client):
        """unload_model does nothing when Ollama is not connected."""
        manager, mock_client = self._make_manager(mock_create_client)
        manager.is_connected = False
        mock_client.generate.reset_mock()

        manager.unload_model()

        mock_client.generate.assert_not_called()

    @patch("src.core.ai.ollama_model_manager._create_ollama_client")
    def test_unload_swallows_exceptions(self, mock_create_client):
        """unload_model does not raise if generate fails (non-critical)."""
        manager, mock_client = self._make_manager(mock_create_client)
        mock_client.generate.side_effect = Exception("connection reset")

        # Should not raise
        manager.unload_model()


# ---------------------------------------------------------------------------
# ollama.ResponseError handling in generate_text / generate_structured
# ---------------------------------------------------------------------------


class TestResponseErrorHandling:
    """ollama.ResponseError is caught and re-raised as RuntimeError."""

    def _make_manager(self, mock_create_client):
        mock_client = MagicMock()
        mock_client.list.return_value = MagicMock(models=[])
        mock_create_client.return_value = mock_client

        from src.core.ai.ollama_model_manager import OllamaModelManager

        manager = OllamaModelManager()
        manager.model_name = "gemma3:12b"
        manager.is_connected = True
        return manager, mock_client

    @patch("src.core.ai.ollama_model_manager._create_ollama_client")
    def test_generate_text_converts_response_error(self, mock_create_client):
        """generate_text wraps ollama.ResponseError as RuntimeError."""
        import ollama

        manager, mock_client = self._make_manager(mock_create_client)
        mock_client.generate.side_effect = ollama.ResponseError("model not found", status_code=404)

        with pytest.raises(RuntimeError, match="404"):
            manager.generate_text("Test prompt", max_tokens=50)

    @patch("src.core.ai.ollama_model_manager._create_ollama_client")
    def test_generate_text_response_error_message_includes_error_text(self, mock_create_client):
        """RuntimeError message includes the original error text from Ollama."""
        import ollama

        manager, mock_client = self._make_manager(mock_create_client)
        mock_client.generate.side_effect = ollama.ResponseError("model not found", status_code=404)

        with pytest.raises(RuntimeError, match="model not found"):
            manager.generate_text("Test prompt", max_tokens=50)

    @patch("src.core.ai.ollama_model_manager._create_ollama_client")
    def test_generate_structured_converts_response_error(self, mock_create_client):
        """generate_structured wraps ollama.ResponseError as RuntimeError."""
        import ollama

        manager, mock_client = self._make_manager(mock_create_client)
        mock_client.generate.side_effect = ollama.ResponseError("model overloaded", status_code=503)

        with pytest.raises(RuntimeError, match="503"):
            manager.generate_structured("Return JSON", max_tokens=100)

    @patch("src.core.ai.ollama_model_manager._create_ollama_client")
    def test_generate_structured_response_error_not_swallowed(self, mock_create_client):
        """generate_structured raises (not returns None) on ResponseError."""
        import ollama

        manager, mock_client = self._make_manager(mock_create_client)
        mock_client.generate.side_effect = ollama.ResponseError("model overloaded", status_code=503)

        # Must raise, not silently return None like generic exceptions
        with pytest.raises(RuntimeError):
            manager.generate_structured("Return JSON", max_tokens=100)


# ---------------------------------------------------------------------------
# generate_structured schema param
# ---------------------------------------------------------------------------


class TestGenerateStructuredSchema:
    """generate_structured() passes schema dict as format when provided."""

    def _make_manager(self, mock_create_client):
        mock_client = MagicMock()
        mock_client.list.return_value = MagicMock(models=[])
        mock_client.generate.return_value = MagicMock(
            get=lambda k, d="": '{"vocabulary": []}' if k == "response" else d
        )
        mock_create_client.return_value = mock_client

        from src.core.ai.ollama_model_manager import OllamaModelManager

        manager = OllamaModelManager()
        manager.model_name = "gemma3:12b"
        manager.is_connected = True
        return manager, mock_client

    @patch("src.core.ai.ollama_model_manager._create_ollama_client")
    def test_schema_passed_as_format_when_provided(self, mock_create_client):
        """When schema is given, it is passed as format= to generate()."""
        manager, mock_client = self._make_manager(mock_create_client)

        schema = {"type": "object", "properties": {"terms": {"type": "array"}}}
        manager.generate_structured("Extract terms", schema=schema)

        call_kwargs = mock_client.generate.call_args.kwargs
        assert call_kwargs.get("format") == schema

    @patch("src.core.ai.ollama_model_manager._create_ollama_client")
    def test_format_json_used_when_no_schema(self, mock_create_client):
        """When schema is None, format='json' is used as fallback."""
        manager, mock_client = self._make_manager(mock_create_client)

        manager.generate_structured("Extract terms", schema=None)

        call_kwargs = mock_client.generate.call_args.kwargs
        assert call_kwargs.get("format") == "json"
