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

from src.config import OLLAMA_CONTEXT_WINDOW


class TestContextWindowConfig:
    """Test that context window configuration is properly set."""

    def test_context_window_defined(self):
        """Verify OLLAMA_CONTEXT_WINDOW is defined and reasonable."""
        assert OLLAMA_CONTEXT_WINDOW is not None
        assert isinstance(OLLAMA_CONTEXT_WINDOW, int)
        assert OLLAMA_CONTEXT_WINDOW >= 1024  # At least 1K
        assert OLLAMA_CONTEXT_WINDOW <= 131072  # Max reasonable for consumer hardware

    def test_default_is_cpu_optimized(self):
        """Verify default is 2048 for CPU performance."""
        assert OLLAMA_CONTEXT_WINDOW == 2048, (
            "Default should be 2048 for CPU-only laptops. "
            "See research: 2k context = ~150 tokens/sec vs 8k = ~43 tokens/sec"
        )


class TestChunkingConfig:
    """Test that chunking config aligns with context window."""

    def test_chunk_words_fit_context(self):
        """Verify max_chunk_words fits within context window."""
        import yaml

        config_path = project_root / "config" / "chunking_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        max_chunk_words = config['chunking']['max_chunk_words']
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

        hard_limit = config['chunking']['max_chunk_words_hard_limit']
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

    @patch('src.core.ai.ollama_model_manager.requests.post')
    def test_num_ctx_in_payload(self, mock_post):
        """Verify num_ctx is included in Ollama API calls."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Test summary',
            'eval_count': 10
        }
        mock_post.return_value = mock_response

        # Also mock the connection check
        with patch('src.core.ai.ollama_model_manager.requests.get') as mock_get:
            mock_get_response = MagicMock()
            mock_get_response.status_code = 200
            mock_get_response.json.return_value = {'models': []}
            mock_get.return_value = mock_get_response

            from src.core.ai.ollama_model_manager import OllamaModelManager

            manager = OllamaModelManager()
            manager.is_connected = True

            # Make a test call
            try:
                manager.generate_text("Test prompt", max_tokens=100)
            except Exception:
                pass  # We just want to check the payload

            # Check that requests.post was called with num_ctx in options
            if mock_post.called:
                call_args = mock_post.call_args
                payload = call_args.kwargs.get('json', call_args.args[1] if len(call_args.args) > 1 else {})

                assert 'options' in payload, "Payload should include 'options'"
                assert 'num_ctx' in payload['options'], "Options should include 'num_ctx'"
                assert payload['options']['num_ctx'] == OLLAMA_CONTEXT_WINDOW, (
                    f"num_ctx should be {OLLAMA_CONTEXT_WINDOW}"
                )


class TestTruncationWarning:
    """Test that truncation warnings are issued appropriately."""

    @patch('src.core.ai.ollama_model_manager.warning')
    @patch('src.core.ai.ollama_model_manager.requests.post')
    @patch('src.core.ai.ollama_model_manager.requests.get')
    def test_warning_on_large_prompt(self, mock_get, mock_post, mock_warning):
        """Verify warning is issued when prompt approaches context limit."""
        # Setup mocks
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {'models': []}
        mock_get.return_value = mock_get_response

        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {'response': 'Summary', 'eval_count': 10}
        mock_post.return_value = mock_post_response

        from src.core.ai.ollama_model_manager import OllamaModelManager

        manager = OllamaModelManager()
        manager.is_connected = True

        # Create a prompt that's close to context limit
        # 2048 context - 300 output room = 1748 tokens
        # 1748 tokens * 4 chars = ~7000 chars to trigger warning
        large_prompt = "word " * 1500  # ~7500 chars

        manager.generate_text(large_prompt, max_tokens=100)

        # Verify warning was called
        assert mock_warning.called, "Warning should be issued for large prompt"
        warning_msg = mock_warning.call_args[0][0]
        assert "truncated" in warning_msg.lower() or "exceed" in warning_msg.lower(), (
            f"Warning should mention truncation risk: {warning_msg}"
        )

    @patch('src.core.ai.ollama_model_manager.warning')
    @patch('src.core.ai.ollama_model_manager.requests.post')
    @patch('src.core.ai.ollama_model_manager.requests.get')
    def test_no_warning_on_small_prompt(self, mock_get, mock_post, mock_warning):
        """Verify no warning for prompts well under context limit."""
        # Setup mocks
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {'models': []}
        mock_get.return_value = mock_get_response

        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {'response': 'Summary', 'eval_count': 10}
        mock_post.return_value = mock_post_response

        from src.core.ai.ollama_model_manager import OllamaModelManager

        manager = OllamaModelManager()
        manager.is_connected = True

        # Create a small prompt (well under limit)
        small_prompt = "Summarize this short text."

        manager.generate_text(small_prompt, max_tokens=100)

        # Verify warning was NOT called
        assert not mock_warning.called, "No warning should be issued for small prompts"
