"""
Shared test fixtures for the CasePrepd test suite.

Resets module-level singletons before every test so that no test
inherits leftover state from a previous test.
"""

import pytest


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all module-level singletons before and after each test."""
    _do_reset()
    yield
    _do_reset()


def _do_reset():
    """Clear cached singleton instances across the codebase."""
    # UserPreferencesManager -- holds user settings (context size, GPU, etc.)
    try:
        from src.user_preferences import reset_singleton as reset_prefs

        reset_prefs()
    except ImportError:
        pass

    # AIService -- singleton wrapper for Ollama / GPU operations
    try:
        from src.services.ai_service import AIService

        AIService.reset_singleton()
    except ImportError:
        pass
