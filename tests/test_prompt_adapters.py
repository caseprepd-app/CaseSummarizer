"""
Tests for Prompt Focus Extraction and Stage Prompt Builder Modules

Tests the thread-through prompt template architecture:
- FocusExtractor ABC and AIFocusExtractor implementation
- StagePromptBuilder ABC and MultiDocStagePromptBuilder implementation
- Caching behavior for focus extraction
- Prompt generation with focus emphasis

Uses mock model_manager to avoid actual Ollama calls during testing.
"""

from unittest.mock import Mock

import pytest

from src.core.prompting import (
    AIFocusExtractor,
    FocusExtractor,
    MultiDocStagePromptBuilder,
    StagePromptBuilder,
)


class TestFocusExtractorABC:
    """Test the FocusExtractor abstract base class."""

    def test_focus_extractor_is_abstract(self):
        """FocusExtractor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FocusExtractor()

    def test_custom_extractor_must_implement_extract_focus(self):
        """Custom extractors must implement extract_focus method."""

        class IncompleteExtractor(FocusExtractor):
            pass

        with pytest.raises(TypeError):
            IncompleteExtractor()

    def test_custom_extractor_can_be_created(self):
        """Custom extractors with extract_focus work correctly."""

        class MockExtractor(FocusExtractor):
            def extract_focus(self, template: str, preset_id: str) -> dict:
                return {"emphasis": "test emphasis", "instructions": "1. Test instruction"}

        extractor = MockExtractor()
        result = extractor.extract_focus("template content", "test-preset")

        assert result["emphasis"] == "test emphasis"
        assert "Test instruction" in result["instructions"]


class TestAIFocusExtractor:
    """Test AIFocusExtractor implementation."""

    def test_requires_model_manager(self):
        """AIFocusExtractor requires a model_manager."""
        with pytest.raises(ValueError):
            AIFocusExtractor(model_manager=None)

    def test_extract_focus_calls_model(self):
        """extract_focus uses model_manager for AI extraction."""
        mock_model = Mock()
        mock_model.generate_text.return_value = """
EMPHASIS: injuries, medical treatment, timeline, damages
INSTRUCTIONS:
1. Identify all injuries mentioned
2. Note medical treatments
3. Track timeline of events
4. Calculate damages
"""
        extractor = AIFocusExtractor(model_manager=mock_model)
        # Clear class-level cache to ensure fresh extraction
        AIFocusExtractor.clear_cache()

        result = extractor.extract_focus("Test template content", "test-preset")

        assert "injuries" in result["emphasis"].lower()
        assert "1." in result["instructions"]
        mock_model.generate_text.assert_called_once()

    def test_caches_by_content_hash(self):
        """Results are cached by template content hash, not preset_id."""
        mock_model = Mock()
        mock_model.generate_text.return_value = """
EMPHASIS: test focus
INSTRUCTIONS:
1. Test instruction
"""
        extractor = AIFocusExtractor(model_manager=mock_model)
        AIFocusExtractor.clear_cache()

        # First call with same content, different preset_ids
        result1 = extractor.extract_focus("Same template content", "preset-a")
        result2 = extractor.extract_focus("Same template content", "preset-b")

        # Should only call model once (cached by content hash)
        assert mock_model.generate_text.call_count == 1
        assert result1 == result2

    def test_different_content_triggers_new_extraction(self):
        """Different template content triggers new AI extraction."""
        mock_model = Mock()
        mock_model.generate_text.return_value = "EMPHASIS: test\nINSTRUCTIONS:\n1. Test"

        extractor = AIFocusExtractor(model_manager=mock_model)
        AIFocusExtractor.clear_cache()

        extractor.extract_focus("Content A", "preset")
        extractor.extract_focus("Content B", "preset")

        # Should call model twice (different content hashes)
        assert mock_model.generate_text.call_count == 2

    def test_graceful_fallback_on_error(self):
        """Falls back to generic focus when AI extraction fails."""
        mock_model = Mock()
        mock_model.generate_text.side_effect = Exception("Model error")

        extractor = AIFocusExtractor(model_manager=mock_model)
        AIFocusExtractor.clear_cache()

        result = extractor.extract_focus("Test template", "test-preset")

        # Should return fallback values
        assert "key facts" in result["emphasis"].lower()
        assert "Synthesize" in result["instructions"]

    def test_parses_emphasis_correctly(self):
        """Correctly parses EMPHASIS line from AI response."""
        mock_model = Mock()
        mock_model.generate_text.return_value = """
EMPHASIS: medical records, injury severity, treatment costs, recovery timeline
INSTRUCTIONS:
1. Review all medical documentation
2. Assess injury severity
"""
        extractor = AIFocusExtractor(model_manager=mock_model)
        AIFocusExtractor.clear_cache()

        result = extractor.extract_focus("Template", "preset")

        assert "medical records" in result["emphasis"]
        assert "injury severity" in result["emphasis"]

    def test_parses_instructions_correctly(self):
        """Correctly parses INSTRUCTIONS block from AI response."""
        mock_model = Mock()
        mock_model.generate_text.return_value = """
EMPHASIS: test
INSTRUCTIONS:
1. First instruction
2. Second instruction
3. Third instruction
"""
        extractor = AIFocusExtractor(model_manager=mock_model)
        AIFocusExtractor.clear_cache()

        result = extractor.extract_focus("Template", "preset")

        assert "1. First instruction" in result["instructions"]
        assert "2. Second instruction" in result["instructions"]
        assert "3. Third instruction" in result["instructions"]


class TestStagePromptBuilderABC:
    """Test the StagePromptBuilder abstract base class."""

    def test_prompt_adapter_is_abstract(self):
        """StagePromptBuilder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            StagePromptBuilder()

    def test_must_implement_all_methods(self):
        """Custom adapters must implement all abstract methods."""

        class IncompleteAdapter(StagePromptBuilder):
            def create_chunk_prompt(self, *args, **kwargs):
                return "chunk"

            # Missing create_document_final_prompt and create_meta_summary_prompt

        with pytest.raises(TypeError):
            IncompleteAdapter()


class TestMultiDocStagePromptBuilder:
    """Test MultiDocStagePromptBuilder implementation."""

    @pytest.fixture
    def mock_template_manager(self):
        """Create a mock template manager."""
        manager = Mock()
        manager.load_template.return_value = """
Analyze this legal document focusing on:
- Injuries and medical treatment
- Timeline of events
- Damages claimed
"""
        return manager

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock model manager."""
        manager = Mock()
        manager.generate_text.return_value = """
EMPHASIS: injuries, medical treatment, timeline, damages
INSTRUCTIONS:
1. Identify all injuries
2. Note treatments
3. Track events
"""
        return manager

    @pytest.fixture
    def adapter(self, mock_template_manager, mock_model_manager):
        """Create adapter with mocks."""
        AIFocusExtractor.clear_cache()
        return MultiDocStagePromptBuilder(
            template_manager=mock_template_manager, model_manager=mock_model_manager
        )

    def test_create_chunk_prompt_includes_focus(self, adapter):
        """Chunk prompt includes extracted focus emphasis."""
        prompt = adapter.create_chunk_prompt(
            preset_id="injuries-focus",
            model_name="phi-3-mini",
            global_context="Document overview here",
            local_context="Previous section summary",
            chunk_text="The plaintiff reported severe back pain...",
            max_words=75,
        )

        assert "injuries" in prompt.lower()
        assert "back pain" in prompt
        assert "75" in prompt

    def test_create_document_final_prompt_includes_focus(self, adapter):
        """Document final prompt includes focus emphasis."""
        prompt = adapter.create_document_final_prompt(
            preset_id="injuries-focus",
            model_name="phi-3-mini",
            chunk_summaries="Summary 1\n\nSummary 2",
            filename="complaint.pdf",
            max_words=200,
        )

        assert "injuries" in prompt.lower()
        assert "complaint.pdf" in prompt
        assert "200" in prompt

    def test_create_meta_summary_prompt_includes_instructions(self, adapter):
        """Meta-summary prompt includes extracted instructions."""
        prompt = adapter.create_meta_summary_prompt(
            preset_id="injuries-focus",
            model_name="phi-3-mini",
            formatted_summaries="Doc 1 summary\n\nDoc 2 summary",
            max_words=500,
            doc_count=2,
        )

        assert "Identify all injuries" in prompt
        assert "2 documents" in prompt
        assert "Doc 1 summary" in prompt

    def test_caches_focus_per_preset_and_model(self, adapter, mock_model_manager):
        """Focus is cached per preset/model combination."""
        # First call
        adapter.create_chunk_prompt(
            preset_id="preset-a",
            model_name="model-1",
            global_context="",
            local_context="",
            chunk_text="Text",
            max_words=75,
        )

        # Same preset/model - should use cache
        adapter.create_chunk_prompt(
            preset_id="preset-a",
            model_name="model-1",
            global_context="",
            local_context="",
            chunk_text="Different text",
            max_words=75,
        )

        # Only called AI once for focus extraction (template load triggers this)
        # The generate_text is only called once per unique template content
        assert mock_model_manager.generate_text.call_count == 1

    def test_accepts_custom_focus_extractor(self, mock_template_manager, mock_model_manager):
        """Can inject a custom focus extractor."""

        class CustomExtractor(FocusExtractor):
            def extract_focus(self, template: str, preset_id: str) -> dict:
                return {"emphasis": "custom emphasis", "instructions": "1. Custom instruction"}

        adapter = MultiDocStagePromptBuilder(
            template_manager=mock_template_manager,
            model_manager=mock_model_manager,
            focus_extractor=CustomExtractor(),
        )

        prompt = adapter.create_meta_summary_prompt(
            preset_id="any", model_name="any", formatted_summaries="", max_words=100, doc_count=1
        )

        assert "Custom instruction" in prompt

    def test_clear_cache_resets_focus_cache(self, adapter, mock_model_manager):
        """clear_cache() removes cached focus entries."""
        # Generate and cache
        adapter.create_chunk_prompt(
            preset_id="test",
            model_name="model",
            global_context="",
            local_context="",
            chunk_text="Text",
            max_words=75,
        )

        # Clear and regenerate
        adapter.clear_cache()
        AIFocusExtractor.clear_cache()

        adapter.create_chunk_prompt(
            preset_id="test",
            model_name="model",
            global_context="",
            local_context="",
            chunk_text="Text",
            max_words=75,
        )

        # Should call AI twice (cache was cleared)
        assert mock_model_manager.generate_text.call_count == 2


class TestIntegrationImports:
    """Test that all components import correctly."""

    def test_focus_extractor_imports(self):
        """FocusExtractor components import correctly."""
        from src.core.prompting import AIFocusExtractor, FocusExtractor

        assert FocusExtractor is not None
        assert AIFocusExtractor is not None

    def test_prompt_adapter_imports(self):
        """StagePromptBuilder components import correctly."""
        from src.core.prompting import MultiDocStagePromptBuilder, StagePromptBuilder

        assert StagePromptBuilder is not None
        assert MultiDocStagePromptBuilder is not None

    def test_summarizer_accepts_adapter_params(self):
        """ProgressiveDocumentSummarizer accepts adapter parameters."""
        from src.core.summarization import ProgressiveDocumentSummarizer

        mock_model = Mock()
        mock_adapter = Mock()

        # Should not raise
        summarizer = ProgressiveDocumentSummarizer(
            model_manager=mock_model, prompt_adapter=mock_adapter, preset_id="test-preset"
        )

        assert summarizer.prompt_adapter == mock_adapter
        assert summarizer.preset_id == "test-preset"

    def test_orchestrator_accepts_adapter_params(self):
        """MultiDocumentOrchestrator accepts adapter parameters."""
        from src.core.summarization import MultiDocumentOrchestrator

        mock_model = Mock()
        mock_adapter = Mock()
        mock_summarizer = Mock()

        # Should not raise
        orchestrator = MultiDocumentOrchestrator(
            document_summarizer=mock_summarizer,
            model_manager=mock_model,
            prompt_adapter=mock_adapter,
            preset_id="test-preset",
        )

        assert orchestrator.prompt_adapter == mock_adapter
        assert orchestrator.preset_id == "test-preset"
