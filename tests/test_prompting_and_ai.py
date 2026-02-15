"""Tests for prompting/AI: template manager, adapters, prompt formatter, post-processor, focus."""

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# PromptTemplateManager
# ---------------------------------------------------------------------------


VALID_TEMPLATE = (
    "<|system|>\nYou are a legal summarizer.\n<|end|>\n"
    "<|user|>\nSummarize in {min_words}-{max_words} words (range {max_words_range}):\n"
    "{case_text}\n<|end|>\n<|assistant|>\n"
)


class TestPromptTemplateManager:
    """PromptTemplateManager loads and formats prompt templates."""

    def _make(self, tmp_path):
        from src.core.prompting.template_manager import PromptTemplateManager

        return PromptTemplateManager(prompts_base_dir=tmp_path)

    def test_creation(self, tmp_path):
        mgr = self._make(tmp_path)
        assert mgr is not None
        assert mgr.prompts_base_dir == tmp_path

    def test_get_available_models_empty_dir(self, tmp_path):
        mgr = self._make(tmp_path)
        models = mgr.get_available_models()
        assert models == []

    def test_get_available_models_with_subdirs(self, tmp_path):
        (tmp_path / "phi-3-mini").mkdir()
        (tmp_path / "phi-3-mini" / "factual-summary.txt").write_text("template")
        mgr = self._make(tmp_path)
        models = mgr.get_available_models()
        assert "phi-3-mini" in models

    def test_get_available_presets(self, tmp_path):
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        (model_dir / "factual-summary.txt").write_text("template content")
        (model_dir / "narrative.txt").write_text("template content")

        mgr = self._make(tmp_path)
        presets = mgr.get_available_presets("test-model")
        assert len(presets) == 2
        ids = [p["id"] for p in presets]
        assert "factual-summary" in ids
        assert "narrative" in ids

    def test_underscore_files_excluded(self, tmp_path):
        """Files starting with _ are excluded from presets."""
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        (model_dir / "factual-summary.txt").write_text("template")
        (model_dir / "_template.txt").write_text("helper template")
        (model_dir / "_README.txt").write_text("instructions")

        mgr = self._make(tmp_path)
        presets = mgr.get_available_presets("test-model")
        ids = [p["id"] for p in presets]
        assert "_template" not in ids
        assert "_README" not in ids
        assert "factual-summary" in ids

    def test_load_template(self, tmp_path):
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        (model_dir / "factual-summary.txt").write_text(VALID_TEMPLATE)

        mgr = self._make(tmp_path)
        template = mgr.load_template("test-model", "factual-summary")
        assert "legal summarizer" in template

    def test_load_template_caching(self, tmp_path):
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        (model_dir / "test.txt").write_text(VALID_TEMPLATE)

        mgr = self._make(tmp_path)
        t1 = mgr.load_template("test-model", "test")
        t2 = mgr.load_template("test-model", "test")
        assert t1 == t2

    def test_clear_cache(self, tmp_path):
        mgr = self._make(tmp_path)
        mgr._cache["key"] = "value"
        mgr.clear_cache()
        assert len(mgr._cache) == 0

    def test_format_template(self, tmp_path):
        mgr = self._make(tmp_path)
        template = "Summarize in {min_words}-{max_words} words:\n{case_text}"
        result = mgr.format_template(
            template,
            min_words=100,
            max_words=200,
            max_words_range=50,
            case_text="The plaintiff filed a lawsuit.",
        )
        assert "100" in result
        assert "200" in result
        assert "The plaintiff filed a lawsuit." in result

    def test_get_default_preset_id(self, tmp_path):
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        (model_dir / "alpha.txt").write_text("t")
        (model_dir / "beta.txt").write_text("t")

        mgr = self._make(tmp_path)
        preset = mgr.get_default_preset_id("test-model")
        assert preset is not None

    def test_get_default_preset_id_no_presets(self, tmp_path):
        mgr = self._make(tmp_path)
        preset = mgr.get_default_preset_id("nonexistent")
        assert preset is None

    def test_user_prompts_override(self, tmp_path):
        """User prompts directory overrides built-in."""
        builtin_dir = tmp_path / "builtin"
        user_dir = tmp_path / "user"
        model_builtin = builtin_dir / "test-model"
        model_user = user_dir / "test-model"
        model_builtin.mkdir(parents=True)
        model_user.mkdir(parents=True)
        (model_builtin / "preset.txt").write_text(
            VALID_TEMPLATE.replace("legal summarizer", "built-in")
        )
        (model_user / "preset.txt").write_text(
            VALID_TEMPLATE.replace("legal summarizer", "user override")
        )

        from src.core.prompting.template_manager import PromptTemplateManager

        mgr = PromptTemplateManager(prompts_base_dir=builtin_dir, user_prompts_dir=user_dir)
        template = mgr.load_template("test-model", "preset")
        assert "user override" in template


# ---------------------------------------------------------------------------
# prompt_formatter (wrap_prompt_for_model)
# ---------------------------------------------------------------------------


class TestPromptFormatter:
    """wrap_prompt_for_model wraps prompts for model-specific formats."""

    def _wrap(self, model_name, prompt):
        from src.core.ai.prompt_formatter import wrap_prompt_for_model

        return wrap_prompt_for_model(model_name, prompt)

    def test_llama_format(self):
        result = self._wrap("llama2:7b", "Hello")
        assert "[INST]" in result
        assert "[/INST]" in result
        assert "Hello" in result

    def test_mistral_format(self):
        result = self._wrap("mistral:7b", "Hello")
        assert "[INST]" in result

    def test_gemma_raw(self):
        result = self._wrap("gemma:7b", "Hello")
        assert result == "Hello"

    def test_neural_chat_format(self):
        result = self._wrap("neural-chat:7b", "Hello")
        assert "### User:" in result
        assert "### Assistant:" in result

    def test_unknown_model_passthrough(self):
        result = self._wrap("some-unknown-model:latest", "Hello")
        assert "Hello" in result

    def test_case_insensitive(self):
        result = self._wrap("Llama2:7B", "Hello")
        assert "[INST]" in result


# ---------------------------------------------------------------------------
# SummaryPostProcessor
# ---------------------------------------------------------------------------


class TestSummaryPostProcessor:
    """SummaryPostProcessor enforces summary length limits."""

    def _make(self, generate_fn=None):
        from src.core.ai.summary_post_processor import SummaryPostProcessor

        if generate_fn is None:
            generate_fn = MagicMock(return_value="Short summary.")
        return SummaryPostProcessor(generate_text_fn=generate_fn, tolerance=0.2, max_attempts=2)

    def test_creation(self):
        proc = self._make()
        assert proc is not None

    def test_get_word_count(self):
        proc = self._make()
        assert proc.get_word_count("one two three") == 3
        assert proc.get_word_count("") == 0

    def test_is_within_tolerance_under(self):
        proc = self._make()
        # 100 words, target 100 -> within 20% tolerance
        text = " ".join(["word"] * 100)
        assert proc.is_within_tolerance(text, 100) is True

    def test_is_within_tolerance_over(self):
        proc = self._make()
        # 200 words, target 100 -> 100% over, exceeds 20% tolerance
        text = " ".join(["word"] * 200)
        assert proc.is_within_tolerance(text, 100) is False

    def test_enforce_length_within_tolerance(self):
        proc = self._make()
        text = " ".join(["word"] * 100)
        result = proc.enforce_length(text, 100)
        # Should return as-is since within tolerance
        assert result == text

    def test_enforce_length_calls_generate(self):
        """When over tolerance, should call generate to condense."""
        condensed = " ".join(["condensed"] * 50)
        gen_fn = MagicMock(return_value=condensed)
        proc = self._make(generate_fn=gen_fn)

        long_text = " ".join(["word"] * 200)
        result = proc.enforce_length(long_text, 100)
        gen_fn.assert_called()


# ---------------------------------------------------------------------------
# FocusExtractor
# ---------------------------------------------------------------------------


class TestFocusExtractorABC:
    """FocusExtractor abstract base class."""

    def test_abc_cannot_instantiate(self):
        from src.core.prompting.focus_extractor import FocusExtractor

        with pytest.raises(TypeError):
            FocusExtractor()

    def test_subclass_must_implement_extract_focus(self):
        from src.core.prompting.focus_extractor import FocusExtractor

        class BadExtractor(FocusExtractor):
            pass

        with pytest.raises(TypeError):
            BadExtractor()


class TestAIFocusExtractor:
    """AIFocusExtractor uses AI to extract focus areas."""

    def _make(self):
        from src.core.prompting.focus_extractor import AIFocusExtractor

        mock_mgr = MagicMock()
        mock_mgr.generate_text.return_value = (
            "EMPHASIS: injuries, timeline, damages\n"
            "INSTRUCTIONS:\n"
            "1. Identify all parties\n"
            "2. Note key injuries\n"
            "3. Document timeline\n"
        )
        return AIFocusExtractor(model_manager=mock_mgr)

    def test_creation_requires_model_manager(self):
        from src.core.prompting.focus_extractor import AIFocusExtractor

        with pytest.raises((ValueError, TypeError)):
            AIFocusExtractor(model_manager=None)

    def test_extract_focus_returns_dict(self):
        extractor = self._make()
        result = extractor.extract_focus("template text", "preset-1")
        assert "emphasis" in result
        assert "instructions" in result

    def test_extract_focus_cached(self):
        extractor = self._make()
        r1 = extractor.extract_focus("same template", "p1")
        r2 = extractor.extract_focus("same template", "p1")
        # Second call should use cache (same template content)
        assert r1 == r2

    def test_clear_cache(self):
        from src.core.prompting.focus_extractor import AIFocusExtractor

        AIFocusExtractor.clear_cache()
        # Should not raise

    def test_generic_fallback(self):
        extractor = self._make()
        result = extractor._generic_fallback()
        assert "emphasis" in result
        assert "instructions" in result
        assert len(result["emphasis"]) > 0


# ---------------------------------------------------------------------------
# StagePromptBuilder / MultiDocStagePromptBuilder
# ---------------------------------------------------------------------------


class TestStagePromptBuilderABC:
    """StagePromptBuilder is abstract."""

    def test_abc_cannot_instantiate(self):
        from src.core.prompting.adapters import StagePromptBuilder

        with pytest.raises(TypeError):
            StagePromptBuilder()


class TestMultiDocStagePromptBuilder:
    """MultiDocStagePromptBuilder creates stage-specific prompts."""

    def _make(self):
        from src.core.prompting.adapters import MultiDocStagePromptBuilder

        mock_template_mgr = MagicMock()
        mock_template_mgr.load_template.return_value = "A template about {case_text}"
        mock_model_mgr = MagicMock()
        mock_model_mgr.generate_text.return_value = (
            "EMPHASIS: key facts, damages\nINSTRUCTIONS:\n1. Find facts\n2. Note damages\n"
        )
        return MultiDocStagePromptBuilder(
            template_manager=mock_template_mgr, model_manager=mock_model_mgr
        )

    def test_creation(self):
        builder = self._make()
        assert builder is not None

    def test_create_chunk_prompt(self):
        builder = self._make()
        prompt = builder.create_chunk_prompt(
            preset_id="factual-summary",
            model_name="phi-3-mini",
            global_context="Case about injury",
            local_context="Previous section summary",
            chunk_text="The plaintiff slipped and fell.",
            max_words=200,
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_create_document_final_prompt(self):
        builder = self._make()
        prompt = builder.create_document_final_prompt(
            preset_id="factual-summary",
            model_name="phi-3-mini",
            chunk_summaries="Summary of section 1.\nSummary of section 2.",
            filename="deposition.pdf",
            max_words=500,
        )
        assert isinstance(prompt, str)
        assert "deposition.pdf" in prompt

    def test_create_meta_summary_prompt(self):
        builder = self._make()
        prompt = builder.create_meta_summary_prompt(
            preset_id="factual-summary",
            model_name="phi-3-mini",
            formatted_summaries="Doc1: summary\nDoc2: summary",
            max_words=1000,
            doc_count=2,
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_clear_cache(self):
        builder = self._make()
        builder.clear_cache()
        assert len(builder._focus_cache) == 0
