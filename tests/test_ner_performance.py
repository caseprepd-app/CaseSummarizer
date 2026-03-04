"""
Tests for NER performance optimizations.

Validates:
- Unused spaCy pipeline components are disabled during NER extraction
- Chunk size default increased to 100KB
- Entity extraction still works with disabled components
"""

import inspect
from unittest.mock import MagicMock, patch

from src.core.vocabulary.algorithms.ner_algorithm import (
    _NER_DISABLED_COMPONENTS,
    NERAlgorithm,
)


class TestNERDisabledComponents:
    """Tests for the disabled pipeline components optimization."""

    def test_ner_disabled_components_constant(self):
        """Verify _NER_DISABLED_COMPONENTS contains exactly the expected components."""
        expected = ["tagger", "parser", "attribute_ruler", "lemmatizer", "topicrank"]
        assert expected == _NER_DISABLED_COMPONENTS

    def test_ner_pipe_disables_components(self):
        """Verify nlp.pipe() is called with disable= parameter."""
        mock_nlp = MagicMock()
        mock_nlp.pipe.return_value = iter([])  # No docs

        algo = NERAlgorithm(nlp=mock_nlp)
        algo.extract("Some test text for NER processing.")

        mock_nlp.pipe.assert_called_once()
        call_kwargs = mock_nlp.pipe.call_args
        assert "disable" in call_kwargs.kwargs or (len(call_kwargs.args) > 2), (
            "nlp.pipe() must be called with disable= parameter"
        )

        # Check the actual value
        disable_value = call_kwargs.kwargs.get("disable")
        assert disable_value == _NER_DISABLED_COMPONENTS

    def test_ner_extracts_entities_with_disabled_components(self):
        """Integration-style: extraction works when pipe returns docs with entities."""
        # Create a mock doc with entities and tokens
        mock_ent = MagicMock()
        mock_ent.label_ = "PERSON"
        mock_ent.text = "John Smith"
        mock_ent.start_char = 0
        mock_ent.end_char = 10
        mock_ent.start = 0
        mock_ent.end = 2

        mock_token = MagicMock()
        mock_token.text = "John"
        mock_token.i = 0
        mock_token.is_alpha = True
        mock_token.is_space = False
        mock_token.is_punct = False
        mock_token.is_digit = False
        mock_token.ent_type_ = "PERSON"

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_token]))
        mock_doc.__len__ = MagicMock(return_value=1)

        mock_nlp = MagicMock()
        mock_nlp.pipe.return_value = iter([mock_doc])

        algo = NERAlgorithm(nlp=mock_nlp)
        result = algo.extract("John Smith testified in court.")

        assert len(result.candidates) > 0
        assert any(c.term == "John Smith" for c in result.candidates)


class TestNERChunkSize:
    """Tests for the chunk size optimization."""

    def test_ner_chunk_size_default_100kb(self):
        """Verify _chunk_text default is 100KB."""
        sig = inspect.signature(NERAlgorithm._chunk_text)
        default = sig.parameters["chunk_size_kb"].default
        assert default == 100, f"Expected chunk_size_kb default=100, got {default}"

    def test_extract_uses_100kb_chunks(self):
        """Verify extract() passes 100KB to _chunk_text when no chunks provided."""
        mock_nlp = MagicMock()
        mock_nlp.pipe.return_value = iter([])

        algo = NERAlgorithm(nlp=mock_nlp)

        with patch.object(algo, "_chunk_text", return_value=["chunk1"]) as mock_chunk:
            algo.extract("test text")
            mock_chunk.assert_called_once_with("test text", chunk_size_kb=100)
