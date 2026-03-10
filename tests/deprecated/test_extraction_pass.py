"""
Tests for src.core.summarization.extraction_pass — two-pass extraction processor.

Covers:
- Basic extraction from chunks (mock LLM)
- Prompt format contains chunk text
- Redundant chunks skipped via ChunkScores
- Cancellation via stop_check
- String chunks and object chunks (.text attribute)
- Progress reporting callback
- Result count matches input count
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

from src.core.summarization.extraction_pass import (
    ExtractionPassProcessor,
)
from src.core.utils.chunk_scoring import ChunkScores


@dataclass
class FakeChunk:
    """Simulates a chunk object with .text attribute."""

    text: str


class TestExtractionPassBasic:
    """Test basic extraction functionality with mock LLM."""

    def test_extracts_from_all_chunks(self):
        """Should call LLM once per chunk and return one result per chunk."""
        mock_model = MagicMock()
        mock_model.generate_text.return_value = "CLAIMS: Negligence alleged"

        processor = ExtractionPassProcessor(model_manager=mock_model)
        chunks = [FakeChunk("chunk 1 text"), FakeChunk("chunk 2 text")]
        results = processor.extract_from_chunks(chunks)

        assert len(results) == 2
        assert mock_model.generate_text.call_count == 2

    def test_results_are_stripped(self):
        """LLM output whitespace should be stripped."""
        mock_model = MagicMock()
        mock_model.generate_text.return_value = "  FACTS: Injury on Jan 5  \n"

        processor = ExtractionPassProcessor(model_manager=mock_model)
        results = processor.extract_from_chunks([FakeChunk("text")])

        assert results[0] == "FACTS: Injury on Jan 5"

    def test_empty_chunks_list(self):
        """Empty input returns empty output."""
        mock_model = MagicMock()
        processor = ExtractionPassProcessor(model_manager=mock_model)
        results = processor.extract_from_chunks([])

        assert results == []
        mock_model.generate_text.assert_not_called()


class TestExtractionPassPromptFormat:
    """Test that prompts are correctly formatted."""

    def test_prompt_contains_chunk_text(self):
        """The prompt sent to LLM should contain the chunk's text."""
        mock_model = MagicMock()
        mock_model.generate_text.return_value = "extracted"

        processor = ExtractionPassProcessor(model_manager=mock_model)
        processor.extract_from_chunks([FakeChunk("The plaintiff alleges fraud.")])

        called_prompt = mock_model.generate_text.call_args[1]["prompt"]
        assert "The plaintiff alleges fraud." in called_prompt

    def test_prompt_contains_extraction_headers(self):
        """The prompt should contain CLAIMS, FACTS, RELIEF, TESTIMONY headers."""
        mock_model = MagicMock()
        mock_model.generate_text.return_value = "extracted"

        processor = ExtractionPassProcessor(model_manager=mock_model)
        processor.extract_from_chunks([FakeChunk("some text")])

        called_prompt = mock_model.generate_text.call_args[1]["prompt"]
        assert "CLAIMS:" in called_prompt
        assert "FACTS:" in called_prompt
        assert "RELIEF:" in called_prompt
        assert "TESTIMONY:" in called_prompt

    def test_max_tokens_set(self):
        """generate_text should be called with max_tokens=150."""
        mock_model = MagicMock()
        mock_model.generate_text.return_value = "extracted"

        processor = ExtractionPassProcessor(model_manager=mock_model)
        processor.extract_from_chunks([FakeChunk("text")])

        assert mock_model.generate_text.call_args[1]["max_tokens"] == 150


class TestExtractionPassRedundancySkipping:
    """Test that redundant chunks are skipped."""

    def test_redundant_chunk_skipped(self):
        """Chunks flagged as redundant should not call the LLM."""
        mock_model = MagicMock()
        mock_model.generate_text.return_value = "extracted"

        scores = ChunkScores(
            skip=[False, True, False],
            skip_reason=["", "redundant with chunk 1", ""],
        )
        processor = ExtractionPassProcessor(model_manager=mock_model, chunk_scores=scores)
        chunks = [FakeChunk("a"), FakeChunk("b"), FakeChunk("c")]
        results = processor.extract_from_chunks(chunks)

        assert len(results) == 3
        assert results[0] == "extracted"
        assert results[1] == "", "Redundant chunk should produce empty string"
        assert results[2] == "extracted"
        assert mock_model.generate_text.call_count == 2, (
            "LLM should only be called for non-redundant chunks"
        )

    def test_all_chunks_redundant(self):
        """If all chunks are redundant, no LLM calls should be made."""
        mock_model = MagicMock()
        scores = ChunkScores(
            skip=[True, True],
            skip_reason=["dup", "dup"],
        )
        processor = ExtractionPassProcessor(model_manager=mock_model, chunk_scores=scores)
        results = processor.extract_from_chunks([FakeChunk("a"), FakeChunk("b")])

        assert results == ["", ""]
        mock_model.generate_text.assert_not_called()


class TestExtractionPassCancellation:
    """Test stop_check cancellation."""

    def test_cancellation_stops_processing(self):
        """When stop_check returns True, remaining chunks get empty strings."""
        mock_model = MagicMock()
        mock_model.generate_text.return_value = "extracted"

        call_count = 0

        def stop_after_one():
            nonlocal call_count
            call_count += 1
            return call_count > 1  # Allow first chunk, cancel before second

        processor = ExtractionPassProcessor(model_manager=mock_model, stop_check=stop_after_one)
        chunks = [FakeChunk("a"), FakeChunk("b"), FakeChunk("c")]
        results = processor.extract_from_chunks(chunks)

        assert len(results) == 3, "Should return result for every chunk"
        assert results[0] == "extracted"
        assert results[1] == "", "Cancelled chunks should be empty"
        assert results[2] == "", "Cancelled chunks should be empty"

    def test_no_stop_check_processes_all(self):
        """Without stop_check, all chunks are processed."""
        mock_model = MagicMock()
        mock_model.generate_text.return_value = "ok"

        processor = ExtractionPassProcessor(model_manager=mock_model)
        results = processor.extract_from_chunks([FakeChunk("a"), FakeChunk("b"), FakeChunk("c")])
        assert all(r == "ok" for r in results)
        assert mock_model.generate_text.call_count == 3


class TestExtractionPassChunkTypes:
    """Test handling of different chunk types."""

    def test_string_chunks(self):
        """Plain strings should be handled (no .text attribute)."""
        mock_model = MagicMock()
        mock_model.generate_text.return_value = "extracted"

        processor = ExtractionPassProcessor(model_manager=mock_model)
        results = processor.extract_from_chunks(["plain string chunk"])

        assert len(results) == 1
        called_prompt = mock_model.generate_text.call_args[1]["prompt"]
        assert "plain string chunk" in called_prompt

    def test_object_chunks_with_text_attr(self):
        """Chunk objects with .text attribute should use that."""
        mock_model = MagicMock()
        mock_model.generate_text.return_value = "extracted"

        processor = ExtractionPassProcessor(model_manager=mock_model)
        results = processor.extract_from_chunks([FakeChunk("object text")])

        called_prompt = mock_model.generate_text.call_args[1]["prompt"]
        assert "object text" in called_prompt


class TestExtractionPassProgressReporting:
    """Test status_reporter callback."""

    def test_progress_callback_called(self):
        """status_reporter should be called with percent and message."""
        mock_model = MagicMock()
        mock_model.generate_text.return_value = "ok"
        reporter = MagicMock()

        processor = ExtractionPassProcessor(model_manager=mock_model)
        processor.extract_from_chunks([FakeChunk("a"), FakeChunk("b")], status_reporter=reporter)

        assert reporter.call_count == 2
        # First call: chunk 1/2, percent = 0
        first_call = reporter.call_args_list[0]
        assert first_call[0][0] == 0  # 0% for first chunk
        assert "chunk 1/2" in first_call[0][1]

    def test_no_reporter_no_crash(self):
        """Processing works fine without a status_reporter."""
        mock_model = MagicMock()
        mock_model.generate_text.return_value = "ok"

        processor = ExtractionPassProcessor(model_manager=mock_model)
        results = processor.extract_from_chunks([FakeChunk("a")])
        assert results == ["ok"]
