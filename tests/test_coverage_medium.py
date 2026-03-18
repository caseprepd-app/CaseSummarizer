"""
Tests for 6 MEDIUM-priority coverage gaps.

Covers:
1. (removed — prompting deprecated)
2. src/config.py — get_count_bin(), get_count_bin_features(), load_model_configs()
3. src/core/extraction/ocr_processor.py — process_image(), Tesseract path detection
4. src/services/workers.py — worker execute() signatures, QueueMessage generation
5. src/core/retrieval/algorithms/faiss_semantic.py — FAISSRetriever index/retrieve
6. src/core/utils/text_utils.py — combine_document_texts() logic
"""

import inspect
import threading
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. PromptConfig tests — REMOVED (src/core/prompting/ deprecated)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. config.py: get_count_bin / get_count_bin_features / load_model_configs
# ---------------------------------------------------------------------------


class TestGetCountBin:
    """Tests for get_count_bin() boundary values."""

    def test_count_0_returns_bin_1(self):
        """Count 0 falls into bin_1 (single/zero occurrence)."""
        from src.config import get_count_bin

        assert get_count_bin(0) == "bin_1"

    def test_count_1_returns_bin_1(self):
        """Count 1 falls into bin_1."""
        from src.config import get_count_bin

        assert get_count_bin(1) == "bin_1"

    def test_count_2_returns_bin_2_3(self):
        """Count 2 falls into bin_2_3."""
        from src.config import get_count_bin

        assert get_count_bin(2) == "bin_2_3"

    def test_count_3_returns_bin_2_3(self):
        """Count 3 falls into bin_2_3 (upper boundary)."""
        from src.config import get_count_bin

        assert get_count_bin(3) == "bin_2_3"

    def test_count_4_returns_bin_4_6(self):
        """Count 4 falls into bin_4_6 (lower boundary)."""
        from src.config import get_count_bin

        assert get_count_bin(4) == "bin_4_6"

    def test_count_6_returns_bin_4_6(self):
        """Count 6 falls into bin_4_6 (upper boundary)."""
        from src.config import get_count_bin

        assert get_count_bin(6) == "bin_4_6"

    def test_count_7_returns_bin_7_20(self):
        """Count 7 falls into bin_7_20 (lower boundary)."""
        from src.config import get_count_bin

        assert get_count_bin(7) == "bin_7_20"

    def test_count_20_returns_bin_7_20(self):
        """Count 20 falls into bin_7_20 (upper boundary)."""
        from src.config import get_count_bin

        assert get_count_bin(20) == "bin_7_20"

    def test_count_21_returns_bin_21_plus(self):
        """Count 21 falls into bin_21_plus."""
        from src.config import get_count_bin

        assert get_count_bin(21) == "bin_21_plus"

    def test_count_100_returns_bin_21_plus(self):
        """Large count falls into bin_21_plus."""
        from src.config import get_count_bin

        assert get_count_bin(100) == "bin_21_plus"


class TestGetCountBinFeatures:
    """Tests for get_count_bin_features() one-hot encoding."""

    def test_one_hot_encoding_bin_1(self):
        """Count 1 produces one-hot with first position active."""
        from src.config import get_count_bin_features

        features = get_count_bin_features(1)
        assert features == (1.0, 0.0, 0.0, 0.0, 0.0)

    def test_one_hot_encoding_bin_2_3(self):
        """Count 2 produces one-hot with second position active."""
        from src.config import get_count_bin_features

        features = get_count_bin_features(2)
        assert features == (0.0, 1.0, 0.0, 0.0, 0.0)

    def test_one_hot_encoding_bin_4_6(self):
        """Count 5 produces one-hot with third position active."""
        from src.config import get_count_bin_features

        features = get_count_bin_features(5)
        assert features == (0.0, 0.0, 1.0, 0.0, 0.0)

    def test_one_hot_encoding_bin_7_20(self):
        """Count 10 produces one-hot with fourth position active."""
        from src.config import get_count_bin_features

        features = get_count_bin_features(10)
        assert features == (0.0, 0.0, 0.0, 1.0, 0.0)

    def test_one_hot_encoding_bin_21_plus(self):
        """Count 50 produces one-hot with fifth position active."""
        from src.config import get_count_bin_features

        features = get_count_bin_features(50)
        assert features == (0.0, 0.0, 0.0, 0.0, 1.0)

    def test_one_hot_sum_is_one(self):
        """Every one-hot encoding sums to exactly 1.0."""
        from src.config import get_count_bin_features

        for count in [0, 1, 2, 3, 4, 6, 7, 20, 21, 100]:
            features = get_count_bin_features(count)
            assert sum(features) == 1.0, f"Failed for count={count}"

    def test_one_hot_is_length_5(self):
        """Every one-hot encoding has exactly 5 elements."""
        from src.config import get_count_bin_features

        for count in [1, 5, 15, 50]:
            features = get_count_bin_features(count)
            assert len(features) == 5


class TestLoadModelConfigs:
    """Tests for load_model_configs() with mocked YAML file."""

    def test_load_valid_yaml(self, tmp_path):
        """load_model_configs loads models from valid YAML file."""
        import src.config as config_module

        yaml_content = "models:\n  llama3: {context_size: 8192}\n  mistral: {context_size: 4096}\n"
        yaml_file = tmp_path / "models.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        original_configs = config_module.MODEL_CONFIGS
        try:
            with patch.object(config_module, "MODEL_CONFIG_FILE", yaml_file):
                config_module.load_model_configs()
                assert "llama3" in config_module.MODEL_CONFIGS
                assert "mistral" in config_module.MODEL_CONFIGS
                assert config_module.MODEL_CONFIGS["llama3"]["context_size"] == 8192
        finally:
            config_module.MODEL_CONFIGS = original_configs

    def test_load_missing_file_returns_empty(self, tmp_path):
        """load_model_configs sets MODEL_CONFIGS to {} when file is missing."""
        import src.config as config_module

        original_configs = config_module.MODEL_CONFIGS
        try:
            with patch.object(config_module, "MODEL_CONFIG_FILE", tmp_path / "missing.yaml"):
                config_module.load_model_configs()
                assert config_module.MODEL_CONFIGS == {}
        finally:
            config_module.MODEL_CONFIGS = original_configs

    def test_load_corrupted_yaml_returns_empty(self, tmp_path):
        """load_model_configs sets MODEL_CONFIGS to {} when YAML is invalid."""
        import src.config as config_module

        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("{{{{invalid yaml", encoding="utf-8")

        original_configs = config_module.MODEL_CONFIGS
        try:
            with patch.object(config_module, "MODEL_CONFIG_FILE", bad_file):
                config_module.load_model_configs()
                assert config_module.MODEL_CONFIGS == {}
        finally:
            config_module.MODEL_CONFIGS = original_configs


# ---------------------------------------------------------------------------
# 3. OCRProcessor: process_image with mocked pytesseract
# ---------------------------------------------------------------------------


class TestOCRProcessorProcessImage:
    """Tests for OCRProcessor.process_image() with mocked Tesseract."""

    def _make_processor(self, preprocessing_enabled=False):
        """Create an OCRProcessor with mocked dictionary validator."""
        with patch(
            "src.core.extraction.ocr_processor.OCR_PREPROCESSING_ENABLED",
            preprocessing_enabled,
        ):
            from src.core.extraction.ocr_processor import OCRProcessor

            mock_dictionary = MagicMock()
            mock_dictionary.calculate_confidence.return_value = 85.0
            return OCRProcessor(dictionary=mock_dictionary)

    @patch("src.core.extraction.ocr_processor._configure_tesseract")
    @patch("src.core.extraction.ocr_processor.pytesseract", create=True)
    def test_process_image_success(self, mock_pytesseract, mock_configure):
        """process_image returns success with text and confidence."""
        import sys

        # Create a fake pytesseract module for the import inside process_image
        mock_module = MagicMock()
        mock_module.image_to_string.return_value = "This is OCR text from an image."
        sys.modules["pytesseract"] = mock_module

        try:
            processor = self._make_processor(preprocessing_enabled=False)
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image

            result = processor.process_image(mock_image)

            assert result["status"] == "success"
            assert result["text"] == "This is OCR text from an image."
            assert result["method"] == "image_ocr"
            assert result["confidence"] == 85
            assert result["error_message"] is None
        finally:
            del sys.modules["pytesseract"]

    @patch("src.core.extraction.ocr_processor._configure_tesseract")
    def test_process_image_empty_text_returns_error(self, mock_configure):
        """process_image returns error when OCR produces empty text."""
        import sys

        mock_module = MagicMock()
        mock_module.image_to_string.return_value = "   \n  "  # Whitespace only
        sys.modules["pytesseract"] = mock_module

        try:
            processor = self._make_processor(preprocessing_enabled=False)
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image

            result = processor.process_image(mock_image)

            assert result["status"] == "error"
            assert result["text"] is None
            assert "Could not extract text" in result["error_message"]
            assert result["confidence"] == 0
        finally:
            del sys.modules["pytesseract"]

    @patch("src.core.extraction.ocr_processor._configure_tesseract")
    def test_process_image_exception_returns_error(self, mock_configure):
        """process_image returns error dict when pytesseract raises."""
        import sys

        mock_module = MagicMock()
        mock_module.image_to_string.side_effect = RuntimeError("Tesseract not found")
        sys.modules["pytesseract"] = mock_module

        try:
            processor = self._make_processor(preprocessing_enabled=False)
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image

            result = processor.process_image(mock_image)

            assert result["status"] == "error"
            assert result["text"] is None
            assert "Tesseract not found" in result["error_message"]
        finally:
            del sys.modules["pytesseract"]


class TestConfigureTesseract:
    """Tests for _configure_tesseract Tesseract path detection."""

    def test_bundled_exe_takes_priority(self):
        """When bundled exe exists, it is used regardless of PATH."""
        import sys

        mock_pytesseract = MagicMock()
        sys.modules["pytesseract"] = mock_pytesseract

        try:
            from src.core.extraction.ocr_processor import _configure_tesseract

            bundled_path = MagicMock()
            bundled_path.exists.return_value = True

            with (
                patch(
                    "src.core.extraction.ocr_processor.TESSERACT_BUNDLED_EXE",
                    bundled_path,
                    create=True,
                ),
                patch("src.config.TESSERACT_BUNDLED_EXE", bundled_path),
                patch("src.core.extraction.ocr_processor._tesseract_patched", True),
            ):
                _configure_tesseract()

            assert mock_pytesseract.tesseract_cmd == str(bundled_path)
        finally:
            del sys.modules["pytesseract"]

    def test_path_tesseract_used_when_no_bundled(self):
        """When no bundled exe but tesseract is on PATH, no cmd is set."""
        import sys

        mock_pytesseract = MagicMock()
        # Reset tesseract_cmd to track if it changes
        mock_pytesseract.tesseract_cmd = "tesseract"
        sys.modules["pytesseract"] = mock_pytesseract

        try:
            from src.core.extraction.ocr_processor import _configure_tesseract

            bundled_path = MagicMock()
            bundled_path.exists.return_value = False

            with (
                patch(
                    "src.core.extraction.ocr_processor.TESSERACT_BUNDLED_EXE",
                    bundled_path,
                    create=True,
                ),
                patch("src.config.TESSERACT_BUNDLED_EXE", bundled_path),
                patch("src.core.extraction.ocr_processor._tesseract_patched", True),
                patch("shutil.which", return_value="/usr/bin/tesseract"),
            ):
                _configure_tesseract()

            # Should NOT have changed tesseract_cmd since it's on PATH
            assert mock_pytesseract.tesseract_cmd == "tesseract"
        finally:
            del sys.modules["pytesseract"]


# ---------------------------------------------------------------------------
# 4. Workers: execute() signatures and QueueMessage generation
# ---------------------------------------------------------------------------


class TestWorkerSignatures:
    """Tests for worker execute() methods existing with correct signatures."""

    def test_processing_worker_has_execute(self):
        """ProcessingWorker has execute() method with no required args."""
        from src.services.workers import ProcessingWorker

        assert hasattr(ProcessingWorker, "execute")
        sig = inspect.signature(ProcessingWorker.execute)
        # Only 'self' parameter
        params = [p for p in sig.parameters if p != "self"]
        assert params == [], f"execute() should take no args besides self, got: {params}"

    def test_qa_worker_has_execute(self):
        """SemanticWorker has execute() method with no required args."""
        from src.services.workers import SemanticWorker

        assert hasattr(SemanticWorker, "execute")
        sig = inspect.signature(SemanticWorker.execute)
        params = [p for p in sig.parameters if p != "self"]
        assert params == []

    def test_progressive_extraction_worker_has_execute(self):
        """ProgressiveExtractionWorker has execute() method with no required args."""
        from src.services.workers import ProgressiveExtractionWorker

        assert hasattr(ProgressiveExtractionWorker, "execute")
        sig = inspect.signature(ProgressiveExtractionWorker.execute)
        params = [p for p in sig.parameters if p != "self"]
        assert params == []


class TestProgressiveExtractionWorkerQueueMessages:
    """Tests for QueueMessage generation in ProgressiveExtractionWorker."""

    def test_extraction_started_message_format(self):
        """QueueMessage.extraction_started() returns correct tuple format."""
        from src.services.queue_messages import QueueMessage

        msg = QueueMessage.extraction_started()
        assert isinstance(msg, tuple)
        assert msg[0] == "extraction_started"

    def test_ner_complete_message_format(self):
        """QueueMessage.ner_complete() returns correct tuple format with data."""
        from src.services.queue_messages import QueueMessage

        vocab_data = [{"term": "Smith", "type": "Person"}]
        filtered = [{"term": "x", "type": "Technical"}]
        msg = QueueMessage.ner_complete(vocab_data, filtered)
        assert isinstance(msg, tuple)
        assert msg[0] == "ner_complete"
        assert msg[1] == {"vocab": vocab_data, "filtered": filtered}

    def test_processing_worker_empty_files_sends_finished(self):
        """ProcessingWorker with 0 files sends processing_finished immediately."""
        from src.services.workers import ProcessingWorker

        ui_queue = Queue()
        worker = ProcessingWorker.__new__(ProcessingWorker)
        worker.file_paths = []
        worker.ui_queue = ui_queue
        worker.processed_results = []
        worker._stop_event = threading.Event()

        worker.execute()

        msg = ui_queue.get_nowait()
        assert msg[0] == "processing_finished"
        assert msg[1] == []


# ---------------------------------------------------------------------------
# 5. FAISSRetriever: index_documents and retrieve with mocks
# ---------------------------------------------------------------------------


class TestFAISSRetriever:
    """Tests for FAISSRetriever.index_documents() and retrieve() with mocks."""

    def _make_chunks(self, count=3):
        """Create a list of DocumentChunk objects for testing."""
        from src.core.retrieval.base import DocumentChunk

        return [
            DocumentChunk(
                text=f"This is chunk number {i} about legal proceedings.",
                chunk_id=f"chunk_{i}",
                filename=f"doc_{i}.pdf",
                chunk_num=i,
                section_name="Body",
                word_count=8,
            )
            for i in range(count)
        ]

    def test_index_documents_empty_raises_value_error(self):
        """index_documents() raises ValueError when given empty chunk list."""
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        retriever = FAISSRetriever(embeddings=MagicMock())
        with pytest.raises(ValueError, match="Cannot index empty chunk list"):
            retriever.index_documents([])

    def test_is_indexed_false_before_indexing(self):
        """is_indexed returns False before index_documents is called."""
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        retriever = FAISSRetriever(embeddings=MagicMock())
        assert retriever.is_indexed is False

    @patch("src.core.retrieval.algorithms.faiss_semantic.FAISS", create=True)
    def test_index_documents_builds_vector_store(self, mock_faiss_cls):
        """index_documents() creates FAISS index from chunks."""
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        mock_embeddings = MagicMock()
        mock_vector_store = MagicMock()
        mock_faiss_cls.from_documents.return_value = mock_vector_store

        retriever = FAISSRetriever(embeddings=mock_embeddings)
        chunks = self._make_chunks(3)

        with patch(
            "src.core.retrieval.algorithms.faiss_semantic.FAISS",
            mock_faiss_cls,
        ):
            # We need to patch the imports inside index_documents
            mock_faiss_module = MagicMock()
            mock_faiss_module.from_documents.return_value = mock_vector_store
            mock_distance = MagicMock()

            with patch.dict(
                "sys.modules",
                {
                    "langchain_community.vectorstores": MagicMock(FAISS=mock_faiss_module),
                    "langchain_community.vectorstores.utils": MagicMock(
                        DistanceStrategy=mock_distance
                    ),
                    "langchain_core.documents": MagicMock(),
                },
            ):
                retriever.index_documents(chunks)

        assert retriever._chunks == chunks
        assert len(retriever._chunks) == 3

    def test_retrieve_without_index_raises_runtime_error(self):
        """retrieve() raises RuntimeError when called before indexing."""
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        retriever = FAISSRetriever(embeddings=MagicMock())
        with pytest.raises(RuntimeError, match="Index not built"):
            retriever.retrieve("test query", k=5)

    def test_retrieve_returns_algorithm_retrieval_result(self):
        """retrieve() returns AlgorithmRetrievalResult with chunks."""
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever
        from src.core.retrieval.base import AlgorithmRetrievalResult

        mock_embeddings = MagicMock()
        retriever = FAISSRetriever(embeddings=mock_embeddings)
        retriever._chunks = self._make_chunks(2)

        # Mock the vector store
        mock_doc = MagicMock()
        mock_doc.page_content = "Legal proceedings text."
        mock_doc.metadata = {
            "chunk_id": "chunk_0",
            "filename": "doc_0.pdf",
            "chunk_num": 0,
            "section_name": "Body",
            "word_count": 8,
        }

        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search_with_score.return_value = [
            (mock_doc, 0.85),
        ]
        retriever._vector_store = mock_vector_store

        result = retriever.retrieve("Who is the plaintiff?", k=5)

        assert isinstance(result, AlgorithmRetrievalResult)
        assert len(result.chunks) == 1
        assert result.chunks[0].relevance_score == 0.85
        assert result.chunks[0].filename == "doc_0.pdf"
        assert result.chunks[0].source_algorithm == "FAISS"

    def test_retrieve_clamps_scores(self):
        """retrieve() clamps scores to [0.0, 1.0] range."""
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        retriever = FAISSRetriever(embeddings=MagicMock())
        retriever._chunks = self._make_chunks(1)

        mock_doc = MagicMock()
        mock_doc.page_content = "Text."
        mock_doc.metadata = {
            "chunk_id": "chunk_0",
            "filename": "doc.pdf",
            "chunk_num": 0,
            "section_name": "N/A",
            "word_count": 1,
        }

        mock_vector_store = MagicMock()
        # Score > 1.0 should be clamped
        mock_vector_store.similarity_search_with_score.return_value = [
            (mock_doc, 1.5),
        ]
        retriever._vector_store = mock_vector_store

        result = retriever.retrieve("query", k=1)
        assert result.chunks[0].relevance_score == 1.0

    def test_get_config_includes_faiss_fields(self):
        """get_config() includes FAISS-specific fields."""
        from src.core.retrieval.algorithms.faiss_semantic import FAISSRetriever

        retriever = FAISSRetriever(embeddings=MagicMock())
        config = retriever.get_config()

        assert config["name"] == "FAISS"
        assert "index_size" in config
        assert "embedding_model" in config
        assert config["distance_metric"] == "cosine"


# ---------------------------------------------------------------------------
# 6. text_utils: combine_document_texts
# ---------------------------------------------------------------------------


class TestCombineDocumentTexts:
    """Tests for combine_document_texts() logic."""

    def test_prefers_preprocessed_text(self):
        """Uses preprocessed_text over extracted_text when available."""
        from src.core.utils.text_utils import combine_document_texts

        docs = [
            {
                "filename": "a.pdf",
                "extracted_text": "raw text",
                "preprocessed_text": "clean text",
            }
        ]
        result = combine_document_texts(docs, preprocess=False)
        assert result == "clean text"

    def test_falls_back_to_extracted_text(self):
        """Uses extracted_text when preprocessed_text is not present."""
        from src.core.utils.text_utils import combine_document_texts

        docs = [{"filename": "a.pdf", "extracted_text": "raw text"}]
        # preprocess=False to avoid calling the pipeline
        result = combine_document_texts(docs, preprocess=False)
        assert result == "raw text"

    def test_include_headers(self):
        """Includes filename headers when include_headers=True."""
        from src.core.utils.text_utils import combine_document_texts

        docs = [
            {"filename": "a.pdf", "preprocessed_text": "Hello"},
            {"filename": "b.pdf", "preprocessed_text": "World"},
        ]
        result = combine_document_texts(docs, include_headers=True, preprocess=False)
        assert "--- a.pdf ---" in result
        assert "--- b.pdf ---" in result
        assert "Hello" in result
        assert "World" in result

    def test_skips_empty_documents(self):
        """Documents without text are skipped."""
        from src.core.utils.text_utils import combine_document_texts

        docs = [
            {"filename": "a.pdf", "preprocessed_text": "Content"},
            {"filename": "b.pdf", "extracted_text": ""},
            {"filename": "c.pdf"},
        ]
        result = combine_document_texts(docs, preprocess=False)
        assert result == "Content"

    def test_empty_list_returns_empty_string(self):
        """Empty document list returns empty string."""
        from src.core.utils.text_utils import combine_document_texts

        result = combine_document_texts([], preprocess=False)
        assert result == ""

    def test_separator_between_documents(self):
        """Documents are joined with separator (default double newline)."""
        from src.core.utils.text_utils import combine_document_texts

        docs = [
            {"filename": "a.pdf", "preprocessed_text": "First"},
            {"filename": "b.pdf", "preprocessed_text": "Second"},
        ]
        result = combine_document_texts(docs, preprocess=False)
        assert result == "First\n\nSecond"

    def test_custom_separator(self):
        """Custom separator is used between documents."""
        from src.core.utils.text_utils import combine_document_texts

        docs = [
            {"filename": "a.pdf", "preprocessed_text": "AAA"},
            {"filename": "b.pdf", "preprocessed_text": "BBB"},
        ]
        result = combine_document_texts(docs, separator=" | ", preprocess=False)
        assert result == "AAA | BBB"

    def test_preprocessed_text_skips_preprocessing_pipeline(self):
        """When preprocessed_text is present, preprocessing pipeline is NOT called."""
        from src.core.utils.text_utils import combine_document_texts

        docs = [{"filename": "a.pdf", "preprocessed_text": "Already clean"}]

        # The import of create_default_pipeline is conditional and only fires
        # when no preprocessed_text exists. With preprocessed_text present,
        # the pipeline code path is never reached.
        with patch("src.core.preprocessing.create_default_pipeline") as mock_pipeline:
            result = combine_document_texts(docs, preprocess=True)

        mock_pipeline.assert_not_called()
        assert result == "Already clean"
