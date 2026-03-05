"""
Deployment Safety Tests

Verifies that model-loading code paths handle missing bundled models gracefully,
prevent silent network downloads on end-user machines, and use consistent
encoding for CSV I/O.
"""

import ast
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fix 1a: FAISS embeddings -- local_files_only guard
# ---------------------------------------------------------------------------
class TestFAISSEmbeddingsGuard:
    """Verify FAISS embedding loader prevents silent downloads."""

    def test_local_files_only_when_bundled_model_exists(self):
        """When bundled model path exists, model_kwargs must include local_files_only=True."""
        import src.core.retrieval.algorithms.faiss_semantic as mod

        # Reset cached embeddings
        mod._shared_embeddings = None

        mock_embeddings = MagicMock()
        with (
            patch.object(mod, "EMBEDDING_MODEL_LOCAL_PATH") as mock_path,
            patch(
                "langchain_huggingface.HuggingFaceEmbeddings", return_value=mock_embeddings
            ) as mock_cls,
        ):
            mock_path.exists.return_value = True
            result = mod.get_embeddings_model()

        assert result is mock_embeddings
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model_kwargs"]["local_files_only"] is True
        # Clean up
        mod._shared_embeddings = None

    def test_no_local_files_only_when_bundled_missing(self):
        """When bundled model is absent, local_files_only should NOT be set."""
        import src.core.retrieval.algorithms.faiss_semantic as mod

        mod._shared_embeddings = None

        mock_embeddings = MagicMock()
        with (
            patch.object(mod, "EMBEDDING_MODEL_LOCAL_PATH") as mock_path,
            patch(
                "langchain_huggingface.HuggingFaceEmbeddings", return_value=mock_embeddings
            ) as mock_cls,
        ):
            mock_path.exists.return_value = False
            mod.get_embeddings_model()

        call_kwargs = mock_cls.call_args[1]
        assert "local_files_only" not in call_kwargs["model_kwargs"]
        mod._shared_embeddings = None

    def test_clear_error_on_load_failure(self):
        """Failed model load raises RuntimeError with helpful message."""
        import src.core.retrieval.algorithms.faiss_semantic as mod

        mod._shared_embeddings = None

        with (
            patch.object(mod, "EMBEDDING_MODEL_LOCAL_PATH") as mock_path,
            patch("langchain_huggingface.HuggingFaceEmbeddings", side_effect=OSError("No model")),
        ):
            mock_path.exists.return_value = False
            with pytest.raises(RuntimeError, match="Embedding model not available"):
                mod.get_embeddings_model()

        mod._shared_embeddings = None

    def test_cached_embeddings_returned_on_second_call(self):
        """Second call returns cached instance without re-loading."""
        import src.core.retrieval.algorithms.faiss_semantic as mod

        sentinel = MagicMock()
        mod._shared_embeddings = sentinel
        assert mod.get_embeddings_model() is sentinel
        mod._shared_embeddings = None

    def test_runtime_error_chains_original_exception(self):
        """RuntimeError.__cause__ should be the original OSError for debugging."""
        import src.core.retrieval.algorithms.faiss_semantic as mod

        mod._shared_embeddings = None
        original = OSError("connection refused")

        with (
            patch.object(mod, "EMBEDDING_MODEL_LOCAL_PATH") as mock_path,
            patch("langchain_huggingface.HuggingFaceEmbeddings", side_effect=original),
        ):
            mock_path.exists.return_value = False
            with pytest.raises(RuntimeError) as exc_info:
                mod.get_embeddings_model()

        assert exc_info.value.__cause__ is original
        mod._shared_embeddings = None

    def test_error_message_includes_download_instructions(self):
        """RuntimeError message should tell user how to fix it."""
        import src.core.retrieval.algorithms.faiss_semantic as mod

        mod._shared_embeddings = None

        with (
            patch.object(mod, "EMBEDDING_MODEL_LOCAL_PATH") as mock_path,
            patch("langchain_huggingface.HuggingFaceEmbeddings", side_effect=OSError("fail")),
        ):
            mock_path.exists.return_value = True
            with pytest.raises(RuntimeError, match="download_models"):
                mod.get_embeddings_model()

        mod._shared_embeddings = None

    def test_device_kwarg_always_present(self):
        """model_kwargs always includes device regardless of local/remote."""
        import src.core.retrieval.algorithms.faiss_semantic as mod

        for local_exists in (True, False):
            mod._shared_embeddings = None
            with (
                patch.object(mod, "EMBEDDING_MODEL_LOCAL_PATH") as mock_path,
                patch(
                    "langchain_huggingface.HuggingFaceEmbeddings", return_value=MagicMock()
                ) as mock_cls,
            ):
                mock_path.exists.return_value = local_exists
                mod.get_embeddings_model()

            call_kwargs = mock_cls.call_args[1]
            assert "device" in call_kwargs["model_kwargs"]
            mod._shared_embeddings = None

    def test_failed_load_does_not_cache(self):
        """If loading fails, _shared_embeddings stays None for retry."""
        import src.core.retrieval.algorithms.faiss_semantic as mod

        mod._shared_embeddings = None

        with (
            patch.object(mod, "EMBEDDING_MODEL_LOCAL_PATH") as mock_path,
            patch("langchain_huggingface.HuggingFaceEmbeddings", side_effect=OSError("fail")),
        ):
            mock_path.exists.return_value = False
            with pytest.raises(RuntimeError):
                mod.get_embeddings_model()

        assert mod._shared_embeddings is None

    def test_get_embedding_model_path_prefers_local(self):
        """_get_embedding_model_path() returns local path string when it exists."""
        import src.core.retrieval.algorithms.faiss_semantic as mod

        with patch.object(mod, "EMBEDDING_MODEL_LOCAL_PATH") as mock_path:
            mock_path.exists.return_value = True
            mock_path.__str__ = lambda self: "/models/nomic"
            result = mod._get_embedding_model_path()
        assert result == str(mock_path)

    def test_get_embedding_model_path_falls_back_to_name(self):
        """_get_embedding_model_path() returns HF model name when local is missing."""
        import src.core.retrieval.algorithms.faiss_semantic as mod

        with patch.object(mod, "EMBEDDING_MODEL_LOCAL_PATH") as mock_path:
            mock_path.exists.return_value = False
            result = mod._get_embedding_model_path()
        assert result == mod.EMBEDDING_MODEL_NAME


# ---------------------------------------------------------------------------
# Fix 1b: Cross-encoder reranker -- local_files_only guard
# ---------------------------------------------------------------------------
class TestCrossEncoderGuard:
    """Verify cross-encoder reranker prevents silent downloads."""

    def test_local_files_only_when_bundled(self):
        """Bundled model path -> local_files_only=True as direct kwarg."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        mock_model = MagicMock()

        with (
            patch(
                "src.core.retrieval.cross_encoder_reranker.RERANKER_MODEL_LOCAL_PATH"
            ) as mock_path,
            patch("sentence_transformers.CrossEncoder", return_value=mock_model) as mock_cls,
        ):
            mock_path.exists.return_value = True
            mock_path.__str__ = lambda self: "/bundled/model"
            reranker._load_model()

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["local_files_only"] is True

    def test_no_local_files_only_when_not_bundled(self):
        """Non-bundled path -> local_files_only should not be passed."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        mock_model = MagicMock()

        with (
            patch(
                "src.core.retrieval.cross_encoder_reranker.RERANKER_MODEL_LOCAL_PATH"
            ) as mock_path,
            patch("sentence_transformers.CrossEncoder", return_value=mock_model) as mock_cls,
        ):
            mock_path.exists.return_value = False
            reranker._load_model()

        call_kwargs = mock_cls.call_args[1]
        assert "local_files_only" not in call_kwargs

    def test_rerank_fallback_on_load_failure(self):
        """If model fails to load, rerank() returns original chunks (graceful degradation)."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        chunks = [MagicMock(), MagicMock(), MagicMock()]

        with patch.object(reranker, "_load_model", side_effect=OSError("Network unreachable")):
            result = reranker.rerank("test query", chunks, top_k=2)

        # Should return first top_k chunks as fallback
        assert result == chunks[:2]

    def test_load_model_sets_hf_env_vars(self):
        """_load_model() always sets HF_HOME and TRANSFORMERS_CACHE env vars."""
        import os

        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        with (
            patch(
                "src.core.retrieval.cross_encoder_reranker.RERANKER_MODEL_LOCAL_PATH"
            ) as mock_path,
            patch("sentence_transformers.CrossEncoder", return_value=MagicMock()),
        ):
            mock_path.exists.return_value = False
            reranker._load_model()

        # HF_HOME should be set to project cache dir
        assert "HF_HOME" in os.environ
        assert "TRANSFORMERS_CACHE" in os.environ

    def test_no_local_files_only_kwarg_when_not_bundled(self):
        """When not bundled, local_files_only should not be in kwargs."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        with (
            patch(
                "src.core.retrieval.cross_encoder_reranker.RERANKER_MODEL_LOCAL_PATH"
            ) as mock_path,
            patch("sentence_transformers.CrossEncoder", return_value=MagicMock()) as mock_cls,
        ):
            mock_path.exists.return_value = False
            reranker._load_model()

        call_kwargs = mock_cls.call_args[1]
        assert "local_files_only" not in call_kwargs

    def test_is_available_returns_false_on_failure(self):
        """is_available() returns False when model cannot load."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        with patch.object(reranker, "_load_model", side_effect=OSError("No model")):
            assert reranker.is_available() is False

    def test_rerank_empty_chunks_returns_empty(self):
        """rerank() with empty list returns empty list without loading model."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", [], top_k=5)
        assert result == []
        # Model should NOT have been loaded
        assert reranker._model is None

    def test_max_length_passed_to_cross_encoder(self):
        """RERANKER_MAX_LENGTH config is passed through to CrossEncoder constructor."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        with (
            patch(
                "src.core.retrieval.cross_encoder_reranker.RERANKER_MODEL_LOCAL_PATH"
            ) as mock_path,
            patch("sentence_transformers.CrossEncoder", return_value=MagicMock()) as mock_cls,
        ):
            mock_path.exists.return_value = True
            mock_path.__str__ = lambda self: "/bundled"
            reranker._load_model()

        call_kwargs = mock_cls.call_args[1]
        assert "max_length" in call_kwargs


# ---------------------------------------------------------------------------
# Fix 2: WordNet synsets resilience
# ---------------------------------------------------------------------------
class TestWordNetResilience:
    """Verify NER algorithm handles missing WordNet data gracefully."""

    def test_wordnet_lookup_error_treated_as_rare(self):
        """LookupError from wordnet.synsets() should treat word as rare (return True)."""
        from src.core.vocabulary.algorithms.ner_algorithm import NERAlgorithm

        algo = NERAlgorithm()

        # Create a mock token that passes all filters up to the wordnet check
        mock_token = MagicMock()
        mock_token.is_alpha = True
        mock_token.is_space = False
        mock_token.is_punct = False
        mock_token.is_digit = False
        mock_token.text = "xylophone"
        mock_token.pos_ = "NOUN"
        mock_token.ent_type_ = ""

        with patch("src.core.vocabulary.algorithms.ner_algorithm.wordnet") as mock_wn:
            mock_wn.synsets.side_effect = LookupError("Resource wordnet not found")
            result = algo._is_unusual(mock_token, ent_type=None)

        # Should return True (treat as rare/unusual) instead of crashing
        assert result is True

    def test_wordnet_normal_operation(self):
        """Normal wordnet operation: word with synsets is NOT unusual."""
        from src.core.vocabulary.algorithms.ner_algorithm import NERAlgorithm

        algo = NERAlgorithm()

        mock_token = MagicMock()
        mock_token.is_alpha = True
        mock_token.is_space = False
        mock_token.is_punct = False
        mock_token.is_digit = False
        mock_token.text = "house"
        mock_token.pos_ = "NOUN"
        mock_token.ent_type_ = ""

        with patch("src.core.vocabulary.algorithms.ner_algorithm.wordnet") as mock_wn:
            mock_wn.synsets.return_value = ["syn1"]  # Has synsets = common word
            with patch(
                "src.core.vocabulary.algorithms.ner_algorithm.matches_token_filter",
                return_value=False,
            ):
                result = algo._is_unusual(mock_token, ent_type=None)

        assert result is False

    def test_wordnet_source_code_has_try_except(self):
        """Verify the try/except LookupError is present in the source code."""
        src_path = (
            Path(__file__).parent.parent
            / "src"
            / "core"
            / "vocabulary"
            / "algorithms"
            / "ner_algorithm.py"
        )
        source = src_path.read_text(encoding="utf-8")
        assert "except LookupError" in source
        assert "wordnet.synsets" in source

    def test_wordnet_empty_synsets_means_rare(self):
        """Word with no synsets (empty list) is considered rare/unusual."""
        from src.core.vocabulary.algorithms.ner_algorithm import NERAlgorithm

        algo = NERAlgorithm()

        mock_token = MagicMock()
        mock_token.is_alpha = True
        mock_token.is_space = False
        mock_token.is_punct = False
        mock_token.is_digit = False
        mock_token.text = "defenestrate"
        mock_token.pos_ = "VERB"
        mock_token.ent_type_ = ""

        with patch("src.core.vocabulary.algorithms.ner_algorithm.wordnet") as mock_wn:
            mock_wn.synsets.return_value = []  # No synsets = rare
            with patch(
                "src.core.vocabulary.algorithms.ner_algorithm.matches_token_filter",
                return_value=False,
            ):
                result = algo._is_unusual(mock_token, ent_type=None)

        assert result is True

    def test_non_lookup_errors_still_propagate(self):
        """Errors other than LookupError (e.g. TypeError) should NOT be silently caught."""
        from src.core.vocabulary.algorithms.ner_algorithm import NERAlgorithm

        algo = NERAlgorithm()

        mock_token = MagicMock()
        mock_token.is_alpha = True
        mock_token.is_space = False
        mock_token.is_punct = False
        mock_token.is_digit = False
        mock_token.text = "testword"
        mock_token.pos_ = "NOUN"
        mock_token.ent_type_ = ""

        with patch("src.core.vocabulary.algorithms.ner_algorithm.wordnet") as mock_wn:
            mock_wn.synsets.side_effect = TypeError("unexpected error")
            with (
                patch(
                    "src.core.vocabulary.algorithms.ner_algorithm.matches_token_filter",
                    return_value=False,
                ),
                pytest.raises(TypeError, match="unexpected error"),
            ):
                algo._is_unusual(mock_token, ent_type=None)

    def test_named_entities_bypass_wordnet(self):
        """Tokens with entity types (PERSON, ORG, etc.) skip wordnet entirely."""
        from src.core.vocabulary.algorithms.ner_algorithm import NERAlgorithm

        algo = NERAlgorithm()

        mock_token = MagicMock()
        mock_token.is_alpha = True
        mock_token.is_space = False
        mock_token.is_punct = False
        mock_token.is_digit = False
        mock_token.text = "Smith"
        mock_token.pos_ = "PROPN"
        mock_token.ent_type_ = "PERSON"

        with patch("src.core.vocabulary.algorithms.ner_algorithm.wordnet") as mock_wn:
            mock_wn.synsets.side_effect = AssertionError("Should not be called")
            with patch(
                "src.core.vocabulary.algorithms.ner_algorithm.matches_token_filter",
                return_value=False,
            ):
                result = algo._is_unusual(mock_token, ent_type="PERSON")

        assert result is True
        # wordnet.synsets should never have been called
        mock_wn.synsets.assert_not_called()

    def test_stopwords_filtered_before_wordnet(self):
        """Common stopwords are filtered early, never reaching wordnet check."""
        from src.core.vocabulary.algorithms.ner_algorithm import NERAlgorithm

        algo = NERAlgorithm()

        mock_token = MagicMock()
        mock_token.is_alpha = True
        mock_token.is_space = False
        mock_token.is_punct = False
        mock_token.is_digit = False
        mock_token.text = "the"
        mock_token.pos_ = "DET"
        mock_token.ent_type_ = ""

        with patch("src.core.vocabulary.algorithms.ner_algorithm.wordnet") as mock_wn:
            result = algo._is_unusual(mock_token, ent_type=None)

        assert result is False
        mock_wn.synsets.assert_not_called()


# ---------------------------------------------------------------------------
# Fix 3: Progressive summarizer CSV encoding
# ---------------------------------------------------------------------------
class TestProgressiveSummarizerEncoding:
    """Verify debug CSV export uses utf-8-sig encoding."""

    def test_debug_csv_uses_utf8_sig(self):
        """save_debug_dataframe() must pass encoding='utf-8-sig' to to_csv()."""
        src_path = (
            Path(__file__).parent.parent
            / "src"
            / "core"
            / "summarization"
            / "progressive_summarizer.py"
        )
        source = src_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        found_encoding = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Look for .to_csv() calls
                if isinstance(func, ast.Attribute) and func.attr == "to_csv":
                    for kw in node.keywords:
                        if kw.arg == "encoding":
                            if isinstance(kw.value, ast.Constant) and kw.value.value == "utf-8-sig":
                                found_encoding = True

        assert found_encoding, "to_csv() call must include encoding='utf-8-sig'"

    def test_save_debug_dataframe_writes_bom(self, tmp_path):
        """Functional test: saved CSV file starts with UTF-8 BOM byte sequence."""
        import pandas as pd

        from src.core.summarization.progressive_summarizer import ProgressiveSummarizer

        with patch.object(ProgressiveSummarizer, "__init__", lambda self, **kw: None):
            summarizer = ProgressiveSummarizer()

        summarizer.df = pd.DataFrame(
            {
                "chunk_num": [1],
                "chunk_text": ["Sample text for testing BOM encoding"],
                "chunk_summary": ["A summary of the sample"],
                "progressive_summary": ["Progressive summary so far"],
                "section_detected": ["Introduction"],
                "word_count": [6],
                "processing_time_sec": [0.5],
            }
        )
        summarizer.config = {"processing": {"debug_files_to_keep": 5}}

        result_path = summarizer.save_debug_dataframe(output_dir=tmp_path)

        # Read raw bytes to check for BOM
        raw = result_path.read_bytes()
        assert raw[:3] == b"\xef\xbb\xbf", "CSV file should start with UTF-8 BOM"

    def test_save_debug_dataframe_truncates_text(self, tmp_path):
        """Debug CSV truncates long text columns for readability."""
        import pandas as pd

        from src.core.summarization.progressive_summarizer import ProgressiveSummarizer

        with patch.object(ProgressiveSummarizer, "__init__", lambda self, **kw: None):
            summarizer = ProgressiveSummarizer()

        long_text = "A" * 200
        summarizer.df = pd.DataFrame(
            {
                "chunk_num": [1],
                "chunk_text": [long_text],
                "chunk_summary": [long_text],
                "progressive_summary": [long_text],
                "section_detected": ["Test"],
                "word_count": [1],
                "processing_time_sec": [0.1],
            }
        )
        summarizer.config = {"processing": {"debug_files_to_keep": 5}}

        result_path = summarizer.save_debug_dataframe(output_dir=tmp_path)

        df_read = pd.read_csv(result_path, encoding="utf-8-sig")
        # chunk_text truncated to 100 chars + "..."
        assert len(df_read["chunk_text"].iloc[0]) <= 104


# ---------------------------------------------------------------------------
# Fix 4: Diagnostic script CSV read encoding
# ---------------------------------------------------------------------------
class TestDiagnosticScriptEncoding:
    """Verify diagnose_ml.py reads CSVs with utf-8-sig encoding."""

    def test_read_csv_calls_use_utf8_sig(self):
        """Both pd.read_csv() calls in diagnose_ml.py should use encoding='utf-8-sig'."""
        src_path = Path(__file__).parent.parent / "scripts" / "diagnose_ml.py"
        source = src_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        read_csv_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "read_csv":
                    has_encoding = False
                    for kw in node.keywords:
                        if kw.arg == "encoding":
                            if isinstance(kw.value, ast.Constant) and kw.value.value == "utf-8-sig":
                                has_encoding = True
                    read_csv_calls.append(has_encoding)

        assert len(read_csv_calls) >= 2, (
            f"Expected at least 2 read_csv calls, found {len(read_csv_calls)}"
        )
        assert all(read_csv_calls), "All read_csv() calls must include encoding='utf-8-sig'"

    def test_read_csv_handles_bom_files(self, tmp_path):
        """Functional test: utf-8-sig encoded CSVs are read correctly with BOM."""
        import pandas as pd

        csv_path = tmp_path / "test_feedback.csv"
        df = pd.DataFrame({"term": ["Smith", "Jones"], "feedback": [1, -1], "occurrences": [5, 3]})
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # Reading with utf-8-sig should strip the BOM and parse correctly
        df_read = pd.read_csv(csv_path, encoding="utf-8-sig")
        assert list(df_read.columns) == ["term", "feedback", "occurrences"]
        assert df_read["term"].iloc[0] == "Smith"

    def test_read_csv_without_encoding_may_corrupt_column_names(self, tmp_path):
        """Without utf-8-sig, BOM bytes can corrupt the first column name."""
        import pandas as pd

        csv_path = tmp_path / "test_bom.csv"
        df = pd.DataFrame({"term": ["test"], "feedback": [1], "occurrences": [1]})
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # Reading without encoding kwarg uses utf-8 which may keep BOM in column name
        df_raw = pd.read_csv(csv_path)
        # The first column might have BOM prefix -- this is the bug we're preventing
        first_col = list(df_raw.columns)[0]
        # With utf-8-sig encoding, this should be clean
        df_clean = pd.read_csv(csv_path, encoding="utf-8-sig")
        assert list(df_clean.columns)[0] == "term"
        # This test documents WHY we need utf-8-sig for reading


# ---------------------------------------------------------------------------
# Model loading patterns: verify all paths have try/except guards
# ---------------------------------------------------------------------------
class TestModelLoadingGuards:
    """Verify all model-loading code paths have proper error handling."""

    def test_faiss_get_embeddings_model_has_try_except(self):
        """get_embeddings_model() must wrap HuggingFaceEmbeddings creation in try/except."""
        from src.core.retrieval.algorithms.faiss_semantic import get_embeddings_model

        source = inspect.getsource(get_embeddings_model)
        assert "try:" in source
        assert "except" in source
        assert "RuntimeError" in source

    def test_cross_encoder_rerank_has_try_except(self):
        """rerank() must catch _load_model failures."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        source = inspect.getsource(CrossEncoderReranker.rerank)
        assert "try:" in source
        assert "except" in source

    def test_hallucination_verifier_uses_local_files_only(self):
        """Hallucination verifier (the good pattern) still uses local_files_only."""
        verifier_path = (
            Path(__file__).parent.parent / "src" / "core" / "qa" / "hallucination_verifier.py"
        )
        source = verifier_path.read_text(encoding="utf-8")
        assert "local_files_only" in source

    def test_cross_encoder_load_model_has_local_files_only_in_source(self):
        """_load_model source code contains local_files_only as direct kwarg."""
        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        source = inspect.getsource(CrossEncoderReranker._load_model)
        assert "local_files_only" in source
        assert "init_kwargs" in source

    def test_faiss_source_checks_embedding_path_exists(self):
        """get_embeddings_model() source checks EMBEDDING_MODEL_LOCAL_PATH.exists()."""
        from src.core.retrieval.algorithms.faiss_semantic import get_embeddings_model

        source = inspect.getsource(get_embeddings_model)
        assert "EMBEDDING_MODEL_LOCAL_PATH.exists()" in source

    def test_ner_algorithm_wordnet_has_lookup_error_guard(self):
        """_is_unusual source code catches LookupError around wordnet.synsets()."""
        from src.core.vocabulary.algorithms.ner_algorithm import NERAlgorithm

        source = inspect.getsource(NERAlgorithm._is_unusual)
        assert "LookupError" in source
        assert "wordnet.synsets" in source


# ---------------------------------------------------------------------------
# Encoding consistency: all CSV exports should use utf-8-sig
# ---------------------------------------------------------------------------
class TestCSVEncodingConsistency:
    """Verify all production CSV exports use utf-8-sig encoding."""

    def _find_to_csv_calls_in_file(self, filepath: Path) -> list[tuple[int, bool]]:
        """Find to_csv() calls and check if they have encoding='utf-8-sig'.

        Returns list of (line_number, has_utf8sig) tuples.
        """
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source)
        results = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "to_csv":
                    has_encoding = False
                    for kw in node.keywords:
                        if kw.arg == "encoding":
                            if isinstance(kw.value, ast.Constant) and kw.value.value == "utf-8-sig":
                                has_encoding = True
                    results.append((getattr(node, "lineno", 0), has_encoding))

        return results

    def test_progressive_summarizer_csv_encoding(self):
        """progressive_summarizer.py to_csv uses utf-8-sig."""
        path = (
            Path(__file__).parent.parent
            / "src"
            / "core"
            / "summarization"
            / "progressive_summarizer.py"
        )
        results = self._find_to_csv_calls_in_file(path)
        assert len(results) > 0, "Expected at least one to_csv call"
        for lineno, has_encoding in results:
            assert has_encoding, f"to_csv at line {lineno} missing encoding='utf-8-sig'"
