from unittest.mock import MagicMock, patch

import pytest

from src.core.summarization.progressive_summarizer import ProgressiveSummarizer


# Mock config for testing (if needed, otherwise ProgressiveSummarizer uses its own)
@pytest.fixture
def mock_config_path(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "chunking_config.yaml"
    config_file.write_text("""
    chunking:
        max_chunk_words: 2000
        patterns_file: "non_existent_patterns.txt" # Mocked path, actual content not needed for this test
        min_chunk_words: 500
        max_chunk_words_hard_limit: 3000
    fast_mode:
        enabled: true
        section_aware_batching: false
        adaptive_batching: false
        base_batch_frequency: 5
    summarization:
        progressive_summary_max_sentences: 2
        local_context_max_sentences: 2
    processing:
        debug_files_to_keep: 5
    """)
    return config_file


@pytest.fixture
def progressive_summarizer_instance(mock_config_path):
    # Patch create_unified_chunker during the test to prevent it from loading real dependencies
    with patch(
        "src.core.summarization.progressive_summarizer.create_unified_chunker"
    ) as MockCreateUnifiedChunker:
        # Configure the mock UnifiedChunker
        mock_chunker = MagicMock()
        mock_chunker.chunk_text.return_value = []  # Return empty chunks if called
        MockCreateUnifiedChunker.return_value = mock_chunker
        summarizer = ProgressiveSummarizer(config_path=mock_config_path)
        yield summarizer


def test_generate_summary_metadata_empty_data(progressive_summarizer_instance):
    """
    Test generate_summary_metadata with empty summary_data.
    """
    summary_data = []
    metadata = progressive_summarizer_instance.generate_summary_metadata(summary_data)

    assert metadata["overall_sentiment"] == "Neutral"
    assert metadata["key_themes"] == []
    assert metadata["document_count"] == 0
    assert metadata["average_summary_length"] == 0
    assert metadata["most_frequent_keyword"] is None


def test_generate_summary_metadata_basic_data(progressive_summarizer_instance):
    """
    Test generate_summary_metadata with basic summary_data.
    """
    summary_data = [
        {
            "title": "Doc 1",
            "summary": "This is a short summary of document one. It talks about apples.",
            "keywords": ["apple", "fruit", "red"],
        },
        {
            "title": "Doc 2",
            "summary": "Document two summarizes oranges and bananas.",
            "keywords": ["orange", "fruit", "yellow"],
        },
    ]
    metadata = progressive_summarizer_instance.generate_summary_metadata(summary_data)

    assert metadata["overall_sentiment"] == "Mixed (Placeholder)"
    assert sorted(metadata["key_themes"]) == sorted(["apple", "fruit", "red", "orange", "yellow"])
    assert metadata["document_count"] == 2
    # len('This is a short summary of document one. It talks about apples.'.split()) = 13
    # len('Document two summarizes oranges and bananas.'.split()) = 7
    # (12 + 7) / 2 = 9.5, int should be 9
    assert metadata["average_summary_length"] == 9
    assert metadata["most_frequent_keyword"] == "fruit"


def test_generate_summary_metadata_complex_data(progressive_summarizer_instance):
    """
    Test generate_summary_metadata with more complex summary_data, including missing keywords.
    """
    summary_data = [
        {
            "title": "Case A",
            "summary": "The court ruled in favor of the plaintiff regarding contract breach. Key concepts included negotiation and damages.",
            "keywords": ["court", "plaintiff", "contract", "negotiation", "damages"],
        },
        {
            "title": "Case B",
            "summary": "Defendant appealed the decision on grounds of procedural error. Evidence was deemed inadmissible.",
            "keywords": ["defendant", "appeal", "procedural", "evidence"],
        },
        {
            "title": "Case C",
            "summary": "Settlement reached in mediation. Both parties agreed to terms.",
            "keywords": ["settlement", "mediation", "agreement"],
        },
        {
            "title": "Case D",
            "summary": "Another contract dispute, similar to Case A.",
            "keywords": ["contract", "dispute", "plaintiff"],
        },
        {
            "title": "Case E",
            "summary": "Short summary.",
            "keywords": [],  # Missing keywords
        },
    ]
    metadata = progressive_summarizer_instance.generate_summary_metadata(summary_data)

    assert metadata["overall_sentiment"] == "Mixed (Placeholder)"
    expected_key_themes = sorted(
        [
            "court",
            "plaintiff",
            "contract",
            "negotiation",
            "damages",
            "defendant",
            "appeal",
            "procedural",
            "evidence",
            "settlement",
            "mediation",
            "agreement",
            "dispute",
        ]
    )
    assert sorted(metadata["key_themes"]) == expected_key_themes
    assert metadata["document_count"] == 5

    # Summary lengths:
    # Case A: 16 words
    # Case B: 12 words
    # Case C: 9 words
    # Case D: 7 words
    # Case E: 2 words
    # Total: 16 + 12 + 9 + 7 + 2 = 46
    # Average: 46 / 5 = 9.2, int should be 9
    assert metadata["average_summary_length"] == 9

    # Keywords:
    # court: 1, plaintiff: 2, contract: 2, negotiation: 1, damages: 1
    # defendant: 1, appeal: 1, procedural: 1, evidence: 1
    # settlement: 1, mediation: 1, agreement: 1, dispute: 1
    # Most frequent are 'plaintiff' and 'contract' (both 2 occurrences).
    # Counter.most_common(1) will return one of them, typically the first encountered
    # or based on internal hash order. We just need to assert it's one of them.
    assert metadata["most_frequent_keyword"] in ["plaintiff", "contract"]


def test_generate_summary_metadata_single_item(progressive_summarizer_instance):
    """
    Test generate_summary_metadata with a single item in summary_data.
    """
    summary_data = [
        {
            "title": "Solo Doc",
            "summary": "This document is about cats and dogs.",
            "keywords": ["cat", "dog", "pet"],
        }
    ]
    metadata = progressive_summarizer_instance.generate_summary_metadata(summary_data)

    assert metadata["overall_sentiment"] == "Mixed (Placeholder)"
    assert sorted(metadata["key_themes"]) == sorted(["cat", "dog", "pet"])
    assert metadata["document_count"] == 1
    assert metadata["average_summary_length"] == len(
        ["This", "document", "is", "about", "cats", "and", "dogs."]
    )  # 7 words
    assert metadata["most_frequent_keyword"] in [
        "cat",
        "dog",
        "pet",
    ]  # Can be any if counts are equal
