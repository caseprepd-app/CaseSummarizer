"""
Vocabulary Extraction Package

This package provides functionality for extracting unusual and domain-specific
vocabulary from legal documents. It identifies proper nouns (people, organizations,
locations), medical terms, acronyms, and technical terminology.

Main Components:
- VocabularyExtractor: Core orchestrator for multi-algorithm extraction
- FeedbackManager: Stores user feedback (thumbs up/down) for ML learning
- VocabularyMetaLearner: Learns user preferences from feedback

Multi-Algorithm Architecture (Session 25):
- Base extraction algorithms in src/vocabulary/algorithms/
- Results merged by ResultMerger with weighted confidence
- User feedback trains logistic regression meta-learner

Usage:
    from src.core.vocabulary import VocabularyExtractor

    extractor = VocabularyExtractor()
    vocabulary = extractor.extract(document_text)

    # Feedback loop
    from src.core.vocabulary import get_feedback_manager, get_meta_learner
    feedback_mgr = get_feedback_manager()
    feedback_mgr.record_feedback(term_data, +1)  # Thumbs up

    learner = get_meta_learner()
    if learner.should_retrain():
        learner.train()
"""

from .corpus_manager import CorpusFile, CorpusManager, get_corpus_manager
from .corpus_registry import CorpusInfo, CorpusRegistry, get_corpus_registry
from .feedback_manager import FeedbackManager, get_feedback_manager
from .meta_learner import VocabularyMetaLearner, get_meta_learner
from .name_deduplicator import deduplicate_names
from .name_regularizer import filter_name_fragments, filter_typo_variants, regularize_names
from .vocabulary_extractor import VocabularyExtractor

__all__ = [
    "CorpusFile",
    "CorpusInfo",
    # Corpus Management (Session 29)
    "CorpusManager",
    "CorpusRegistry",
    "FeedbackManager",
    "VocabularyExtractor",
    "VocabularyMetaLearner",
    # Name Deduplication
    "deduplicate_names",
    "filter_name_fragments",
    "filter_typo_variants",
    "get_corpus_manager",
    "get_corpus_registry",
    "get_feedback_manager",
    "get_meta_learner",
    # Name Regularization (Session 63)
    "regularize_names",
]
