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
    from src.vocabulary import VocabularyExtractor

    extractor = VocabularyExtractor()
    vocabulary = extractor.extract(document_text)

    # Feedback loop
    from src.vocabulary import get_feedback_manager, get_meta_learner
    feedback_mgr = get_feedback_manager()
    feedback_mgr.record_feedback(term_data, +1)  # Thumbs up

    learner = get_meta_learner()
    if learner.should_retrain():
        learner.train()
"""

from .vocabulary_extractor import VocabularyExtractor
from .feedback_manager import FeedbackManager, get_feedback_manager
from .meta_learner import VocabularyMetaLearner, get_meta_learner
from .corpus_manager import CorpusManager, CorpusFile, get_corpus_manager
from .corpus_registry import CorpusRegistry, CorpusInfo, get_corpus_registry
from .name_deduplicator import deduplicate_names

__all__ = [
    'VocabularyExtractor',
    'FeedbackManager',
    'get_feedback_manager',
    'VocabularyMetaLearner',
    'get_meta_learner',
    # Corpus Management (Session 29)
    'CorpusManager',
    'CorpusFile',
    'get_corpus_manager',
    'CorpusRegistry',
    'CorpusInfo',
    'get_corpus_registry',
    # Name Deduplication
    'deduplicate_names',
]
