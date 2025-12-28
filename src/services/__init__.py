"""
LocalScribe Services Layer

Provides a clean interface between the UI and core business logic.
Services are thin wrappers that coordinate between components.

Usage:
    from src.services import DocumentService, VocabularyService, QAService

    doc_service = DocumentService()
    results = doc_service.process_documents(file_paths)
"""

from src.services.document_service import DocumentService
from src.services.vocabulary_service import VocabularyService
from src.services.qa_service import QAService
from src.services.settings_service import SettingsService

__all__ = [
    'DocumentService',
    'VocabularyService',
    'QAService',
    'SettingsService',
]
