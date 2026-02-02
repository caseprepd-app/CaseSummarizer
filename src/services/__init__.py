"""
CasePrepd Services Layer

Provides a clean interface between the UI and core business logic.
Services are thin wrappers that coordinate between components.

Usage:
    from src.services import DocumentService, VocabularyService, QAService, AIService

    doc_service = DocumentService()
    results = doc_service.process_documents(file_paths)
"""

from src.services.ai_service import AIService
from src.services.document_service import DocumentService
from src.services.export_service import ExportService, get_export_service
from src.services.model_io_service import (
    export_user_feedback,
    export_user_model,
    import_user_feedback,
    import_user_model,
)
from src.services.qa_service import QAService
from src.services.vocabulary_service import VocabularyService

__all__ = [
    "AIService",
    "DocumentService",
    "ExportService",
    "QAService",
    "VocabularyService",
    "export_user_feedback",
    "export_user_model",
    "get_export_service",
    "import_user_feedback",
    "import_user_model",
]
