"""
Vector Store Package for LocalScribe Q&A System.

Provides FAISS-based vector storage for RAG (Retrieval-Augmented Generation)
question answering over legal documents.

Components:
- VectorStoreBuilder: Creates FAISS indexes from document chunks
- QARetriever: Retrieves relevant context for user questions

Architecture:
- File-based persistence (no database required)
- Compatible with Windows standalone installer
- Uses LangChain for seamless integration

Usage:
    from src.core.vector_store import VectorStoreBuilder, QARetriever

    # Build vector store from documents
    builder = VectorStoreBuilder()
    builder.create_from_documents(documents, embeddings, persist_dir)

    # Query for relevant context
    retriever = QARetriever(persist_dir, embeddings)
    context, sources = retriever.retrieve_context("Who are the plaintiffs?")
"""

from .vector_store_builder import VectorStoreBuilder
from .qa_retriever import QARetriever
from .question_flow import QuestionFlowManager, QuestionAnswer, FlowState

__all__ = [
    "VectorStoreBuilder",
    "QARetriever",
    "QuestionFlowManager",
    "QuestionAnswer",
    "FlowState",
]
