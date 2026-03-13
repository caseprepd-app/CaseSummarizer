"""
Vector Store Package for CasePrepd Semantic Search System.

Provides FAISS-based vector storage for semantic search
over legal documents.

Components:
- VectorStoreBuilder: Creates FAISS indexes from document chunks
- SemanticRetriever: Retrieves relevant context for user questions

Architecture:
- File-based persistence (no database required)
- Compatible with Windows standalone installer
- Uses LangChain for seamless integration

Usage:
    from src.core.vector_store import VectorStoreBuilder, SemanticRetriever

    # Build vector store from documents
    builder = VectorStoreBuilder()
    builder.create_from_documents(documents, embeddings, persist_dir)

    # Query for relevant context
    retriever = SemanticRetriever(persist_dir, embeddings)
    context, sources = retriever.retrieve_context("Who are the plaintiffs?")
"""

from .semantic_retriever import SemanticRetriever
from .vector_store_builder import VectorStoreBuilder

__all__ = [
    "SemanticRetriever",
    "VectorStoreBuilder",
]
