"""
Diagnostic script to test Q&A metadata flow.

This script helps identify why source_summary might be blank in CSV exports
while citation shows partial data in the GUI.

Run this after processing documents to inspect the actual QAResult objects.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import VECTOR_STORE_DIR


def inspect_latest_vector_store():
    """Find and inspect the most recently created vector store."""
    print("\n" + "=" * 80)
    print("Q&A METADATA DIAGNOSTIC")
    print("=" * 80)

    # Find latest vector store
    if not VECTOR_STORE_DIR.exists():
        print(f"\n[!] Vector store directory not found: {VECTOR_STORE_DIR}")
        print("    Process documents first to create a vector store.")
        return

    stores = sorted(VECTOR_STORE_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not stores:
        print(f"\n[!] No vector stores found in: {VECTOR_STORE_DIR}")
        return

    latest_store = stores[0]
    print(f"\n[+] Latest vector store: {latest_store.name}")
    print(f"    Created: {Path(latest_store).stat().st_mtime}")

    # Check if FAISS files exist
    faiss_file = latest_store / "index.faiss"
    pkl_file = latest_store / "index.pkl"

    if not faiss_file.exists():
        print(f"\n[!] index.faiss not found in {latest_store}")
        return

    if not pkl_file.exists():
        print(f"\n[!] index.pkl not found in {latest_store}")
        return

    print("\n[OK] FAISS files exist:")
    print(f"     - index.faiss ({faiss_file.stat().st_size:,} bytes)")
    print(f"     - index.pkl ({pkl_file.stat().st_size:,} bytes)")

    # Load vector store and inspect metadata
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        print("\n[*] Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        print(f"[*] Loading FAISS index from {latest_store}...")
        vector_store = FAISS.load_local(
            folder_path=str(latest_store),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

        # Inspect docstore
        docstore = vector_store.docstore
        index_to_id = vector_store.index_to_docstore_id

        print("\n[OK] Vector store loaded successfully")
        print(f"     Total chunks indexed: {len(index_to_id)}")

        # Sample first 5 chunks to inspect metadata
        print("\n[*] Sample chunk metadata (first 5):")
        print("-" * 80)

        for i, (_idx, doc_id) in enumerate(list(index_to_id.items())[:5]):
            doc = docstore.search(doc_id)
            if doc is None:
                print(f"\n  Chunk {i + 1}: [!] Document not found in docstore")
                continue

            metadata = doc.metadata
            print(f"\n  Chunk {i + 1}:")
            print(f"    filename: {metadata.get('filename', '[MISSING]')}")
            print(f"    chunk_num: {metadata.get('chunk_num', '[MISSING]')}")
            print(f"    section_name: {metadata.get('section_name', '[MISSING]')}")
            print(f"    word_count: {metadata.get('word_count', '[MISSING]')}")
            print(f"    text preview: {doc.page_content[:80]}...")

        # Test retrieval with a sample question
        print("\n" + "=" * 80)
        print("TESTING RETRIEVAL WITH SAMPLE QUESTION")
        print("=" * 80)

        from src.core.vector_store.qa_retriever import QARetriever

        retriever = QARetriever(vector_store_path=latest_store, embeddings=embeddings)

        test_question = "What are the main allegations?"
        print(f"\nQuestion: {test_question}")

        result = retriever.retrieve_context(test_question, k=3)

        print("\n[*] Retrieval Results:")
        print(f"    Chunks retrieved: {result.chunks_retrieved}")
        print(f"    Retrieval time: {result.retrieval_time_ms:.1f}ms")
        print(f"    Context length: {len(result.context)} chars")

        print(f"\n[*] Sources ({len(result.sources)} total):")
        print("-" * 80)

        if not result.sources:
            print("  [!] No sources found!")
        else:
            for i, source in enumerate(result.sources):
                print(f"\n  Source {i + 1}:")
                print(f"    filename: {source.filename or '[EMPTY]'}")
                print(f"    chunk_num: {source.chunk_num}")
                print(f"    section: {source.section or 'N/A'}")
                print(f"    relevance_score: {source.relevance_score:.3f}")
                print(f"    word_count: {source.word_count}")
                print(f"    algorithms: {source.sources}")

        # Test source summary generation
        source_summary = retriever.get_relevant_sources_summary(result)
        print("\n[*] Source Summary:")
        print(f"    {source_summary}")

        if not source_summary or source_summary == "No sources found":
            print("\n  [!] Source summary is blank or 'No sources found'!")
            print("      This is the bug - citations exist but source_summary is empty.")
        else:
            print("\n  [OK] Source summary generated successfully")

    except Exception as e:
        print(f"\n[!] Error during inspection: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    inspect_latest_vector_store()
