"""
Command-line interface for the CasePrepd pipeline.

Runs documents through extraction, vocabulary, semantic indexing, and
key-excerpt extraction without launching the GUI. Workers are driven
in-process by calling .execute() synchronously and draining the
internal Queue for results.

Usage:
    python -m src.cli --input docs/ --output out/
    python -m src.cli --input a.pdf b.pdf --output out/ --only vocab
    python -m src.cli --input docs/ --output out/ --query "knee injury" --format json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from queue import Empty, Queue

from src.logging_config import setup_logging
from src.services.queue_messages import MessageType

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".pdf", ".txt", ".rtf", ".docx", ".png", ".jpg", ".jpeg"}


def collect_inputs(inputs: list[str]) -> list[str]:
    """Expand each input path: a file is kept, a directory is globbed."""
    files: list[str] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTS:
                    files.append(str(child))
        elif p.is_file():
            files.append(str(p))
        else:
            print(f"warning: input not found: {raw}", file=sys.stderr)
    if not files:
        raise SystemExit("error: no supported input files found")
    return files


def drain(q: Queue) -> list[tuple]:
    """Drain a Queue non-blockingly into a list of (msg_type, data) tuples."""
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except Empty:
            return out


def run_processing(file_paths: list[str], q: Queue) -> list[dict]:
    """Run extraction + preprocessing synchronously; return processed docs."""
    from src.services.processing_worker import ProcessingWorker

    worker = ProcessingWorker(file_paths=file_paths, ui_queue=q, ocr_allowed=True)
    print(f"Extracting {len(file_paths)} document(s)...")
    worker.execute()
    return worker.processed_results


def run_vocab_and_index(documents: list[dict], q: Queue) -> dict:
    """Run vocabulary + semantic indexing; return collected results."""
    from src.services.progressive_extraction_worker import ProgressiveExtractionWorker

    combined = "\n\n".join(
        d.get("preprocessed_text") or d.get("extracted_text", "") for d in documents
    )
    worker = ProgressiveExtractionWorker(
        documents=documents,
        combined_text=combined,
        ui_queue=q,
    )
    print("Extracting vocabulary and building semantic index (this can take a few minutes)...")
    worker.execute()

    collected: dict = {"vocab": [], "semantic_ready": None}
    for msg_type, data in drain(q):
        if msg_type == MessageType.NER_COMPLETE:
            collected["vocab"] = data.get("vocab", [])
        elif msg_type == MessageType.SEMANTIC_READY:
            collected["semantic_ready"] = data
    return collected


def run_key_excerpts(documents: list[dict], semantic_data: dict, vocab: list[dict]) -> list[dict]:
    """Run key-excerpt extraction synchronously using indexed chunks."""
    if not semantic_data or not semantic_data.get("chunk_texts"):
        return []
    import numpy as np

    from src.core.summarization.key_sentences import extract_key_passages
    from src.worker_process import _build_scorer_inputs

    print("Extracting key excerpts...")
    embeddings_array = np.array(semantic_data["chunk_embeddings"], dtype=np.float32)
    total_pages = sum((d.get("page_count") or 0) for d in documents)
    results = extract_key_passages(
        chunk_texts=semantic_data["chunk_texts"],
        chunk_embeddings=embeddings_array,
        chunk_metadata=semantic_data["chunk_metadata"],
        total_pages=total_pages,
        scorer_inputs=_build_scorer_inputs(vocab),
    )
    return [
        {"text": ks.text, "source_file": ks.source_file, "position": ks.position, "score": ks.score}
        for ks in results
    ]


def run_query(semantic_data: dict, query: str):
    """Run a single follow-up search; returns SemanticResult or None."""
    if not semantic_data:
        print("warning: semantic index unavailable, skipping --query", file=sys.stderr)
        return None
    from src.core.semantic import SemanticOrchestrator

    print(f"Running search: {query!r}")
    orch = SemanticOrchestrator(
        vector_store_path=semantic_data["vector_store_path"],
        embeddings=semantic_data["embeddings"],
    )
    return orch.ask_followup(query)


def write_human(out: Path, vocab, key_excerpts, search_result, only: set):
    """Write Word/TXT human-facing outputs via core export functions."""
    from src.core.export import (
        WordDocumentBuilder,
        export_combined,
        export_vocabulary,
    )

    if "vocab" in only and vocab:
        b = WordDocumentBuilder()
        export_vocabulary(vocab, b, include_details=False, is_single_doc=False)
        b.save(str(out / "vocabulary.docx"))
    if "excerpts" in only and key_excerpts:
        path = out / "key_excerpts.txt"
        path.write_text(
            "\n\n---\n\n".join(
                f"[{e['source_file']} | score={e['score']:.3f}]\n{e['text']}" for e in key_excerpts
            ),
            encoding="utf-8",
        )
    if "combined" in only:
        results = [search_result] if search_result else []
        b = WordDocumentBuilder()
        export_combined(
            vocab,
            results,
            b,
            include_vocab_details=False,
            include_verification=False,
            summary_text="",
        )
        b.save(str(out / "combined.docx"))


def write_json(out: Path, vocab, key_excerpts, search_result, only: set):
    """Write machine-readable JSON outputs."""
    if "vocab" in only:
        (out / "vocabulary.json").write_text(
            json.dumps(vocab, indent=2, default=str), encoding="utf-8"
        )
    if "excerpts" in only:
        (out / "excerpts.json").write_text(json.dumps(key_excerpts, indent=2), encoding="utf-8")
    if "combined" in only and search_result is not None:
        payload = {
            "question": search_result.question,
            "citation": search_result.citation,
            "source": search_result.source_summary,
            "relevance": search_result.relevance,
        }
        (out / "search.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(prog="caseprepd-cli", description=__doc__)
    p.add_argument("--input", nargs="+", required=True, help="Input file(s) or directory")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument(
        "--only",
        choices=["vocab", "excerpts", "combined"],
        action="append",
        help="Limit outputs (repeatable). Default: all three.",
    )
    p.add_argument("--query", help="Run a search with this phrase (default: none)")
    p.add_argument("--format", choices=["human", "json", "both"], default="human")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    files = collect_inputs(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    only = set(args.only) if args.only else {"vocab", "excerpts", "combined"}

    q: Queue = Queue()
    documents = run_processing(files, q)
    drain(q)
    if not documents:
        raise SystemExit("error: no documents extracted successfully")

    collected = run_vocab_and_index(documents, q)
    vocab = collected["vocab"]
    key_excerpts = run_key_excerpts(documents, collected["semantic_ready"], vocab)
    search_result = run_query(collected["semantic_ready"], args.query) if args.query else None

    if args.format in ("human", "both"):
        write_human(out, vocab, key_excerpts, search_result, only)
    if args.format in ("json", "both"):
        write_json(out, vocab, key_excerpts, search_result, only)

    print(f"Done. Outputs written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
