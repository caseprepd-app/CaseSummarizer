"""
Worker Subprocess Entry Point

Runs pipeline work (extraction, Q&A, summarization) in a separate process
to avoid GIL contention with the Tkinter event loop. Workers write to an
internal queue.Queue; a forwarder thread bridges messages to the
multiprocessing.Queue consumed by the GUI.

Architecture:
    command_queue (mp.Queue)  --> _command_loop  --> spawns worker threads
    internal_queue (Queue)    --> _forwarder_loop --> result_queue (mp.Queue)

State dict (shared by reference within the subprocess):
    embeddings, vector_store_path, chunk_scores, active_worker, worker_lock, shutdown
"""

import logging
import os
import threading
import traceback
from queue import Empty, Queue

logger = logging.getLogger(__name__)

# Sentinels: plain strings placed on command_queue to signal control events.
# Checked by equality comparison in _command_loop() (identity won't work across pickle).
_SHUTDOWN_SENTINEL = "shutdown"
_CANCEL_SENTINEL = "cancel"


def _summarize_command_args(cmd_type, args):
    """Summarize command args for logging without full document text."""
    if not args or not isinstance(args, dict):
        return str(args)
    parts = []
    if cmd_type == "process_files":
        paths = args.get("file_paths", [])
        names = [os.path.basename(p) for p in paths] if paths else []
        parts.append(f"files={len(names)}{names}")
        if "ocr_allowed" in args:
            parts.append(f"ocr_allowed={args['ocr_allowed']}")
    elif cmd_type == "extract":
        parts.append(f"docs={len(args.get('documents', []))}")
        parts.append(f"use_llm={args.get('use_llm', '?')}")
        parts.append(f"doc_confidence={args.get('doc_confidence', '?')}")
    elif cmd_type == "run_qa":
        parts.append(f"answer_mode={args.get('answer_mode', '?')}")
        qs = args.get("questions")
        parts.append(f"questions={len(qs) if qs else 'default'}")
    elif cmd_type == "followup":
        q = args.get("question", "")
        parts.append(f"question='{q[:80]}'")
    elif cmd_type == "summary":
        parts.append(f"docs={len(args.get('documents', []))}")
        ai = args.get("ai_params", {})
        parts.append(f"model={ai.get('model_name', '?')}")
    else:
        parts.append(f"keys={list(args.keys())}")
    return ", ".join(parts)


def worker_process_main(command_queue, result_queue):
    """
    Entry point for the worker subprocess.

    Runs two threads:
    - command_loop: reads commands, spawns worker threads
    - forwarder_loop: bridges internal_queue -> result_queue

    Args:
        command_queue: multiprocessing.Queue for receiving commands from GUI
        result_queue: multiprocessing.Queue for sending messages to GUI
    """
    # Configure logging in subprocess
    logging.basicConfig(
        level=logging.DEBUG,
        format="[WorkerProcess] %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Worker subprocess started (PID: %s)", __import__("os").getpid())

    internal_queue = Queue()
    state = {
        "embeddings": None,
        "vector_store_path": None,
        "chunk_scores": None,
        "active_worker": None,
        "shutdown": threading.Event(),
        "worker_lock": threading.Lock(),
    }

    # Start forwarder thread
    forwarder = threading.Thread(
        target=_forwarder_loop,
        args=(internal_queue, result_queue, command_queue, state),
        daemon=True,
        name="forwarder",
    )
    forwarder.start()

    # Signal GUI that the worker is ready to accept commands
    result_queue.put(("worker_ready", None))
    logger.info("Worker ready signal sent")

    # Run command loop in main thread (blocks until shutdown)
    _command_loop(command_queue, internal_queue, result_queue, state)

    logger.info("Worker subprocess exiting")


def _command_loop(command_queue, internal_queue, result_queue, state):
    """
    Read commands from the GUI and dispatch to worker threads.

    Commands are (cmd_type, args_dict) tuples.
    Blocks on command_queue.get() until shutdown.

    Args:
        command_queue: mp.Queue for incoming commands
        internal_queue: thread Queue that workers write to
        result_queue: mp.Queue for direct replies (e.g. errors)
        state: shared state dict
    """
    while not state["shutdown"].is_set():
        try:
            msg = command_queue.get(timeout=1.0)
        except Empty:
            continue
        except Exception as e:
            logger.warning("Unexpected error reading command queue: %s", e)
            continue

        if msg == _SHUTDOWN_SENTINEL:
            logger.info("Shutdown command received")
            _stop_active_worker(state)
            state["shutdown"].set()
            break

        if msg == _CANCEL_SENTINEL:
            logger.info("Cancel command received")
            _stop_active_worker(state)
            continue

        try:
            cmd_type, args = msg
        except (TypeError, ValueError):
            logger.warning("Invalid command format: %s", msg)
            continue

        logger.debug("Dispatching command: %s", cmd_type)
        logger.debug("  args: %s", _summarize_command_args(cmd_type, args))
        try:
            result_queue.put(("command_ack", {"cmd": cmd_type}))
        except Exception as e:
            logger.error("Failed to send command_ack for %s: %s", cmd_type, e, exc_info=True)
            continue

        try:
            _dispatch_command(cmd_type, args, internal_queue, state)
        except Exception as e:
            logger.error("Command dispatch error (%s): %s", cmd_type, e, exc_info=True)
            result_queue.put(("error", f"Worker error: {e}"))


def _dispatch_command(cmd_type, args, internal_queue, state):
    """
    Create and start the appropriate worker for a command.

    Args:
        cmd_type: Command string (process_files, extract, run_qa, etc.)
        args: Dict of arguments for the worker
        internal_queue: Queue the worker writes messages to
        state: shared subprocess state
    """
    # Stop any running worker before starting a new one
    _stop_active_worker(state)

    if cmd_type == "process_files":
        _run_process_files(args, internal_queue, state)
    elif cmd_type == "extract":
        _run_extraction(args, internal_queue, state)
    elif cmd_type == "run_qa":
        _run_qa(args, internal_queue, state)
    elif cmd_type == "followup":
        _run_followup(args, internal_queue, state)
    elif cmd_type == "summary":
        _run_summary(args, internal_queue, state)
    else:
        logger.warning("Unknown command: %s", cmd_type)
        internal_queue.put(("error", f"Unknown command: {cmd_type}"))


def _run_process_files(args, internal_queue, state):
    """Spawn ProcessingWorker for document extraction."""
    from src.services.workers import ProcessingWorker

    worker = ProcessingWorker(
        file_paths=args["file_paths"],
        ui_queue=internal_queue,
        ocr_allowed=args.get("ocr_allowed", True),
    )
    with state["worker_lock"]:
        state["active_worker"] = worker
    worker.start()
    logger.debug("Worker thread started: %s (thread: %s)", type(worker).__name__, worker.name)


def _run_extraction(args, internal_queue, state):
    """Spawn ProgressiveExtractionWorker for vocabulary extraction."""
    from src.services.workers import ProgressiveExtractionWorker

    # Save checkbox state so trigger_default_qa can check it later
    state["ask_default_questions"] = args.get("ask_default_questions", True)

    worker = ProgressiveExtractionWorker(
        documents=args["documents"],
        combined_text=args["combined_text"],
        ui_queue=internal_queue,
        embeddings=state.get("embeddings"),
        exclude_list_path=args.get("exclude_list_path"),
        medical_terms_path=args.get("medical_terms_path"),
        user_exclude_path=args.get("user_exclude_path"),
        doc_confidence=args.get("doc_confidence", 100.0),
        use_llm=args.get("use_llm", True),
    )
    with state["worker_lock"]:
        state["active_worker"] = worker
    worker.start()
    logger.debug("Worker thread started: %s (thread: %s)", type(worker).__name__, worker.name)


def _run_qa(args, internal_queue, state):
    """Spawn QAWorker for default questions."""
    vector_store_path = state.get("vector_store_path")
    embeddings = state.get("embeddings")

    if not vector_store_path or not embeddings:
        internal_queue.put(("error", "Q&A not ready: no vector store or embeddings"))
        return

    from src.services.workers import QAWorker

    worker = QAWorker(
        vector_store_path=vector_store_path,
        embeddings=embeddings,
        ui_queue=internal_queue,
        answer_mode=args.get("answer_mode", "extraction"),
        questions=args.get("questions"),
        use_default_questions=args.get("use_default_questions", True),
    )
    with state["worker_lock"]:
        state["active_worker"] = worker
    worker.start()
    logger.debug("Worker thread started: %s (thread: %s)", type(worker).__name__, worker.name)


def _run_followup(args, internal_queue, state):
    """Run a follow-up question in a background thread."""
    question = args.get("question", "")
    logger.debug("Follow-up question: %.80s", question)
    vector_store_path = state.get("vector_store_path")
    embeddings = state.get("embeddings")

    if not vector_store_path or not embeddings:
        internal_queue.put(("qa_followup_result", None))
        return

    def do_followup():
        try:
            from src.core.qa import QAOrchestrator
            from src.user_preferences import get_user_preferences

            prefs = get_user_preferences()
            orchestrator = QAOrchestrator(
                vector_store_path=vector_store_path,
                embeddings=embeddings,
                answer_mode=prefs.get("qa_answer_mode", "ollama"),
            )
            result = orchestrator.ask_followup(question)
            logger.debug("Follow-up answered: %d chars", len(result.answer) if result else 0)
            internal_queue.put(("qa_followup_result", result))
        except Exception as e:
            logger.error("Follow-up error: %s", e, exc_info=True)
            internal_queue.put(("qa_followup_result", None))

    thread = threading.Thread(target=do_followup, daemon=True, name="followup")
    thread.start()
    # Don't track as active_worker -- followups are lightweight


def _run_summary(args, internal_queue, state):
    """Spawn MultiDocSummaryWorker for document summarization."""
    from src.services.workers import MultiDocSummaryWorker

    ai_params = args.get("ai_params", {})
    # Inject chunk_scores from subprocess state (saved during qa_ready)
    if "chunk_scores" not in ai_params and state.get("chunk_scores"):
        ai_params["chunk_scores"] = state["chunk_scores"]

    worker = MultiDocSummaryWorker(
        documents=args["documents"],
        ui_queue=internal_queue,
        ai_params=ai_params,
    )
    with state["worker_lock"]:
        state["active_worker"] = worker
    worker.start()
    logger.debug("Worker thread started: %s (thread: %s)", type(worker).__name__, worker.name)


def _stop_active_worker(state):
    """Stop the currently active worker, if any."""
    with state["worker_lock"]:
        worker = state.get("active_worker")
    if worker and hasattr(worker, "is_alive") and worker.is_alive():
        logger.debug("Stopping active worker: %s", type(worker).__name__)
        if hasattr(worker, "stop"):
            worker.stop()
        worker.join(timeout=2.0)
    with state["worker_lock"]:
        state["active_worker"] = None

    # Also stop auto-spawned QA worker if running
    with state["worker_lock"]:
        auto_qa = state.get("auto_qa_worker")
    if auto_qa and hasattr(auto_qa, "is_alive") and auto_qa.is_alive():
        if hasattr(auto_qa, "stop"):
            auto_qa.stop()
        auto_qa.join(timeout=2.0)
    with state["worker_lock"]:
        state["auto_qa_worker"] = None


def _forwarder_loop(internal_queue, result_queue, command_queue, state):
    """
    Forward messages from internal_queue to result_queue.

    Intercepts:
    - qa_ready: saves embeddings/vector_store_path in state, strips
      embeddings from forwarded message (not picklable)
    - trigger_default_qa: auto-spawns QAWorker in subprocess, sends
      trigger_default_qa_started to GUI instead

    All other messages are forwarded as-is.

    Args:
        internal_queue: thread Queue workers write to
        result_queue: mp.Queue for GUI consumption
        command_queue: mp.Queue for receiving additional commands
        state: shared subprocess state
    """
    while not state["shutdown"].is_set():
        try:
            msg = internal_queue.get(timeout=0.5)
        except Empty:
            continue
        except Exception as exc:
            logger.error("Forwarder loop error reading internal queue: %s", exc, exc_info=True)
            try:
                result_queue.put(("error", f"Internal forwarder error: {exc}"))
            except Exception as inner_exc:
                logger.error(
                    "Failed to send error to GUI via result_queue: %s", inner_exc, exc_info=True
                )
            continue

        try:
            msg_type, data = msg
        except (TypeError, ValueError):
            logger.warning("Invalid message format in forwarder: %s", msg)
            continue

        try:
            _forward_message(msg_type, data, internal_queue, result_queue, state)
        except Exception as exc:
            logger.error("Forwarder loop error processing %s: %s", msg_type, exc, exc_info=True)
            try:
                result_queue.put(("error", f"Internal error processing {msg_type}: {exc}"))
            except Exception as inner_exc:
                logger.error(
                    "Failed to send error to GUI via result_queue: %s", inner_exc, exc_info=True
                )


def _forward_message(msg_type, data, internal_queue, result_queue, state):
    """
    Process and forward a single message from the internal queue.

    Extracted from _forwarder_loop for error isolation.

    Args:
        msg_type: Message type string
        data: Message data payload
        internal_queue: thread Queue workers write to
        result_queue: mp.Queue for GUI consumption
        state: shared subprocess state
    """
    if msg_type == "qa_ready":
        # Save embeddings and vector store path in subprocess state
        state["embeddings"] = data.get("embeddings")
        state["vector_store_path"] = data.get("vector_store_path")
        state["chunk_scores"] = data.get("chunk_scores")
        logger.debug(
            "Saved embeddings and vector_store_path in subprocess state (path=%s, chunks=%s)",
            data.get("vector_store_path"),
            data.get("chunk_count", "?"),
        )

        # Forward qa_ready WITHOUT embeddings (not picklable)
        forwarded_data = {
            "vector_store_path": data.get("vector_store_path"),
            "chunk_count": data.get("chunk_count", 0),
            "chunk_scores": data.get("chunk_scores"),
        }
        result_queue.put(("qa_ready", forwarded_data))

    elif msg_type == "trigger_default_qa":
        # Skip if user unchecked the default questions checkbox
        if not state.get("ask_default_questions", True):
            logger.debug("Default questions disabled by user, skipping QAWorker")
            result_queue.put(("qa_complete", []))
            return

        # Auto-spawn QAWorker in subprocess instead of forwarding
        logger.debug("Intercepted trigger_default_qa, auto-spawning QAWorker")
        result_queue.put(("trigger_default_qa_started", None))

        # Spawn QAWorker using saved state
        try:
            from src.services.workers import QAWorker
            from src.user_preferences import get_user_preferences

            embeddings = state.get("embeddings")
            vector_store_path = data.get("vector_store_path") or state.get("vector_store_path")

            if embeddings and vector_store_path:
                prefs = get_user_preferences()
                qa_worker = QAWorker(
                    vector_store_path=vector_store_path,
                    embeddings=embeddings,
                    ui_queue=internal_queue,
                    answer_mode=prefs.get("qa_answer_mode", "ollama"),
                    questions=None,
                    use_default_questions=True,
                )
                with state["worker_lock"]:
                    state["auto_qa_worker"] = qa_worker
                qa_worker.start()
                logger.debug("Default QAWorker started in subprocess")
            else:
                logger.warning("Cannot start default Q&A: missing embeddings or vector_store_path")
                result_queue.put(("qa_complete", []))
        except Exception as e:
            logger.error("Failed to start default QAWorker: %s", e, exc_info=True)
            result_queue.put(("error", f"Default Q&A failed: {e}"))

    else:
        # Forward all other messages as-is
        logger.debug("Forwarding message: %s", msg_type)
        try:
            result_queue.put((msg_type, data))
        except Exception as e:
            logger.error(
                "Failed to forward message %s: %s\n%s",
                msg_type,
                e,
                traceback.format_exc(),
            )
