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
    embeddings, vector_store_path, chunk_scores, active_worker, shutdown
"""

import logging
import threading
import traceback
from queue import Empty, Queue

logger = logging.getLogger(__name__)

# Sentinel for shutdown
_SHUTDOWN = "shutdown"
_CANCEL = "cancel"


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
        except Exception:
            # Timeout or empty -- loop back and check shutdown
            continue

        if msg == _SHUTDOWN:
            logger.info("Shutdown command received")
            _stop_active_worker(state)
            state["shutdown"].set()
            break

        if msg == _CANCEL:
            logger.info("Cancel command received")
            _stop_active_worker(state)
            continue

        try:
            cmd_type, args = msg
        except (TypeError, ValueError):
            logger.warning("Invalid command format: %s", msg)
            continue

        logger.debug("Dispatching command: %s", cmd_type)
        result_queue.put(("command_ack", {"cmd": cmd_type}))

        try:
            _dispatch_command(cmd_type, args, internal_queue, state)
        except Exception as e:
            logger.error("Command dispatch error (%s): %s", cmd_type, e)
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
    state["active_worker"] = worker
    worker.start()


def _run_extraction(args, internal_queue, state):
    """Spawn ProgressiveExtractionWorker for vocabulary extraction."""
    from src.services.workers import ProgressiveExtractionWorker

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
    state["active_worker"] = worker
    worker.start()


def _run_qa(args, internal_queue, state):
    """Spawn QAWorker for default questions."""
    from src.services.workers import QAWorker

    vector_store_path = state.get("vector_store_path")
    embeddings = state.get("embeddings")

    if not vector_store_path or not embeddings:
        internal_queue.put(("error", "Q&A not ready: no vector store or embeddings"))
        return

    worker = QAWorker(
        vector_store_path=vector_store_path,
        embeddings=embeddings,
        ui_queue=internal_queue,
        answer_mode=args.get("answer_mode", "extraction"),
        questions=args.get("questions"),
        use_default_questions=args.get("use_default_questions", True),
    )
    state["active_worker"] = worker
    worker.start()


def _run_followup(args, internal_queue, state):
    """Run a follow-up question in a background thread."""
    question = args.get("question", "")
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
            internal_queue.put(("qa_followup_result", result))
        except Exception as e:
            logger.error("Follow-up error: %s", e)
            internal_queue.put(("qa_followup_result", None))

    thread = threading.Thread(target=do_followup, daemon=True, name="followup")
    thread.start()
    # Don't track as active_worker -- followups are lightweight


def _run_summary(args, internal_queue, state):
    """Spawn MultiDocSummaryWorker for document summarization."""
    from src.services.workers import MultiDocSummaryWorker

    worker = MultiDocSummaryWorker(
        documents=args["documents"],
        ui_queue=internal_queue,
        ai_params=args.get("ai_params", {}),
    )
    state["active_worker"] = worker
    worker.start()


def _stop_active_worker(state):
    """Stop the currently active worker, if any."""
    worker = state.get("active_worker")
    if worker and hasattr(worker, "is_alive") and worker.is_alive():
        logger.debug("Stopping active worker: %s", type(worker).__name__)
        if hasattr(worker, "stop"):
            worker.stop()
        worker.join(timeout=2.0)
    state["active_worker"] = None


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

        try:
            msg_type, data = msg
        except (TypeError, ValueError):
            logger.warning("Invalid message format in forwarder: %s", msg)
            continue

        if msg_type == "qa_ready":
            # Save embeddings and vector store path in subprocess state
            state["embeddings"] = data.get("embeddings")
            state["vector_store_path"] = data.get("vector_store_path")
            state["chunk_scores"] = data.get("chunk_scores")
            logger.debug("Saved embeddings and vector_store_path in subprocess state")

            # Forward qa_ready WITHOUT embeddings (not picklable)
            forwarded_data = {
                "vector_store_path": data.get("vector_store_path"),
                "chunk_count": data.get("chunk_count", 0),
                "chunk_scores": data.get("chunk_scores"),
            }
            result_queue.put(("qa_ready", forwarded_data))

        elif msg_type == "trigger_default_qa":
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
                    state["active_worker"] = qa_worker
                    qa_worker.start()
                    logger.debug("Default QAWorker started in subprocess")
                else:
                    logger.warning(
                        "Cannot start default Q&A: missing embeddings or vector_store_path"
                    )
            except Exception as e:
                logger.error("Failed to start default QAWorker: %s", e)
                result_queue.put(("error", f"Default Q&A failed: {e}"))

        else:
            # Forward all other messages as-is
            try:
                result_queue.put((msg_type, data))
            except Exception as e:
                logger.error(
                    "Failed to forward message %s: %s\n%s",
                    msg_type,
                    e,
                    traceback.format_exc(),
                )
