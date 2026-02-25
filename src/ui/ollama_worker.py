import logging
import multiprocessing
import time
import traceback

from src.config import QUEUE_TIMEOUT_SECONDS
from src.services import AIService

logger = logging.getLogger(__name__)


def ollama_generation_worker_process(
    input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue
):
    """
    Target function for the multiprocessing worker that handles Ollama AI generation.
    """
    model_manager = None
    model_initialized = False
    try:
        # Instantiate OllamaModelManager within the new process via AIService
        ai_service = AIService()
        model_manager = ai_service.get_ollama_manager()
        model_initialized = True
        logger.info("OllamaModelManager instantiated via AIService")

        while True:
            # Check for termination signal or new tasks
            try:
                task = input_queue.get(
                    timeout=QUEUE_TIMEOUT_SECONDS
                )  # Timeout to allow checking for termination
                if task == "TERMINATE":
                    logger.info("Termination signal received, exiting")
                    break

                # Unpack task
                task_type, payload = task

                if task_type == "GENERATE_SUMMARY":
                    # Unpack payload for generate_summary
                    case_text = payload["case_text"]
                    max_words = payload["max_words"]
                    preset_id = payload["preset_id"]

                    logger.debug(
                        "Received GENERATE_SUMMARY task. Preset: %s, Max words: %s",
                        preset_id,
                        max_words,
                    )

                    start_time = time.time()

                    # Instead of directly calling, we'll monitor for termination during the potentially long call
                    # We can't interrupt the requests.post directly, but we can monitor before/after.
                    # Send an initial progress message
                    output_queue.put(
                        ("progress", (0, "Starting AI generation (this may take a while)..."))
                    )

                    # The generate_summary itself is blocking, so we cannot send continuous updates from within it.
                    # However, we can track if a terminate signal comes *before* or *after* the call.
                    # The requests.post call does have a timeout, which is the primary control.

                    # Check for termination right before the blocking call
                    # Use try/except to avoid TOCTOU race with empty() + get_nowait()
                    try:
                        if input_queue.get_nowait() == "TERMINATE":
                            logger.debug(
                                "Termination signal received before generate_summary, aborting task"
                            )
                            output_queue.put(("cancelled", "AI generation task cancelled."))
                            continue  # Skip the generation and await next task/termination
                    except Exception:
                        pass  # No termination signal, proceed with generation

                    summary = model_manager.generate_summary(
                        case_text=case_text, max_words=max_words, preset_id=preset_id
                    )

                    # After generation, check again for termination
                    # Use try/except to avoid TOCTOU race with empty() + get_nowait()
                    try:
                        if input_queue.get_nowait() == "TERMINATE":
                            logger.debug(
                                "Termination signal received after generate_summary, discarding result"
                            )
                            output_queue.put(
                                ("cancelled", "AI generation task cancelled after completion.")
                            )
                            continue
                    except Exception:
                        pass  # No termination signal, send the result

                    elapsed_time = time.time() - start_time
                    logger.info("Summary generated in %.2f seconds", elapsed_time)
                    output_queue.put(
                        (
                            "summary_result",
                            {
                                "type": "individual",
                                "filename": payload.get("filename"),
                                "summary": summary,
                            },
                        )
                    )

                elif task_type == "LOAD_MODEL":
                    model_name = payload["model_name"]
                    logger.debug("Received LOAD_MODEL task for %s", model_name)

                    # Ensure the model manager's model is set for the worker process
                    # This will also trigger the model to be pulled if not available
                    output_queue.put(("progress", (0, f"Loading AI model: {model_name}...")))

                    # Monitor load progress by checking terminate signal
                    load_start_time = time.time()
                    model_loaded = False
                    while not model_loaded:
                        try:
                            if (
                                input_queue.get(timeout=QUEUE_TIMEOUT_SECONDS) == "TERMINATE"
                            ):  # Check termination while loading
                                logger.debug(
                                    "Termination signal received during model loading, aborting"
                                )
                                output_queue.put(("cancelled", "Model loading cancelled."))
                                return  # Exit process if terminated during critical load
                        except multiprocessing.queues.Empty:
                            pass  # No termination signal, continue loading attempt

                        # Attempt to load model - this call blocks until model is ready
                        if model_manager.load_model(model_name):
                            model_loaded = True
                            break
                        else:
                            # Ollama might take some time to pull, or there might be connection issues
                            elapsed = time.time() - load_start_time
                            output_queue.put(
                                (
                                    "progress",
                                    (
                                        0,
                                        f"Loading AI model: {model_name} (elapsed: {elapsed:.1f}s)...",
                                    ),
                                )
                            )
                            time.sleep(5)  # Wait a bit before retrying load/checking status

                    if model_loaded:
                        output_queue.put(("model_loaded", model_name))
                    else:
                        output_queue.put(
                            (
                                "model_load_failed",
                                f"Failed to load {model_name} after multiple attempts.",
                            )
                        )

            except multiprocessing.queues.Empty:
                # If no task, and model manager is loaded, send a heartbeat to keep UI updated
                if model_manager and model_manager.is_connected:
                    output_queue.put(("heartbeat", None))
                pass
            except Exception as e:
                error_msg = f"Ollama worker process error: {e!s}\n{traceback.format_exc()}"
                logger.debug("Worker process error: %s", error_msg)
                output_queue.put(("error", error_msg))

    except Exception as e:
        error_msg = (
            f"Critical error in Ollama worker process setup: {e!s}\n{traceback.format_exc()}"
        )
        logger.debug("Critical setup error: %s", error_msg)
        output_queue.put(("error", error_msg))
    finally:
        # Ensure model_manager is cleaned up if necessary, though Ollama handles this
        if model_manager and model_initialized:
            model_manager.unload_model()
        logger.info("Worker process finished")
