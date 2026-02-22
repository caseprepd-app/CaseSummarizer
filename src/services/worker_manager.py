"""
Worker Process Manager

GUI-side manager for the persistent worker subprocess. Handles process
lifecycle (start, shutdown, restart) and provides a clean API for
sending commands and receiving results via multiprocessing.Queue.

Follows the same pattern as OllamaAIWorkerManager (workers.py:572-686).

Usage:
    manager = WorkerProcessManager()
    manager.start()
    manager.send_command("process_files", {"file_paths": [...]})
    messages = manager.check_for_messages()  # non-blocking drain
    manager.shutdown(blocking=True)
"""

import gc
import logging
import multiprocessing
from queue import Empty

logger = logging.getLogger(__name__)


class WorkerProcessManager:
    """
    Manages the persistent worker subprocess for pipeline tasks.

    The subprocess runs all heavy work (extraction, Q&A, summarization)
    in isolation from the GUI process, eliminating GIL contention.

    Attributes:
        command_queue: mp.Queue for sending commands to subprocess
        result_queue: mp.Queue for receiving messages from subprocess
        process: the subprocess.Process instance (or None)
    """

    def __init__(self):
        """Initialize queues and state. Does not start the subprocess."""
        self.command_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.process = None
        self._started = False
        self._worker_ready = False

    def start(self):
        """
        Spawn the worker subprocess.

        Safe to call multiple times -- skips if already running.
        """
        if self._started and self.process and self.process.is_alive():
            logger.debug("Worker subprocess already running (PID: %s)", self.process.pid)
            return

        # Clear stale messages from previous runs
        self._clear_queue(self.command_queue)
        self._clear_queue(self.result_queue)

        from src.worker_process import worker_process_main

        logger.debug("Starting worker subprocess...")
        self.process = multiprocessing.Process(
            target=worker_process_main,
            args=(self.command_queue, self.result_queue),
            daemon=True,
        )
        self.process.start()
        self._started = True
        logger.info("Worker subprocess started (PID: %s)", self.process.pid)

    def send_command(self, cmd_type, args=None):
        """
        Send a command to the worker subprocess.

        Auto-starts the subprocess if not running.

        Args:
            cmd_type: Command string (process_files, extract, run_qa, etc.)
            args: Dict of arguments for the command
        """
        if not self.is_alive():
            logger.debug("Worker not alive, restarting before command")
            self.restart_if_dead()

        logger.debug("Sending command: %s", cmd_type)
        self.command_queue.put((cmd_type, args or {}))

    def check_for_messages(self):
        """
        Drain all available messages from the result queue (non-blocking).

        Intercepts internal messages (worker_ready) and does not forward them.

        Returns:
            List of (msg_type, data) tuples
        """
        messages = []
        while True:
            try:
                msg = self.result_queue.get_nowait()
            except Empty:
                break

            # Intercept worker_ready — set flag, don't forward to GUI
            try:
                msg_type, _data = msg
            except (TypeError, ValueError):
                messages.append(msg)
                continue

            if msg_type == "worker_ready":
                self._worker_ready = True
                logger.info("Worker subprocess is ready")
            else:
                messages.append(msg)

        return messages

    def cancel(self):
        """Send cancel command to stop the active worker in the subprocess."""
        if self.is_alive():
            logger.debug("Sending cancel command")
            self.command_queue.put("cancel")

    def is_alive(self):
        """
        Check if the worker subprocess is running.

        Returns:
            bool: True if process exists and is alive
        """
        return self.process is not None and self.process.is_alive()

    def is_ready(self):
        """
        Check if the worker subprocess is alive and ready to accept commands.

        Returns:
            bool: True if process is alive and has sent the worker_ready signal
        """
        return self._worker_ready and self.is_alive()

    def restart_if_dead(self):
        """Restart the subprocess if it has crashed or exited."""
        if not self.is_alive():
            logger.warning("Worker subprocess is dead, restarting...")
            self._cleanup_dead_process()
            self.start()

    def shutdown(self, blocking=True):
        """
        Gracefully shut down the worker subprocess.

        Args:
            blocking: If True, wait for process to exit. If False, send
                     shutdown signal and return immediately.
        """
        if not self._started:
            return

        if self.process and self.process.is_alive():
            logger.debug("Sending shutdown command to worker subprocess")
            try:
                self.command_queue.put("shutdown")
                if blocking:
                    self.process.join(timeout=5.0)
                else:
                    self.process.join(timeout=0.5)
            except Exception as e:
                logger.debug("Error during graceful shutdown: %s", e)

            # Force terminate if still alive
            if self.process and self.process.is_alive():
                logger.debug("Force-terminating worker subprocess")
                try:
                    self.process.terminate()
                    self.process.join(timeout=1.0)
                except Exception as e:
                    logger.debug("Error during force terminate: %s", e)

        self._cleanup_dead_process()
        logger.info("Worker subprocess shut down")

    def _cleanup_dead_process(self):
        """Clean up references after process exits."""
        self._clear_queue(self.command_queue)
        self._clear_queue(self.result_queue)
        self.process = None
        self._started = False
        self._worker_ready = False
        gc.collect()

    @staticmethod
    def _clear_queue(queue):
        """Safely drain all messages from a queue."""
        cleared = 0
        while True:
            try:
                queue.get_nowait()
                cleared += 1
            except Empty:
                break
        if cleared > 0:
            logger.debug("Cleared %s items from queue", cleared)
