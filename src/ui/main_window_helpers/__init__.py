"""
Main Window Helper Components.

Session 82: Split from main_window.py for modularity.

This package contains helper classes that provide functionality for MainWindow:
- ollama_mixin: Ollama status display and model management
- file_mixin: File selection, drag-drop, preprocessing
- task_mixin: Task execution and progressive extraction
- export_mixin: Export All and Combined Report
- timer_mixin: Processing timer and activity indicator
"""

from src.ui.main_window_helpers.export_mixin import ExportMixin
from src.ui.main_window_helpers.file_mixin import FileMixin
from src.ui.main_window_helpers.ollama_mixin import OllamaMixin
from src.ui.main_window_helpers.task_mixin import TaskMixin
from src.ui.main_window_helpers.timer_mixin import TimerMixin

__all__ = [
    "ExportMixin",
    "FileMixin",
    "OllamaMixin",
    "TaskMixin",
    "TimerMixin",
]
