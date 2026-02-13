"""
OCR Installation Dialog (Safety Net).

Shows a modal dialog when bundled OCR files are missing or damaged.
This should never appear in normal operation -- it's a fallback for
corrupted installations.

Example:
    >>> dialog = OCRDialog(parent_window)
    >>> dialog.wait_window()
    >>> print(dialog.result)  # "skip" or "snooze"
"""

import logging

import customtkinter as ctk

from src.ui.base_dialog import BaseModalDialog

logger = logging.getLogger(__name__)


class OCRDialog(BaseModalDialog):
    """
    Modal dialog shown when bundled OCR files are missing or damaged.

    Displayed when a scanned document is detected but the bundled
    Tesseract/Poppler binaries cannot be found. The user can choose
    to skip OCR or snooze the prompt.

    Attributes:
        result: User's choice -- "skip" or "snooze".
    """

    def __init__(self, parent):
        """
        Initialize the OCR dialog.

        Args:
            parent: Parent window (dialog is modal to this).
        """
        self.result = "skip"  # Default if window is closed

        super().__init__(
            parent=parent,
            title="OCR Unavailable",
            width=480,
            height=280,
            resizable=False,
        )

        self._build_ui()

        # Block until user responds
        self.wait_window()

    def _build_ui(self):
        """Build the dialog UI with explanation and two buttons."""
        # Explanation text
        msg_frame = ctk.CTkFrame(self, fg_color="transparent")
        msg_frame.pack(fill="x", padx=20, pady=(20, 10))

        ctk.CTkLabel(
            msg_frame,
            text="OCR Files Missing",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w",
        ).pack(fill="x")

        ctk.CTkLabel(
            msg_frame,
            text=(
                "One or more documents appear to be scanned images, but "
                "the OCR files needed to read them are missing or damaged.\n\n"
                "Try reinstalling the application to restore the OCR "
                "components (Tesseract and Poppler).\n\n"
                "Without OCR, scanned pages will be processed with lower "
                "accuracy using only digital text extraction."
            ),
            wraplength=440,
            justify="left",
            anchor="w",
        ).pack(fill="x", pady=(8, 0))

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(10, 20), side="bottom")

        ctk.CTkButton(
            btn_frame,
            text="Skip OCR",
            command=self._on_skip,
            width=120,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_frame,
            text="Don't Ask Again",
            command=self._on_snooze,
            fg_color="gray40",
            width=140,
        ).pack(side="left")

    def _on_skip(self):
        """Skip OCR for this session."""
        logger.debug("User chose to skip OCR")
        self.result = "skip"
        self.close()

    def _on_snooze(self):
        """Snooze the OCR prompt for 90 days."""
        logger.debug("User chose to snooze OCR prompt")
        self.result = "snooze"
        self.close()
