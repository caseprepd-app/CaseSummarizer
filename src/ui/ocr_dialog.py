"""
OCR Installation Dialog.

Shows a modal dialog when Tesseract OCR is needed but not installed.
Offers three choices: download Tesseract, skip OCR for this session,
or snooze the prompt for 90 days.

Example:
    >>> dialog = OCRDialog(parent_window)
    >>> dialog.wait_window()
    >>> print(dialog.result)  # "download", "skip", or "snooze"
"""

import logging
import webbrowser

import customtkinter as ctk

from src.ui.base_dialog import BaseModalDialog

logger = logging.getLogger(__name__)

TESSERACT_DOWNLOAD_URL = "https://github.com/UB-Mannheim/tesseract/wiki"


class OCRDialog(BaseModalDialog):
    """
    Modal dialog prompting the user to install Tesseract OCR.

    Displayed when a scanned document is detected but Tesseract is not
    installed. The user can choose to download, skip, or snooze.

    Attributes:
        result: User's choice — "download", "skip", or "snooze".
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
            title="Tesseract OCR Required",
            width=480,
            height=300,
            resizable=False,
        )

        self._build_ui()

        # Block until user responds
        self.wait_window()

    def _build_ui(self):
        """Build the dialog UI with explanation and three buttons."""
        # Explanation text
        msg_frame = ctk.CTkFrame(self, fg_color="transparent")
        msg_frame.pack(fill="x", padx=20, pady=(20, 10))

        ctk.CTkLabel(
            msg_frame,
            text="Scanned Document Detected",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w",
        ).pack(fill="x")

        ctk.CTkLabel(
            msg_frame,
            text=(
                "One or more documents appear to be scanned images. "
                "Tesseract OCR is needed to read scanned text but isn't "
                "installed on this computer.\n\n"
                "Without Tesseract, scanned pages will be processed with "
                "lower accuracy using only digital text extraction."
            ),
            wraplength=440,
            justify="left",
            anchor="w",
        ).pack(fill="x", pady=(8, 0))

        # Clickable info link
        info_link = ctk.CTkLabel(
            msg_frame,
            text="What is Tesseract OCR? Learn more...",
            text_color=("#1a73e8", "#8ab4f8"),
            cursor="hand2",
            anchor="w",
        )
        info_link.pack(fill="x", pady=(4, 0))
        info_link.bind(
            "<Button-1>",
            lambda e: webbrowser.open("https://tesseract-ocr.github.io/"),
        )

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(10, 20), side="bottom")

        ctk.CTkButton(
            btn_frame,
            text="Download Tesseract",
            command=self._on_download,
            width=160,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_frame,
            text="Not Now",
            command=self._on_skip,
            fg_color="gray40",
            width=120,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_frame,
            text="Don't Ask Again",
            command=self._on_snooze,
            fg_color="gray40",
            width=140,
        ).pack(side="left")

    def _on_download(self):
        """Open the Tesseract download page and close dialog."""
        logger.debug("User chose to download Tesseract")
        webbrowser.open(TESSERACT_DOWNLOAD_URL)
        self.result = "download"
        self.close()

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
