"""
Corpus Management Dialog for LocalScribe.

Full-featured dialog for managing multiple corpora. Allows users to:
- Create, delete, and combine corpora
- Add documents to the active corpus
- Preprocess documents for BM25 vocabulary extraction
- View preprocessing status of all corpus documents

Educational content explains what a corpus is and why it matters.

Privacy: All data is stored locally in %APPDATA%/LocalScribe/corpora/
"""

import os
import subprocess
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk

from src.config import DEBUG_MODE
from src.logging_config import debug_log
from src.ui.theme import FONTS, COLORS
from src.core.vocabulary import (
    CorpusRegistry,
    CorpusInfo,
    CorpusManager,
    CorpusFile,
    get_corpus_registry,
)

# BM25 Wikipedia link
BM25_WIKI_URL = "https://en.wikipedia.org/wiki/Okapi_BM25"


class CorpusDialog(ctk.CTkToplevel):
    """
    Dialog for managing vocabulary corpora.

    Provides full corpus management capabilities including create, delete,
    combine, and document management with preprocessing status.

    Example:
        dialog = CorpusDialog(parent_window)
        dialog.wait_window()  # Blocks until closed
        if dialog.corpus_changed:
            # Refresh BM25 index
            pass
    """

    def __init__(self, parent):
        """
        Initialize corpus management dialog.

        Args:
            parent: Parent window
        """
        super().__init__(parent)

        self.corpus_changed = False
        self.registry = get_corpus_registry()
        self._selected_corpus: str | None = None
        self._corpus_manager: CorpusManager | None = None

        # Window configuration
        self.title("Corpus Management")
        self.geometry("800x650")
        self.minsize(700, 550)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 800) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 650) // 2
        self.geometry(f"+{x}+{y}")

        # Build UI
        self._create_ui()

        # Load initial data
        self._refresh_corpus_list()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if DEBUG_MODE:
            debug_log("[CorpusDialog] Opened")

    def _create_ui(self):
        """Build the dialog UI."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)  # Corpus list expands
        self.grid_rowconfigure(4, weight=2)  # Document list expands more

        # === Section 1: Educational Header ===
        self._create_header()

        # === Section 2: Corpus List ===
        self._create_corpus_section()

        # === Section 3: Document List ===
        self._create_document_section()

        # === Section 4: Status Bar & Close Button ===
        self._create_footer()

    def _create_header(self):
        """Create the educational header explaining what a corpus is."""
        header = ctk.CTkFrame(self, fg_color=("gray90", "gray17"))
        header.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 10))
        header.grid_columnconfigure(0, weight=1)

        # Title
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 5))

        title = ctk.CTkLabel(
            title_frame,
            text="📚 What is a Corpus?",
            font=FONTS["heading_xl"],
            anchor="w"
        )
        title.pack(anchor="w")

        # Explanation text
        explanation = ctk.CTkLabel(
            header,
            text=(
                "A corpus is a collection of YOUR past transcripts that helps LocalScribe\n"
                "understand which words are common in your work vs. unusual for a specific case."
            ),
            font=FONTS["heading_sm"],
            anchor="w",
            justify="left"
        )
        explanation.grid(row=1, column=0, sticky="w", padx=15, pady=(0, 5))

        # Privacy note
        privacy_frame = ctk.CTkFrame(header, fg_color="transparent")
        privacy_frame.grid(row=2, column=0, sticky="w", padx=15, pady=(0, 5))

        check1 = ctk.CTkLabel(
            privacy_frame,
            text="✓ 100% local and offline - never leaves your machine",
            font=FONTS["body"],
            text_color=(COLORS["success"], COLORS["success_light"])
        )
        check1.pack(anchor="w")

        # BM25 link frame
        link_frame = ctk.CTkFrame(header, fg_color="transparent")
        link_frame.grid(row=3, column=0, sticky="w", padx=15, pady=(0, 5))

        check2 = ctk.CTkLabel(
            link_frame,
            text="✓ Powers the BM25 vocabulary algorithm",
            font=FONTS["body"],
            text_color=(COLORS["success"], COLORS["success_light"])
        )
        check2.pack(side="left")

        learn_more = ctk.CTkButton(
            link_frame,
            text="Learn more ↗",
            font=FONTS["small"],
            fg_color="transparent",
            text_color=("#1f6aa5", "#3B8ED0"),
            hover_color=("gray80", "gray30"),
            width=80,
            height=24,
            command=self._open_bm25_wiki
        )
        learn_more.pack(side="left", padx=(5, 0))

        # Power user note
        note = ctk.CTkLabel(
            header,
            text=(
                "💡 Most users need only ONE corpus. Power users may want separate\n"
                "     corpora for different case types (e.g., Criminal vs. Civil)."
            ),
            font=FONTS["body"],
            text_color=COLORS["text_secondary"],
            anchor="w",
            justify="left"
        )
        note.grid(row=4, column=0, sticky="w", padx=15, pady=(5, 15))

    def _create_corpus_section(self):
        """Create the corpus list section."""
        # Section header
        section_header = ctk.CTkFrame(self, fg_color="transparent")
        section_header.grid(row=1, column=0, sticky="ew", padx=15, pady=(10, 5))

        label = ctk.CTkLabel(
            section_header,
            text="Your Corpora",
            font=FONTS["heading"]
        )
        label.pack(side="left")

        # Corpus list frame
        list_frame = ctk.CTkFrame(self)
        list_frame.grid(row=2, column=0, sticky="nsew", padx=15, pady=(0, 10))
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(0, weight=1)

        # Treeview for corpus list
        columns = ("name", "docs", "path")
        self.corpus_tree = ttk.Treeview(
            list_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
            height=4
        )

        self.corpus_tree.heading("name", text="Corpus Name")
        self.corpus_tree.heading("docs", text="Documents")
        self.corpus_tree.heading("path", text="Location")

        self.corpus_tree.column("name", width=150, minwidth=100)
        self.corpus_tree.column("docs", width=80, minwidth=60, anchor="center")
        self.corpus_tree.column("path", width=400, minwidth=200)

        self.corpus_tree.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.corpus_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", pady=5)
        self.corpus_tree.configure(yscrollcommand=scrollbar.set)

        # Bind selection
        self.corpus_tree.bind("<<TreeviewSelect>>", self._on_corpus_select)
        self.corpus_tree.bind("<Double-1>", self._on_corpus_double_click)

        # Corpus action buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=3, column=0, sticky="ew", padx=15, pady=(0, 10))

        self.new_corpus_btn = ctk.CTkButton(
            btn_frame,
            text="+ New Corpus",
            width=120,
            command=self._new_corpus
        )
        self.new_corpus_btn.pack(side="left", padx=(0, 5))

        self.combine_btn = ctk.CTkButton(
            btn_frame,
            text="Combine Selected...",
            width=140,
            fg_color=("gray70", "gray30"),
            command=self._combine_corpora,
            state="disabled"  # Enable when multiple selected
        )
        self.combine_btn.pack(side="left", padx=5)

        self.delete_corpus_btn = ctk.CTkButton(
            btn_frame,
            text="Delete Corpus",
            width=120,
            fg_color=("gray70", "gray30"),
            hover_color=("#c42b1c", "#a52714"),
            command=self._delete_corpus
        )
        self.delete_corpus_btn.pack(side="left", padx=5)

        self.open_folder_btn = ctk.CTkButton(
            btn_frame,
            text="Open Folder",
            width=100,
            fg_color=("gray70", "gray30"),
            command=self._open_corpus_folder
        )
        self.open_folder_btn.pack(side="left", padx=5)

        # Set active button
        self.set_active_btn = ctk.CTkButton(
            btn_frame,
            text="Set as Active",
            width=110,
            fg_color=("#217346", "#1a5c38"),
            hover_color=("#1a5c38", "#145230"),
            command=self._set_active_corpus
        )
        self.set_active_btn.pack(side="right")

    def _create_document_section(self):
        """Create the document list section."""
        # Section header
        self.doc_header = ctk.CTkFrame(self, fg_color="transparent")
        self.doc_header.grid(row=4, column=0, sticky="ew", padx=15, pady=(10, 5))

        self.doc_label = ctk.CTkLabel(
            self.doc_header,
            text="Documents in Corpus",
            font=FONTS["heading"]
        )
        self.doc_label.pack(side="left")

        # Document list frame
        doc_frame = ctk.CTkFrame(self)
        doc_frame.grid(row=5, column=0, sticky="nsew", padx=15, pady=(0, 10))
        doc_frame.grid_columnconfigure(0, weight=1)
        doc_frame.grid_rowconfigure(0, weight=1)

        # Treeview for document list
        doc_columns = ("name", "status", "date")
        self.doc_tree = ttk.Treeview(
            doc_frame,
            columns=doc_columns,
            show="headings",
            selectmode="extended",
            height=6
        )

        self.doc_tree.heading("name", text="Document")
        self.doc_tree.heading("status", text="Status")
        self.doc_tree.heading("date", text="Modified")

        self.doc_tree.column("name", width=350, minwidth=200)
        self.doc_tree.column("status", width=120, minwidth=80, anchor="center")
        self.doc_tree.column("date", width=150, minwidth=100, anchor="center")

        self.doc_tree.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Scrollbar
        doc_scrollbar = ttk.Scrollbar(doc_frame, orient="vertical", command=self.doc_tree.yview)
        doc_scrollbar.grid(row=0, column=1, sticky="ns", pady=5)
        self.doc_tree.configure(yscrollcommand=doc_scrollbar.set)

        # Document action buttons
        doc_btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        doc_btn_frame.grid(row=6, column=0, sticky="ew", padx=15, pady=(0, 10))

        self.add_files_btn = ctk.CTkButton(
            doc_btn_frame,
            text="Add Files...",
            width=100,
            command=self._add_files
        )
        self.add_files_btn.pack(side="left", padx=(0, 5))

        self.preprocess_btn = ctk.CTkButton(
            doc_btn_frame,
            text="Preprocess All",
            width=120,
            fg_color=("gray70", "gray30"),
            command=self._preprocess_all
        )
        self.preprocess_btn.pack(side="left", padx=5)

        self.remove_files_btn = ctk.CTkButton(
            doc_btn_frame,
            text="Remove Selected",
            width=120,
            fg_color=("gray70", "gray30"),
            hover_color=("#c42b1c", "#a52714"),
            command=self._remove_files
        )
        self.remove_files_btn.pack(side="left", padx=5)

    def _create_footer(self):
        """Create the status bar and close button."""
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=7, column=0, sticky="ew", padx=15, pady=(5, 15))
        footer.grid_columnconfigure(0, weight=1)

        # Status label
        self.status_label = ctk.CTkLabel(
            footer,
            text="",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"],
            anchor="w"
        )
        self.status_label.grid(row=0, column=0, sticky="w")

        # Close button
        close_btn = ctk.CTkButton(
            footer,
            text="Close",
            width=100,
            command=self._on_close
        )
        close_btn.grid(row=0, column=1, sticky="e")

    # =========================================================================
    # Data Methods
    # =========================================================================

    def _refresh_corpus_list(self):
        """Refresh the corpus list from registry."""
        # Clear existing
        for item in self.corpus_tree.get_children():
            self.corpus_tree.delete(item)

        # Get corpora
        corpora = self.registry.list_corpora()

        for corpus in corpora:
            # Mark active corpus
            name = corpus.name
            if corpus.is_active:
                name = f"● {corpus.name} (Active)"

            self.corpus_tree.insert("", "end", values=(
                name,
                f"{corpus.doc_count} docs",
                str(corpus.path)
            ), tags=("active",) if corpus.is_active else ())

        # Select active corpus
        active = self.registry.get_active_corpus()
        self._select_corpus_by_name(active)

        # Update status
        self._update_status()

    def _refresh_document_list(self):
        """Refresh the document list for selected corpus."""
        # Clear existing
        for item in self.doc_tree.get_children():
            self.doc_tree.delete(item)

        if not self._selected_corpus:
            self.doc_label.configure(text="Documents in Corpus")
            return

        # Update header
        self.doc_label.configure(text=f'Documents in "{self._selected_corpus}"')

        # Get corpus path and create manager
        try:
            corpus_path = self.registry.get_corpus_path(self._selected_corpus)
            self._corpus_manager = CorpusManager(corpus_dir=corpus_path)
            files = self._corpus_manager.get_corpus_files_with_status()
        except Exception as e:
            debug_log(f"[CorpusDialog] Error loading documents: {e}")
            return

        # Populate list
        pending_count = 0
        for f in files:
            status = "✓ Preprocessed" if f.is_preprocessed else "⏳ Pending"
            if not f.is_preprocessed:
                pending_count += 1

            date_str = f.modified_at.strftime("%Y-%m-%d") if f.modified_at else ""

            self.doc_tree.insert("", "end", values=(
                f.name,
                status,
                date_str
            ))

        # Update preprocess button state
        if pending_count > 0:
            self.preprocess_btn.configure(
                text=f"Preprocess All ({pending_count})",
                state="normal"
            )
        else:
            self.preprocess_btn.configure(
                text="Preprocess All",
                state="disabled"
            )

    def _select_corpus_by_name(self, name: str):
        """Select a corpus in the tree by name."""
        for item in self.corpus_tree.get_children():
            values = self.corpus_tree.item(item, "values")
            # Check if name matches (accounting for "● name (Active)" format)
            if values[0].replace("● ", "").replace(" (Active)", "") == name:
                self.corpus_tree.selection_set(item)
                self.corpus_tree.focus(item)
                self._selected_corpus = name
                self._refresh_document_list()
                break

    def _update_status(self):
        """Update the status bar."""
        corpora = self.registry.list_corpora()
        total_docs = sum(c.doc_count for c in corpora)
        active = self.registry.get_active_corpus()

        try:
            active_path = self.registry.get_corpus_path(active)
            self.status_label.configure(
                text=f"{len(corpora)} corpus(es) | {total_docs} total documents | Active: {active_path}"
            )
        except Exception:
            self.status_label.configure(
                text=f"{len(corpora)} corpus(es) | {total_docs} total documents"
            )

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_corpus_select(self, event):
        """Handle corpus selection in tree."""
        selection = self.corpus_tree.selection()
        if not selection:
            return

        values = self.corpus_tree.item(selection[0], "values")
        # Extract name (remove "● " and " (Active)" if present)
        name = values[0].replace("● ", "").replace(" (Active)", "")
        self._selected_corpus = name
        self._refresh_document_list()

    def _on_corpus_double_click(self, event):
        """Handle double-click on corpus to set as active."""
        self._set_active_corpus()

    def _on_close(self):
        """Handle dialog close."""
        if DEBUG_MODE:
            debug_log("[CorpusDialog] Closed")
        self.destroy()

    # =========================================================================
    # Corpus Actions
    # =========================================================================

    def _new_corpus(self):
        """Create a new corpus."""
        # Simple input dialog
        dialog = ctk.CTkInputDialog(
            text="Enter name for new corpus:",
            title="New Corpus"
        )
        name = dialog.get_input()

        if not name or not name.strip():
            return

        name = name.strip()

        try:
            path = self.registry.create_corpus(name)
            self.corpus_changed = True
            self._refresh_corpus_list()
            self._select_corpus_by_name(name)

            messagebox.showinfo(
                "Corpus Created",
                f'Corpus "{name}" created at:\n\n{path}\n\n'
                "Add your transcript files to this folder."
            )

        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create corpus: {e}")

    def _delete_corpus(self):
        """Delete the selected corpus."""
        if not self._selected_corpus:
            messagebox.showwarning("No Selection", "Please select a corpus to delete.")
            return

        # Confirm deletion
        result = messagebox.askyesnocancel(
            "Delete Corpus",
            f'Delete corpus "{self._selected_corpus}"?\n\n'
            "Choose Yes to delete files too, or No to keep files."
        )

        if result is None:  # Cancel
            return

        try:
            self.registry.delete_corpus(self._selected_corpus, delete_files=result)
            self.corpus_changed = True
            self._selected_corpus = None
            self._refresh_corpus_list()

        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete corpus: {e}")

    def _combine_corpora(self):
        """Combine selected corpora into a new one."""
        # For now, just use all corpora
        corpora = self.registry.list_corpora()
        if len(corpora) < 2:
            messagebox.showinfo("Info", "Need at least 2 corpora to combine.")
            return

        # Get name for combined corpus
        dialog = ctk.CTkInputDialog(
            text="Enter name for combined corpus:",
            title="Combine Corpora"
        )
        new_name = dialog.get_input()

        if not new_name or not new_name.strip():
            return

        try:
            source_names = [c.name for c in corpora]
            path = self.registry.combine_corpora(source_names, new_name.strip())
            self.corpus_changed = True
            self._refresh_corpus_list()

            messagebox.showinfo(
                "Corpora Combined",
                f'Created "{new_name.strip()}" at:\n\n{path}'
            )

        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to combine corpora: {e}")

    def _set_active_corpus(self):
        """Set the selected corpus as active."""
        if not self._selected_corpus:
            messagebox.showwarning("No Selection", "Please select a corpus.")
            return

        try:
            self.registry.set_active_corpus(self._selected_corpus)
            self.corpus_changed = True
            self._refresh_corpus_list()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to set active corpus: {e}")

    def _open_corpus_folder(self):
        """Open the selected corpus folder in file explorer."""
        if not self._selected_corpus:
            messagebox.showwarning("No Selection", "Please select a corpus.")
            return

        try:
            path = self.registry.get_corpus_path(self._selected_corpus)
            if os.name == 'nt':  # Windows
                os.startfile(str(path))
            else:  # macOS/Linux
                subprocess.run(['open' if os.name == 'darwin' else 'xdg-open', str(path)])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open folder: {e}")

    def _open_bm25_wiki(self):
        """Open BM25 Wikipedia page in browser."""
        webbrowser.open(BM25_WIKI_URL)

    # =========================================================================
    # Document Actions
    # =========================================================================

    def _add_files(self):
        """Add files to the selected corpus."""
        if not self._selected_corpus:
            messagebox.showwarning("No Corpus", "Please select or create a corpus first.")
            return

        # File picker
        files = filedialog.askopenfilenames(
            title="Add Documents to Corpus",
            filetypes=[
                ("Documents", "*.pdf *.txt *.rtf"),
                ("PDF files", "*.pdf"),
                ("Text files", "*.txt"),
                ("RTF files", "*.rtf"),
                ("All files", "*.*")
            ]
        )

        if not files:
            return

        try:
            corpus_path = self.registry.get_corpus_path(self._selected_corpus)

            # Copy files to corpus folder
            import shutil
            copied = 0
            for file_path in files:
                src = Path(file_path)
                dst = corpus_path / src.name

                # Handle duplicates
                if dst.exists():
                    result = messagebox.askyesno(
                        "File Exists",
                        f'"{src.name}" already exists. Overwrite?'
                    )
                    if not result:
                        continue

                shutil.copy2(src, dst)
                copied += 1

                # Preprocess immediately
                if self._corpus_manager:
                    try:
                        self._corpus_manager.preprocess_file(dst)
                    except Exception as e:
                        debug_log(f"[CorpusDialog] Preprocess error: {e}")

            self.corpus_changed = True
            self._refresh_corpus_list()
            self._refresh_document_list()

            if copied > 0:
                messagebox.showinfo(
                    "Files Added",
                    f"Added {copied} file(s) to corpus."
                )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to add files: {e}")

    def _preprocess_all(self):
        """Preprocess all pending documents."""
        if not self._corpus_manager:
            return

        try:
            count = self._corpus_manager.preprocess_pending()
            self.corpus_changed = True
            self._refresh_document_list()

            if count > 0:
                messagebox.showinfo(
                    "Preprocessing Complete",
                    f"Preprocessed {count} document(s)."
                )
            else:
                messagebox.showinfo("Info", "No documents needed preprocessing.")

        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing failed: {e}")

    def _remove_files(self):
        """Remove selected documents from corpus."""
        selection = self.doc_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select documents to remove.")
            return

        # Confirm
        count = len(selection)
        result = messagebox.askyesno(
            "Remove Documents",
            f"Remove {count} document(s) from corpus?\n\n"
            "This will delete the files."
        )

        if not result:
            return

        try:
            corpus_path = self.registry.get_corpus_path(self._selected_corpus)

            for item in selection:
                values = self.doc_tree.item(item, "values")
                file_name = values[0]

                # Delete original and preprocessed versions
                original = corpus_path / file_name
                preprocessed = corpus_path / f"{Path(file_name).stem}_preprocessed.txt"

                if original.exists():
                    original.unlink()
                if preprocessed.exists():
                    preprocessed.unlink()

            self.corpus_changed = True
            self._refresh_corpus_list()
            self._refresh_document_list()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove files: {e}")
