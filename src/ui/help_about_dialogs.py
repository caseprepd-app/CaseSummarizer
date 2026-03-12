"""
Help and About dialogs for CasePrepd.

Provides user-facing information dialogs with clickable links
to online documentation.
"""

import webbrowser

import customtkinter as ctk

from src import __version__
from src.config import APP_NAME
from src.ui.base_dialog import BaseModalDialog
from src.ui.theme import COLORS, FONTS

# Online documentation URL
DOCS_URL = "https://sites.google.com/view/caseprepd/home"


class HelpDialog(BaseModalDialog):
    """
    Comprehensive help dialog with tabbed sections.

    Provides detailed information about the app's features,
    workflow, and tips for end users.
    """

    def __init__(self, parent):
        """
        Initialize the Help dialog.

        Args:
            parent: Parent window
        """
        super().__init__(
            parent=parent,
            title=f"{APP_NAME} Help",
            width=700,
            height=550,
            min_width=600,
            min_height=450,
        )
        self._create_ui()

    def _create_ui(self):
        """Build the dialog UI with tabbed sections."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Title
        title = ctk.CTkLabel(
            self,
            text=f"{APP_NAME} Help",
            font=FONTS["heading_xl"],
        )
        title.grid(row=0, column=0, padx=20, pady=(15, 10))

        # Tabview for sections
        self.tabview = ctk.CTkTabview(self, fg_color=COLORS["bg_dark"])
        self.tabview.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        # Create tabs
        self.tabview.add("Overview")
        self.tabview.add("Workflow")
        self.tabview.add("Features")
        self.tabview.add("Tips")

        self._build_overview_tab()
        self._build_workflow_tab()
        self._build_features_tab()
        self._build_tips_tab()

        # Bottom section with link and close button
        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.grid(row=2, column=0, padx=20, pady=(5, 15), sticky="ew")
        bottom_frame.grid_columnconfigure(0, weight=1)

        link_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        link_frame.grid(row=0, column=0, pady=(0, 10))

        link_label = ctk.CTkLabel(
            link_frame,
            text="For the latest updates:",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"],
        )
        link_label.pack(side="left", padx=(0, 5))

        link_button = ctk.CTkButton(
            link_frame,
            text="Visit Online Documentation",
            font=FONTS["small"],
            fg_color="transparent",
            text_color=COLORS["dialog_link"],
            hover_color=COLORS["bg_hover"],
            command=lambda: webbrowser.open(DOCS_URL),
            width=180,
        )
        link_button.pack(side="left")

        close_btn = ctk.CTkButton(
            bottom_frame,
            text="Close",
            command=self.close,
            width=100,
        )
        close_btn.grid(row=1, column=0)

    def _build_tab_content(self, tab_name: str, text: str) -> None:
        """
        Build a read-only textbox tab with the given content.

        Args:
            tab_name: Name of the tab to populate.
            text: Content to display in the tab.
        """
        tab = self.tabview.tab(tab_name)
        tab.grid_columnconfigure(0, weight=1)

        content = ctk.CTkTextbox(
            tab,
            font=FONTS["body"],
            fg_color=COLORS["bg_darker"],
            wrap="word",
            activate_scrollbars=True,
        )
        content.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        tab.grid_rowconfigure(0, weight=1)

        content.insert("1.0", text)
        content.configure(state="disabled")

    def _build_overview_tab(self):
        """Build the Overview tab content."""
        self._build_tab_content(
            "Overview",
            f"""WHAT IS {APP_NAME.upper()}?

{APP_NAME} is a 100% offline document processor designed specifically for court reporters. It helps you prepare for depositions and trials by extracting names, technical vocabulary, and searching your case documents.

WHY OFFLINE?

Legal documents contain sensitive PII (Personally Identifiable Information) and PHI (Protected Health Information). {APP_NAME} ensures your documents NEVER leave your computer. All processing happens locally on your machine, so there's no cloud, no data transmission, and no privacy concerns.

WHAT DOES IT DO?

1. VOCABULARY EXTRACTION
   Automatically identifies names (parties, witnesses, doctors, attorneys) and technical terms (medical, legal, specialized vocabulary) that you'll need to know for accurate transcription.

2. SEMANTIC SEARCH
   Search your documents by meaning and get answers with source citations. Perfect for quickly finding specific information or creating handoff documents for colleagues.

3. KEY SENTENCES
   Automatically extracts the most important sentences from your case documents, giving you a quick overview of the key facts and details.

WHO IS IT FOR?

Court reporters who need to:
- Prepare vocabulary lists before depositions
- Learn unfamiliar medical or legal terminology
- Quickly find information across multiple case documents
- Create reference materials for colleagues

SYSTEM REQUIREMENTS

- Windows PC with 16GB+ RAM
- ~2GB disk space
- No internet required after initial setup""",
        )

    def _build_workflow_tab(self):
        """Build the Workflow tab content."""
        self._build_tab_content(
            "Workflow",
            f"""THE 5-STEP WORKFLOW

STEP 1: SELECT FILES
Drag files onto the file list, click "+ Add Files", or use Ctrl+O.
Supported formats: PDF, TXT, RTF, DOCX, PNG, JPG

You can select multiple documents at once - complaints, answers, exhibits, motions, medical records, etc. {APP_NAME} processes them together to build a complete case picture.

STEP 2: REVIEW FILE QUALITY
{APP_NAME} analyzes each file and shows quality indicators:
- Green checkmark: Good quality, ready to process
- Yellow warning: Lower OCR confidence (may have errors)
- Red X: Failed to process

Files are automatically checked for inclusion. You can uncheck any files you want to exclude.

STEP 3: PROCESS
Click "Process Documents". This runs:
- Vocabulary extraction (named entities)
- Semantic search indexing (builds searchable passages)
- Key excerpts (representative passages)

Optionally check "Run default searches" to auto-run your saved questions.

Processing time depends on document size and your computer's speed.

You'll see progress updates as it works.

STEP 5: REVIEW & EXPORT
Results appear in tabs:
- Vocabulary: Table of extracted terms
- Semantic Search: Search results and follow-up input
- Key Excerpts: Representative passages extracted from your documents

Export your results to Word, PDF, CSV, or other formats.

KEYBOARD SHORTCUTS

Ctrl+O    Select Files
Ctrl+F    Find in Results
Ctrl+,    Open Settings
Ctrl+Q    Exit Application""",
        )

    def _build_features_tab(self):
        """Build the Features tab content."""
        self._build_tab_content(
            "Features",
            f"""VOCABULARY EXTRACTION

How It Works:
{APP_NAME} uses multiple algorithms to find important terms:
- NER (Named Entity Recognition): Identifies people, places, organizations
- RAKE: Finds technical phrases and key terms
- BM25 Corpus Analysis: Compares against your past work to find case-specific terms

The "Quality Score" predicts how useful each term is. Higher scores = more likely to be relevant vocabulary you need.

Feedback System:
Use the thumbs up/down buttons to rate terms. {APP_NAME} learns your preferences and improves its predictions over time. The more you rate, the better it gets.

---

SEMANTIC SEARCH

Default Searches:
{APP_NAME} comes with pre-configured searches common to legal cases. You can customize these in Settings > Search tab.

Follow-Up Searches:
After processing, type any search in the input box at the bottom of the Search tab. {APP_NAME} searches your documents and provides answers with source citations.

---

CORPUS (YOUR TRANSCRIPT LIBRARY)

What Is It?
The corpus is a collection of your past transcripts that {APP_NAME} uses as a baseline. When you process new documents, it can identify terms that are unusual compared to your typical work.

How To Use It:
1. Go to Settings > Corpus
2. Add past transcript files to your corpus folder
3. Once you have 5+ documents, BM25 analysis activates
4. New cases will highlight terms unique to that case

Your corpus stays 100% local and private.

---

EXPORT OPTIONS

All results can be exported to:
- Word (.docx): Formatted documents
- PDF: Professional reports
- CSV: Spreadsheet data
- TXT: Plain text
- HTML: Web-viewable format

Tip: Configure which columns appear in exports via Settings > Vocabulary.""",
        )

    def _build_tips_tab(self):
        """Build the Tips tab content."""
        self._build_tab_content(
            "Tips",
            """TIPS FOR BEST RESULTS

DOCUMENT QUALITY
- Digital PDFs work best (text can be selected)
- Scanned documents are OCR'd automatically but may have errors
- If OCR confidence is low, the original scan quality may be poor

VOCABULARY EXTRACTION
- Rate terms with thumbs up/down to train your personal model
- Right-click terms to permanently exclude common false positives
- Use the filter box to search for specific terms
- Build your corpus over time for better case-specific detection

SEMANTIC SEARCH
- Be specific in your searches for better results
- Use follow-up searches to drill down into details
- Check the source citations to verify answers

PERFORMANCE
- Close other programs during processing for faster results
- Larger documents take longer - be patient

TROUBLESHOOTING

Poor vocabulary results?
- Check document quality (OCR confidence)
- Rate more terms to train the model

Processing seems stuck?
- Large documents take time - check the progress indicator
- System resources are shown in the bottom status bar""",
        )


class AboutDialog(BaseModalDialog):
    """
    About dialog showing application information.

    Displays version, description, and credits with a link
    to the product website.
    """

    def __init__(self, parent):
        """
        Initialize the About dialog.

        Args:
            parent: Parent window
        """
        super().__init__(
            parent=parent,
            title=f"About {APP_NAME}",
            width=450,
            height=320,
            min_width=350,
            min_height=280,
        )
        self._create_ui()

    def _create_ui(self):
        """Build the dialog UI."""
        self.grid_columnconfigure(0, weight=1)

        # App name and version
        title = ctk.CTkLabel(
            self,
            text=APP_NAME,
            font=FONTS["heading_xl"],
        )
        title.grid(row=0, column=0, padx=20, pady=(25, 5))

        version = ctk.CTkLabel(
            self,
            text=f"Version {__version__}",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"],
        )
        version.grid(row=1, column=0, padx=20, pady=(0, 15))

        # Description
        desc = ctk.CTkLabel(
            self,
            text="100% Offline Legal Document Processor\nfor Court Reporters",
            font=FONTS["body"],
            justify="center",
        )
        desc.grid(row=2, column=0, padx=20, pady=5)

        # Privacy note
        privacy = ctk.CTkLabel(
            self,
            text="Your documents never leave your computer.",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"],
        )
        privacy.grid(row=3, column=0, padx=20, pady=(5, 15))

        # Website link
        link_button = ctk.CTkButton(
            self,
            text="Visit Website",
            font=FONTS["body"],
            fg_color="transparent",
            text_color=COLORS["dialog_link"],
            hover_color=COLORS["bg_hover"],
            command=lambda: webbrowser.open(DOCS_URL),
            width=120,
        )
        link_button.grid(row=4, column=0, pady=5)

        # Close button
        close_btn = ctk.CTkButton(
            self,
            text="Close",
            command=self.close,
            width=100,
        )
        close_btn.grid(row=5, column=0, pady=(15, 20))


class SearchTipsDialog(BaseModalDialog):
    """
    Dialog with research-based tips for effective semantic search.

    Opened from the 'Search Tips' button near the search bar.
    """

    def __init__(self, parent):
        """
        Initialize the Search Tips dialog.

        Args:
            parent: Parent window
        """
        super().__init__(
            parent=parent,
            title="Semantic Search Tips",
            width=500,
            height=420,
            resizable=False,
        )
        self._create_ui()

    def _create_ui(self):
        """Build the dialog UI with scrollable search tips."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        title = ctk.CTkLabel(
            self,
            text="Tips for Better Local Semantic Search Results",
            font=FONTS["heading"],
        )
        title.grid(row=0, column=0, padx=20, pady=(15, 10))

        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.grid(row=1, column=0, padx=15, pady=5, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)

        tips = [
            (
                "One topic per search",
                "Search for one specific topic at a time.\n"
                'Good: "injuries sustained by the plaintiff"\n'
                'Avoid: "injuries and warnings and timeline"',
            ),
            (
                "Use natural language",
                "Write phrases, not keyword lists. The engine matches meaning.\n"
                'Good: "warnings given before the accident"\n'
                'Avoid: "warning accident before"',
            ),
            (
                "Be specific",
                "Narrow queries get more relevant results.\n"
                'Good: "medical treatment after the fall"\n'
                'Avoid: "what happened"',
            ),
            (
                "Try different phrasings",
                "Synonyms and different angles surface different passages.\n"
                "Rephrase if results aren't what you expected.",
            ),
            (
                "Check multiple results",
                "The top result isn't always the best — scan a few and\n"
                "check the source citations to verify context.",
            ),
        ]

        for i, (heading, body) in enumerate(tips):
            h = ctk.CTkLabel(scroll, text=heading, font=FONTS["heading_sm"], anchor="w")
            h.grid(row=i * 2, column=0, sticky="w", pady=(8 if i else 0, 2))
            b = ctk.CTkLabel(scroll, text=body, font=FONTS["body"], anchor="w", justify="left")
            b.grid(row=i * 2 + 1, column=0, sticky="w", padx=(10, 0))

        ok_btn = ctk.CTkButton(self, text="OK", command=self.close, width=100)
        ok_btn.grid(row=2, column=0, pady=(10, 15))
