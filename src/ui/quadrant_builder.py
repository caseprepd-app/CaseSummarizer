"""
Quadrant Builder Module

Builds individual quadrants for the main window central widget.
Separates UI layout construction from window orchestration.
"""

import customtkinter as ctk

from src.ui.dynamic_output import DynamicOutputWidget
from src.ui.theme import COLORS, FONTS
from src.ui.tooltip_helper import create_tooltip
from src.ui.widgets import FileReviewTable, ModelSelectionWidget, OutputOptionsWidget


def build_document_selection_quadrant(parent_frame):
    """
    Build the top-left quadrant: Document Selection with FileReviewTable.

    Returns:
        dict with 'frame', 'table', and metadata about the quadrant
    """
    # Header
    files_label = ctk.CTkLabel(parent_frame, text="📄 Document Selection", font=FONTS["heading_lg"])
    files_label.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))

    create_tooltip(
        files_label,
        "Digital PDF: Text extracted directly. Scanned PDF: Uses Tesseract OCR (confidence evaluation may result in higher errors). TXT/RTF: Direct text extraction.\n\n"
        "Batch: Up to 100 docs. ProcessingTime ~ (avg_pages x model_size). Supports .pdf, .txt, .rtf.",
    )

    # File table
    file_table = FileReviewTable(parent_frame)
    file_table.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    return {
        "frame": parent_frame,
        "widget": file_table,
        "label": files_label,
        "name": "Document Selection",
    }


def build_model_selection_quadrant(parent_frame, model_manager, prompt_template_manager=None):
    """
    Build the top-right quadrant: Model & Prompt Selection.

    Args:
        parent_frame: The parent CTkFrame
        model_manager: The ModelManager instance
        prompt_template_manager: The PromptTemplateManager instance (optional)

    Returns:
        dict with 'widget' (ModelSelectionWidget) and metadata
    """
    # Header
    model_label = ctk.CTkLabel(
        parent_frame, text="🤖 Model & Prompt Selection", font=FONTS["heading_lg"]
    )
    model_label.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))

    create_tooltip(
        model_label,
        "MODEL: Any Ollama model supported. 1B=fast/basic, 7B=quality, 13B=best.\n\n"
        "PROMPT: Choose a summarization style or create your own. "
        "Add custom .txt files to your prompts folder - see _README.txt for instructions.",
    )

    # Model selection widget (now includes prompt style dropdown)
    model_selection = ModelSelectionWidget(parent_frame, model_manager, prompt_template_manager)
    model_selection.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

    # Set up tooltip for prompt selector after widget is created
    model_selection.setup_tooltip(create_tooltip)

    return {
        "frame": parent_frame,
        "widget": model_selection,
        "label": model_label,
        "name": "AI Model Selection",
    }


def build_output_display_quadrant(parent_frame):
    """
    Build the bottom-left quadrant: Generated Outputs display.

    Returns:
        dict with 'widget' (DynamicOutputWidget) and metadata
    """
    # Header
    output_display_label = ctk.CTkLabel(
        parent_frame, text="📝 Generated Outputs", font=FONTS["heading_lg"]
    )
    output_display_label.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))

    create_tooltip(
        output_display_label,
        "Individual summaries: Per-document outputs (from parallel processing). Meta-summary: Hierarchical summary of all docs (blocking final step). "
        "Vocabulary: CSV of technical terms (category, definition, relevance). Dropdown switches between output types. Copy/Save buttons available.",
    )

    # Output display widget
    summary_results = DynamicOutputWidget(parent_frame)
    summary_results.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    return {
        "frame": parent_frame,
        "widget": summary_results,
        "label": output_display_label,
        "name": "Generated Outputs",
    }


def build_output_options_quadrant(parent_frame):
    """
    Build the bottom-right quadrant: Output Options and Generate button.

    Returns:
        dict with 'widget' (OutputOptionsWidget), 'button', and metadata
    """
    # Header
    output_options_label = ctk.CTkLabel(
        parent_frame, text="⚙️ Output Options", font=FONTS["heading_lg"]
    )
    output_options_label.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))

    create_tooltip(
        output_options_label,
        "Word count: 50-500 words per summary (adjusts token budget). Outputs: Toggle which results to generate (save time by disabling unneeded outputs). "
        "Parallel processing uses CPU fraction from Settings. Monitor system impact via status bar CPU/RAM display.",
    )

    # Output options widget
    output_options = OutputOptionsWidget(parent_frame)
    output_options.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

    # Generate button (text updated dynamically based on checkbox count)
    generate_btn = ctk.CTkButton(
        parent_frame,
        text="Generate 2 Outputs",  # Default: meta-summary + vocab (2 checked by default)
        font=FONTS["body_bold"],
    )
    generate_btn.grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 5))
    generate_btn.configure(state="disabled")

    # Wire up output_options to update generate button text dynamically
    output_options.set_generate_button(generate_btn)

    # Cancel button (always visible, disabled when not processing)
    # Keeping it visible prevents UI stutter from grid/grid_remove operations
    cancel_btn = ctk.CTkButton(
        parent_frame,
        text="Cancel Processing",
        font=FONTS["body_bold"],
        fg_color=COLORS["btn_disabled"],
        hover_color=COLORS["btn_disabled_hover"],
        state="disabled",  # Disabled by default
    )
    cancel_btn.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 10))
    # NOT using grid_remove() - button stays visible but greyed out

    return {
        "frame": parent_frame,
        "widget": output_options,
        "button": generate_btn,
        "cancel_button": cancel_btn,
        "label": output_options_label,
        "name": "Output Options",
    }


def create_central_widget_layout(main_window, model_manager, prompt_template_manager=None):
    """
    Create and layout the four-quadrant central widget.

    This function orchestrates the creation of all four quadrants and manages
    the grid layout. It returns a tuple of the widgets for the main window to use.

    Args:
        main_window: The parent MainWindow (CTk) instance
        model_manager: The ModelManager instance for model selection
        prompt_template_manager: The PromptTemplateManager instance (optional)

    Returns:
        tuple: (main_content_frame, file_table, model_selection, summary_results,
                output_options, generate_btn, cancel_btn)
    """
    # Create main content frame
    main_content_frame = ctk.CTkFrame(main_window, corner_radius=0, fg_color="transparent")
    main_content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

    # Configure 2x2 grid
    main_content_frame.grid_columnconfigure(0, weight=1)  # Left column
    main_content_frame.grid_columnconfigure(1, weight=1)  # Right column
    main_content_frame.grid_rowconfigure(0, weight=1)  # Top row
    main_content_frame.grid_rowconfigure(1, weight=1)  # Bottom row

    # Create frames for each quadrant
    top_left_frame = ctk.CTkFrame(main_content_frame)
    top_right_frame = ctk.CTkFrame(main_content_frame)
    bottom_left_frame = ctk.CTkFrame(main_content_frame)
    bottom_right_frame = ctk.CTkFrame(main_content_frame)

    # Place frames in the grid
    top_left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=(0, 5))
    top_right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=(0, 5))
    bottom_left_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5), pady=(5, 0))
    bottom_right_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0), pady=(5, 0))

    # Configure quadrant frames with borders and internal layout
    for frame in [top_left_frame, top_right_frame, bottom_left_frame, bottom_right_frame]:
        frame.grid_rowconfigure(0, weight=0)  # Header row
        frame.grid_rowconfigure(1, weight=1)  # Content row
        frame.grid_columnconfigure(0, weight=1)
        # Add subtle border
        frame.configure(border_width=1, border_color="#404040")

    # Build quadrants
    doc_quad = build_document_selection_quadrant(top_left_frame)
    model_quad = build_model_selection_quadrant(
        top_right_frame, model_manager, prompt_template_manager
    )
    output_quad = build_output_display_quadrant(bottom_left_frame)
    options_quad = build_output_options_quadrant(bottom_right_frame)

    return (
        main_content_frame,
        doc_quad["widget"],  # file_table
        model_quad["widget"],  # model_selection
        output_quad["widget"],  # summary_results
        options_quad["widget"],  # output_options
        options_quad["button"],  # generate_btn
        options_quad["cancel_button"],  # cancel_btn
    )
