"""
Interactive HTML Export Builder

Generates self-contained HTML files with embedded CSS and JavaScript
for filtering, sorting, and interactive features.

Supports configurable columns matching GUI settings, shared column_config
for consistency with GUI, sort warnings, and Term column protection.
"""

import html
import json
import logging
from datetime import datetime
from pathlib import Path

from src.config import (
    COLUMN_DEFINITIONS,
    NUMERIC_COLUMNS,
    PROTECTED_COLUMNS,
    SORT_WARNING_COLUMNS,
)

logger = logging.getLogger(__name__)


def _escape(text: str) -> str:
    """Escape HTML special characters."""
    return html.escape(str(text)) if text is not None and text != "" else ""


# ============================================================================
# Vocabulary HTML Builder (configurable columns)
# ============================================================================

# Build column list from shared config (name, data_key tuples)
# Excludes Keep/Skip feedback columns which are GUI-only
VOCAB_HTML_COLUMNS = [
    (c.name, c.data_key)
    for c in COLUMN_DEFINITIONS
    if c.name not in ("Keep", "Skip")  # Feedback columns are GUI-only
]

VOCAB_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Names & Vocabulary</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
        }}
        .summary {{
            color: #666;
            margin-bottom: 20px;
        }}
        .controls {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
        }}
        .controls input[type="text"] {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            width: 250px;
        }}
        .controls label {{
            display: flex;
            align-items: center;
            gap: 5px;
            cursor: pointer;
            font-size: 14px;
        }}
        .controls button {{
            padding: 8px 16px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        .controls button:hover {{
            background: #2980b9;
        }}
        .filter-group {{
            display: flex;
            gap: 10px;
            align-items: center;
            padding-left: 15px;
            border-left: 2px solid #eee;
            flex-wrap: wrap;
        }}
        .filter-group-label {{
            font-weight: 500;
            color: #666;
            font-size: 13px;
        }}
        .column-toggles {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .column-toggles label {{
            font-size: 12px;
            padding: 4px 8px;
            background: #f0f0f0;
            border-radius: 4px;
            transition: background 0.2s;
        }}
        .column-toggles label:hover {{
            background: #e0e0e0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: #2c3e50;
            color: white;
            padding: 12px 15px;
            text-align: left;
            cursor: pointer;
            user-select: none;
            white-space: nowrap;
        }}
        th:hover {{
            background: #34495e;
        }}
        th .sort-arrow {{
            margin-left: 5px;
            opacity: 0.5;
        }}
        th.sorted .sort-arrow {{
            opacity: 1;
        }}
        td {{
            padding: 10px 15px;
            border-bottom: 1px solid #eee;
            white-space: nowrap;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        tr.person {{
            background: #e8f4f8;
        }}
        tr.person:hover {{
            background: #d4ebf2;
        }}
        .count-display {{
            color: #666;
            font-size: 14px;
            margin-left: auto;
        }}
        .hidden {{
            display: none;
        }}
        .col-hidden {{
            display: none;
        }}
        @media print {{
            .controls {{ display: none; }}
            body {{ background: white; padding: 10px; }}
            table {{ box-shadow: none; }}
        }}
    </style>
</head>
<body>
    <h1>Names & Vocabulary</h1>
    <div class="summary">{summary}</div>

    <div class="controls">
        <input type="text" id="search" placeholder="Search terms..." oninput="filterTable()">

        <div class="filter-group">
            <span class="filter-group-label">Show:</span>
            <label><input type="checkbox" id="persons-only" onchange="filterTable()"> People only</label>
            <label><input type="checkbox" id="terms-only" onchange="filterTable()"> Terms only</label>
        </div>

        <div class="filter-group">
            <span class="filter-group-label">Columns:</span>
            <div class="column-toggles" id="column-toggles">
                {column_toggles}
            </div>
        </div>

        <button onclick="window.print()">Print</button>
        <span class="count-display" id="count-display"></span>
    </div>

    <table id="vocab-table">
        <thead>
            <tr>
{table_headers}
            </tr>
        </thead>
        <tbody>
{table_rows}
        </tbody>
    </table>

    <script>
        const numericColumns = {numeric_columns_json};
        const columnOrder = {column_order_json};
        const sortWarningColumns = {sort_warning_columns_json};
        let sortColumn = -1;
        let sortAsc = true;

        function toggleColumn(colName) {{
            const checkbox = document.getElementById('col-' + colName.replace(/[^a-zA-Z0-9]/g, ''));
            const isVisible = checkbox.checked;
            const colIndex = columnOrder.indexOf(colName);
            if (colIndex === -1) return;

            // Toggle header
            const headers = document.querySelectorAll('#vocab-table thead th');
            headers[colIndex].classList.toggle('col-hidden', !isVisible);

            // Toggle all cells in that column
            const rows = document.querySelectorAll('#vocab-table tbody tr');
            rows.forEach(row => {{
                if (row.cells[colIndex]) {{
                    row.cells[colIndex].classList.toggle('col-hidden', !isVisible);
                }}
            }});
        }}

        function filterTable() {{
            const search = document.getElementById('search').value.toLowerCase();
            const personsOnly = document.getElementById('persons-only').checked;
            const termsOnly = document.getElementById('terms-only').checked;

            const personColIndex = columnOrder.indexOf('Is Person');
            const rows = document.querySelectorAll('#vocab-table tbody tr');
            let visibleCount = 0;

            rows.forEach(row => {{
                const term = row.cells[0].textContent.toLowerCase();
                const isPerson = personColIndex >= 0 && row.cells[personColIndex]
                    ? row.cells[personColIndex].textContent.trim() === 'Yes'
                    : false;

                let show = true;

                // Search filter
                if (search && !term.includes(search)) {{
                    show = false;
                }}

                // Person/term filter (mutually exclusive)
                if (personsOnly && !isPerson) show = false;
                if (termsOnly && isPerson) show = false;

                row.classList.toggle('hidden', !show);
                if (show) visibleCount++;
            }});

            document.getElementById('count-display').textContent =
                `Showing ${{visibleCount}} of ${{rows.length}} entries`;
        }}

        function sortTable(colIndex) {{
            const table = document.getElementById('vocab-table');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const headers = table.querySelectorAll('th');
            const colName = columnOrder[colIndex];
            const isNumeric = numericColumns.includes(colName);

            // Show warning when sorting by columns where low values appear first
            if (sortWarningColumns.includes(colName)) {{
                if (!confirm("Sorting by '" + colName + "' will show lower-quality results first.\\n\\nContinue?")) {{
                    return;  // User cancelled
                }}
            }}

            // Toggle sort direction if same column
            if (sortColumn === colIndex) {{
                sortAsc = !sortAsc;
            }} else {{
                sortColumn = colIndex;
                sortAsc = true;
            }}

            // Update header styling
            headers.forEach((h, i) => {{
                h.classList.toggle('sorted', i === colIndex);
                const arrow = h.querySelector('.sort-arrow');
                if (arrow) arrow.textContent = (i === colIndex && !sortAsc) ? '▲' : '▼';
            }});

            // Sort rows
            rows.sort((a, b) => {{
                let aVal = a.cells[colIndex].textContent.trim();
                let bVal = b.cells[colIndex].textContent.trim();

                // Numeric sort
                if (isNumeric) {{
                    aVal = parseFloat(aVal) || 0;
                    bVal = parseFloat(bVal) || 0;
                    return sortAsc ? aVal - bVal : bVal - aVal;
                }}

                // String sort
                return sortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            }});

            // Re-append sorted rows
            rows.forEach(row => tbody.appendChild(row));
        }}

        // Initial count
        filterTable();
    </script>
</body>
</html>"""


def build_vocabulary_html(
    vocab_data: list[dict],
    visible_columns: list[str] | None = None,
) -> str:
    """
    Build vocabulary HTML content string.

    Args:
        vocab_data: List of vocabulary dicts
        visible_columns: List of column names to show initially (from GUI).
                        If None, uses default columns.

    Returns:
        HTML content as string
    """
    # Default visible columns if not specified
    if visible_columns is None:
        visible_columns = ["Term", "Score", "Is Person", "Found By"]

    # Get all column display names in order
    column_names = [col[0] for col in VOCAB_HTML_COLUMNS]

    # Build summary
    person_count = sum(1 for v in vocab_data if v.get("Is Person") == "Yes")
    term_count = len(vocab_data) - person_count
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    summary = f"{len(vocab_data)} entries ({person_count} persons, {term_count} terms) — Generated {timestamp}"

    # Build column toggle checkboxes
    # Protected columns (like Term) have disabled checkbox
    toggle_parts = []
    for col_name, _ in VOCAB_HTML_COLUMNS:
        col_id = col_name.replace(" ", "").replace("#", "").replace("/", "")
        is_visible = col_name in visible_columns

        if col_name in PROTECTED_COLUMNS:
            # Protected column: always visible, checkbox disabled
            toggle_parts.append(
                f'<label><input type="checkbox" id="col-{col_id}" '
                f"checked disabled> {col_name} (required)</label>"
            )
        else:
            checked = " checked" if is_visible else ""
            toggle_parts.append(
                f'<label><input type="checkbox" id="col-{col_id}" '
                f"onchange=\"toggleColumn('{col_name}')\"{checked}> {col_name}</label>"
            )
    column_toggles = "\n                ".join(toggle_parts)

    # Build table headers
    header_parts = []
    for i, (col_name, _) in enumerate(VOCAB_HTML_COLUMNS):
        hidden_class = "" if col_name in visible_columns else ' class="col-hidden"'
        header_parts.append(
            f'                <th onclick="sortTable({i})"{hidden_class}>'
            f'{col_name} <span class="sort-arrow">▼</span></th>'
        )
    table_headers = "\n".join(header_parts)

    # Build table rows with all columns
    rows = []
    for v in vocab_data:
        is_person = v.get("Is Person", "") == "Yes"
        row_class = ' class="person"' if is_person else ""

        cells = []
        for col_name, data_key in VOCAB_HTML_COLUMNS:
            hidden_class = "" if col_name in visible_columns else ' class="col-hidden"'
            value = v.get(data_key, "")
            cells.append(f"<td{hidden_class}>{_escape(value)}</td>")

        rows.append(f"            <tr{row_class}>{''.join(cells)}</tr>")

    table_rows = "\n".join(rows)

    # Generate HTML with JSON data for JavaScript
    # Include sort warning columns for confirm dialog
    return VOCAB_HTML_TEMPLATE.format(
        summary=_escape(summary),
        column_toggles=column_toggles,
        table_headers=table_headers,
        table_rows=table_rows,
        numeric_columns_json=json.dumps(list(NUMERIC_COLUMNS)),
        column_order_json=json.dumps(column_names),
        sort_warning_columns_json=json.dumps(list(SORT_WARNING_COLUMNS)),
    )


def export_vocabulary_html(
    vocab_data: list[dict],
    file_path: str,
    visible_columns: list[str] | None = None,
    title: str = "Names & Vocabulary",
) -> bool:
    """
    Export vocabulary to interactive HTML file.

    Args:
        vocab_data: List of vocabulary dicts
        file_path: Output file path (.html)
        visible_columns: List of column names to show initially (from GUI).
                        If None, uses default columns.
        title: Document title

    Returns:
        True if successful, False otherwise
    """
    try:
        html_content = build_vocabulary_html(vocab_data, visible_columns)
        Path(file_path).write_text(html_content, encoding="utf-8")
        return True
    except Exception as e:
        logger.error("Failed to export vocabulary HTML to '%s': %s", file_path, e)
        return False


# ============================================================================
# Q&A HTML Builder
# ============================================================================

QA_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Questions & Answers</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
        }}
        h1 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
        }}
        .summary {{
            color: #666;
            margin-bottom: 20px;
        }}
        .controls {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
        }}
        .controls input[type="text"] {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            width: 300px;
        }}
        .controls button {{
            padding: 8px 16px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        .controls button:hover {{
            background: #2980b9;
        }}
        .controls button.secondary {{
            background: #95a5a6;
        }}
        .controls button.secondary:hover {{
            background: #7f8c8d;
        }}
        .count-display {{
            color: #666;
            font-size: 14px;
            margin-left: auto;
        }}
        .qa-item {{
            background: white;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .qa-header {{
            background: #2c3e50;
            color: white;
            padding: 12px 15px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .qa-header:hover {{
            background: #34495e;
        }}
        .qa-header .toggle {{
            font-size: 12px;
            opacity: 0.7;
        }}
        .qa-content {{
            padding: 15px;
        }}
        .qa-content.collapsed {{
            display: none;
        }}
        .question {{
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .label {{
            font-weight: 500;
            color: #666;
            margin-top: 15px;
            margin-bottom: 5px;
            font-size: 13px;
            text-transform: uppercase;
        }}
        .answer {{
            line-height: 1.6;
        }}
        .citation {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            font-size: 14px;
            color: #555;
            border-left: 3px solid #3498db;
        }}
        .source {{
            font-size: 13px;
            color: #888;
            font-style: italic;
        }}
        .reliability {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
            margin-left: 10px;
        }}
        .reliability-high {{ background: #d4edda; color: #155724; }}
        .reliability-medium {{ background: #fff3cd; color: #856404; }}
        .reliability-low {{ background: #f8d7da; color: #721c24; }}
        .verified {{ color: rgb(40, 167, 69); }}
        .uncertain {{ color: rgb(255, 193, 7); }}
        .suspicious {{ color: rgb(253, 126, 20); }}
        .unreliable {{ color: rgb(220, 53, 69); }}
        .hallucinated {{ color: rgb(136, 136, 136); text-decoration: line-through; }}
        .hidden {{
            display: none;
        }}
        .legend {{
            background: white;
            padding: 10px 15px;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 13px;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .legend-title {{
            font-weight: 500;
            color: #666;
        }}
        @media print {{
            .controls {{ display: none; }}
            body {{ background: white; padding: 10px; max-width: none; }}
            .qa-item {{ box-shadow: none; border: 1px solid #ddd; }}
            .qa-content.collapsed {{ display: block !important; }}
        }}
    </style>
</head>
<body>
    <h1>Questions & Answers</h1>
    <div class="summary">{summary}</div>

    <div class="controls">
        <input type="text" id="search" placeholder="Search questions and answers..." oninput="filterQA()">
        <button onclick="toggleAll(true)">Expand All</button>
        <button onclick="toggleAll(false)" class="secondary">Collapse All</button>
        <button onclick="window.print()">Print</button>
        <span class="count-display" id="count-display"></span>
    </div>

    <div id="qa-container">
{qa_items}
    </div>

{legend}

    <script>
        function filterQA() {{
            const search = document.getElementById('search').value.toLowerCase();
            const items = document.querySelectorAll('.qa-item');
            let visibleCount = 0;

            items.forEach(item => {{
                const text = item.textContent.toLowerCase();
                const show = !search || text.includes(search);
                item.classList.toggle('hidden', !show);
                if (show) visibleCount++;
            }});

            document.getElementById('count-display').textContent =
                `Showing ${{visibleCount}} of ${{items.length}} Q&A pairs`;
        }}

        function toggleItem(header) {{
            const content = header.nextElementSibling;
            const toggle = header.querySelector('.toggle');
            const isCollapsed = content.classList.toggle('collapsed');
            toggle.textContent = isCollapsed ? '▶ Show' : '▼ Hide';
        }}

        function toggleAll(expand) {{
            const contents = document.querySelectorAll('.qa-content');
            const toggles = document.querySelectorAll('.toggle');

            contents.forEach(content => {{
                content.classList.toggle('collapsed', !expand);
            }});

            toggles.forEach(toggle => {{
                toggle.textContent = expand ? '▼ Hide' : '▶ Show';
            }});
        }}

        // Initial count
        filterQA();
    </script>
</body>
</html>"""


def export_qa_html(
    results: list,
    file_path: str,
    include_verification: bool = True,
    title: str = "Questions & Answers",
) -> bool:
    """
    Export Q&A results to interactive HTML file.

    Args:
        results: List of QAResult objects
        file_path: Output file path (.html)
        include_verification: Include verification coloring
        title: Document title

    Returns:
        True if successful, False otherwise
    """
    try:
        # Build summary
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        summary = f"{len(results)} Q&A pairs — Generated {timestamp}"

        # Track if we need legend
        has_verification = False

        # Build Q&A items
        items = []
        for i, result in enumerate(results, 1):
            # Build answer with verification coloring if available
            answer_html = ""
            reliability_badge = ""

            if include_verification and hasattr(result, "verification") and result.verification:
                has_verification = True
                v = result.verification

                # Reliability badge
                reliability = v.overall_reliability
                if reliability >= 0.7:
                    rel_class = "reliability-high"
                    rel_text = f"HIGH ({int(reliability * 100)}%)"
                elif reliability >= 0.5:
                    rel_class = "reliability-medium"
                    rel_text = f"MEDIUM ({int(reliability * 100)}%)"
                else:
                    rel_class = "reliability-low"
                    rel_text = f"LOW ({int(reliability * 100)}%)"
                reliability_badge = f'<span class="reliability {rel_class}">{rel_text}</span>'

                # Color-coded spans
                if v.answer_rejected:
                    answer_html = f'<span class="unreliable">{_escape(result.quick_answer)}</span>'
                else:
                    for span in v.spans:
                        prob = span.hallucination_prob
                        if prob < 0.3:
                            css_class = "verified"
                        elif prob < 0.5:
                            css_class = "uncertain"
                        elif prob < 0.7:
                            css_class = "suspicious"
                        elif prob < 0.85:
                            css_class = "unreliable"
                        else:
                            css_class = "hallucinated"
                        answer_html += f'<span class="{css_class}">{_escape(span.text)}</span>'
            else:
                answer_html = _escape(result.quick_answer)

            # Build Q&A item
            citation = _escape(result.citation) if result.citation else "(no citation)"
            source = _escape(result.source_summary) if result.source_summary else "(source unknown)"

            item = f"""        <div class="qa-item">
            <div class="qa-header" onclick="toggleItem(this)">
                <span>Q{i}: {_escape(result.question)[:80]}{"..." if len(result.question) > 80 else ""}{reliability_badge}</span>
                <span class="toggle">▼ Hide</span>
            </div>
            <div class="qa-content">
                <div class="question">{_escape(result.question)}</div>
                <div class="label">Answer</div>
                <div class="answer">{answer_html}</div>
                <div class="label">Citation</div>
                <div class="citation">{citation}</div>
                <div class="label">Source</div>
                <div class="source">{source}</div>
            </div>
        </div>"""
            items.append(item)

        qa_items = "\n".join(items)

        # Build legend if verification was used
        if has_verification:
            legend = """    <div class="legend">
        <span class="legend-title">Verification:</span>
        <span class="verified">■ Verified</span>
        <span class="uncertain">■ Uncertain</span>
        <span class="suspicious">■ Suspicious</span>
        <span class="unreliable">■ Unreliable</span>
        <span class="hallucinated">■ Hallucinated</span>
    </div>"""
        else:
            legend = ""

        # Generate HTML
        html_content = QA_HTML_TEMPLATE.format(
            summary=_escape(summary), qa_items=qa_items, legend=legend
        )

        Path(file_path).write_text(html_content, encoding="utf-8")
        return True
    except Exception as e:
        logger.error("Failed to export Q&A HTML to '%s': %s", file_path, e)
        return False
