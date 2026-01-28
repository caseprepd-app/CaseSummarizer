"""
Combined Tabbed HTML Export Builder.

Generates a single self-contained HTML file with dynamic tabs for
Vocabulary, Q&A, and Summary sections. Only renders tabs that have data.

Each tab has its own controls (search, filter, sort, print).
JavaScript is scoped to prevent cross-tab interference.
"""

import json
from datetime import datetime

from src.config import NUMERIC_COLUMNS, PROTECTED_COLUMNS, SORT_WARNING_COLUMNS
from src.core.export.html_builder import VOCAB_HTML_COLUMNS, _escape


def build_combined_html(
    vocab_data: list[dict],
    qa_results: list,
    summary_text: str,
    visible_columns: list[str] | None = None,
    include_verification: bool = True,
) -> str:
    """
    Build a tabbed HTML document combining Vocabulary, Q&A, and Summary.

    Only includes tabs for sections that have data. The first available
    tab is active by default.

    Args:
        vocab_data: List of vocabulary dicts (score-filtered)
        qa_results: List of QAResult objects (already filtered to answered only)
        summary_text: Summary text string
        visible_columns: Columns to show initially in vocab table
        include_verification: Whether to show verification badges in Q&A

    Returns:
        Complete HTML document as string
    """
    # Determine which tabs to show
    has_vocab = bool(vocab_data)
    has_qa = bool(qa_results)
    has_summary = bool(summary_text and summary_text.strip())

    # Build tab definitions: (id, label, content_html)
    tabs = []
    if has_vocab:
        tabs.append(("vocab", "Vocabulary", _build_vocab_section(vocab_data, visible_columns)))
    if has_qa:
        tabs.append(("qa", "Q&A", _build_qa_section(qa_results, include_verification)))
    if has_summary:
        tabs.append(("summary", "Summary", _build_summary_section(summary_text)))

    if not tabs:
        return "<html><body><p>No data to export.</p></body></html>"

    # Build tab buttons
    tab_buttons = []
    for i, (tab_id, label, _) in enumerate(tabs):
        active_class = " active" if i == 0 else ""
        tab_buttons.append(
            f'<button class="tab-btn{active_class}" '
            f"onclick=\"switchTab('{tab_id}')\" "
            f'data-tab="{tab_id}">{label}</button>'
        )
    tab_buttons_html = "\n            ".join(tab_buttons)

    # Build tab content divs
    tab_contents = []
    for i, (tab_id, _, content) in enumerate(tabs):
        display = "" if i == 0 else ' style="display:none"'
        tab_contents.append(
            f'    <div id="tab-{tab_id}" class="tab-content"{display}>\n{content}\n    </div>'
        )
    tab_contents_html = "\n".join(tab_contents)

    # Build vocab JS data (only if vocab tab exists)
    vocab_js = ""
    if has_vocab:
        column_names = [col[0] for col in VOCAB_HTML_COLUMNS]
        vocab_js = f"""
        // Vocabulary table data
        const numericColumns = {json.dumps(list(NUMERIC_COLUMNS))};
        const columnOrder = {json.dumps(column_names)};
        const sortWarningColumns = {json.dumps(list(SORT_WARNING_COLUMNS))};
        let sortColumn = -1;
        let sortAsc = true;
"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    return COMBINED_HTML_TEMPLATE.format(
        timestamp=_escape(timestamp),
        tab_buttons=tab_buttons_html,
        tab_contents=tab_contents_html,
        vocab_js=vocab_js,
        has_vocab="true" if has_vocab else "false",
        has_qa="true" if has_qa else "false",
    )


def _build_vocab_section(vocab_data: list[dict], visible_columns: list[str] | None) -> str:
    """
    Build the Vocabulary tab HTML content.

    Args:
        vocab_data: List of vocabulary dicts
        visible_columns: Columns to show initially

    Returns:
        HTML fragment for the vocabulary section
    """
    if visible_columns is None:
        visible_columns = ["Term", "Score", "Is Person", "Found By"]

    person_count = sum(1 for v in vocab_data if v.get("Is Person") == "Yes")
    term_count = len(vocab_data) - person_count

    # Controls
    toggle_parts = []
    for col_name, _ in VOCAB_HTML_COLUMNS:
        col_id = col_name.replace(" ", "").replace("#", "").replace("/", "")
        is_visible = col_name in visible_columns
        if col_name in PROTECTED_COLUMNS:
            toggle_parts.append(
                f'<label><input type="checkbox" id="col-{col_id}" '
                f"checked disabled> {col_name}</label>"
            )
        else:
            checked = " checked" if is_visible else ""
            toggle_parts.append(
                f'<label><input type="checkbox" id="col-{col_id}" '
                f"onchange=\"toggleColumn('{col_name}')\"{checked}> {col_name}</label>"
            )
    column_toggles = "\n                    ".join(toggle_parts)

    # Table headers
    header_parts = []
    for i, (col_name, _) in enumerate(VOCAB_HTML_COLUMNS):
        hidden_class = "" if col_name in visible_columns else ' class="col-hidden"'
        header_parts.append(
            f'                    <th onclick="sortTable({i})"{hidden_class}>'
            f'{col_name} <span class="sort-arrow">&#x25BC;</span></th>'
        )
    table_headers = "\n".join(header_parts)

    # Table rows
    rows = []
    for v in vocab_data:
        is_person = v.get("Is Person", "") == "Yes"
        row_class = ' class="person"' if is_person else ""
        cells = []
        for col_name, data_key in VOCAB_HTML_COLUMNS:
            hidden_class = "" if col_name in visible_columns else ' class="col-hidden"'
            value = v.get(data_key, "")
            cells.append(f"<td{hidden_class}>{_escape(value)}</td>")
        rows.append(f"                <tr{row_class}>{''.join(cells)}</tr>")
    table_rows = "\n".join(rows)

    return f"""        <div class="section-controls">
            <input type="text" id="vocab-search" placeholder="Search terms..."
                   oninput="filterVocabTable()">
            <div class="filter-group">
                <span class="filter-group-label">Show:</span>
                <label><input type="checkbox" id="persons-only"
                       onchange="filterVocabTable()"> People only</label>
                <label><input type="checkbox" id="terms-only"
                       onchange="filterVocabTable()"> Terms only</label>
            </div>
            <div class="filter-group">
                <span class="filter-group-label">Columns:</span>
                <div class="column-toggles">
                    {column_toggles}
                </div>
            </div>
            <button onclick="window.print()">Print</button>
            <span class="count-display" id="vocab-count">
                {len(vocab_data)} entries ({person_count} persons, {term_count} terms)
            </span>
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
        </table>"""


def _build_qa_section(qa_results: list, include_verification: bool) -> str:
    """
    Build the Q&A tab HTML content.

    Args:
        qa_results: List of QAResult objects
        include_verification: Whether to show verification badges

    Returns:
        HTML fragment for the Q&A section
    """
    has_verification = False
    items = []

    for i, result in enumerate(qa_results, 1):
        answer_html = ""
        reliability_badge = ""

        if include_verification and hasattr(result, "verification") and result.verification:
            has_verification = True
            v = result.verification
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

        citation = _escape(result.citation) if result.citation else "(no citation)"
        source = _escape(result.source_summary) if result.source_summary else "(source unknown)"
        q_truncated = _escape(result.question[:80])
        ellipsis = "..." if len(result.question) > 80 else ""

        items.append(f"""        <div class="qa-item">
            <div class="qa-header" onclick="toggleQAItem(this)">
                <span>Q{i}: {q_truncated}{ellipsis}{reliability_badge}</span>
                <span class="toggle">&#x25BC; Hide</span>
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
        </div>""")

    items_html = "\n".join(items)

    legend = ""
    if has_verification:
        legend = """        <div class="legend">
            <span class="legend-title">Verification:</span>
            <span class="verified">&#x25A0; Verified</span>
            <span class="uncertain">&#x25A0; Uncertain</span>
            <span class="suspicious">&#x25A0; Suspicious</span>
            <span class="unreliable">&#x25A0; Unreliable</span>
            <span class="hallucinated">&#x25A0; Hallucinated</span>
        </div>"""

    return f"""        <div class="section-controls">
            <input type="text" id="qa-search" placeholder="Search questions and answers..."
                   oninput="filterQA()">
            <button onclick="toggleAllQA(true)">Expand All</button>
            <button onclick="toggleAllQA(false)" class="secondary">Collapse All</button>
            <button onclick="window.print()">Print</button>
            <span class="count-display" id="qa-count">{len(qa_results)} Q&amp;A pairs</span>
        </div>
        <div id="qa-container">
{items_html}
        </div>
{legend}"""


def _build_summary_section(summary_text: str) -> str:
    """
    Build the Summary tab HTML content.

    Args:
        summary_text: Summary text string

    Returns:
        HTML fragment for the summary section
    """
    paragraphs = [p.strip() for p in summary_text.strip().split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in summary_text.strip().split("\n") if p.strip()]

    para_html = "\n".join(f"            <p>{_escape(p)}</p>" for p in paragraphs)

    return f"""        <div class="section-controls">
            <button onclick="window.print()">Print</button>
        </div>
        <div class="summary-content">
{para_html}
        </div>"""


# =============================================================================
# HTML Template
# =============================================================================

COMBINED_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Case Report</title>
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
            margin: 0 0 5px 0;
            color: #2c3e50;
        }}
        .timestamp {{
            color: #888;
            font-size: 13px;
            margin-bottom: 20px;
        }}

        /* ---- Tab bar ---- */
        .tab-bar {{
            display: flex;
            gap: 4px;
            margin-bottom: 0;
        }}
        .tab-btn {{
            padding: 12px 28px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            background: #ddd;
            color: #555;
            transition: background 0.2s, color 0.2s;
        }}
        .tab-btn:hover {{
            background: #ccc;
            color: #333;
        }}
        .tab-btn.active {{
            background: #2c3e50;
            color: white;
            box-shadow: 0 -3px 0 0 #3498db inset;
        }}

        /* ---- Tab content ---- */
        .tab-content {{
            background: white;
            border-radius: 0 8px 8px 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-height: 300px;
        }}

        /* ---- Shared controls ---- */
        .section-controls {{
            background: #f8f9fa;
            padding: 12px 15px;
            border-radius: 6px;
            margin-bottom: 15px;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
        }}
        .section-controls input[type="text"] {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            width: 250px;
        }}
        .section-controls button {{
            padding: 8px 16px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        .section-controls button:hover {{
            background: #2980b9;
        }}
        .section-controls button.secondary {{
            background: #95a5a6;
        }}
        .section-controls button.secondary:hover {{
            background: #7f8c8d;
        }}
        .filter-group {{
            display: flex;
            gap: 10px;
            align-items: center;
            padding-left: 12px;
            border-left: 2px solid #ddd;
            flex-wrap: wrap;
        }}
        .filter-group-label {{
            font-weight: 500;
            color: #666;
            font-size: 13px;
        }}
        .column-toggles {{
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }}
        .column-toggles label {{
            font-size: 12px;
            padding: 3px 7px;
            background: #eee;
            border-radius: 4px;
            cursor: pointer;
        }}
        .column-toggles label:hover {{
            background: #ddd;
        }}
        .count-display {{
            color: #666;
            font-size: 14px;
            margin-left: auto;
        }}

        /* ---- Vocabulary table ---- */
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
        }}
        th {{
            background: #2c3e50;
            color: white;
            padding: 10px 12px;
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
            padding: 8px 12px;
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
        .col-hidden {{
            display: none;
        }}

        /* ---- Q&A items ---- */
        .qa-item {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            margin-bottom: 12px;
            overflow: hidden;
        }}
        .qa-header {{
            background: #2c3e50;
            color: white;
            padding: 10px 15px;
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
            margin-top: 12px;
            margin-bottom: 4px;
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
        .legend {{
            background: #f8f9fa;
            padding: 10px 15px;
            border-radius: 6px;
            margin-top: 15px;
            font-size: 13px;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .legend-title {{
            font-weight: 500;
            color: #666;
        }}

        /* ---- Summary ---- */
        .summary-content {{
            line-height: 1.7;
            font-size: 15px;
            max-width: 800px;
        }}
        .summary-content p {{
            margin: 0 0 12px 0;
        }}

        /* ---- Utility ---- */
        .hidden {{
            display: none;
        }}

        @media print {{
            .tab-bar, .section-controls {{ display: none; }}
            body {{ background: white; padding: 10px; }}
            .tab-content {{ display: block !important; box-shadow: none; border: none; }}
            .qa-content.collapsed {{ display: block !important; }}
        }}
    </style>
</head>
<body>
    <h1>Case Report</h1>
    <div class="timestamp">Generated {timestamp}</div>

    <div class="tab-bar">
            {tab_buttons}
    </div>

{tab_contents}

    <script>
        // ---- Tab switching ----
        function switchTab(tabId) {{
            // Hide all tab content
            document.querySelectorAll('.tab-content').forEach(el => {{
                el.style.display = 'none';
            }});
            // Deactivate all buttons
            document.querySelectorAll('.tab-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            // Show selected tab
            const target = document.getElementById('tab-' + tabId);
            if (target) target.style.display = '';
            // Activate button
            const btn = document.querySelector('.tab-btn[data-tab="' + tabId + '"]');
            if (btn) btn.classList.add('active');
        }}

        // ---- Vocabulary functions ----
        const hasVocab = {has_vocab};
        const hasQA = {has_qa};
        {vocab_js}

        function toggleColumn(colName) {{
            if (!hasVocab) return;
            const checkbox = document.getElementById('col-' + colName.replace(/[^a-zA-Z0-9]/g, ''));
            const isVisible = checkbox.checked;
            const colIndex = columnOrder.indexOf(colName);
            if (colIndex === -1) return;

            const headers = document.querySelectorAll('#vocab-table thead th');
            if (headers[colIndex]) headers[colIndex].classList.toggle('col-hidden', !isVisible);

            document.querySelectorAll('#vocab-table tbody tr').forEach(row => {{
                if (row.cells[colIndex]) {{
                    row.cells[colIndex].classList.toggle('col-hidden', !isVisible);
                }}
            }});
        }}

        function filterVocabTable() {{
            if (!hasVocab) return;
            const search = document.getElementById('vocab-search').value.toLowerCase();
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
                if (search && !term.includes(search)) show = false;
                if (personsOnly && !isPerson) show = false;
                if (termsOnly && isPerson) show = false;
                row.classList.toggle('hidden', !show);
                if (show) visibleCount++;
            }});

            document.getElementById('vocab-count').textContent =
                'Showing ' + visibleCount + ' of ' + rows.length + ' entries';
        }}

        function sortTable(colIndex) {{
            if (!hasVocab) return;
            const table = document.getElementById('vocab-table');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const headers = table.querySelectorAll('th');
            const colName = columnOrder[colIndex];
            const isNumeric = numericColumns.includes(colName);

            if (sortWarningColumns.includes(colName)) {{
                if (!confirm("Sorting by '" + colName + "' will show lower-quality results first.\\n\\nContinue?")) {{
                    return;
                }}
            }}

            if (sortColumn === colIndex) {{
                sortAsc = !sortAsc;
            }} else {{
                sortColumn = colIndex;
                sortAsc = true;
            }}

            headers.forEach((h, i) => {{
                h.classList.toggle('sorted', i === colIndex);
                const arrow = h.querySelector('.sort-arrow');
                if (arrow) arrow.textContent = (i === colIndex && !sortAsc) ? '\\u25B2' : '\\u25BC';
            }});

            rows.sort((a, b) => {{
                let aVal = a.cells[colIndex].textContent.trim();
                let bVal = b.cells[colIndex].textContent.trim();
                if (isNumeric) {{
                    aVal = parseFloat(aVal) || 0;
                    bVal = parseFloat(bVal) || 0;
                    return sortAsc ? aVal - bVal : bVal - aVal;
                }}
                return sortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            }});

            rows.forEach(row => tbody.appendChild(row));
        }}

        // ---- Q&A functions ----
        function filterQA() {{
            if (!hasQA) return;
            const search = document.getElementById('qa-search').value.toLowerCase();
            const items = document.querySelectorAll('#qa-container .qa-item');
            let visibleCount = 0;

            items.forEach(item => {{
                const text = item.textContent.toLowerCase();
                const show = !search || text.includes(search);
                item.classList.toggle('hidden', !show);
                if (show) visibleCount++;
            }});

            document.getElementById('qa-count').textContent =
                'Showing ' + visibleCount + ' of ' + items.length + ' Q&A pairs';
        }}

        function toggleQAItem(header) {{
            const content = header.nextElementSibling;
            const toggle = header.querySelector('.toggle');
            const isCollapsed = content.classList.toggle('collapsed');
            toggle.innerHTML = isCollapsed ? '&#x25B6; Show' : '&#x25BC; Hide';
        }}

        function toggleAllQA(expand) {{
            document.querySelectorAll('#qa-container .qa-content').forEach(content => {{
                content.classList.toggle('collapsed', !expand);
            }});
            document.querySelectorAll('#qa-container .toggle').forEach(toggle => {{
                toggle.innerHTML = expand ? '&#x25BC; Hide' : '&#x25B6; Show';
            }});
        }}
    </script>
</body>
</html>"""
