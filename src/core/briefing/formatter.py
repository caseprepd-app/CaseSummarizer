"""
Briefing Formatter for Case Briefing Generator.

Formats the aggregated briefing data into the final Case Briefing Sheet
output. Supports multiple output formats for different use cases.

The Case Briefing Sheet is designed for court reporters who need
quick prep before proceedings (~30 minutes).

Output Sections:
1. CASE TYPE - Classification (e.g., "Medical Malpractice")
2. PARTIES - Plaintiffs and Defendants
3. WHAT HAPPENED - Narrative summary
4. ALLEGATIONS - Key claims
5. DEFENSES - Key defenses
6. NAMES TO KNOW - People grouped by role
7. (Optional) Processing metadata
"""

from dataclasses import dataclass
from datetime import datetime

from .aggregator import AggregatedBriefingData, PersonEntry
from .orchestrator import BriefingResult
from .synthesizer import SynthesisResult


@dataclass
class FormattedBriefing:
    """
    Formatted Case Briefing Sheet output.

    Attributes:
        text: Plain text formatted briefing
        sections: Dict of section name -> content for UI display
        metadata: Processing metadata (timing, sources, etc.)
    """

    text: str
    sections: dict
    metadata: dict


class BriefingFormatter:
    """
    Formats Case Briefing Sheets for display and export.

    Produces structured output suitable for:
    - Plain text display and export
    - UI panel rendering (section-by-section)
    - Markdown export

    Example:
        formatter = BriefingFormatter()
        formatted = formatter.format(briefing_result)
        print(formatted.text)

        # Or access sections individually
        print(formatted.sections["narrative"])
    """

    # Section header styling
    HEADER_CHAR = "="
    SUBHEADER_CHAR = "-"
    SECTION_WIDTH = 60

    def __init__(self, include_metadata: bool = False):
        """
        Initialize the formatter.

        Args:
            include_metadata: Whether to include processing stats in output
        """
        self.include_metadata = include_metadata

    def format(self, result: BriefingResult) -> FormattedBriefing:
        """
        Format a BriefingResult into displayable output.

        Args:
            result: BriefingResult from BriefingOrchestrator

        Returns:
            FormattedBriefing with text and structured sections
        """
        if not result.success or not result.aggregated_data:
            return self._format_error(result)

        data = result.aggregated_data
        narrative = result.narrative

        # Build sections
        sections = {}

        sections["case_type"] = self._format_case_type(data.case_type)
        sections["parties"] = self._format_parties(data.plaintiffs, data.defendants)
        sections["narrative"] = self._format_narrative(narrative)
        sections["allegations"] = self._format_list_section("ALLEGATIONS", data.allegations)
        sections["defenses"] = self._format_list_section("DEFENSES", data.defenses)
        sections["names"] = self._format_names(data.people_by_category)
        sections["vocabulary"] = self._format_vocabulary(data.vocabulary)

        if self.include_metadata:
            sections["metadata"] = self._format_metadata(result)

        # Build full text
        text_parts = [
            self._make_header("CASE BRIEFING SHEET"),
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            sections["case_type"],
            "",
            sections["parties"],
            "",
            self._make_subheader("WHAT HAPPENED"),
            sections["narrative"],
            "",
            sections["allegations"],
            "",
            sections["defenses"],
            "",
            sections["names"],
            "",
            sections["vocabulary"],
        ]

        if self.include_metadata:
            text_parts.extend(["", sections["metadata"]])

        text = "\n".join(text_parts)

        # Build metadata dict
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "total_time_seconds": result.total_time_seconds,
            "chunk_count": result.chunk_count,
            "extraction_count": result.extraction_count,
            "source_documents": data.source_documents,
            "narrative_method": narrative.method if narrative else "none",
        }

        return FormattedBriefing(
            text=text,
            sections=sections,
            metadata=metadata,
        )

    def format_for_export(self, result: BriefingResult, format_type: str = "txt") -> str:
        """
        Format for file export.

        Args:
            result: BriefingResult to format
            format_type: "txt" or "md" (markdown)

        Returns:
            Formatted string for export
        """
        if format_type == "md":
            return self._format_markdown(result)
        return self.format(result).text

    def _format_error(self, result: BriefingResult) -> FormattedBriefing:
        """Format an error result."""
        error_text = f"""
{self._make_header("CASE BRIEFING SHEET")}

ERROR: Briefing generation failed

{result.error_message}

Please check that:
1. Documents contain readable text
2. Ollama is running and accessible
3. The selected model is available
"""
        return FormattedBriefing(
            text=error_text.strip(),
            sections={"error": result.error_message},
            metadata={"success": False, "error": result.error_message},
        )

    def _format_case_type(self, case_type: str) -> str:
        """Format case type section."""
        if not case_type or case_type == "unknown":
            return "CASE TYPE: Not determined"
        return f"CASE TYPE: {case_type.title()}"

    def _format_parties(self, plaintiffs: list[str], defendants: list[str]) -> str:
        """Format parties section."""
        lines = [self._make_subheader("PARTIES")]

        if plaintiffs:
            lines.append(f"Plaintiff(s): {', '.join(plaintiffs)}")
        else:
            lines.append("Plaintiff(s): Not identified")

        if defendants:
            lines.append(f"Defendant(s): {', '.join(defendants)}")
        else:
            lines.append("Defendant(s): Not identified")

        return "\n".join(lines)

    def _format_narrative(self, narrative: SynthesisResult | None) -> str:
        """Format the WHAT HAPPENED narrative."""
        if not narrative or not narrative.narrative:
            return "No narrative generated. Check document content."

        return narrative.narrative

    def _format_list_section(self, title: str, items: list[str]) -> str:
        """Format a bulleted list section."""
        lines = [self._make_subheader(title)]

        if not items:
            lines.append("None identified in documents.")
        else:
            for i, item in enumerate(items[:10], 1):  # Limit to 10 items
                # Truncate long items
                display = item if len(item) <= 200 else item[:197] + "..."
                lines.append(f"  {i}. {display}")

            if len(items) > 10:
                lines.append(f"  ... and {len(items) - 10} more")

        return "\n".join(lines)

    def _format_names(self, people_by_category: dict[str, list[PersonEntry]]) -> str:
        """Format the NAMES TO KNOW section."""
        lines = [self._make_subheader("NAMES TO KNOW")]

        if not people_by_category:
            lines.append("No key individuals identified.")
            return "\n".join(lines)

        # Category display order and labels
        category_order = [
            ("PARTY", "Parties"),
            ("MEDICAL", "Medical Personnel"),
            ("WITNESS", "Witnesses"),
            ("OTHER", "Other Individuals"),
        ]

        for category_key, category_label in category_order:
            people = people_by_category.get(category_key, [])
            if not people:
                continue

            lines.append(f"\n  {category_label}:")
            for person in people[:10]:  # Limit per category
                line = f"    - {person.canonical_name}"
                if person.role:
                    line += f" ({person.role})"
                lines.append(line)

            if len(people) > 10:
                lines.append(f"    ... and {len(people) - 10} more")

        return "\n".join(lines)

    def _format_vocabulary(self, vocabulary: list[str]) -> str:
        """Format the VOCABULARY section with technical/unusual terms."""
        lines = [self._make_subheader("VOCABULARY")]

        if not vocabulary:
            lines.append("No unusual terminology identified.")
            return "\n".join(lines)

        # Display terms in a compact comma-separated format for easy scanning
        # Limit to 30 terms for readability
        display_terms = vocabulary[:30]
        lines.append("  " + ", ".join(display_terms))

        if len(vocabulary) > 30:
            lines.append(f"  ... and {len(vocabulary) - 30} more terms")

        return "\n".join(lines)

    def _format_metadata(self, result: BriefingResult) -> str:
        """Format processing metadata."""
        lines = [self._make_subheader("PROCESSING INFO")]

        lines.append(f"  Total time: {result.total_time_seconds:.1f} seconds")
        lines.append(f"  Chunks processed: {result.chunk_count}")
        lines.append(f"  Successful extractions: {result.extraction_count}")

        if result.aggregated_data:
            lines.append(f"  Source documents: {len(result.aggregated_data.source_documents)}")
            for doc in result.aggregated_data.source_documents:
                lines.append(f"    - {doc}")

        if result.timing:
            lines.append("\n  Timing breakdown:")
            for phase, ms in result.timing.items():
                if phase != "total":
                    lines.append(f"    {phase}: {ms:.0f}ms")

        return "\n".join(lines)

    def _format_markdown(self, result: BriefingResult) -> str:
        """Format as Markdown for export."""
        if not result.success or not result.aggregated_data:
            return f"# Case Briefing Sheet\n\n**Error:** {result.error_message}"

        data = result.aggregated_data
        narrative = result.narrative

        lines = [
            "# Case Briefing Sheet",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
            "---",
            "",
            f"## Case Type: {data.case_type.title() if data.case_type else 'Unknown'}",
            "",
            "## Parties",
            "",
            f"**Plaintiff(s):** {', '.join(data.plaintiffs) if data.plaintiffs else 'Not identified'}",
            "",
            f"**Defendant(s):** {', '.join(data.defendants) if data.defendants else 'Not identified'}",
            "",
            "---",
            "",
            "## What Happened",
            "",
            narrative.narrative if narrative and narrative.narrative else "*No narrative generated*",
            "",
            "---",
            "",
            "## Allegations",
            "",
        ]

        if data.allegations:
            for allegation in data.allegations[:10]:
                lines.append(f"- {allegation}")
        else:
            lines.append("*None identified*")

        lines.extend([
            "",
            "## Defenses",
            "",
        ])

        if data.defenses:
            for defense in data.defenses[:10]:
                lines.append(f"- {defense}")
        else:
            lines.append("*None identified*")

        lines.extend([
            "",
            "---",
            "",
            "## Names to Know",
            "",
        ])

        category_order = [
            ("PARTY", "Parties"),
            ("MEDICAL", "Medical Personnel"),
            ("WITNESS", "Witnesses"),
            ("OTHER", "Other Individuals"),
        ]

        for category_key, category_label in category_order:
            people = data.people_by_category.get(category_key, [])
            if people:
                lines.append(f"### {category_label}")
                lines.append("")
                for person in people[:10]:
                    role_part = f" - {person.role}" if person.role else ""
                    lines.append(f"- **{person.canonical_name}**{role_part}")
                lines.append("")

        # Vocabulary section
        lines.extend([
            "---",
            "",
            "## Vocabulary",
            "",
        ])

        if data.vocabulary:
            lines.append("*Technical and unusual terms for reference:*")
            lines.append("")
            lines.append(", ".join(data.vocabulary[:30]))
            if len(data.vocabulary) > 30:
                lines.append("")
                lines.append(f"*... and {len(data.vocabulary) - 30} more terms*")
        else:
            lines.append("*No unusual terminology identified*")

        return "\n".join(lines)

    def _make_header(self, text: str) -> str:
        """Create a main header."""
        line = self.HEADER_CHAR * self.SECTION_WIDTH
        return f"{line}\n{text.center(self.SECTION_WIDTH)}\n{line}"

    def _make_subheader(self, text: str) -> str:
        """Create a section subheader."""
        return f"{text}\n{self.SUBHEADER_CHAR * len(text)}"

    def get_section_titles(self) -> list[str]:
        """
        Get list of section titles for UI tab creation.

        Returns:
            List of section identifier strings
        """
        return [
            "case_type",
            "parties",
            "narrative",
            "allegations",
            "defenses",
            "names",
            "vocabulary",
        ]
