"""
Narrative Synthesizer for Case Briefing Generator.

Generates the "WHAT HAPPENED" section of the Case Briefing Sheet by
synthesizing the aggregated data into a coherent narrative summary.

The narrative is designed for court reporters who need to quickly
understand the key events and context of a case before a proceeding.

Key Features:
- LLM-based narrative generation using Ollama
- Structured prompt with all relevant context
- Fallback to template-based narrative if LLM fails
- Configurable target length
"""

from dataclasses import dataclass

from src.core.ai.ollama_model_manager import OllamaModelManager
from src.logging_config import debug_log

from .aggregator import AggregatedBriefingData, PersonEntry


@dataclass
class SynthesisResult:
    """
    Result of narrative synthesis.

    Attributes:
        narrative: The generated "WHAT HAPPENED" narrative text
        success: Whether synthesis succeeded
        method: How narrative was generated ("llm" or "template")
        word_count: Approximate word count
    """

    narrative: str
    success: bool = True
    method: str = "llm"
    word_count: int = 0

    def __post_init__(self):
        """Calculate word count."""
        if self.narrative and not self.word_count:
            self.word_count = len(self.narrative.split())


class NarrativeSynthesizer:
    """
    Generates narrative summaries from aggregated case data.

    Uses Ollama to create a natural language "WHAT HAPPENED" section
    that summarizes the key events and context of a legal case.

    Example:
        synthesizer = NarrativeSynthesizer()
        result = synthesizer.synthesize(aggregated_data)
        print(result.narrative)
    """

    # Prompt template for narrative generation
    SYNTHESIS_PROMPT = """You are a legal assistant helping court reporters prepare for proceedings.

Based on the extracted case information below, write a concise "WHAT HAPPENED" narrative summary.

INSTRUCTIONS:
- Write 2-4 paragraphs summarizing the key events and claims
- Focus on WHAT happened, not legal arguments
- Include relevant dates if available
- Mention key parties by name
- Keep it factual and neutral
- Target length: {target_words} words
- Do NOT include headers or bullet points - write flowing prose

CASE INFORMATION:
Case Type: {case_type}

Plaintiffs: {plaintiffs}

Defendants: {defendants}

Key Allegations:
{allegations}

Key Defenses:
{defenses}

Key Facts:
{key_facts}

Relevant Dates: {dates}

WRITE THE NARRATIVE NOW (no preamble, just the narrative):"""

    def __init__(
        self,
        ollama_manager: OllamaModelManager | None = None,
        target_words: int = 200,
        max_tokens: int = 500,
    ):
        """
        Initialize the narrative synthesizer.

        Args:
            ollama_manager: OllamaModelManager instance (creates new if None)
            target_words: Target word count for narrative
            max_tokens: Maximum tokens for LLM response
        """
        self.ollama_manager = ollama_manager or OllamaModelManager()
        self.target_words = target_words
        self.max_tokens = max_tokens

        debug_log(f"[NarrativeSynthesizer] Initialized: target_words={target_words}")

    def synthesize(self, data: AggregatedBriefingData) -> SynthesisResult:
        """
        Generate narrative from aggregated case data.

        Attempts LLM-based synthesis first, falls back to template if needed.

        Args:
            data: AggregatedBriefingData from the aggregator

        Returns:
            SynthesisResult with narrative text
        """
        debug_log("[NarrativeSynthesizer] Starting narrative synthesis")

        # Try LLM synthesis first
        try:
            result = self._synthesize_with_llm(data)
            if result.success and result.narrative:
                debug_log(
                    f"[NarrativeSynthesizer] LLM synthesis complete: {result.word_count} words"
                )
                return result
        except Exception as e:
            debug_log(f"[NarrativeSynthesizer] LLM synthesis failed: {e}")

        # Fallback to template
        debug_log("[NarrativeSynthesizer] Falling back to template synthesis")
        return self._synthesize_with_template(data)

    def _synthesize_with_llm(self, data: AggregatedBriefingData) -> SynthesisResult:
        """
        Generate narrative using Ollama LLM.

        Args:
            data: Aggregated case data

        Returns:
            SynthesisResult with LLM-generated narrative
        """
        # Build prompt
        prompt = self.SYNTHESIS_PROMPT.format(
            target_words=self.target_words,
            case_type=data.case_type or "Unknown",
            plaintiffs=self._format_list(data.plaintiffs) or "Not identified",
            defendants=self._format_list(data.defendants) or "Not identified",
            allegations=self._format_numbered_list(data.allegations) or "None extracted",
            defenses=self._format_numbered_list(data.defenses) or "None extracted",
            key_facts=self._format_numbered_list(data.key_facts) or "None extracted",
            dates=self._format_list(data.dates) or "None identified",
        )

        # Generate with Ollama (use regular generate for flowing prose)
        response = self.ollama_manager.generate(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=0.3,  # Slightly creative for natural prose
        )

        if not response:
            return SynthesisResult(narrative="", success=False, method="llm")

        # Clean up the response
        narrative = self._clean_narrative(response)

        return SynthesisResult(
            narrative=narrative,
            success=True,
            method="llm",
        )

    def _synthesize_with_template(self, data: AggregatedBriefingData) -> SynthesisResult:
        """
        Generate narrative using template (fallback).

        Creates a structured summary when LLM is unavailable.

        Args:
            data: Aggregated case data

        Returns:
            SynthesisResult with template-based narrative
        """
        parts = []

        # Opening sentence with case type and parties
        case_type = data.case_type or "legal case"
        if data.plaintiffs and data.defendants:
            parts.append(
                f"This is a {case_type} case brought by {self._format_list(data.plaintiffs)} "
                f"against {self._format_list(data.defendants)}."
            )
        elif data.plaintiffs:
            parts.append(
                f"This is a {case_type} case involving {self._format_list(data.plaintiffs)}."
            )
        else:
            parts.append(f"This is a {case_type} case.")

        # Key allegations
        if data.allegations:
            parts.append("")
            if len(data.allegations) == 1:
                parts.append(f"The plaintiff alleges that {data.allegations[0].lower()}")
            else:
                parts.append("The key allegations in this case include:")
                for allegation in data.allegations[:5]:  # Limit to top 5
                    parts.append(f"- {allegation}")

        # Key defenses
        if data.defenses:
            parts.append("")
            if len(data.defenses) == 1:
                parts.append(f"The defendant contends that {data.defenses[0].lower()}")
            else:
                parts.append("The defense asserts:")
                for defense in data.defenses[:3]:  # Limit to top 3
                    parts.append(f"- {defense}")

        # Key facts
        if data.key_facts:
            parts.append("")
            parts.append("Key facts include:")
            for fact in data.key_facts[:5]:  # Limit to top 5
                parts.append(f"- {fact}")

        # Dates if available
        if data.dates:
            parts.append("")
            parts.append(f"Relevant dates: {self._format_list(data.dates[:5])}")

        narrative = "\n".join(parts)

        return SynthesisResult(
            narrative=narrative,
            success=True,
            method="template",
        )

    def _format_list(self, items: list[str], conjunction: str = "and") -> str:
        """
        Format list for natural reading.

        Args:
            items: List of strings
            conjunction: Word to join last two items

        Returns:
            Formatted string (e.g., "A, B, and C")
        """
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} {conjunction} {items[1]}"
        return f"{', '.join(items[:-1])}, {conjunction} {items[-1]}"

    def _format_numbered_list(self, items: list[str]) -> str:
        """
        Format as numbered list for prompt.

        Args:
            items: List of strings

        Returns:
            Numbered list string
        """
        if not items:
            return ""
        return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items[:10]))

    def _clean_narrative(self, text: str) -> str:
        """
        Clean up LLM-generated narrative.

        Removes common artifacts and normalizes formatting.

        Args:
            text: Raw LLM output

        Returns:
            Cleaned narrative text
        """
        # Remove common preambles
        preambles = [
            "Here is the narrative:",
            "Here's the narrative:",
            "WHAT HAPPENED:",
            "What Happened:",
            "Narrative:",
        ]
        for preamble in preambles:
            if text.strip().startswith(preamble):
                text = text[len(preamble) :].strip()

        # Remove trailing artifacts
        text = text.strip()

        # Remove any markdown headers that slipped through
        lines = text.split("\n")
        cleaned_lines = [
            line
            for line in lines
            if not line.strip().startswith("#") and not line.strip().startswith("##")
        ]
        text = "\n".join(cleaned_lines)

        return text.strip()

    def format_people_section(
        self,
        people_by_category: dict[str, list[PersonEntry]],
    ) -> str:
        """
        Format the "NAMES TO KNOW" section of the briefing.

        Groups people by category (MEDICAL, PARTY, WITNESS, OTHER).

        Args:
            people_by_category: Dict from AggregatedBriefingData

        Returns:
            Formatted string for briefing output
        """
        if not people_by_category:
            return "No key individuals identified."

        # Define category order and display names
        category_order = [
            ("PARTY", "Parties"),
            ("MEDICAL", "Medical Personnel"),
            ("WITNESS", "Witnesses"),
            ("OTHER", "Other Individuals"),
        ]

        sections = []

        for category_key, category_name in category_order:
            people = people_by_category.get(category_key, [])
            if not people:
                continue

            section_lines = [f"{category_name}:"]
            for person in people:
                line = f"  - {person.canonical_name}"
                if person.role:
                    line += f" ({person.role})"
                section_lines.append(line)

            sections.append("\n".join(section_lines))

        return "\n\n".join(sections) if sections else "No key individuals identified."
