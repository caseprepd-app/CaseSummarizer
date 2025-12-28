"""
LLM-based Vocabulary Extractor (Session 45 Update)

Uses Ollama to extract names and technical vocabulary from document chunks
with a single prompt per chunk for efficiency. This replaces the previous
4-prompt-per-chunk approach used in Case Briefing.

Session 45 Update:
- Added support for UnifiedChunk objects from unified_chunker
- New combined extraction format with separate "people" and "vocabulary" keys
- Better structure for reconciliation with NER results

Design philosophy:
- One focused prompt that extracts NAMES and TECHNICAL VOCABULARY
- Results stored in Python lists (no truncation like "and 70 more...")
- Categories from shared config (config/categories.json)
- Designed for court reporters who need word lists for transcription prep

Usage:
    from src.core.extraction.llm_extractor import LLMVocabExtractor
    from src.core.chunking import UnifiedChunker

    # With unified chunks (Session 45)
    chunker = UnifiedChunker()
    chunks = chunker.chunk_text(document_text)

    extractor = LLMVocabExtractor(use_combined_prompt=True)
    result = extractor.extract_from_unified_chunks(chunks)

    for person in result.people:
        print(f"{person.name} ({person.role})")
    for term in result.terms:
        print(f"{term.term} ({term.type})")
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src.core.ai.ollama_model_manager import OllamaModelManager
from src.categories import get_category_list, get_llm_prompt_categories, normalize_category
from src.logging_config import debug_log


# Default prompt file path (Session 45: combined extraction)
PROMPT_FILE = Path(__file__).parent.parent.parent.parent / "config" / "extraction_prompts" / "combined_extraction.txt"

# Chunk size for processing (characters)
DEFAULT_CHUNK_SIZE = 2000

# Maximum tokens for LLM response
DEFAULT_MAX_TOKENS = 1000


@dataclass
class LLMPerson:
    """A person/organization extracted by the LLM (Session 45).

    Attributes:
        name: Full name of the person or organization
        role: Role in the case (plaintiff, defendant, attorney, witness, etc.)
        source_chunk: Index of the chunk this person was extracted from
        confidence: Confidence score (0-1), default 0.8 for LLM extractions
    """
    name: str
    role: str
    source_chunk: int = 0
    confidence: float = 0.8

    # Valid roles for normalization
    VALID_ROLES = {
        "plaintiff", "defendant", "attorney", "judge", "witness",
        "treating_physician", "expert", "nurse", "family_member",
        "employer", "organization", "other"
    }

    def __post_init__(self):
        """Validate and normalize the role."""
        self.name = self.name.strip()
        role_lower = self.role.lower().strip().replace(" ", "_")
        # Normalize common variations
        if role_lower in ("doctor", "physician", "dr"):
            role_lower = "treating_physician"
        elif role_lower in ("company", "corporation", "firm", "hospital"):
            role_lower = "organization"
        elif role_lower not in self.VALID_ROLES:
            role_lower = "other"
        self.role = role_lower


@dataclass
class LLMTerm:
    """A vocabulary term extracted by the LLM.

    Attributes:
        term: The extracted term text (medical, legal, technical vocabulary)
        type: Category from shared config (Medical, Legal, Technical, Acronym, Unknown)
        source_chunk: Index of the chunk this term was extracted from
        confidence: Confidence score (0-1), default 0.8 for LLM extractions
    """
    term: str
    type: str
    source_chunk: int = 0
    confidence: float = 0.8

    def __post_init__(self):
        """Validate and normalize the category."""
        self.type = normalize_category(self.type)
        self.term = self.term.strip()


@dataclass
class LLMExtractionResult:
    """Result from LLM extraction of a document (Session 45 Update).

    Attributes:
        people: List of extracted people/organizations with roles
        terms: List of extracted vocabulary terms (medical, legal, technical)
        chunk_count: Number of chunks processed
        processing_time_ms: Total processing time in milliseconds
        success: Whether extraction completed successfully
        error_message: Error message if success is False
    """
    people: list[LLMPerson] = field(default_factory=list)
    terms: list[LLMTerm] = field(default_factory=list)
    chunk_count: int = 0
    processing_time_ms: float = 0.0
    success: bool = True
    error_message: str = ""


class LLMVocabExtractor:
    """
    Extracts vocabulary using a single LLM prompt per chunk.

    This class provides efficient vocabulary extraction by using one focused
    prompt that extracts both names and technical terms in a single pass.

    Key features:
    - Single prompt per chunk (vs. 4 prompts in Case Briefing)
    - Results stored in Python lists (no truncation)
    - Uses shared category config for consistency with NER
    - Progress callback support for UI updates

    Example:
        extractor = LLMVocabExtractor()

        def on_progress(current, total):
            print(f"Processing chunk {current}/{total}")

        result = extractor.extract(document_text, progress_callback=on_progress)

        if result.success:
            for term in result.terms:
                print(f"{term.term}: {term.type}")
    """

    def __init__(
        self,
        ollama_manager: Optional[OllamaModelManager] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        prompt_file: Optional[Path] = None,
    ):
        """
        Initialize the LLM vocab extractor.

        Args:
            ollama_manager: OllamaModelManager instance (creates new if None)
            max_tokens: Maximum tokens for LLM response
            prompt_file: Path to prompt template file (uses default if None)
        """
        self.ollama_manager = ollama_manager or OllamaModelManager()
        self.max_tokens = max_tokens
        self.prompt_file = prompt_file or PROMPT_FILE
        self._prompt_template = self._load_prompt_template()

        debug_log(f"[LLMVocabExtractor] Initialized with max_tokens={max_tokens}")

    def _load_prompt_template(self) -> str:
        """Load the prompt template from file."""
        if self.prompt_file.exists():
            template = self.prompt_file.read_text(encoding="utf-8")
            debug_log(f"[LLMVocabExtractor] Loaded prompt from {self.prompt_file}")
            return template

        debug_log(f"[LLMVocabExtractor] Prompt file not found, using default")
        return self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """Return default prompt if file is missing."""
        categories = get_llm_prompt_categories()
        return f"""Extract NAMES and TECHNICAL VOCABULARY from this legal document chunk.

For court reporters who need to prepare word lists for transcription accuracy.

EXTRACT TWO TYPES OF TERMS:

1. NAMES (people and organizations):
   - Full names of individuals (attorneys, witnesses, parties, doctors)
   - Organization names (hospitals, law firms, companies)
   - Do NOT include: titles alone (Dr., Mr.), single letters, generic roles

2. TECHNICAL VOCABULARY (terms needing definition):
   - Medical terms (diagnoses, procedures, medications, anatomy)
   - Legal terms (motions, causes of action, legal concepts)
   - Industry-specific jargon
   - Do NOT include: common English words, names of people/places

CATEGORIES: {categories}

Return ONLY valid JSON:
{{"terms": [{{"term": "John Smith", "type": "Person"}}, {{"term": "pulmonary embolism", "type": "Medical"}}]}}

DOCUMENT CHUNK:
{{chunk_text}}"""

    def extract(
        self,
        text: str,
        chunk_size_chars: int = DEFAULT_CHUNK_SIZE,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> LLMExtractionResult:
        """
        Extract people and vocabulary from text using single LLM prompt per chunk.

        Results are stored in memory as lists of LLMPerson and LLMTerm objects.

        Args:
            text: Full document text to process
            chunk_size_chars: Size of each chunk in characters
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            LLMExtractionResult with all extracted people and terms
        """
        start_time = time.time()
        all_people: list[LLMPerson] = []
        all_terms: list[LLMTerm] = []

        # Chunk the text
        chunks = self._chunk_text(text, chunk_size_chars)
        chunk_count = len(chunks)

        debug_log(f"[LLMVocabExtractor] Processing {chunk_count} chunks")

        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i + 1, chunk_count)

            chunk_people, chunk_terms = self._extract_chunk(chunk, i)
            all_people.extend(chunk_people)
            all_terms.extend(chunk_terms)

            debug_log(
                f"[LLMVocabExtractor] Chunk {i+1}/{chunk_count}: "
                f"extracted {len(chunk_people)} people, {len(chunk_terms)} terms"
            )

        processing_time = (time.time() - start_time) * 1000

        debug_log(
            f"[LLMVocabExtractor] Complete: {len(all_people)} people, {len(all_terms)} terms "
            f"from {chunk_count} chunks in {processing_time:.1f}ms"
        )

        return LLMExtractionResult(
            people=all_people,
            terms=all_terms,
            chunk_count=chunk_count,
            processing_time_ms=processing_time,
            success=True,
        )

    def _chunk_text(self, text: str, chunk_size: int) -> list[str]:
        """
        Split text into chunks for processing.

        Attempts to split on sentence boundaries when possible.

        Args:
            text: Full text to chunk
            chunk_size: Target chunk size in characters

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        current_pos = 0

        while current_pos < len(text):
            # Get chunk of target size
            end_pos = min(current_pos + chunk_size, len(text))

            if end_pos < len(text):
                # Try to find a sentence boundary
                chunk = text[current_pos:end_pos]

                # Look for sentence-ending punctuation
                for punct in ['. ', '.\n', '? ', '?\n', '! ', '!\n']:
                    last_punct = chunk.rfind(punct)
                    if last_punct > chunk_size * 0.5:  # At least half the chunk
                        end_pos = current_pos + last_punct + len(punct)
                        break

            chunks.append(text[current_pos:end_pos].strip())
            current_pos = end_pos

        return [c for c in chunks if c]  # Remove empty chunks

    def _extract_chunk(self, chunk: str, chunk_id: int) -> tuple[list[LLMPerson], list[LLMTerm]]:
        """
        Extract people and terms from a single chunk (Session 45 combined format).

        Args:
            chunk: Text chunk to process
            chunk_id: Index of this chunk (for source tracking)

        Returns:
            Tuple of (people_list, terms_list) extracted from this chunk
        """
        # Build prompt using replace() to avoid curly brace issues with JSON
        prompt = self._prompt_template.replace("{chunk_text}", chunk)

        try:
            response = self.ollama_manager.generate_structured(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=0.0,  # Deterministic for extraction
            )

            if response is None:
                debug_log(f"[LLMVocabExtractor] No response for chunk {chunk_id}")
                return [], []

            return self._parse_response(response, chunk_id)

        except Exception as e:
            debug_log(f"[LLMVocabExtractor] Error extracting chunk {chunk_id}: {e}")
            return [], []

    def _parse_response(self, response: dict | str, chunk_id: int) -> tuple[list[LLMPerson], list[LLMTerm]]:
        """
        Parse LLM JSON response into LLMPerson and LLMTerm objects (Session 45).

        Handles combined format: {"people": [...], "vocabulary": [...]}

        Args:
            response: LLM response (dict or JSON string)
            chunk_id: Source chunk index

        Returns:
            Tuple of (people_list, terms_list)
        """
        people = []
        terms = []

        # Handle string response
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                debug_log(f"[LLMVocabExtractor] Failed to parse JSON from chunk {chunk_id}")
                return [], []

        if not isinstance(response, dict):
            debug_log(f"[LLMVocabExtractor] Unexpected response type: {type(response)}")
            return [], []

        # Extract people from response
        raw_people = response.get("people", [])
        for raw_person in raw_people:
            if not isinstance(raw_person, dict):
                continue

            name = raw_person.get("name", "").strip()
            role = raw_person.get("role", "other")

            # Skip empty or too-short names
            if not name or len(name) < 2:
                continue

            # Skip generic titles alone
            if name.upper() in ["DR", "MR", "MS", "MRS", "MISS", "THE", "PLAINTIFF", "DEFENDANT"]:
                continue

            people.append(LLMPerson(
                name=name,
                role=role,
                source_chunk=chunk_id,
            ))

        # Extract vocabulary terms from response
        raw_terms = response.get("vocabulary", [])
        # Also check for legacy "terms" key for backward compatibility
        if not raw_terms:
            raw_terms = response.get("terms", [])

        for raw_term in raw_terms:
            if not isinstance(raw_term, dict):
                continue

            term_text = raw_term.get("term", "").strip()
            term_type = raw_term.get("type", "Unknown")

            # Skip empty or too-short terms
            if not term_text or len(term_text) < 2:
                continue

            # Skip obvious noise
            if term_text.upper() in ["Q", "A", "THE", "AND", "OR", "BUT"]:
                continue

            terms.append(LLMTerm(
                term=term_text,
                type=term_type,
                source_chunk=chunk_id,
            ))

        return people, terms

    def extract_from_chunks(
        self,
        chunks: list[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> LLMExtractionResult:
        """
        Extract people and vocabulary from pre-chunked text.

        Use this when you already have document chunks as plain strings.

        Args:
            chunks: List of text chunks
            progress_callback: Optional callback(current, total) for progress

        Returns:
            LLMExtractionResult with all extracted people and terms
        """
        start_time = time.time()
        all_people: list[LLMPerson] = []
        all_terms: list[LLMTerm] = []
        chunk_count = len(chunks)

        debug_log(f"[LLMVocabExtractor] Processing {chunk_count} pre-chunked segments")

        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i + 1, chunk_count)

            chunk_people, chunk_terms = self._extract_chunk(chunk, i)
            all_people.extend(chunk_people)
            all_terms.extend(chunk_terms)

        processing_time = (time.time() - start_time) * 1000

        return LLMExtractionResult(
            people=all_people,
            terms=all_terms,
            chunk_count=chunk_count,
            processing_time_ms=processing_time,
            success=True,
        )

    def extract_from_unified_chunks(
        self,
        chunks: list,  # list[UnifiedChunk] - avoid import for type hint
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> LLMExtractionResult:
        """
        Extract people and vocabulary from UnifiedChunk objects (Session 45).

        Use this when you have chunks from UnifiedChunker for integrated
        processing with Q&A indexing.

        Args:
            chunks: List of UnifiedChunk objects from unified_chunker
            progress_callback: Optional callback(current, total) for progress

        Returns:
            LLMExtractionResult with all extracted people and terms
        """
        start_time = time.time()
        all_people: list[LLMPerson] = []
        all_terms: list[LLMTerm] = []
        chunk_count = len(chunks)

        debug_log(f"[LLMVocabExtractor] Processing {chunk_count} unified chunks")

        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i + 1, chunk_count)

            # UnifiedChunk has .text attribute
            chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
            chunk_id = chunk.chunk_num if hasattr(chunk, 'chunk_num') else i

            chunk_people, chunk_terms = self._extract_chunk(chunk_text, chunk_id)
            all_people.extend(chunk_people)
            all_terms.extend(chunk_terms)

            debug_log(
                f"[LLMVocabExtractor] Chunk {i+1}/{chunk_count}: "
                f"extracted {len(chunk_people)} people, {len(chunk_terms)} terms"
            )

        processing_time = (time.time() - start_time) * 1000

        debug_log(
            f"[LLMVocabExtractor] Complete: {len(all_people)} people, {len(all_terms)} terms "
            f"from {chunk_count} unified chunks in {processing_time:.1f}ms"
        )

        return LLMExtractionResult(
            people=all_people,
            terms=all_terms,
            chunk_count=chunk_count,
            processing_time_ms=processing_time,
            success=True,
        )
