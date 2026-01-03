"""
Chunk Extractor for Case Briefing Generator.

Extracts structured information from document chunks using Ollama's
structured output mode. Uses PARALLEL PROMPT CHAINING for better accuracy.

Architecture:
- 4 focused prompts per chunk (parties, names, allegations, facts)
- All (chunk, prompt) pairs processed in parallel via ThreadPoolExecutor
- Results merged by chunk_id after completion

This is the MAP phase of the Map-Reduce pattern.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.core.ai.ollama_model_manager import OllamaModelManager
from src.logging_config import debug_log

from .chunker import BriefingChunk


# External prompt files directory
EXTRACTION_PROMPTS_DIR = Path("config/extraction_prompts")

# Prompt types and their corresponding files
PROMPT_TYPES = {
    "parties": "01_parties.txt",
    "names": "02_names.txt",
    "allegations": "03_allegations.txt",
    "facts": "04_facts.txt",
}


@dataclass
class PartialExtraction:
    """
    Partial extraction result from a single focused prompt.

    Attributes:
        chunk_id: ID of the source chunk
        prompt_type: Which prompt was used ("parties", "names", etc.)
        data: Parsed JSON response from Ollama
        success: Whether extraction succeeded
    """

    chunk_id: int
    prompt_type: str
    data: dict = field(default_factory=dict)
    success: bool = True


@dataclass
class ChunkExtraction:
    """
    Extracted data from a single document chunk.

    Represents the merged output of all 4 focused prompts for one chunk.
    Multiple ChunkExtraction objects are later merged by the DataAggregator.

    Attributes:
        chunk_id: ID of the source chunk
        source_document: Original filename
        document_type: Type of legal document
        parties: Dict with plaintiffs and defendants lists
        allegations: List of allegation strings
        defenses: List of defense strings
        names_mentioned: List of name dicts with name, role, category
        key_facts: List of key fact strings
        dates_mentioned: List of date strings
        case_type_hints: List of case type indicator strings
        vocabulary: List of unusual/technical terms for layperson reference
        extraction_success: Whether extraction succeeded
        raw_response: Raw JSON response for debugging
    """

    chunk_id: int
    source_document: str
    document_type: str
    parties: dict = field(default_factory=lambda: {"plaintiffs": [], "defendants": []})
    allegations: list[str] = field(default_factory=list)
    defenses: list[str] = field(default_factory=list)
    names_mentioned: list[dict] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    dates_mentioned: list[str] = field(default_factory=list)
    case_type_hints: list[str] = field(default_factory=list)
    vocabulary: list[str] = field(default_factory=list)
    extraction_success: bool = True
    raw_response: dict | None = None


class ChunkExtractor:
    """
    Extracts structured information from document chunks via Ollama.

    Uses PARALLEL PROMPT CHAINING: 4 focused prompts per chunk,
    all running in parallel for maximum throughput.

    Example:
        extractor = ChunkExtractor()
        extractions = extractor.extract_batch(chunks)
        for e in extractions:
            print(e.parties)
    """

    def __init__(
        self,
        ollama_manager: OllamaModelManager | None = None,
        max_tokens: int = 500,  # Reduced since prompts are simpler
    ):
        """
        Initialize the chunk extractor.

        Args:
            ollama_manager: OllamaModelManager instance (creates new if None)
            max_tokens: Maximum tokens for extraction response
        """
        self.ollama_manager = ollama_manager or OllamaModelManager()
        self.max_tokens = max_tokens
        self._prompt_templates = self._load_prompt_templates()

        debug_log(
            f"[ChunkExtractor] Initialized with {len(self._prompt_templates)} prompts, "
            f"max_tokens={max_tokens}"
        )

    def _load_prompt_templates(self) -> dict[str, str]:
        """
        Load all focused prompt templates from external files.

        Returns:
            Dict mapping prompt_type to prompt template string
        """
        templates = {}

        for prompt_type, filename in PROMPT_TYPES.items():
            filepath = EXTRACTION_PROMPTS_DIR / filename
            try:
                if filepath.exists():
                    templates[prompt_type] = filepath.read_text(encoding="utf-8")
                    debug_log(f"[ChunkExtractor] Loaded prompt: {filename}")
                else:
                    debug_log(f"[ChunkExtractor] WARNING: Prompt file not found: {filepath}")
                    templates[prompt_type] = self._fallback_prompt(prompt_type)
            except Exception as e:
                debug_log(f"[ChunkExtractor] Error loading {filename}: {e}")
                templates[prompt_type] = self._fallback_prompt(prompt_type)

        return templates

    def _fallback_prompt(self, prompt_type: str) -> str:
        """Return minimal fallback prompt if file not found."""
        fallbacks = {
            "parties": 'Extract plaintiffs and defendants. Return JSON: {"plaintiffs": [], "defendants": []}\n\n{chunk_text}',
            "names": 'Extract names mentioned. Return JSON: {"names_mentioned": []}\n\n{chunk_text}',
            "allegations": 'Extract allegations and defenses. Return JSON: {"allegations": [], "defenses": []}\n\n{chunk_text}',
            "facts": 'Extract facts and dates. Return JSON: {"key_facts": [], "dates_mentioned": [], "case_type_hints": [], "vocabulary": []}\n\n{chunk_text}',
        }
        return fallbacks.get(prompt_type, "{chunk_text}")

    def extract(self, chunk: BriefingChunk) -> ChunkExtraction:
        """
        Extract structured data from a single chunk using all 4 prompts sequentially.

        This method is primarily for testing. Use extract_batch() for production.

        Args:
            chunk: BriefingChunk to process

        Returns:
            ChunkExtraction with extracted data
        """
        debug_log(f"[ChunkExtractor] Processing chunk {chunk.chunk_id} (sequential)")

        partials = []
        for prompt_type in PROMPT_TYPES:
            partial = self._extract_single(chunk, prompt_type)
            partials.append(partial)

        return self._merge_partials(chunk, partials)

    def extract_batch(
        self,
        chunks: list[BriefingChunk],
        progress_callback: Callable[[int, int], None] | None = None,
        parallel: bool = True,
        max_workers: int | None = None,
    ) -> list[ChunkExtraction]:
        """
        Extract structured data from multiple chunks using parallel prompt chaining.

        Creates N×4 tasks (N chunks × 4 prompts) and processes them all in parallel.
        Results are merged by chunk_id after completion.

        Args:
            chunks: List of BriefingChunk objects
            progress_callback: Optional callback(current, total) for progress
            parallel: Whether to use parallel processing (default True)
            max_workers: Max concurrent extractions (None = auto-detect)

        Returns:
            List of ChunkExtraction objects, ordered by chunk_id
        """
        if not chunks:
            return []

        num_chunks = len(chunks)
        num_prompts = len(PROMPT_TYPES)
        total_tasks = num_chunks * num_prompts

        # Auto-calculate workers if not specified
        if max_workers is None:
            from src.system_resources import get_optimal_workers
            # Each Ollama extraction uses ~2GB RAM, but prompts are lighter now
            max_workers = get_optimal_workers(task_ram_gb=1.5, max_workers=12)

        debug_log(
            f"[ChunkExtractor] Parallel prompt chaining: {num_chunks} chunks × "
            f"{num_prompts} prompts = {total_tasks} tasks, {max_workers} workers"
        )

        # Use sequential processing if parallel is disabled or trivial workload
        if not parallel or total_tasks <= 4 or max_workers <= 1:
            return self._extract_sequential(chunks, progress_callback)

        return self._extract_parallel(chunks, progress_callback, max_workers)

    def _extract_sequential(
        self,
        chunks: list[BriefingChunk],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[ChunkExtraction]:
        """
        Extract chunks sequentially (fallback mode).

        Args:
            chunks: Chunks to process
            progress_callback: Progress callback

        Returns:
            List of extractions in order
        """
        extractions = []
        total = len(chunks) * len(PROMPT_TYPES)
        current = 0

        for chunk in chunks:
            partials = []
            for prompt_type in PROMPT_TYPES:
                partial = self._extract_single(chunk, prompt_type)
                partials.append(partial)
                current += 1
                if progress_callback:
                    progress_callback(current, total)

            extraction = self._merge_partials(chunk, partials)
            extractions.append(extraction)

        debug_log(f"[ChunkExtractor] Sequential extraction complete: {len(extractions)} chunks")
        return extractions

    def _extract_parallel(
        self,
        chunks: list[BriefingChunk],
        progress_callback: Callable[[int, int], None] | None = None,
        max_workers: int = 6,
    ) -> list[ChunkExtraction]:
        """
        Extract using parallel prompt chaining.

        Flattens all (chunk, prompt_type) pairs into a single task queue
        for maximum parallelism.

        Args:
            chunks: Chunks to process
            progress_callback: Progress callback
            max_workers: Maximum concurrent workers

        Returns:
            List of extractions ordered by chunk_id
        """
        total_tasks = len(chunks) * len(PROMPT_TYPES)
        completed_count = 0
        count_lock = threading.Lock()

        # Store partial results by (chunk_id, prompt_type)
        partial_results: dict[tuple[int, str], PartialExtraction] = {}

        debug_log(
            f"[ChunkExtractor] Starting parallel extraction: {total_tasks} tasks, "
            f"{max_workers} workers"
        )

        def extract_task(chunk: BriefingChunk, prompt_type: str) -> PartialExtraction:
            """Extract a single (chunk, prompt_type) pair."""
            nonlocal completed_count
            result = self._extract_single(chunk, prompt_type)

            with count_lock:
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, total_tasks)

            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all (chunk, prompt_type) pairs
            future_to_task = {}
            for chunk in chunks:
                for prompt_type in PROMPT_TYPES:
                    future = executor.submit(extract_task, chunk, prompt_type)
                    future_to_task[future] = (chunk.chunk_id, prompt_type)

            # Collect results as they complete
            for future in as_completed(future_to_task):
                chunk_id, prompt_type = future_to_task[future]
                try:
                    partial = future.result()
                    partial_results[(chunk_id, prompt_type)] = partial
                except Exception as e:
                    debug_log(
                        f"[ChunkExtractor] Error for chunk {chunk_id}, {prompt_type}: {e}"
                    )
                    partial_results[(chunk_id, prompt_type)] = PartialExtraction(
                        chunk_id=chunk_id,
                        prompt_type=prompt_type,
                        data={},
                        success=False,
                    )

        # Merge partials by chunk_id
        extractions = []
        for chunk in chunks:
            partials = [
                partial_results.get(
                    (chunk.chunk_id, pt),
                    PartialExtraction(chunk_id=chunk.chunk_id, prompt_type=pt, success=False),
                )
                for pt in PROMPT_TYPES
            ]
            extraction = self._merge_partials(chunk, partials)
            extractions.append(extraction)

        debug_log(
            f"[ChunkExtractor] Parallel extraction complete: {len(extractions)} chunks "
            f"from {total_tasks} tasks"
        )
        return extractions

    def _extract_single(
        self,
        chunk: BriefingChunk,
        prompt_type: str,
    ) -> PartialExtraction:
        """
        Run a single focused extraction prompt on a chunk.

        Args:
            chunk: The chunk to process
            prompt_type: Which prompt to use ("parties", "names", etc.)

        Returns:
            PartialExtraction with results
        """
        template = self._prompt_templates.get(prompt_type, "")
        if not template:
            return PartialExtraction(
                chunk_id=chunk.chunk_id,
                prompt_type=prompt_type,
                success=False,
            )

        # Build prompt using replace() to avoid curly brace issues
        prompt = template.replace("{chunk_text}", chunk.text)

        try:
            response = self.ollama_manager.generate_structured(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=0.0,  # Deterministic for extraction
            )

            if response is None:
                debug_log(
                    f"[ChunkExtractor] No response for chunk {chunk.chunk_id}, {prompt_type}"
                )
                return PartialExtraction(
                    chunk_id=chunk.chunk_id,
                    prompt_type=prompt_type,
                    success=False,
                )

            return PartialExtraction(
                chunk_id=chunk.chunk_id,
                prompt_type=prompt_type,
                data=response,
                success=True,
            )

        except Exception as e:
            debug_log(
                f"[ChunkExtractor] Error for chunk {chunk.chunk_id}, {prompt_type}: {e}"
            )
            return PartialExtraction(
                chunk_id=chunk.chunk_id,
                prompt_type=prompt_type,
                success=False,
            )

    def _merge_partials(
        self,
        chunk: BriefingChunk,
        partials: list[PartialExtraction],
    ) -> ChunkExtraction:
        """
        Merge partial extractions from all 4 prompts into one ChunkExtraction.

        Args:
            chunk: The source chunk
            partials: List of PartialExtraction objects (one per prompt type)

        Returns:
            Complete ChunkExtraction with merged data
        """
        import json

        # Combine all partial data
        merged_data = {}
        any_success = False

        for partial in partials:
            if partial.success and partial.data:
                data = partial.data

                # Handle case where data is a string (raw JSON) instead of dict
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        debug_log(
                            f"[ChunkExtractor] Failed to parse JSON for chunk "
                            f"{chunk.chunk_id}, {partial.prompt_type}: {data[:100]}"
                        )
                        continue

                # Only update if data is a dict
                if isinstance(data, dict):
                    merged_data.update(data)
                    any_success = True
                else:
                    debug_log(
                        f"[ChunkExtractor] Unexpected data type for chunk "
                        f"{chunk.chunk_id}, {partial.prompt_type}: {type(data)}"
                    )

        if not any_success:
            return self._empty_extraction(chunk, success=False)

        return self._parse_response(chunk, merged_data)

    def _parse_response(
        self,
        chunk: BriefingChunk,
        response: dict[str, Any],
    ) -> ChunkExtraction:
        """
        Parse merged JSON response into ChunkExtraction dataclass.

        Handles missing fields gracefully by using defaults.

        Args:
            chunk: Source chunk for metadata
            response: Merged JSON response from all prompts

        Returns:
            ChunkExtraction with parsed data
        """
        # Extract parties
        parties_data = response.get("parties", {})
        # Handle both nested and flat party formats
        if isinstance(parties_data, dict):
            parties = {
                "plaintiffs": self._ensure_list(parties_data.get("plaintiffs", [])),
                "defendants": self._ensure_list(parties_data.get("defendants", [])),
            }
        else:
            # Flat format from parties prompt
            parties = {
                "plaintiffs": self._ensure_list(response.get("plaintiffs", [])),
                "defendants": self._ensure_list(response.get("defendants", [])),
            }

        # Extract names with normalization
        names_raw = response.get("names_mentioned", [])
        names_mentioned = []
        for name_entry in self._ensure_list(names_raw):
            if isinstance(name_entry, dict):
                names_mentioned.append({
                    "name": str(name_entry.get("name", "")),
                    "role": str(name_entry.get("role", "")),
                    "category": str(name_entry.get("category", "OTHER")).upper(),
                })
            elif isinstance(name_entry, str):
                # Handle simple string names
                names_mentioned.append({
                    "name": name_entry,
                    "role": "",
                    "category": "OTHER",
                })

        return ChunkExtraction(
            chunk_id=chunk.chunk_id,
            source_document=chunk.source_document,
            document_type=chunk.document_type,
            parties=parties,
            allegations=self._ensure_string_list(response.get("allegations", [])),
            defenses=self._ensure_string_list(response.get("defenses", [])),
            names_mentioned=names_mentioned,
            key_facts=self._ensure_string_list(response.get("key_facts", [])),
            dates_mentioned=self._ensure_string_list(response.get("dates_mentioned", [])),
            case_type_hints=self._ensure_string_list(response.get("case_type_hints", [])),
            vocabulary=self._ensure_string_list(response.get("vocabulary", [])),
            extraction_success=True,
            raw_response=response,
        )

    def _empty_extraction(
        self,
        chunk: BriefingChunk,
        success: bool = False,
    ) -> ChunkExtraction:
        """
        Create an empty extraction for failed/skipped chunks.

        Args:
            chunk: Source chunk for metadata
            success: Whether this should be marked as successful

        Returns:
            ChunkExtraction with empty data
        """
        return ChunkExtraction(
            chunk_id=chunk.chunk_id,
            source_document=chunk.source_document,
            document_type=chunk.document_type,
            extraction_success=success,
        )

    def _ensure_list(self, value: Any) -> list:
        """Ensure value is a list."""
        if isinstance(value, list):
            return value
        if value is None:
            return []
        return [value]

    def _ensure_string_list(self, value: Any) -> list[str]:
        """Ensure value is a list of strings."""
        items = self._ensure_list(value)
        return [str(item) for item in items if item]

    # LOG-025: Removed dead code _count_items() method (never called)
