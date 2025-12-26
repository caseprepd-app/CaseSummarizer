"""
NER-based Vocabulary Extraction Algorithm

Uses spaCy's Named Entity Recognition (NER) to extract vocabulary from legal documents.
This is the primary extraction algorithm, identifying:
- Named entities (PERSON, ORG, GPE, LOC)
- Medical terms (from curated list)
- Acronyms (all-caps words)
- Rare/unusual words (not in common vocabulary)

FILTERING SCOPE:
This algorithm handles SINGLE-WORD filtering only:
- Stopwords (shared STOPWORDS from tokenizer.py)
- Rarity threshold (Google word frequency rank)
- Exclude lists (legal terms, user exclusions)
- Pattern matching (legal boilerplate, variations)

Multi-word PHRASE filtering is done CENTRALLY by rarity_filter.py after all
algorithms contribute their candidates. This ensures consistent filtering.
"""

import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import nltk
import spacy
from nltk.corpus import wordnet

from src.config import (
    GOOGLE_WORD_FREQUENCY_FILE,
    SPACY_DOWNLOAD_TIMEOUT_SEC,
    SPACY_SOCKET_TIMEOUT_SEC,
    SPACY_THREAD_TIMEOUT_SEC,
    VOCAB_ALGORITHM_WEIGHTS,
    VOCABULARY_BATCH_SIZE,
    VOCABULARY_RARITY_THRESHOLD,
)
from src.categories import get_ner_mapping
from src.logging_config import debug_log
from src.utils.tokenizer import STOPWORDS
from src.utils.pattern_filter import (
    VARIATION_FILTER,
    TITLE_ABBREVIATIONS,
    MIN_ENTITY_LENGTH,
    MAX_ENTITY_LENGTH,
    matches_entity_filter,
    matches_token_filter,
    is_valid_acronym,
)
from src.vocabulary.algorithms import register_algorithm
from src.vocabulary.algorithms.base import (
    AlgorithmResult,
    BaseExtractionAlgorithm,
    CandidateTerm,
)

# Constants for spaCy model
SPACY_MODEL_NAME = "en_core_web_lg"
SPACY_MODEL_VERSION = "3.8.0"


@register_algorithm("NER")
class NERAlgorithm(BaseExtractionAlgorithm):
    """
    Named Entity Recognition algorithm using spaCy.

    Extracts:
    - PERSON entities (names)
    - ORG entities (organizations)
    - GPE/LOC entities (places/locations)
    - Medical terms from curated list
    - Acronyms (all-caps words, 2+ chars)
    - Rare words not in common vocabulary

    This is the primary extraction algorithm with highest confidence
    for named entities.
    """

    name = "NER"
    weight = VOCAB_ALGORITHM_WEIGHTS.get("NER", 1.0)  # Primary algorithm

    def __init__(
        self,
        nlp=None,
        exclude_list: set[str] | None = None,
        user_exclude_list: set[str] | None = None,
        medical_terms: set[str] | None = None,
        common_words_blacklist: set[str] | None = None,
        frequency_dataset: dict[str, int] | None = None,
        frequency_rank_map: dict[str, int] | None = None,
        rarity_threshold: int = VOCABULARY_RARITY_THRESHOLD,
    ):
        """
        Initialize NER algorithm.

        Args:
            nlp: Pre-loaded spaCy model. If None, will be loaded on first use.
            exclude_list: Common legal terms to exclude.
            user_exclude_list: User-specified terms to exclude.
            medical_terms: Known medical terms (guaranteed inclusion).
            common_words_blacklist: Common medical/legal words to exclude.
            frequency_dataset: Word -> frequency count mapping.
            frequency_rank_map: Word -> rank mapping (cached for O(1) lookup).
            rarity_threshold: Minimum rank to consider a word rare.
        """
        self._nlp = nlp
        self.exclude_list = exclude_list or set()
        self.user_exclude_list = user_exclude_list or set()
        self.medical_terms = medical_terms or set()
        self.common_words_blacklist = common_words_blacklist or set()
        self.frequency_dataset = frequency_dataset or {}
        self.frequency_rank_map = frequency_rank_map or {}
        self.rarity_threshold = rarity_threshold

    @property
    def nlp(self):
        """Lazy-load spaCy model on first access."""
        if self._nlp is None:
            self._nlp = self._load_spacy_model()
        return self._nlp

    @nlp.setter
    def nlp(self, value):
        self._nlp = value

    def extract(self, text: str, **kwargs) -> AlgorithmResult:
        """
        Extract named entities and unusual words from text.

        Args:
            text: Document text to analyze
            **kwargs:
                - doc: Pre-processed spaCy Doc object (optional)
                - chunks: Pre-chunked text list (optional)

        Returns:
            AlgorithmResult with candidate terms
        """
        start_time = time.time()

        # Use provided chunks or chunk the text
        chunks = kwargs.get('chunks')
        if chunks is None:
            chunks = self._chunk_text(text, chunk_size_kb=50)

        candidates = []
        term_frequencies: dict[str, int] = defaultdict(int)
        total_tokens = 0
        total_entities = 0

        # Process chunks using nlp.pipe() for efficiency
        for doc in self.nlp.pipe(chunks, batch_size=VOCABULARY_BATCH_SIZE):
            total_tokens += len(doc)
            total_entities += len(doc.ents)

            # Extract from this chunk
            chunk_candidates = self._extract_from_doc(doc, term_frequencies)
            candidates.extend(chunk_candidates)

        processing_time_ms = (time.time() - start_time) * 1000

        return AlgorithmResult(
            candidates=candidates,
            processing_time_ms=processing_time_ms,
            metadata={
                "total_tokens": total_tokens,
                "total_entities": total_entities,
                "chunks_processed": len(chunks),
                "unique_terms": len(set(c.term.lower() for c in candidates)),
            }
        )

    def _extract_from_doc(
        self, doc, term_frequencies: dict[str, int]
    ) -> list[CandidateTerm]:
        """
        Extract candidates from a single spaCy Doc.

        Args:
            doc: spaCy Doc object
            term_frequencies: Shared frequency counter (modified in place)

        Returns:
            List of CandidateTerm objects
        """
        candidates = []

        # Extract named entities first (prioritize multi-word entities)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
                term_text = self._clean_entity_text(ent.text)

                if not term_text:
                    continue

                if self._matches_entity_filter(term_text):
                    continue

                # Single-word entities need additional filtering
                words = term_text.split()
                if len(words) == 1:
                    # Session 53: Filter stopwords for ALL single-word entities
                    # (spaCy sometimes tags common words like "bill" as PERSON)
                    if term_text.lower() in STOPWORDS:
                        continue
                    # Non-PERSON entities also need rarity check
                    if ent.label_ != "PERSON" and not self._is_word_rare_enough(term_text):
                        continue

                lower_term = term_text.lower()
                if lower_term in self.exclude_list or lower_term in self.user_exclude_list:
                    continue

                term_frequencies[lower_term] += 1

                candidates.append(CandidateTerm(
                    term=term_text,
                    source_algorithm=self.name,
                    confidence=0.85,  # High confidence for NER entities
                    suggested_type=self._map_entity_type(ent.label_),
                    frequency=1,  # Will be aggregated later
                    metadata={
                        "ent_label": ent.label_,
                        "ent_start_char": ent.start_char,
                        "ent_end_char": ent.end_char,
                    }
                ))

        # Extract unusual single tokens not part of entities
        for token in doc:
            # Skip tokens that are part of ANY entity span (even if entity was filtered)
            # This prevents common words like "leg" from bypassing rarity check
            # when spaCy tagged them as entities but they were filtered for being too common
            is_part_of_entity = any(
                ent.start <= token.i < ent.end
                for ent in doc.ents
            )
            if is_part_of_entity:
                continue

            if self._is_unusual(token, ent_type=token.ent_type_):
                term_text = token.text

                if term_text.lower() in self.exclude_list:
                    continue

                term_frequencies[term_text.lower()] += 1

                # Determine suggested type
                suggested_type = self._get_suggested_type(token)

                candidates.append(CandidateTerm(
                    term=term_text,
                    source_algorithm=self.name,
                    confidence=0.6,  # Lower confidence for single tokens
                    suggested_type=suggested_type,
                    frequency=1,
                    metadata={
                        "token_pos": token.pos_,
                        "token_ent_type": token.ent_type_,
                    }
                ))

        return candidates

    def _map_entity_type(self, ent_label: str) -> str:
        """Map spaCy entity label to our simplified types.

        Uses shared categories config from config/categories.json to ensure
        consistent categorization across NER and LLM extraction methods.
        """
        mapping = get_ner_mapping()
        return mapping.get(ent_label, "Unknown")

    def _get_suggested_type(self, token) -> str:
        """Get suggested type for a single token.

        Uses shared categories from config/categories.json.
        """
        lower_text = token.text.lower()

        if lower_text in self.medical_terms:
            return "Medical"

        if is_valid_acronym(token.text):
            return "Technical"  # Acronyms are typically technical/legal terms

        # Default to Unknown for unclassified terms
        return "Unknown"

    # ========================================================================
    # FILTERING METHODS - Moved from VocabularyExtractor
    # ========================================================================

    def _clean_entity_text(self, entity_text: str) -> str:
        """Clean spaCy entity text to remove leading/trailing junk."""
        cleaned = entity_text.strip()
        cleaned = ' '.join(cleaned.split())
        cleaned = re.sub(r'^(and/or|and|or)\s+', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+(and/or|and|or)$', '', cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip('.,;:!?()[]{}"\'/\\')
        return cleaned.strip()

    def _matches_entity_filter(self, entity_text: str) -> bool:
        """Check if an entity should be filtered out based on pattern matching."""
        return matches_entity_filter(entity_text)

    def _matches_variation_filter(self, word: str) -> bool:
        """Check if a word matches common variation patterns."""
        return VARIATION_FILTER.matches(word.lower())

    def _is_word_rare_enough(self, word: str) -> bool:
        """Check if word is rare enough based on frequency rank."""
        if not self.frequency_rank_map:
            return True  # No frequency data, assume rare

        lower_word = word.lower()
        rank = self.frequency_rank_map.get(lower_word)

        if rank is None:
            return True  # Not in dataset = very rare

        return rank >= self.rarity_threshold

    def _is_unusual(self, token, ent_type: str | None = None) -> bool:
        """Determine if a token represents an unusual/noteworthy term."""
        if not token.is_alpha or token.is_space or token.is_punct or token.is_digit:
            return False

        lower_text = token.text.lower()

        # Session 53: Filter common stopwords early
        if lower_text in STOPWORDS:
            return False

        if lower_text in self.exclude_list:
            return False

        if lower_text in self.user_exclude_list:
            return False

        if lower_text in self.common_words_blacklist:
            return False

        if self._matches_variation_filter(token.text):
            return False

        # Use centralized token filter for pattern matching
        if matches_token_filter(token.text):
            return False

        # Named entities are always accepted
        if ent_type in ["PERSON", "ORG", "GPE", "LOC"]:
            return True

        # Medical terms always accepted
        if lower_text in self.medical_terms:
            return True

        # Acronyms (except title abbreviations)
        if is_valid_acronym(token.text):
            return True

        # Frequency-based rarity check
        if self.frequency_dataset and not self._is_word_rare_enough(token.text):
            return False

        # WordNet fallback
        if wordnet.synsets(lower_text):
            return False

        return True

    # ========================================================================
    # SPACY MODEL LOADING - Moved from VocabularyExtractor
    # ========================================================================

    def _load_spacy_model(self):
        """Load or download the spaCy model."""
        try:
            nlp = spacy.load(SPACY_MODEL_NAME)
            debug_log(f"[NER] Loaded spaCy model: {SPACY_MODEL_NAME}")
            return nlp
        except OSError:
            debug_log(f"[NER] Model {SPACY_MODEL_NAME} not found, downloading...")
            return self._download_and_load_model()

    def _download_and_load_model(self):
        """Download spaCy model using subprocess."""
        python_executable = sys.executable

        original_socket_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(SPACY_SOCKET_TIMEOUT_SEC)

        download_error = [None]
        downloaded_model = [None]

        def download_thread():
            try:
                result = subprocess.run(
                    [python_executable, "-m", "spacy", "download", SPACY_MODEL_NAME],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=SPACY_DOWNLOAD_TIMEOUT_SEC
                )
                debug_log(f"[NER] Download output: {result.stdout[:500]}")
                downloaded_model[0] = spacy.load(SPACY_MODEL_NAME)
            except Exception as e:
                download_error[0] = str(e)

        thread = threading.Thread(target=download_thread)
        thread.start()
        thread.join(timeout=SPACY_THREAD_TIMEOUT_SEC)

        socket.setdefaulttimeout(original_socket_timeout)

        if download_error[0]:
            raise RuntimeError(f"Failed to download spaCy model: {download_error[0]}")

        if downloaded_model[0] is None:
            raise RuntimeError("spaCy model download timed out")

        return downloaded_model[0]

    def _chunk_text(self, text: str, chunk_size_kb: int = 50) -> list[str]:
        """Split text into chunks for efficient processing."""
        chunk_size_chars = chunk_size_kb * 1024
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para) + 2

            if current_size + para_size > chunk_size_chars and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def get_config(self) -> dict[str, Any]:
        """Return algorithm configuration."""
        return {
            **super().get_config(),
            "rarity_threshold": self.rarity_threshold,
            "exclude_list_size": len(self.exclude_list),
            "medical_terms_size": len(self.medical_terms),
            "has_frequency_data": bool(self.frequency_dataset),
        }
