"""
Query Transformer for LocalScribe Q&A.

Uses LlamaIndex + Ollama to expand vague user queries into multiple specific
search terms. This improves retrieval for imprecise questions like "what
happened to the person?" which could mean plaintiff, defendant, witness, etc.

Architecture:
- Takes a single user query
- Uses LLM to generate query variants (legal-domain aware)
- Returns original query + expanded variants for hybrid retrieval

Example:
    transformer = QueryTransformer()
    queries = transformer.transform("What happened to the person?")
    # Returns: [
    #   "What happened to the person?",  # Original
    #   "plaintiff injuries damages",     # Expanded
    #   "defendant actions conduct",      # Expanded
    #   "accident victim harm"            # Expanded
    # ]
"""

import re  # PERF-005: Move to module level
import time
from dataclasses import dataclass, field
from typing import Optional

from src.config import DEBUG_MODE
from src.logging_config import debug_log


# Default config values
DEFAULT_VARIANT_COUNT = 3
DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass
class QueryTransformResult:
    """Result of query transformation."""

    original_query: str
    expanded_queries: list[str] = field(default_factory=list)
    success: bool = False
    processing_time_ms: float = 0.0
    error_message: str = ""

    @property
    def all_queries(self) -> list[str]:
        """Get original query plus all expanded queries."""
        return [self.original_query] + self.expanded_queries


class QueryTransformer:
    """
    Transforms vague queries into specific search terms using LlamaIndex + Ollama.

    This addresses the common problem of reporters asking imprecise questions like:
    - "What happened?" → needs context about what aspect
    - "Who is responsible?" → could be plaintiff, defendant, third party
    - "What were the injuries?" → needs to search medical terms

    The transformer uses a legal-domain aware prompt to generate query variants
    that capture different interpretations of the user's question.

    Attributes:
        variant_count: Number of query variants to generate
        timeout_seconds: Maximum time to wait for LLM response
        enabled: Whether transformation is enabled (falls back to original if False)

    Example:
        transformer = QueryTransformer()

        # Check if Ollama is available
        if transformer.is_available():
            result = transformer.transform("What happened?")
            for query in result.all_queries:
                # Retrieve using each query variant
                pass
    """

    # Legal-domain prompt for query expansion
    PROMPT_TEMPLATE = """You are a legal document search assistant. A court reporter needs to find information in legal documents.

Given this question: "{query}"

Generate {variant_count} alternative search queries that would help find relevant information. Consider:
- Different parties (plaintiff, defendant, witnesses, experts)
- Legal terminology variations
- Common document sections (allegations, damages, defenses)
- Medical or technical terms if relevant

Return ONLY the search queries, one per line, no numbering or explanations.

Alternative queries:"""

    def __init__(
        self,
        variant_count: int = DEFAULT_VARIANT_COUNT,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        enabled: bool = True,
    ):
        """
        Initialize the query transformer.

        Args:
            variant_count: Number of query variants to generate (1-5)
            timeout_seconds: Maximum time for LLM response
            enabled: Whether to enable query transformation
        """
        self.variant_count = max(1, min(5, variant_count))
        self.timeout_seconds = timeout_seconds
        self.enabled = enabled

        # Lazy-loaded LlamaIndex components
        self._llm: Optional[object] = None
        self._available: Optional[bool] = None

        if DEBUG_MODE:
            debug_log(f"[QueryTransformer] Initialized (enabled={enabled}, variants={variant_count})")

    def _init_llm(self) -> bool:
        """
        Initialize LlamaIndex Ollama LLM.

        Returns:
            True if initialization succeeded, False otherwise
        """
        if self._llm is not None:
            return True

        try:
            from llama_index.llms.ollama import Ollama

            # Get model name from config
            from src.config import OLLAMA_MODEL_NAME

            self._llm = Ollama(
                model=OLLAMA_MODEL_NAME,
                request_timeout=self.timeout_seconds,
                temperature=0.3,  # Some creativity for query variants
            )

            if DEBUG_MODE:
                debug_log(f"[QueryTransformer] LlamaIndex Ollama initialized with model: {OLLAMA_MODEL_NAME}")

            return True

        except ImportError as e:
            debug_log(f"[QueryTransformer] LlamaIndex not available: {e}")
            return False
        except Exception as e:
            debug_log(f"[QueryTransformer] Failed to initialize LLM: {e}")
            return False

    def is_available(self) -> bool:
        """
        Check if query transformation is available.

        Returns:
            True if LlamaIndex + Ollama is available and working
        """
        if self._available is not None:
            return self._available

        if not self.enabled:
            self._available = False
            return False

        self._available = self._init_llm()
        return self._available

    def transform(self, query: str) -> QueryTransformResult:
        """
        Transform a query into multiple search variants.

        If transformation fails or is disabled, returns just the original query.

        Args:
            query: The user's original question

        Returns:
            QueryTransformResult with original and expanded queries
        """
        start_time = time.perf_counter()

        result = QueryTransformResult(original_query=query)

        # Check if enabled and available
        if not self.enabled:
            result.success = True  # Not an error, just disabled
            result.processing_time_ms = (time.perf_counter() - start_time) * 1000
            return result

        if not self.is_available():
            result.error_message = "Query transformation not available"
            result.processing_time_ms = (time.perf_counter() - start_time) * 1000
            return result

        try:
            # Build prompt
            prompt = self.PROMPT_TEMPLATE.format(
                query=query,
                variant_count=self.variant_count
            )

            if DEBUG_MODE:
                debug_log(f"[QueryTransformer] Transforming: '{query[:50]}...'")

            # Call LLM
            response = self._llm.complete(prompt)
            response_text = response.text.strip()

            # Parse response into query variants
            variants = self._parse_variants(response_text)

            result.expanded_queries = variants[:self.variant_count]
            result.success = True

            if DEBUG_MODE:
                debug_log(f"[QueryTransformer] Generated {len(result.expanded_queries)} variants")
                for i, v in enumerate(result.expanded_queries):
                    debug_log(f"  [{i + 1}] {v}")

        except Exception as e:
            result.error_message = str(e)
            debug_log(f"[QueryTransformer] Transform failed: {e}")

        result.processing_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    def _parse_variants(self, response: str) -> list[str]:
        """
        Parse LLM response into query variants.

        Handles various response formats:
        - One query per line
        - Numbered lists (1. query, 2. query)
        - Bullet points (- query, * query)

        Args:
            response: Raw LLM response text

        Returns:
            List of cleaned query strings
        """
        variants = []

        for line in response.split('\n'):
            # Clean the line
            line = line.strip()

            if not line:
                continue

            # Remove common prefixes
            # Numbered: "1.", "1)", "1:"
            # Bullets: "-", "*", "•"
            # PERF-005: Use module-level re import
            line = re.sub(r'^[\d]+[.)\:]?\s*', '', line)
            line = re.sub(r'^[-*•]\s*', '', line)
            line = line.strip()

            # Skip if too short or looks like a header
            if len(line) < 5:
                continue
            if line.endswith(':'):
                continue

            # Skip if it's a meta-comment
            lower = line.lower()
            if any(skip in lower for skip in ['here are', 'alternative', 'query:', 'search:']):
                continue

            variants.append(line)

        return variants

    def get_status(self) -> dict:
        """
        Get transformer status for diagnostics.

        Returns:
            Dictionary with configuration and availability info
        """
        return {
            "enabled": self.enabled,
            "available": self.is_available() if self.enabled else False,
            "variant_count": self.variant_count,
            "timeout_seconds": self.timeout_seconds,
        }
