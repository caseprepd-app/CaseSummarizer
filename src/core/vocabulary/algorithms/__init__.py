"""
Vocabulary Extraction Algorithms Package

This package provides a pluggable framework for vocabulary extraction algorithms.
Each algorithm is registered via decorator and can be instantiated by name.

Usage:
    from src.core.vocabulary.algorithms import (
        create_default_algorithms,
        get_algorithm,
        get_available_algorithms,
    )

    # Get default algorithm set
    algorithms = create_default_algorithms()

    # Or instantiate specific algorithm
    ner = get_algorithm("NER", exclude_list=my_excludes)

Registration:
    @register_algorithm("MyAlgorithm")
    class MyAlgorithm(BaseExtractionAlgorithm):
        name = "MyAlgorithm"
        ...
"""

from typing import Type

from src.core.vocabulary.algorithms.base import (
    AlgorithmResult,
    BaseExtractionAlgorithm,
    CandidateTerm,
)

# Registry of available algorithms (class references, not instances)
_ALGORITHM_REGISTRY: dict[str, Type[BaseExtractionAlgorithm]] = {}


def register_algorithm(name: str):
    """
    Decorator to register an algorithm class.

    The decorated class will be available via get_algorithm() using the
    registered name. This enables dynamic algorithm discovery and configuration.

    Args:
        name: Unique name for the algorithm (e.g., "NER", "RAKE", "TF-IDF")

    Returns:
        Decorator function that registers the class

    Example:
        @register_algorithm("RAKE")
        class RAKEAlgorithm(BaseExtractionAlgorithm):
            name = "RAKE"
            ...

    Raises:
        ValueError: If name is already registered (prevents accidental overwrites)
    """

    def decorator(cls: Type[BaseExtractionAlgorithm]) -> Type[BaseExtractionAlgorithm]:
        if name in _ALGORITHM_REGISTRY:
            raise ValueError(
                f"Algorithm '{name}' is already registered. "
                f"Existing: {_ALGORITHM_REGISTRY[name].__name__}, New: {cls.__name__}"
            )
        _ALGORITHM_REGISTRY[name] = cls
        return cls

    return decorator


def get_algorithm(name: str, **kwargs) -> BaseExtractionAlgorithm:
    """
    Instantiate an algorithm by its registered name.

    Args:
        name: Registered algorithm name (case-sensitive)
        **kwargs: Constructor arguments passed to the algorithm class

    Returns:
        Configured algorithm instance ready for extraction

    Raises:
        KeyError: If algorithm name is not registered

    Example:
        ner = get_algorithm("NER", rarity_threshold=180000)
        rake = get_algorithm("RAKE", max_phrase_length=3)
    """
    if name not in _ALGORITHM_REGISTRY:
        available = ", ".join(sorted(_ALGORITHM_REGISTRY.keys()))
        raise KeyError(
            f"Unknown algorithm '{name}'. Available algorithms: {available or '(none registered)'}"
        )
    return _ALGORITHM_REGISTRY[name](**kwargs)


def get_available_algorithms() -> list[str]:
    """
    Return list of all registered algorithm names.

    Returns:
        Sorted list of algorithm names that can be passed to get_algorithm()
    """
    return sorted(_ALGORITHM_REGISTRY.keys())


def create_default_algorithms(**config) -> list[BaseExtractionAlgorithm]:
    """
    Create the default set of extraction algorithms.

    This is the recommended way to get a working algorithm set. It creates
    instances of all standard algorithms with sensible defaults.

    Args:
        **config: Optional configuration overrides. Supported keys:
            - ner_enabled: Enable NER algorithm (default: True)
            - rake_enabled: Enable RAKE algorithm (default: True)
            - ner_weight: NER algorithm weight (default: 1.0)
            - rake_weight: RAKE algorithm weight (default: 0.7)
            - Additional algorithm-specific config passed through

    Returns:
        List of configured algorithm instances in recommended execution order

    Note:
        Import algorithms here to trigger their @register_algorithm decorators.
        This is intentionally lazy to avoid circular imports.
    """
    # Import algorithms to register them (triggers @register_algorithm decorators)
    # These imports are here to avoid circular dependencies
    from src.core.vocabulary.algorithms.ner_algorithm import NERAlgorithm
    from src.core.vocabulary.algorithms.rake_algorithm import RAKEAlgorithm

    algorithms = []

    # NER Algorithm (primary - highest weight)
    if config.get("ner_enabled", True):
        ner_kwargs = {
            k.replace("ner_", ""): v
            for k, v in config.items()
            if k.startswith("ner_") and k != "ner_enabled"
        }
        ner = NERAlgorithm(**ner_kwargs)
        ner.weight = config.get("ner_weight", 1.0)
        algorithms.append(ner)

    # RAKE Algorithm (secondary - lower weight by default)
    if config.get("rake_enabled", True):
        rake_kwargs = {
            k.replace("rake_", ""): v
            for k, v in config.items()
            if k.startswith("rake_") and k != "rake_enabled"
        }
        rake = RAKEAlgorithm(**rake_kwargs)
        rake.weight = config.get("rake_weight", 0.7)
        algorithms.append(rake)

    return algorithms


# Re-export key classes for convenient imports
__all__ = [
    "AlgorithmResult",
    "BaseExtractionAlgorithm",
    "CandidateTerm",
    "create_default_algorithms",
    "get_algorithm",
    "get_available_algorithms",
    "register_algorithm",
]
