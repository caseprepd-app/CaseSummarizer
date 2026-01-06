"""
Configuration Management Package

Provides unified configuration loading and management for LocalScribe.
"""

from .loader import load_yaml, load_yaml_with_fallback, save_yaml

__all__ = ["load_yaml", "load_yaml_with_fallback", "save_yaml"]
