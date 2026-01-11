"""
Configuration Management Package

Provides unified configuration loading and management for CasePrepd.
"""

from .loader import load_yaml, load_yaml_with_fallback, save_yaml

__all__ = ["load_yaml", "load_yaml_with_fallback", "save_yaml"]
