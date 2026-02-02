"""Extractors for noun phrases and dependencies."""

from .noun_phrase_extractor import extract_noun_phrases
from .dependency_analyzer import get_introductions_from_text, collect_inherited_terms

__all__ = [
    "extract_noun_phrases",
    "get_introductions_from_text",
    "collect_inherited_terms",
]
