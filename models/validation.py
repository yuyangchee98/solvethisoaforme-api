"""Validation models for noun phrases and antecedent errors."""

from pydantic import BaseModel


class NounPhrase(BaseModel):
    """A noun phrase extracted from claim text."""
    text: str           # Full text including determiner
    np: str             # Just the noun phrase (without determiner)
    determiner: str | None
    start: int
    end: int
    type: str           # "introduction", "reference", "bare", "error"


class AntecedentError(BaseModel):
    """An antecedent basis error detected in a claim."""
    text: str
    np: str
    start: int
    end: int
    reason: str
    suggestion: str | None = None  # Closest matching term if available
    suggestion_score: float | None = None  # Similarity score
