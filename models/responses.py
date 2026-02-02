"""Response models for the API."""

from pydantic import BaseModel
from .validation import NounPhrase, AntecedentError


class ClaimAnalysis(BaseModel):
    """Analysis results for a single claim."""
    claim_number: int
    claim_text: str
    introductions: list[NounPhrase]
    references: list[NounPhrase]
    inherited_terms: list[str]
    antecedent_errors: list[AntecedentError]


class AnalyzeClaimsResponse(BaseModel):
    """Response containing analysis for all claims."""
    analyses: list[ClaimAnalysis]
    total_errors: int
