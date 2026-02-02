"""Request models for the API."""

from pydantic import BaseModel


class ParsedClaim(BaseModel):
    """A parsed patent claim with dependencies."""
    number: int
    text: str
    depends_on: list[int]


class AnalyzeClaimsRequest(BaseModel):
    """Request to analyze claims for antecedent basis errors."""
    claims: list[ParsedClaim]
    enabled_validators: set[str] | None = None  # Optional: control which validators to run
