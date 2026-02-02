"""Analyze claim dependencies and collect inherited terms."""

from core.nlp_models import nlp
from models.requests import ParsedClaim
from extractors.noun_phrase_extractor import extract_noun_phrases


def get_introductions_from_text(text: str) -> set[str]:
    """Extract introduced terms from claim text.

    Args:
        text: Claim text to analyze

    Returns:
        Set of introduced term strings (lowercase)
    """
    doc = nlp(text)
    nps = extract_noun_phrases(doc)

    terms = set()
    for np in nps:
        if np.type in ("introduction", "bare"):
            terms.add(np.np)
    return terms


def collect_inherited_terms(
    claim_number: int,
    claims_map: dict[int, ParsedClaim],
    intro_cache: dict[int, set[str]],
    visited: set[int] | None = None
) -> set[str]:
    """Recursively collect introduced terms from ancestor claims.

    Args:
        claim_number: The claim number to analyze
        claims_map: Map of claim numbers to ParsedClaim objects
        intro_cache: Cache of already-computed introductions
        visited: Set of already-visited claim numbers (prevents cycles)

    Returns:
        Set of all terms introduced in this claim and its ancestors
    """
    if visited is None:
        visited = set()

    if claim_number in visited:
        return set()
    visited.add(claim_number)

    claim = claims_map.get(claim_number)
    if not claim:
        return set()

    # Get this claim's introductions
    if claim_number not in intro_cache:
        intro_cache[claim_number] = get_introductions_from_text(claim.text)

    terms = set(intro_cache[claim_number])

    # Add parent terms
    for parent_num in claim.depends_on:
        parent_terms = collect_inherited_terms(parent_num, claims_map, intro_cache, visited)
        terms.update(parent_terms)

    return terms
