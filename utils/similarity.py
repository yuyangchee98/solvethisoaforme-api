"""Similarity utilities for finding closest matches."""

from core.nlp_models import nlp_vectors


def find_closest_match(term: str, candidates: set[str], threshold: float = 0.5) -> tuple[str | None, float | None]:
    """Find the most similar term from candidates using word vectors.

    Args:
        term: The term to find a match for
        candidates: Set of candidate terms to compare against
        threshold: Minimum similarity score (0-1) to consider a match

    Returns:
        Tuple of (best_match, score) or (None, None) if no match above threshold
    """
    if not candidates:
        return None, None

    term_doc = nlp_vectors(term)
    best_match = None
    best_score = 0.0

    for candidate in candidates:
        candidate_doc = nlp_vectors(candidate)
        score = term_doc.similarity(candidate_doc)
        if score > best_score:
            best_score = score
            best_match = candidate

    if best_score >= threshold:
        return best_match, best_score
    return None, None
