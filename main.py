"""
Patent Claim NLP API

Extracts noun phrases, identifies introductions/references,
and checks antecedent basis using spaCy.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy

# Load spaCy transformer model (most accurate for patent text)
nlp = spacy.load("en_core_web_trf")

# Load medium model for word vector similarity (suggestions)
nlp_vectors = spacy.load("en_core_web_md")

app = FastAPI(title="Patent Claim NLP API")

# Allow CORS for webapp
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class NounPhrase(BaseModel):
    text: str           # Full text including determiner
    np: str             # Just the noun phrase (without determiner)
    determiner: str | None
    start: int
    end: int
    type: str           # "introduction", "reference", "bare", "error"


class AntecedentError(BaseModel):
    text: str
    np: str
    start: int
    end: int
    reason: str
    suggestion: str | None = None  # Closest matching term if available
    suggestion_score: float | None = None  # Similarity score


class ClaimAnalysis(BaseModel):
    claim_number: int
    claim_text: str
    introductions: list[NounPhrase]
    references: list[NounPhrase]
    inherited_terms: list[str]
    antecedent_errors: list[AntecedentError]


class ParsedClaim(BaseModel):
    number: int
    text: str
    depends_on: list[int]


class AnalyzeClaimsRequest(BaseModel):
    claims: list[ParsedClaim]


class AnalyzeClaimsResponse(BaseModel):
    analyses: list[ClaimAnalysis]
    total_errors: int


def extract_noun_phrases(doc) -> list[NounPhrase]:
    """Extract all noun phrases with their determiners."""
    results = []

    for chunk in doc.noun_chunks:
        # Find the determiner
        determiner = None
        det_token = None
        for token in chunk:
            if token.dep_ == "det":
                determiner = token.text.lower()
                det_token = token
                break

        # Get NP without determiner
        if det_token:
            np_start = det_token.idx + len(det_token.text_with_ws) - chunk.start_char
            np_text = chunk.text[np_start:].strip()
        else:
            np_text = chunk.text.strip()

        # Determine type
        if determiner in ("a", "an"):
            np_type = "introduction"
        elif determiner in ("the", "said"):
            np_type = "reference"
        elif determiner in ("such",):
            np_type = "error"
        elif determiner is None:
            np_type = "bare"
        else:
            np_type = "other"

        results.append(NounPhrase(
            text=chunk.text,
            np=np_text.lower(),
            determiner=determiner,
            start=chunk.start_char,
            end=chunk.end_char,
            type=np_type,
        ))

    return results


def find_closest_match(term: str, candidates: set[str], threshold: float = 0.5) -> tuple[str | None, float | None]:
    """Find the most similar term from candidates using word vectors."""
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


def find_pronouns(doc) -> list[AntecedentError]:
    """Find pronouns (always errors in patent claims)."""
    errors = []
    error_pronouns = {"it", "its", "they", "their", "them"}

    for token in doc:
        if token.text.lower() in error_pronouns and token.pos_ == "PRON":
            errors.append(AntecedentError(
                text=token.text,
                np=token.text.lower(),
                start=token.idx,
                end=token.idx + len(token.text),
                reason=f"Pronoun '{token.text}' not allowed in patent claims",
            ))

    return errors


def get_introductions_from_text(text: str) -> set[str]:
    """Extract introduced terms from claim text."""
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
    """Recursively collect introduced terms from ancestor claims."""
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


@app.post("/analyze-claims", response_model=AnalyzeClaimsResponse)
def analyze_claims(request: AnalyzeClaimsRequest):
    """Analyze all claims for antecedent basis errors."""

    # Build claims map
    claims_map = {c.number: c for c in request.claims}
    intro_cache: dict[int, set[str]] = {}

    analyses = []
    total_errors = 0

    for claim in request.claims:
        doc = nlp(claim.text)
        nps = extract_noun_phrases(doc)

        # Get inherited terms from ancestors
        inherited = collect_inherited_terms(claim.number, claims_map, intro_cache)

        # Separate introductions and references
        introductions = [np for np in nps if np.type == "introduction"]
        references = [np for np in nps if np.type == "reference"]

        # Check for antecedent errors
        errors: list[AntecedentError] = []

        # Check pronouns
        errors.extend(find_pronouns(doc))

        # Check "such" phrases
        for np in nps:
            if np.type == "error":
                errors.append(AntecedentError(
                    text=np.text,
                    np=np.np,
                    start=np.start,
                    end=np.end,
                    reason=f"'such {np.np}' is not a valid reference",
                ))

        # Check references against inherited terms
        for ref in references:
            if ref.np not in inherited:
                # Find closest match for suggestion
                suggestion, score = find_closest_match(ref.np, inherited)
                errors.append(AntecedentError(
                    text=ref.text,
                    np=ref.np,
                    start=ref.start,
                    end=ref.end,
                    reason=f"No antecedent for '{ref.np}'",
                    suggestion=suggestion,
                    suggestion_score=round(score, 3) if score else None,
                ))

        total_errors += len(errors)

        analyses.append(ClaimAnalysis(
            claim_number=claim.number,
            claim_text=claim.text,
            introductions=introductions,
            references=references,
            inherited_terms=sorted(list(inherited)),
            antecedent_errors=errors,
        ))

    return AnalyzeClaimsResponse(
        analyses=analyses,
        total_errors=total_errors,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": "en_core_web_trf"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
