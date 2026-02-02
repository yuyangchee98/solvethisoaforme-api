"""
Patent Claim NLP API

Extracts noun phrases, identifies introductions/references,
and checks antecedent basis using spaCy.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models import AnalyzeClaimsRequest, AnalyzeClaimsResponse
from core.analyzer import ClaimAnalyzer, register_default_validators

# Initialize FastAPI app
app = FastAPI(title="Patent Claim NLP API")

# Allow CORS for webapp
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all validators at startup
register_default_validators()

# Initialize analyzer
analyzer = ClaimAnalyzer()


@app.post("/analyze-claims", response_model=AnalyzeClaimsResponse)
def analyze_claims(request: AnalyzeClaimsRequest):
    """Analyze all claims for antecedent basis errors.

    Args:
        request: Contains claims to analyze and optional validator selection

    Returns:
        Analysis results for all claims including errors found
    """
    # Run analysis with pluggable validators
    analyses = analyzer.analyze_claims(
        claims=request.claims,
        enabled_validators=request.enabled_validators,
    )

    # Calculate total errors
    total_errors = sum(len(analysis.antecedent_errors) for analysis in analyses)

    return AnalyzeClaimsResponse(
        analyses=analyses,
        total_errors=total_errors,
    )


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "model": "en_core_web_trf"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
