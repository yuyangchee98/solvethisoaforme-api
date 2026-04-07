"""
Patent Claim NLP API

Extracts noun phrases, identifies introductions/references,
and checks antecedent basis using spaCy.
"""

from dotenv import load_dotenv
load_dotenv()

import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models import AnalyzeClaimsRequest, AnalyzeClaimsResponse
from core.analyzer import ClaimAnalyzer, register_default_validators
from sessions import init_db, close_db
from oa_response import oa_response_router
from oa_response.client_manager import get_client_manager
from reviewer import reviewer_router
from auth.db import init_auth_db
from auth.users import fastapi_users, auth_backend, current_active_user
from auth.schemas import UserRead, UserCreate, UserUpdate
from billing.router import router as billing_router
from patent_reader import router as patent_reader_router
from annotation_router import router as annotation_router

FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:4321")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    import asyncio

    # Startup
    register_default_validators()
    await init_db()
    await init_auth_db()

    # Start background cleanup loop for idle OA response clients
    cleanup_task = asyncio.create_task(get_client_manager().run_cleanup_loop())

    yield

    # Shutdown
    cleanup_task.cancel()
    await get_client_manager().shutdown()
    await close_db()


# Initialize FastAPI app
app = FastAPI(title="Patent Claim NLP API", lifespan=lifespan)

# Allow CORS for webapp
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth routers
app.include_router(
    fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"]
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)

# Billing router
app.include_router(billing_router)

# Include OA response router
app.include_router(oa_response_router)

# Include Reviewer router (side-by-side doc reader)
app.include_router(reviewer_router)

# Patent reader (public, no auth required)
app.include_router(patent_reader_router)

# Patent annotations (auth required)
app.include_router(annotation_router)

# Initialize analyzer
analyzer = ClaimAnalyzer()


@app.post("/analyze-claims", response_model=AnalyzeClaimsResponse)
def analyze_claims(request: AnalyzeClaimsRequest, user=Depends(current_active_user)):
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
