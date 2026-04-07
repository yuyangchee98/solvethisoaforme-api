"""FastAPI router for the Reviewer side-by-side reader.

Endpoints:
- POST   /reviewer/sessions                       Create a reviewer session
- POST   /reviewer/sessions/{id}/strategy         Upload a strategy .md (multipart)
- POST   /reviewer/sessions/{id}/strategy-text    Upload a strategy via pasted text (JSON)
- POST   /reviewer/sessions/{id}/sources          Upload a source PDF (multipart)
- GET    /reviewer/sessions/{id}/files            Recursively list all workspace files
- GET    /reviewer/sessions/{id}/files/{path}     Fetch a file by relative path
- DELETE /reviewer/sessions/{id}                  Delete a reviewer-kind session

File GET endpoints accept *any* session the user owns — that's how the
"OA handoff" path works (the reviewer opens an OA session's files
directly without copying them into a new reviewer session).
"""

import logging
import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from sessions import (
    get_session_manager,
    CreateSessionResponse,
    DeleteSessionResponse,
    WorkspaceFilesResponse,
)
from processors import get_processor_registry
from auth.users import current_active_user
from auth.models import User

log = logging.getLogger(__name__)

router = APIRouter(prefix="/reviewer", tags=["reviewer"])


async def require_subscription(user: User = Depends(current_active_user)) -> User:
    """Dependency that requires an active subscription."""
    if user.subscription_status not in ("active", "trialing"):
        raise HTTPException(
            status_code=403,
            detail="Active subscription required",
        )
    return user


def _sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal."""
    filename = Path(filename).name
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    if not filename or filename.startswith("."):
        filename = "file_" + filename
    return filename


# ── Session CRUD ──────────────────────────────────────────────────────


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(user: User = Depends(require_subscription)):
    """Create a new reviewer-kind session with strategy/ and sources/ subdirs."""
    manager = get_session_manager()
    session = await manager.create_session(user_id=str(user.id), kind="reviewer")
    return CreateSessionResponse(
        id=session.id,
        workspace_path=str(manager.get_workspace_path(session.id)),
        created_at=session.created_at,
    )


@router.delete("/sessions/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(
    session_id: str, user: User = Depends(require_subscription)
):
    """Delete a reviewer session and its workspace.

    Only reviewer-kind sessions can be deleted through this endpoint;
    OA sessions must be deleted via the OA router.
    """
    manager = get_session_manager()
    session = await manager.get_session(session_id, user_id=str(user.id))
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.kind != "reviewer":
        raise HTTPException(
            status_code=400,
            detail="This endpoint only deletes reviewer sessions",
        )

    deleted = await manager.delete_session(session_id, user_id=str(user.id))
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return DeleteSessionResponse(status="deleted")


# ── Uploads ───────────────────────────────────────────────────────────


class StrategyUploadResponse(BaseModel):
    filename: str
    path: str


class SourceUploadResponse(BaseModel):
    filename: str
    path: str
    extracted: bool
    extracted_path: str | None = None
    pages: int | None = None
    error: str | None = None


class StrategyTextUpload(BaseModel):
    filename: str
    content: str


def _require_reviewer_session(manager, session, session_id: str):
    """Validate a session exists, is owned, and is reviewer-kind for upload."""
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.kind != "reviewer":
        raise HTTPException(
            status_code=400,
            detail="Uploads are only allowed on reviewer-kind sessions",
        )


_STRATEGY_ACCEPTED_EXTS = (".md", ".mdx", ".docx", ".pdf")


@router.post(
    "/sessions/{session_id}/strategy", response_model=StrategyUploadResponse
)
async def upload_strategy(
    session_id: str,
    file: UploadFile = File(...),
    user: User = Depends(require_subscription),
):
    """Upload a strategy document.

    Accepts .md/.mdx directly, or .docx/.pdf which are run through the
    processor registry to produce an `.extracted.md` in the strategy
    directory. The extracted markdown becomes the primary doc used by
    the reader; the original is also kept so the user can download it.
    """
    manager = get_session_manager()
    session = await manager.get_session(session_id, user_id=str(user.id))
    _require_reviewer_session(manager, session, session_id)

    filename = _sanitize_filename(file.filename or "strategy.md")
    lower = filename.lower()
    if not lower.endswith(_STRATEGY_ACCEPTED_EXTS):
        raise HTTPException(
            status_code=400,
            detail=(
                "Strategy doc must be one of: "
                + ", ".join(_STRATEGY_ACCEPTED_EXTS)
            ),
        )

    workspace = manager.get_workspace_path(session_id)
    strategy_dir = workspace / "strategy"
    strategy_dir.mkdir(parents=True, exist_ok=True)

    file_path = strategy_dir / filename
    content = await file.read()
    file_path.write_bytes(content)

    # For non-markdown formats, run the processor to produce the
    # extracted markdown sibling. The reader's classifySources picks
    # the .md file as the primary doc via its "inside strategy/"
    # heuristic.
    if not lower.endswith((".md", ".mdx")):
        media_type = file.content_type or "application/octet-stream"
        result = get_processor_registry().process_if_needed(
            file_path, media_type, strategy_dir
        )
        if result is None or not result.extracted_text:
            err = (result.error if result else None) or (
                f"No processor available for {filename}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract markdown from {filename}: {err}",
            )
        # The classifier prefers whichever .md lives in strategy/ — the
        # processor writes `{stem}.extracted.md` next to the original,
        # which is what we want.
        primary_path = result.extracted_path or file_path
        return StrategyUploadResponse(
            filename=primary_path.name,
            path=str(primary_path.relative_to(workspace)),
        )

    return StrategyUploadResponse(
        filename=filename,
        path=str(file_path.relative_to(workspace)),
    )


@router.post(
    "/sessions/{session_id}/strategy-text", response_model=StrategyUploadResponse
)
async def upload_strategy_text(
    session_id: str,
    body: StrategyTextUpload,
    user: User = Depends(require_subscription),
):
    """Upload a strategy doc from pasted markdown text."""
    manager = get_session_manager()
    session = await manager.get_session(session_id, user_id=str(user.id))
    _require_reviewer_session(manager, session, session_id)

    filename = _sanitize_filename(body.filename or "strategy.md")
    if not filename.lower().endswith((".md", ".mdx")):
        filename = filename + ".md"

    workspace = manager.get_workspace_path(session_id)
    strategy_dir = workspace / "strategy"
    strategy_dir.mkdir(parents=True, exist_ok=True)

    file_path = strategy_dir / filename
    file_path.write_text(body.content, encoding="utf-8")

    return StrategyUploadResponse(
        filename=filename,
        path=str(file_path.relative_to(workspace)),
    )


@router.post(
    "/sessions/{session_id}/sources", response_model=SourceUploadResponse
)
async def upload_source(
    session_id: str,
    file: UploadFile = File(...),
    user: User = Depends(require_subscription),
):
    """Upload a source document (PDF). Runs the processor pipeline for extraction."""
    manager = get_session_manager()
    session = await manager.get_session(session_id, user_id=str(user.id))
    _require_reviewer_session(manager, session, session_id)

    filename = _sanitize_filename(file.filename or "source.pdf")
    workspace = manager.get_workspace_path(session_id)
    sources_dir = workspace / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    file_path = sources_dir / filename
    content = await file.read()
    file_path.write_bytes(content)

    # Run extraction (PDF → .extracted.md, OCR fallback, etc.)
    media_type = file.content_type or "application/octet-stream"
    result = get_processor_registry().process_if_needed(
        file_path, media_type, sources_dir
    )

    extracted = False
    extracted_path = None
    pages = None
    error = None
    if result is not None:
        if result.error:
            error = result.error
            log.warning("Processor error for %s: %s", filename, result.error)
        if result.extracted_text and result.extracted_path is not None:
            extracted = True
            extracted_path = str(result.extracted_path.relative_to(workspace))
        if result.metadata:
            pages = result.metadata.get("pages")

    return SourceUploadResponse(
        filename=filename,
        path=str(file_path.relative_to(workspace)),
        extracted=extracted,
        extracted_path=extracted_path,
        pages=pages,
        error=error,
    )


# ── File listing & fetching (works on ANY session the user owns) ──────


@router.get("/sessions/{session_id}/files", response_model=WorkspaceFilesResponse)
async def list_files(
    session_id: str, user: User = Depends(require_subscription)
):
    """Recursively list every file in a session's workspace.

    Works on both reviewer-kind sessions (their strategy/ + sources/ dirs)
    and OA-kind sessions (their input/ + rejections/ dirs) — as long as
    the current user owns the session.
    """
    manager = get_session_manager()
    session = await manager.get_session(session_id, user_id=str(user.id))
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    files = manager.list_workspace_files_recursive(session_id)
    return WorkspaceFilesResponse(files=files)


@router.get("/sessions/{session_id}/files/{file_path:path}")
async def download_file(
    session_id: str,
    file_path: str,
    user: User = Depends(require_subscription),
):
    """Serve a file from a session's workspace.

    Works on any session the user owns (reviewer or OA) so the OA-handoff
    flow can read files straight from the OA session without copying.
    """
    manager = get_session_manager()
    session = await manager.get_session(session_id, user_id=str(user.id))
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    workspace = manager.get_workspace_path(session_id)
    full_path = workspace / file_path

    # Path traversal protection
    try:
        full_path.resolve().relative_to(workspace.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not full_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    # Best-effort content type — .md served as text so the frontend's
    # plain `response.text()` call works, PDFs as application/pdf,
    # everything else falls through to octet-stream.
    suffix = full_path.suffix.lower()
    if suffix in (".md", ".mdx", ".txt"):
        media_type = "text/plain; charset=utf-8"
    elif suffix == ".pdf":
        media_type = "application/pdf"
    else:
        media_type = "application/octet-stream"

    return FileResponse(
        path=full_path,
        filename=full_path.name,
        media_type=media_type,
    )
