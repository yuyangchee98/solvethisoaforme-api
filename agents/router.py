"""FastAPI router for agent session endpoints."""

import json
import re
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from sessions import (
    get_session_manager,
    MessageRole,
    DocumentType,
    CreateSessionResponse,
    SessionResponse,
    SessionListResponse,
    DeleteSessionResponse,
    MessageResponse,
    MessageListResponse,
    UploadedDocumentResponse,
    WorkspaceFilesResponse,
)
from .orchestrator import run_orchestrator_turn

router = APIRouter(prefix="/agents", tags=["agents"])


def _sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal.

    Args:
        filename: The original filename

    Returns:
        Sanitized filename safe for filesystem use
    """
    # Remove path components
    filename = Path(filename).name
    # Replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Ensure it's not empty
    if not filename or filename.startswith("."):
        filename = "file_" + filename
    return filename


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session():
    """Create a new session with workspace directories."""
    manager = get_session_manager()
    session = await manager.create_session()

    return CreateSessionResponse(
        id=session.id,
        workspace_path=str(manager.get_workspace_path(session.id)),
        created_at=session.created_at,
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions():
    """List all sessions."""
    manager = get_session_manager()
    sessions = await manager.list_sessions()

    return SessionListResponse(
        sessions=[
            SessionResponse(
                id=s.id,
                status=s.status,
                created_at=s.created_at,
                updated_at=s.updated_at,
                workspace_path=str(manager.get_workspace_path(s.id)),
            )
            for s in sessions
        ]
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get a specific session."""
    manager = get_session_manager()
    session = await manager.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse(
        id=session.id,
        status=session.status,
        created_at=session.created_at,
        updated_at=session.updated_at,
        workspace_path=str(manager.get_workspace_path(session.id)),
    )


@router.delete("/sessions/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(session_id: str):
    """Delete a session and its workspace."""
    manager = get_session_manager()
    deleted = await manager.delete_session(session_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    return DeleteSessionResponse(status="deleted")


@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    content: str = Form(...),
    attachments: list[UploadFile] = File(default=[]),
):
    """Send a message to a session, optionally with file attachments.

    Returns an SSE stream of agent response events.

    Args:
        session_id: The session ID
        content: The message content
        attachments: Optional list of file attachments

    Returns:
        StreamingResponse with SSE events
    """
    manager = get_session_manager()

    # Verify session exists
    session = await manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Save the user message
    message = await manager.save_message(session_id, MessageRole.USER, content)

    # Process attachments
    workspace = manager.get_workspace_path(session_id)
    input_dir = workspace / "input"

    for attachment in attachments:
        if attachment.filename:
            # Sanitize filename
            safe_filename = _sanitize_filename(attachment.filename)

            # Save file
            file_path = input_dir / safe_filename
            file_content = await attachment.read()
            file_path.write_bytes(file_content)

            # Save document metadata
            await manager.save_uploaded_document(
                session_id=session_id,
                message_id=message.id,
                filename=safe_filename,
                original_filename=attachment.filename,
                document_type=DocumentType.OTHER,
                file_path=str(file_path),
                file_size=len(file_content),
            )

    # Get conversation history for context
    history = await manager.get_conversation_history(session_id)

    async def event_stream():
        """Generate SSE events from the orchestrator."""
        full_response = ""

        async for event in run_orchestrator_turn(workspace, history, content):
            if event["type"] == "text":
                full_response += event["content"]
            yield f"data: {json.dumps(event)}\n\n"

        # Save assistant response after streaming completes
        if full_response:
            await manager.save_message(
                session_id, MessageRole.ASSISTANT, full_response
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions/{session_id}/messages", response_model=MessageListResponse)
async def get_messages(session_id: str):
    """Get all messages for a session."""
    manager = get_session_manager()

    # Verify session exists
    session = await manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = await manager.get_messages(session_id)

    # Get attachments for each message
    response_messages = []
    for msg in messages:
        docs = await manager.get_documents_for_message(msg.id)
        response_messages.append(
            MessageResponse(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                created_at=msg.created_at,
                attachments=[
                    UploadedDocumentResponse(
                        id=doc.id,
                        filename=doc.filename,
                        original_filename=doc.original_filename,
                        document_type=doc.document_type,
                        file_size=doc.file_size,
                        created_at=doc.created_at,
                    )
                    for doc in docs
                ],
            )
        )

    return MessageListResponse(messages=response_messages)


@router.get("/sessions/{session_id}/files", response_model=WorkspaceFilesResponse)
async def list_workspace_files(session_id: str, path: str = ""):
    """List files in a session's workspace.

    Args:
        session_id: The session ID
        path: Optional subdirectory path
    """
    manager = get_session_manager()

    # Verify session exists
    session = await manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    files = manager.list_workspace_files(session_id, path)

    return WorkspaceFilesResponse(files=files)


@router.get("/sessions/{session_id}/files/{file_path:path}")
async def download_file(session_id: str, file_path: str):
    """Download a file from the session workspace.

    Args:
        session_id: The session ID
        file_path: Path to the file within the workspace
    """
    manager = get_session_manager()

    # Verify session exists
    session = await manager.get_session(session_id)
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

    return FileResponse(
        path=full_path,
        filename=full_path.name,
        media_type="application/octet-stream",
    )
