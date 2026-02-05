"""FastAPI router for agent session endpoints."""

import json
import re
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

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


# assistant-ui message format
class MessagePart(BaseModel):
    type: str
    text: str | None = None
    # For image parts (backend format)
    image: str | None = None  # base64 encoded image
    # For file parts (backend format)
    data: str | None = None  # base64 encoded file
    # Frontend format uses 'url' instead of data/image
    url: str | None = None
    # Accept both mimeType (backend) and mediaType (frontend)
    mimeType: str | None = None
    mediaType: str | None = None
    filename: str | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    parts: list[MessagePart] | None = None
    content: str | None = None  # Fallback for standard format
    id: str | int | None = None
    metadata: dict | None = None

    def get_text(self) -> str:
        """Extract text content from either parts or content field."""
        if self.parts:
            texts = [p.text for p in self.parts if p.type == "text" and p.text]
            return " ".join(texts)
        return self.content or ""

    def get_file_parts(self) -> list[MessagePart]:
        """Extract file/image parts from the message."""
        if not self.parts:
            return []
        return [p for p in self.parts if p.type in ("file", "image")]


class ChatRequest(BaseModel):
    messages: list[ChatMessage]

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
    request: ChatRequest,
):
    """Send a message to a session using Vercel AI SDK format.

    Returns an SSE stream of agent response events.

    Args:
        session_id: The session ID
        request: ChatRequest with messages array (Vercel AI SDK format)

    Returns:
        StreamingResponse with SSE events
    """
    manager = get_session_manager()

    # Verify session exists
    session = await manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Extract the last user message content
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")

    last_user_message = user_messages[-1]
    content = last_user_message.get_text()
    has_files = bool(last_user_message.get_file_parts())

    # Allow file-only uploads (no text content required if files are attached)
    if not content and not has_files:
        raise HTTPException(status_code=400, detail="Empty message content")

    # Get workspace path
    workspace = manager.get_workspace_path(session_id)
    input_dir = workspace / "input"

    # Process file attachments (base64 encoded)
    import base64
    uploaded_filenames = []
    for part in last_user_message.get_file_parts():
        filename = part.filename or f"file_{len(uploaded_filenames)}"
        filename = _sanitize_filename(filename)

        # Get the raw data from whichever field is populated
        # Frontend uses 'url', backend format uses 'data' or 'image'
        raw_data = None
        if part.type == "image":
            raw_data = part.image or part.url
        elif part.type == "file":
            raw_data = part.data or part.url

        if not raw_data:
            continue

        # Strip data URL prefix if present (e.g., "data:image/png;base64,...")
        if "," in raw_data:
            raw_data = raw_data.split(",", 1)[1]

        file_content = base64.b64decode(raw_data)

        # Save file to input directory
        file_path = input_dir / filename
        file_path.write_bytes(file_content)
        uploaded_filenames.append(filename)

    # Build message content with file upload notification for the agent
    agent_content = content
    if uploaded_filenames:
        file_list = ", ".join(uploaded_filenames)
        agent_content = f"[Uploaded files saved to input/: {file_list}]\n\n{content}"

    # Save the user message (include file info if text content is empty)
    saved_content = content if content else f"[Uploaded: {', '.join(uploaded_filenames)}]"
    await manager.save_message(session_id, MessageRole.USER, saved_content)

    # Get conversation history for context
    history = await manager.get_conversation_history(session_id)

    async def event_stream():
        """Generate SSE events from orchestrator output."""
        import uuid as uuid_mod
        message_id = str(uuid_mod.uuid4())
        full_response = ""

        # Start message event
        yield f"data: {json.dumps({'type': 'start', 'messageId': message_id})}\n\n"

        async for event in run_orchestrator_turn(workspace, history, agent_content):
            # Accumulate text for saving
            if event.get("type") == "text-delta":
                full_response += event.get("delta", "")

            yield f"data: {json.dumps(event)}\n\n"

        # Signal end of stream
        yield "data: [DONE]\n\n"

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
