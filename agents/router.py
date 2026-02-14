"""FastAPI router for agent session endpoints."""

import json
import re
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
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
from processors import get_processor_registry
from auth.users import current_active_user
from auth.models import User
from .orchestrator import run_orchestrator_turn


async def require_subscription(user: User = Depends(current_active_user)) -> User:
    """Dependency that requires an active subscription."""
    if user.subscription_status not in ("active", "trialing"):
        raise HTTPException(
            status_code=403,
            detail="Active subscription required",
        )
    return user


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
async def create_session(user: User = Depends(require_subscription)):
    """Create a new session with workspace directories."""
    manager = get_session_manager()
    session = await manager.create_session(user_id=str(user.id))

    return CreateSessionResponse(
        id=session.id,
        workspace_path=str(manager.get_workspace_path(session.id)),
        created_at=session.created_at,
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(user: User = Depends(require_subscription)):
    """List all sessions for the current user."""
    manager = get_session_manager()
    sessions = await manager.list_sessions(user_id=str(user.id))

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
async def get_session(session_id: str, user: User = Depends(require_subscription)):
    """Get a specific session."""
    manager = get_session_manager()
    session = await manager.get_session(session_id, user_id=str(user.id))

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
async def delete_session(session_id: str, user: User = Depends(require_subscription)):
    """Delete a session and its workspace."""
    manager = get_session_manager()
    deleted = await manager.delete_session(session_id, user_id=str(user.id))

    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    return DeleteSessionResponse(status="deleted")


@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    request: ChatRequest,
    user: User = Depends(require_subscription),
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

    # Verify session exists and belongs to user
    session = await manager.get_session(session_id, user_id=str(user.id))
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
    uploaded_files = []  # For native Claude content blocks
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
        # and extract the actual media type from it (more reliable than
        # the mediaType field, which upstream libs may hardcode incorrectly)
        base64_data = raw_data
        data_url_media_type = None
        if raw_data.startswith("data:") and "," in raw_data:
            header, base64_data = raw_data.split(",", 1)
            # header is e.g. "data:image/jpeg;base64"
            mime_part = header.removeprefix("data:").split(";")[0]
            if "/" in mime_part:
                data_url_media_type = mime_part

        file_content = base64.b64decode(base64_data)

        # Save file to input directory
        file_path = input_dir / filename
        file_path.write_bytes(file_content)
        uploaded_filenames.append(filename)

        # Use the actual media type from the data URL (ground truth),
        # falling back to the declared fields only if there's no data URL
        media_type = data_url_media_type or part.mimeType or part.mediaType or "application/octet-stream"
        file_info = {
            "filename": filename,
            "path": str(file_path),
            "media_type": media_type,
            "data": base64_data,  # Keep base64 for Claude content blocks
        }

        # Run document processor if one matches (e.g. .docx → markdown)
        result = get_processor_registry().process_if_needed(file_path, media_type, input_dir)
        if result is not None:
            if result.extracted_text:
                file_info["extracted_text"] = result.extracted_text
            if result.error:
                import logging
                logging.getLogger(__name__).warning(
                    "Processor error for %s: %s", filename, result.error
                )

        uploaded_files.append(file_info)

    # Build message content with file upload notification for the agent
    agent_content = content
    if uploaded_filenames:
        file_list = ", ".join(uploaded_filenames)
        extracted_files = [f["filename"].replace(Path(f["filename"]).suffix, ".extracted.md")
                           for f in uploaded_files if "extracted_text" in f]
        extra = ""
        if extracted_files:
            extra = f" Extracted text also saved as: {', '.join(extracted_files)}."
        agent_content = f"[Uploaded files saved to input/: {file_list}.{extra}]\n\n{content}"

    # Save the user message (include file info if text content is empty)
    saved_content = content if content else f"[Uploaded: {', '.join(uploaded_filenames)}]"
    saved_message = await manager.save_message(session_id, MessageRole.USER, saved_content)

    # Save document metadata for each uploaded file
    for file_info in uploaded_files:
        file_on_disk = input_dir / file_info["filename"]
        await manager.save_uploaded_document(
            session_id=session_id,
            message_id=saved_message.id,
            filename=file_info["filename"],
            original_filename=file_info["filename"],
            document_type=DocumentType.OTHER,
            file_path=str(file_on_disk),
            file_size=file_on_disk.stat().st_size,
        )

    # Get conversation history for context
    history = await manager.get_conversation_history(session_id)

    async def event_stream():
        """Generate SSE events from orchestrator output."""
        import uuid as uuid_mod
        message_id = str(uuid_mod.uuid4())
        full_response = ""
        tool_calls: dict[str, dict] = {}  # toolCallId -> data

        # Start message event
        yield f"data: {json.dumps({'type': 'start', 'messageId': message_id})}\n\n"

        async for event in run_orchestrator_turn(workspace, history, agent_content, uploaded_files):
            event_type = event.get("type")

            # Accumulate text for saving
            if event_type == "text-delta":
                full_response += event.get("delta", "")

            # Capture tool call input
            elif event_type == "tool-input-available":
                tool_call_id = event.get("toolCallId")
                if tool_call_id:
                    tool_calls[tool_call_id] = {
                        "toolCallId": tool_call_id,
                        "toolName": event.get("toolName"),
                        "input": event.get("input"),
                        "output": None,
                    }

            # Capture tool call output
            elif event_type == "tool-output-available":
                tool_call_id = event.get("toolCallId")
                if tool_call_id and tool_call_id in tool_calls:
                    tool_calls[tool_call_id]["output"] = event.get("output")

            yield f"data: {json.dumps(event)}\n\n"

        # Signal end of stream
        yield "data: [DONE]\n\n"

        # Save assistant response after streaming completes
        if full_response or tool_calls:
            await manager.save_message(
                session_id,
                MessageRole.ASSISTANT,
                full_response,
                tool_calls=list(tool_calls.values()) if tool_calls else None,
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
async def get_messages(session_id: str, user: User = Depends(require_subscription)):
    """Get all messages for a session."""
    manager = get_session_manager()

    # Verify session exists and belongs to user
    session = await manager.get_session(session_id, user_id=str(user.id))
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
                tool_calls=msg.tool_calls,
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
async def list_workspace_files(
    session_id: str, path: str = "", user: User = Depends(require_subscription)
):
    """List files in a session's workspace.

    Args:
        session_id: The session ID
        path: Optional subdirectory path
    """
    manager = get_session_manager()

    # Verify session exists and belongs to user
    session = await manager.get_session(session_id, user_id=str(user.id))
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    files = manager.list_workspace_files(session_id, path)

    return WorkspaceFilesResponse(files=files)


@router.get("/sessions/{session_id}/files/{file_path:path}")
async def download_file(
    session_id: str, file_path: str, user: User = Depends(require_subscription)
):
    """Download a file from the session workspace.

    Args:
        session_id: The session ID
        file_path: Path to the file within the workspace
    """
    manager = get_session_manager()

    # Verify session exists and belongs to user
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

    return FileResponse(
        path=full_path,
        filename=full_path.name,
        media_type="application/octet-stream",
    )
