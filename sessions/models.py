"""Pydantic models for session data."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    """Status of a session."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class DocumentType(str, Enum):
    """Type of uploaded document."""

    CLAIM = "claim"
    PRIOR_ART = "prior_art"
    OTHER = "other"


class MessageRole(str, Enum):
    """Role of a message sender."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# Domain models


class Session(BaseModel):
    """A conversation session."""

    id: str
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime
    updated_at: datetime


class ToolCall(BaseModel):
    """A tool call within a message."""

    toolCallId: str
    toolName: str
    input: dict
    output: str | None = None


class Message(BaseModel):
    """A message in a session."""

    id: int
    session_id: str
    role: MessageRole
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    created_at: datetime


class UploadedDocument(BaseModel):
    """An uploaded document associated with a session."""

    id: int
    session_id: str
    message_id: int | None = None
    filename: str
    original_filename: str
    document_type: DocumentType
    file_path: str
    file_size: int
    created_at: datetime


# Request/Response models


class CreateSessionResponse(BaseModel):
    """Response when creating a new session."""

    id: str
    workspace_path: str
    created_at: datetime


class SessionResponse(BaseModel):
    """Response containing session details."""

    id: str
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    workspace_path: str


class SessionListResponse(BaseModel):
    """Response containing a list of sessions."""

    sessions: list[SessionResponse]


class DeleteSessionResponse(BaseModel):
    """Response when deleting a session."""

    status: str = "deleted"


class MessageResponse(BaseModel):
    """Response containing a message."""

    id: int
    role: MessageRole
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    created_at: datetime
    attachments: list["UploadedDocumentResponse"] = Field(default_factory=list)


class MessageListResponse(BaseModel):
    """Response containing a list of messages."""

    messages: list[MessageResponse]


class UploadedDocumentResponse(BaseModel):
    """Response containing uploaded document metadata."""

    id: int
    filename: str
    original_filename: str
    document_type: DocumentType
    file_size: int
    created_at: datetime


class WorkspaceFileInfo(BaseModel):
    """Information about a file in the workspace."""

    name: str
    path: str
    size: int
    is_directory: bool


class WorkspaceFilesResponse(BaseModel):
    """Response containing workspace files."""

    files: list[WorkspaceFileInfo]


# Update forward references
MessageResponse.model_rebuild()
