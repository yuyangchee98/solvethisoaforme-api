"""Session management module for the conversational agent system."""

from .database import init_db, close_db, get_db, get_data_path
from .models import (
    SessionStatus,
    DocumentType,
    MessageRole,
    Session,
    Message,
    UploadedDocument,
    CreateSessionResponse,
    SessionResponse,
    SessionListResponse,
    DeleteSessionResponse,
    MessageResponse,
    MessageListResponse,
    UploadedDocumentResponse,
    WorkspaceFileInfo,
    WorkspaceFilesResponse,
)
from .manager import SessionManager, get_session_manager

__all__ = [
    # Database
    "init_db",
    "close_db",
    "get_db",
    "get_data_path",
    # Enums
    "SessionStatus",
    "DocumentType",
    "MessageRole",
    # Domain models
    "Session",
    "Message",
    "UploadedDocument",
    # Request/Response models
    "CreateSessionResponse",
    "SessionResponse",
    "SessionListResponse",
    "DeleteSessionResponse",
    "MessageResponse",
    "MessageListResponse",
    "UploadedDocumentResponse",
    "WorkspaceFileInfo",
    "WorkspaceFilesResponse",
    # Manager
    "SessionManager",
    "get_session_manager",
]
