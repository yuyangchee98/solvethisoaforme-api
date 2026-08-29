"""Session manager for CRUD operations and filesystem management."""

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .database import get_db, get_data_path
from .models import (
    Session,
    SessionStatus,
    Message,
    MessagePart,
    MessageRole,
    ToolCall,
    UploadedDocument,
    DocumentType,
    WorkspaceFileInfo,
)

# Singleton instance
_session_manager: "SessionManager | None" = None


class SessionManager:
    """Manages session lifecycle, messages, and workspace files."""

    # Default workspace subdirectories per session kind
    WORKSPACE_DIRS = ["input", "rejections", "prior_art_working"]

    async def create_session(
        self,
        user_id: str | None = None,
        kind: str = "oa_agent",
        subdirs: list[str] | None = None,
    ) -> Session:
        """Create a new session with workspace directories.

        Args:
            user_id: Optional user ID to associate with the session
            kind: Session kind, persisted in the session_kind column.
            subdirs: Explicit list of workspace subdirectories to create.
                Defaults to WORKSPACE_DIRS.

        Returns:
            The newly created session
        """
        db = await get_db()

        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Insert session record
        await db.execute(
            """
            INSERT INTO sessions (id, status, created_at, updated_at, user_id, session_kind)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, SessionStatus.ACTIVE.value, now, now, user_id, kind),
        )
        await db.commit()

        # Create workspace directories
        workspace = self.get_workspace_path(session_id)
        workspace.mkdir(parents=True, exist_ok=True)

        if subdirs is None:
            subdirs = self.WORKSPACE_DIRS

        for subdir in subdirs:
            (workspace / subdir).mkdir(exist_ok=True)

        return Session(
            id=session_id,
            status=SessionStatus.ACTIVE,
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
            kind=kind,
        )

    async def get_session(
        self, session_id: str, user_id: str | None = None
    ) -> Session | None:
        """Get a session by ID, optionally verifying ownership.

        Args:
            session_id: The session ID
            user_id: If provided, verify the session belongs to this user

        Returns:
            The session if found (and owned by user_id if given), None otherwise
        """
        db = await get_db()

        if user_id:
            cursor = await db.execute(
                "SELECT id, status, created_at, updated_at, session_kind FROM sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            )
        else:
            cursor = await db.execute(
                "SELECT id, status, created_at, updated_at, session_kind FROM sessions WHERE id = ?",
                (session_id,),
            )
        row = await cursor.fetchone()

        if row is None:
            return None

        return Session(
            id=row[0],
            status=SessionStatus(row[1]),
            created_at=datetime.fromisoformat(row[2]),
            updated_at=datetime.fromisoformat(row[3]),
            kind=row[4] or "oa_agent",
        )

    async def list_sessions(
        self, user_id: str | None = None, kind: str | None = None
    ) -> list[Session]:
        """List sessions, optionally filtered by user and/or kind.

        Args:
            user_id: If provided, only return sessions for this user
            kind: If provided, only return sessions with this session_kind

        Returns:
            List of sessions ordered by creation date (newest first)
        """
        db = await get_db()

        where_clauses: list[str] = []
        params: list[str] = []
        if user_id:
            where_clauses.append("user_id = ?")
            params.append(user_id)
        if kind:
            where_clauses.append("session_kind = ?")
            params.append(kind)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        query = f"""
            SELECT id, status, created_at, updated_at, session_kind
            FROM sessions
            {where_sql}
            ORDER BY created_at DESC
        """

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

        return [
            Session(
                id=row[0],
                status=SessionStatus(row[1]),
                created_at=datetime.fromisoformat(row[2]),
                updated_at=datetime.fromisoformat(row[3]),
                kind=row[4] or "oa_agent",
            )
            for row in rows
        ]

    async def delete_session(
        self, session_id: str, user_id: str | None = None
    ) -> bool:
        """Delete a session and its workspace.

        Args:
            session_id: The session ID to delete
            user_id: If provided, verify ownership before deleting

        Returns:
            True if session was deleted, False if not found
        """
        db = await get_db()

        # Check if session exists (with ownership check if user_id provided)
        session = await self.get_session(session_id, user_id=user_id)
        if session is None:
            return False

        # Delete from database (cascades to messages and documents)
        await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await db.commit()

        # Delete workspace directory
        workspace = self.get_workspace_path(session_id)
        if workspace.exists():
            shutil.rmtree(workspace)

        return True

    async def save_cli_session_id(self, session_id: str, cli_session_id: str) -> None:
        """Save the CLI subprocess session ID for conversation resume.

        Args:
            session_id: Our application session ID
            cli_session_id: The Claude CLI's internal session ID
        """
        db = await get_db()
        await db.execute(
            "UPDATE sessions SET cli_session_id = ? WHERE id = ?",
            (cli_session_id, session_id),
        )
        await db.commit()

    async def get_cli_session_id(self, session_id: str) -> str | None:
        """Get the CLI subprocess session ID for conversation resume.

        Args:
            session_id: Our application session ID

        Returns:
            The CLI session ID if saved, None otherwise
        """
        db = await get_db()
        cursor = await db.execute(
            "SELECT cli_session_id FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def save_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        tool_calls: list[dict] | None = None,
        parts: list[dict] | None = None,
    ) -> Message:
        """Save a message to a session.

        Args:
            session_id: The session ID
            role: The message role (user/assistant/system)
            content: The message content
            tool_calls: Optional list of tool call dicts
            parts: Optional ordered list of parts (text segments + tool call refs)

        Returns:
            The saved message
        """
        db = await get_db()

        now = datetime.now(timezone.utc).isoformat()
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None
        parts_json = json.dumps(parts) if parts else None

        cursor = await db.execute(
            """
            INSERT INTO messages (session_id, role, content, tool_calls, parts, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, role.value, content, tool_calls_json, parts_json, now),
        )
        await db.commit()

        # Update session updated_at
        await db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )
        await db.commit()

        # Convert tool_calls dicts to ToolCall models
        tool_call_models = []
        if tool_calls:
            tool_call_models = [ToolCall(**tc) for tc in tool_calls]

        # Convert parts dicts to MessagePart models
        part_models = None
        if parts:
            part_models = [MessagePart(**p) for p in parts]

        return Message(
            id=cursor.lastrowid,
            session_id=session_id,
            role=role,
            content=content,
            tool_calls=tool_call_models,
            parts=part_models,
            created_at=datetime.fromisoformat(now),
        )

    async def get_messages(self, session_id: str) -> list[Message]:
        """Get all messages for a session.

        Args:
            session_id: The session ID

        Returns:
            List of messages ordered by creation date
        """
        db = await get_db()

        cursor = await db.execute(
            """
            SELECT id, session_id, role, content, tool_calls, parts, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()

        messages = []
        for row in rows:
            # Parse tool_calls JSON if present
            tool_calls_data = row[4]
            tool_call_models = []
            if tool_calls_data:
                tool_calls_list = json.loads(tool_calls_data)
                tool_call_models = [ToolCall(**tc) for tc in tool_calls_list]

            # Parse parts JSON if present
            parts_data = row[5]
            part_models = None
            if parts_data:
                parts_list = json.loads(parts_data)
                part_models = [MessagePart(**p) for p in parts_list]

            messages.append(
                Message(
                    id=row[0],
                    session_id=row[1],
                    role=MessageRole(row[2]),
                    content=row[3],
                    tool_calls=tool_call_models,
                    parts=part_models,
                    created_at=datetime.fromisoformat(row[6]),
                )
            )

        return messages

    async def save_uploaded_document(
        self,
        session_id: str,
        message_id: int | None,
        filename: str,
        original_filename: str,
        document_type: DocumentType,
        file_path: str,
        file_size: int,
    ) -> UploadedDocument:
        """Save metadata about an uploaded document.

        Args:
            session_id: The session ID
            message_id: Optional associated message ID
            filename: The stored filename (may be sanitized)
            original_filename: The original uploaded filename
            document_type: Type of document
            file_path: Path where file is stored
            file_size: Size of file in bytes

        Returns:
            The saved document metadata
        """
        db = await get_db()

        now = datetime.now(timezone.utc).isoformat()

        cursor = await db.execute(
            """
            INSERT INTO uploaded_documents
            (session_id, message_id, filename, original_filename, document_type, file_path, file_size, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                message_id,
                filename,
                original_filename,
                document_type.value,
                file_path,
                file_size,
                now,
            ),
        )
        await db.commit()

        return UploadedDocument(
            id=cursor.lastrowid,
            session_id=session_id,
            message_id=message_id,
            filename=filename,
            original_filename=original_filename,
            document_type=document_type,
            file_path=file_path,
            file_size=file_size,
            created_at=datetime.fromisoformat(now),
        )

    async def get_uploaded_documents(self, session_id: str) -> list[UploadedDocument]:
        """Get all uploaded documents for a session.

        Args:
            session_id: The session ID

        Returns:
            List of uploaded documents
        """
        db = await get_db()

        cursor = await db.execute(
            """
            SELECT id, session_id, message_id, filename, original_filename,
                   document_type, file_path, file_size, created_at
            FROM uploaded_documents
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()

        return [
            UploadedDocument(
                id=row[0],
                session_id=row[1],
                message_id=row[2],
                filename=row[3],
                original_filename=row[4],
                document_type=DocumentType(row[5]),
                file_path=row[6],
                file_size=row[7],
                created_at=datetime.fromisoformat(row[8]),
            )
            for row in rows
        ]

    async def get_documents_for_message(
        self, message_id: int
    ) -> list[UploadedDocument]:
        """Get uploaded documents associated with a specific message.

        Args:
            message_id: The message ID

        Returns:
            List of uploaded documents for the message
        """
        db = await get_db()

        cursor = await db.execute(
            """
            SELECT id, session_id, message_id, filename, original_filename,
                   document_type, file_path, file_size, created_at
            FROM uploaded_documents
            WHERE message_id = ?
            ORDER BY created_at ASC
            """,
            (message_id,),
        )
        rows = await cursor.fetchall()

        return [
            UploadedDocument(
                id=row[0],
                session_id=row[1],
                message_id=row[2],
                filename=row[3],
                original_filename=row[4],
                document_type=DocumentType(row[5]),
                file_path=row[6],
                file_size=row[7],
                created_at=datetime.fromisoformat(row[8]),
            )
            for row in rows
        ]

    def list_workspace_files(
        self, session_id: str, subpath: str = ""
    ) -> list[WorkspaceFileInfo]:
        """List files in a session's workspace.

        Args:
            session_id: The session ID
            subpath: Optional subdirectory path

        Returns:
            List of file/directory info
        """
        workspace = self.get_workspace_path(session_id)
        target = workspace / subpath if subpath else workspace

        if not target.exists() or not target.is_dir():
            return []

        # Ensure we're not escaping the workspace
        try:
            target.resolve().relative_to(workspace.resolve())
        except ValueError:
            return []

        files = []
        for item in sorted(target.iterdir()):
            files.append(
                WorkspaceFileInfo(
                    name=item.name,
                    path=str(item.relative_to(workspace)),
                    size=item.stat().st_size if item.is_file() else 0,
                    is_directory=item.is_dir(),
                )
            )

        return files


    def get_workspace_path(self, session_id: str) -> Path:
        """Get the workspace path for a session.

        Args:
            session_id: The session ID

        Returns:
            Path to the session's workspace directory
        """
        return get_data_path() / "sessions" / session_id


def get_session_manager() -> SessionManager:
    """Get the singleton session manager instance.

    Returns:
        The session manager instance
    """
    global _session_manager

    if _session_manager is None:
        _session_manager = SessionManager()

    return _session_manager
